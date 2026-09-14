"""Golden regression: the reference policy still scores what it scored.

`tasks/task04/dev/oracle_ref/` is the checkpoint the reference PPO recipe produced
inside its 1 h budget, and the dance entry in `config.MOTION_THRESHOLDS`
records what it measured on the hidden seeds under the shipped thresholds. If a
change to the physics, observation, metrics or reward moves that number, one of two things is
true and both need a human: the harness drifted, or the thresholds need
recalibrating.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from harness import config, evaluator, export, motion, robot, scorer  # noqa: E402

ORACLE = "tasks/task04/dev/oracle_ref"
ROBOT_DIR = "third_party/motiontrack/unitree_g1"
NPZ = "third_party/motiontrack/motions/dance1_subject2.npz"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(NPZ) and os.path.exists(ORACLE)),
    reason="run `make sim-motiontrack` first")


@pytest.fixture(scope="module")
def report():
    return scorer.score_submission(ORACLE, ROBOT_DIR, NPZ)


def test_the_reference_policy_is_a_valid_submission():
    assert export.validate(ORACLE) == []
    meta = export.load_meta(ORACLE)
    assert meta["n_params"] < config.__dict__.get("MAX_PARAMS", 10_000_000)
    assert meta["minutes"] <= 61, "the reference must respect the budget it sets"


def test_the_reference_still_scores_what_it_scored(report):
    expected = config.thresholds_for_motion(NPZ).oracle_ref_score
    assert report["reward"] == pytest.approx(expected, abs=0.02)
    assert not report["gated"]
    assert all(report["gate"].values())


def test_the_reference_actually_tracks(report):
    """Not just 'scores well' — the underlying physical claims."""
    assert report["tracking_score"] > 0.5
    assert report["mpjpe_m"] < 0.12, "mean tracked-body error, in metres"
    assert report["survived_s"] > 10.0, "of a 20 s clip, under hidden randomization"


def test_the_scale_has_headroom_at_both_ends(report):
    """A benchmark where the reference already scores 1.0 measures nothing above
    it, and one where it scores 0 measures nothing below."""
    assert 0.3 < report["reward"] < 0.85
    assert config.thresholds_for_motion(NPZ).null_ref_score == 0.0


def test_the_reference_beats_every_degenerate_policy(report):
    """The ordering that makes the reward a gradient rather than a coin flip."""
    null_score = config.thresholds_for_motion(NPZ).null_ref_score
    assert report["reward"] > 10 * max(null_score, 0.01)


def test_scoring_the_reference_twice_agrees(report):
    """CPU evaluation is bit-reproducible; this is what the determinism gate rests
    on, checked against a real policy rather than a constant one."""
    again = scorer.score_submission(ORACLE, ROBOT_DIR, NPZ)
    assert again["reward"] == report["reward"]
    assert again["episodes"] == report["episodes"]


def test_the_reference_survives_the_full_clip_on_a_gentle_seed():
    """Under a design seed the policy should complete the clip, which is the
    evidence that the hidden-seed shortfall is randomization and not incapacity."""
    clip = motion.Motion.load(NPZ)
    policy = evaluator.OnnxPolicy(ORACLE)
    log = evaluator.run_episode(robot.build(ROBOT_DIR), clip, policy, seed=101,
                                history=policy.history)
    summary = log.summary()
    assert summary["survived_s"] > 15.0
    assert summary["tracking_score"] > 0.7
