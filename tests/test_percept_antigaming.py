"""Heavy task06 battery, baseline, and malformed-submission regression tests."""
import os
import textwrap

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest

# The module-scoped `bat` fixture renders the FULL eval battery (~7 min on a CI
# runner) and nearly every test here consumes it, so excluding individual tests
# would still pay the fixture. Whole module -> nightly lane.
pytestmark = pytest.mark.heavy

from harness import battery, reference, scorer, spec


@pytest.fixture(scope="module")
def bat():
    return battery.build_battery()


def _write(directory, body):
    directory.mkdir(exist_ok=True)
    (directory / "estimator.py").write_text(textwrap.dedent(body))


def test_full_battery_has_requested_size_and_occlusion(bat):
    assert len(bat.stage_a) == 100
    assert len(bat.episodes) == 10
    assert len(bat.stage_b) == 600
    flags = np.asarray([frame.occluded for frame in bat.stage_b])
    assert flags.mean() >= 0.10


def test_reference_beats_naive_baseline(bat, tmp_path):
    ref = tmp_path / "reference"
    naive = tmp_path / "naive"
    reference.build_reference_submission("a", str(ref))
    reference.build_probe_submission("naive_color", str(naive))
    ref_report = scorer.score_submission(str(ref), "a", battery=bat)
    naive_report = scorer.score_submission(str(naive), "a", battery=bat)
    assert not ref_report["gated"]
    assert ref_report["reward"] > naive_report["reward"]


def test_tracking_beats_frozen_pose_on_stage_b(bat, tmp_path):
    ref = tmp_path / "reference"
    frozen = tmp_path / "frozen"
    reference.build_reference_submission("d", str(ref))
    reference.build_probe_submission("last_pose", str(frozen))
    ref_report = scorer.score_submission(str(ref), "d", battery=bat)
    frozen_report = scorer.score_submission(str(frozen), "d", battery=bat)
    assert ref_report["checkpoints"]["B.pose"] > frozen_report["checkpoints"]["B.pose"]


def test_wrong_signature_fails_load_gate(bat, tmp_path):
    directory = tmp_path / "signature"
    _write(directory, """
        class E:
            def reset(self): pass
            def update(self, rgb): return (0.0, 0.0, 0.0)
        def make_estimator(): return E()
    """)
    report = scorer.score_submission(str(directory), "d", battery=bat)
    assert report["gated"] and report["gate_failed"] == ["G.load"]
    assert report["reward"] == 0.0


def test_factory_failure_trips_the_only_gate(bat, tmp_path):
    directory = tmp_path / "factory"
    _write(directory, """
        def make_estimator(): raise RuntimeError("cannot construct")
    """)
    report = scorer.score_submission(str(directory), "d", battery=bat)
    assert report["gated"] and report["gate_failed"] == ["G.load"]
    assert report["reward"] == 0.0


def test_model_for_code_task_scores_zero(bat, tmp_path):
    directory = tmp_path / "modelonly"
    directory.mkdir()
    (directory / "model.pt").write_bytes(b"junk")
    report = scorer.score_submission(str(directory), "a", battery=bat)
    assert report["reward"] == 0.0 and report["gated"]


def test_eval_seeds_are_distinct_and_private_scale():
    seeds = battery.load_eval_seeds()
    assert len(seeds) == 10 and len(set(seeds)) == 10
    assert not set(seeds) & set(spec.DESIGN_SEEDS)
    assert all(seed > 2**40 for seed in seeds)


def test_eval_frames_differ_from_design_frames(bat):
    from harness import episodes
    design = episodes.single_frame(spec.DESIGN_SEEDS[0])
    for frame in bat.stage_a[:4]:
        assert not np.array_equal(frame.obs["rgb"], design.obs["rgb"])
