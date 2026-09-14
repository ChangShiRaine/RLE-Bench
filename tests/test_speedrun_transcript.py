"""Tests for the human-facing evaluation record.

This record exists to be *read by a person* debugging a run, and it is explicitly not
evidence: scoring never reads it. So what matters here is that it captures the story
(which controllers ran, why control came back, what the env looked like at the end),
that it degrades rather than explodes when the environment misbehaves, and that
tampering with it is at least detectable via the digest recorded in the ledger seal.
"""

from __future__ import annotations

import json

from harness import transcript as T
from harness.session import SegmentResult, Termination


class FakeEnv:
    def __init__(self, success=False, lang="Close the left drawer.", broken=False):
        self._success = success
        self._lang = lang
        self._broken = broken

    def _check_success(self):
        if self._broken:
            raise RuntimeError("sim exploded")
        return self._success

    def get_ep_meta(self):
        if self._broken:
            raise RuntimeError("no meta")
        return {"lang": self._lang, "layout_id": 3, "style_id": 7}


OBS = {
    "robot0_eef_pos": [1.0, 2.0, 3.0],
    "robot0_gripper_qpos": [0.02, -0.02],
    "drawer_obj_pos": [0.5, 0.5, 0.8],
    "distr_counter_1_quat": [0, 0, 0, 1],
    "object-state": [1, 2, 3],
}


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# -- diagnosis ---------------------------------------------------------------

def test_diagnosis_captures_what_a_human_looks_at_first():
    d = T.diagnose(FakeEnv(success=True), OBS)
    assert d["success"] is True
    assert d["instruction"] == "Close the left drawer."
    assert d["layout_id"] == 3 and d["style_id"] == 7
    assert d["robot"]["robot0_eef_pos"] == [1.0, 2.0, 3.0]
    # Object poses are collected generically -- task01 spans 317 tasks, so nothing
    # here may special-case one of them.
    assert "drawer_obj_pos" in d["objects"]
    assert "distr_counter_1_quat" in d["objects"]
    assert "robot0_eef_pos" not in d["objects"]      # robot is reported separately


def test_diagnosis_survives_a_broken_environment():
    """A diagnosis that raises would take down the trial it was only describing."""
    d = T.diagnose(FakeEnv(broken=True), OBS)
    assert d["success"] is None
    assert "sim exploded" in d["success_error"]
    assert d["robot"]                                  # still reports what it can


def test_diagnosis_reports_articulated_fixture_state():
    """For drawer/door tasks the normalized open/closed value is the single most
    informative number."""

    class Fixture:
        def get_door_state(self, env=None):
            return {"door": 0.83}

    env = FakeEnv()
    env.drawer = Fixture()
    assert T.diagnose(env, OBS)["fixture_state"] == {"drawer": {"door": 0.83}}


def test_diagnosis_without_an_observation():
    d = T.diagnose(FakeEnv(success=False), None)
    assert d["success"] is False
    assert "robot" not in d


# -- transcript --------------------------------------------------------------

def test_transcript_records_the_shape_of_a_trial(tmp_path):
    tr = T.Transcript(tmp_path / "transcript.jsonl", save_images=False)
    env = FakeEnv(success=True)
    tr.trial_start(0, scene=1, layout=1, style=1, seed=42, env=env, obs=OBS)
    tr.segment(0, 0, SegmentResult(steps=12, reason=Termination.STEP_COMPLETE,
                                   terminated_by="agent", final=False,
                                   success=False, obs=OBS, info={"note": "reached"}))
    tr.trial_end(0, ended_by="agent_reset", success=True, steps=12, env=env, obs=OBS)
    tr.summary({"success_rate": 1.0, "trials": 1})

    recs = read(tr.path)
    assert [r["kind"] for r in recs] == ["trial_start", "segment", "trial_end",
                                         "summary"]
    assert recs[0]["seed"] == 42
    assert recs[1]["steps"] == 12
    assert recs[1]["terminated_by"] == "agent"
    # Diagnostics are kept for the reader but are never scored.
    assert recs[1]["controller_info"]["note"] == "reached"
    assert recs[2]["ended_by"] == "agent_reset"
    assert recs[2]["diagnosis"]["success"] is True


def test_transcript_digest_changes_with_content(tmp_path):
    """The digest goes into the ledger seal, so tampering with the human record is
    detectable even though it is not scored."""
    tr = T.Transcript(tmp_path / "t.jsonl", save_images=False)
    tr.trial_start(0, 1, 1, 1, 42)
    first = tr.digest()
    tr.trial_end(0, "max_steps", False, 5)
    assert tr.digest() != first

    tr.path.write_text(tr.path.read_text().replace('"steps": 5', '"steps": 0'))
    assert tr.digest() != first


def test_digest_is_none_when_nothing_was_written(tmp_path):
    assert T.Transcript(tmp_path / "none.jsonl", save_images=False).digest() is None


def test_frame_saving_is_best_effort(tmp_path):
    """Frames are the first thing a human wants, but a missing encoder or an odd
    observation must never break a run."""
    assert T.save_frames(None, tmp_path / "frames", "x") == []
    assert T.save_frames({"bad_image": "not an array"}, tmp_path / "frames", "x") == []
    assert T.save_frames({"nope": [1, 2, 3]}, tmp_path / "frames", "x") == []


def test_frames_are_written_for_image_observations(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    obs = {"robot0_agentview_left_image": rng.integers(0, 256, (8, 8, 3),
                                                       dtype=np.uint8)}
    names = T.save_frames(obs, tmp_path / "frames", "trial000_end")
    assert names == ["trial000_end_robot0_agentview_left_image.png"]
    assert (tmp_path / "frames" / names[0]).exists()
