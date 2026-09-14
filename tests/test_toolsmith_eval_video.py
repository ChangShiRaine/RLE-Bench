"""Evaluation-trial videos: the scored phase's camera streams, recorded root-side
and exported by the verifier. Runs without MuJoCo, against fake observations."""
from __future__ import annotations

import json
import shutil

import pytest

from harness import eval_video as V
import numpy as np


def fake_obs(value: int = 0) -> dict:
    def cam(seed):
        return np.random.default_rng(seed).integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    return {"robot0_agentview_left_image": cam(value),
            "robot0_agentview_right_image": cam(value + 100),
            "robot0_eye_in_hand_image": cam(value + 200),
            "robot0_proprio-state": np.zeros(32)}

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def test_a_disabled_recorder_writes_nothing(tmp_path):
    rec = V.EvalVideoRecorder(tmp_path / "media", enabled=False)
    rec.episode_started("evaluation", 0, fake_obs())
    rec.frame(fake_obs())
    rec.event("trial_end")
    rec.close()
    assert not (tmp_path / "media").exists()


@needs_ffmpeg
def test_only_evaluation_trials_are_recorded(tmp_path):
    rec = V.EvalVideoRecorder(tmp_path / "media", enabled=True)
    rec.episode_started("development", 0, fake_obs())
    for i in range(6):
        rec.frame(fake_obs(i))
    rec.episode_started("evaluation", 0, fake_obs())
    for i in range(9):
        rec.frame(fake_obs(i))
    rec.event("trial_end", ended_by="success")
    rec.episode_started("evaluation", 1, fake_obs())
    for i in range(3):
        rec.frame(fake_obs(i))
    rec.close()                       # the last trial ends with the run
    index = json.loads((tmp_path / "media" / "index.json").read_text())
    assert index["files"] == ["trial000.mp4", "trial001.mp4"]
    assert not index["skipped"]
    assert sorted(p.name for p in (tmp_path / "media").glob("*.mp4")) == index["files"]


def test_recorders_fan_out_and_isolate_failures(tmp_path):
    class Broken:
        enabled = True

        def frame(self, obs):
            raise RuntimeError("boom")

    rec = V.EvalVideoRecorder(tmp_path / "media", enabled=False)
    both = V.Recorders(Broken(), rec, None)
    both.frame(fake_obs())            # Broken raises, the rest still runs
    both.diagnosis(None, None)        # missing on Broken, present on the recorder
    assert both.enabled

