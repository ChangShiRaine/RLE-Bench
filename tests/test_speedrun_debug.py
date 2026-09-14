"""Tests for the live debug recorder.

Two things matter here and nothing else does: that a DISABLED recorder is genuinely
inert (debug output must never be able to change or slow a scored run), and that an
ENABLED one produces files a person can actually read while the run is still going.

All of it runs without MuJoCo, against fake observations.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from harness import debug as D
from harness import ledger as L
from harness.session import Budgets, MeteredSession
from test_speedrun_session import FakeEnv


def fake_obs(value: int = 0) -> dict:
    """An observation shaped like RoboCasa's: three 128x128 camera views plus state."""
    def cam(seed):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    return {
        "robot0_agentview_left_image": cam(value),
        "robot0_agentview_right_image": cam(value + 100),
        "robot0_eye_in_hand_image": cam(value + 200),
        "robot0_proprio-state": np.zeros(32),
    }


# -- the disabled path -------------------------------------------------------

def test_a_disabled_recorder_writes_absolutely_nothing(tmp_path):
    """Debug is opt-in, so the default must not touch the filesystem at all."""
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=False)
    assert rec.enabled is False

    rec.episode_started("development", 1, fake_obs())
    for _ in range(50):
        rec.frame(fake_obs())
    rec.event("step", steps=1)
    rec.status({"phase": "development"})
    rec.diagnosis(None, fake_obs())
    rec.flush_episode()
    rec.close()

    assert not (tmp_path / "dbg").exists(), "a disabled recorder created files"


def test_a_recorder_with_no_root_is_disabled_even_if_asked_to_be_on():
    assert D.DebugRecorder(root=None, enabled=True).enabled is False


def test_a_session_with_no_recorder_still_runs(tmp_path):
    """The hooks are unconditional, so the null recorder has to absorb every call."""
    sess = MeteredSession(
        task="T", budgets=Budgets(interaction_steps=100, submissions=1),
        ledger=L.Ledger(tmp_path / "c.jsonl"),
        env_factory=lambda task, split="pretrain", scene=None: FakeEnv(success_after=-1),
        trial_seeds=[1], max_episode_steps=10)
    sess.reset()
    sess.step([[0.0] * 12] * 1)
    assert sess.state.steps_used == 2          # the reset, then the action
    sess.close()


# -- frames and video --------------------------------------------------------

def test_three_cameras_tile_into_one_wide_frame():
    img = D.tile_frame(fake_obs())
    assert img.shape == (128, 384, 3), "expected the three views side by side"
    assert img.dtype == np.uint8


def test_tiling_an_observation_with_no_images_returns_nothing():
    """Development can hand back observations with no camera keys; that is not an error."""
    assert D.tile_frame({"robot0_proprio-state": np.zeros(4)}) is None
    assert D.tile_frame(None) is None


def test_frames_are_subsampled(tmp_path):
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=3)
    for _ in range(9):
        rec.frame(fake_obs())
    assert len(rec._frames) == 3, "every_n_steps=3 should keep one frame in three"


# The recorder degrades gracefully without imageio (a [debug] note, no mp4), so
# the two tests that assert on written files need the real encoder.
needs_imageio = pytest.mark.skipif(
    importlib.util.find_spec("imageio") is None,
    reason="imageio not installed; the recorder skips video writing without it")


@needs_imageio
def test_reset_is_the_episode_boundary(tmp_path):
    """Short episodes retain their filenames and flush at reset."""
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=1)
    episodes = tmp_path / "dbg" / "episodes"

    rec.episode_started("development", 1, fake_obs())
    for i in range(4):
        rec.frame(fake_obs(i))
    assert not list(episodes.glob("*.mp4")), "nothing written mid-episode"

    rec.episode_started("development", 2, fake_obs())      # the reset
    assert [p.name for p in episodes.glob("*.mp4")] == ["development_ep0001.mp4"]
    assert rec._frames, "the new episode's first frame should already be buffered"

    for i in range(4):
        rec.frame(fake_obs(i))
    rec.close()
    assert sorted(p.name for p in episodes.glob("*.mp4")) == [
        "development_ep0001.mp4", "development_ep0002.mp4"]
    assert (episodes / "development_ep0001.mp4").stat().st_size > 0


@needs_imageio
def test_video_is_named_for_the_phase(tmp_path):
    """Development and evaluation episodes must be tellable apart at a glance."""
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=1)
    rec.episode_started("evaluation", 0, fake_obs())
    rec.frame(fake_obs())
    rec.close()
    assert (tmp_path / "dbg" / "episodes" / "evaluation_ep0000.mp4").exists()


# -- the text streams --------------------------------------------------------

def test_events_and_status_are_readable_while_the_run_is_going(tmp_path):
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True)
    rec.event("reset", ok=True)
    rec.event("step", steps=12, reason="step_complete")
    rec.status({"phase": "development", "steps_used": 12})

    lines = (tmp_path / "dbg" / "events.jsonl").read_text().splitlines()
    assert [json.loads(l)["op"] for l in lines] == ["reset", "step"]
    assert json.loads(lines[1])["steps"] == 12

    status = json.loads((tmp_path / "dbg" / "status.json").read_text())
    assert status["phase"] == "development" and status["steps_used"] == 12


def test_status_is_replaced_atomically_and_leaves_no_temp_file(tmp_path):
    """It is written to be tailed, so a reader must never catch it half-written."""
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True)
    for i in range(5):
        rec.status({"phase": "development", "steps_used": i})
    assert not list((tmp_path / "dbg").glob("*.tmp"))
    assert json.loads((tmp_path / "dbg" / "status.json").read_text())["steps_used"] == 4


def test_the_workspace_is_snapshotted_per_episode(tmp_path):
    """So a controller can be read as it was when it ran, not only in its final form."""
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "skills.py").write_text("v1")
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, workspace=ws)

    rec.episode_started("development", 1)
    (ws / "skills.py").write_text("v2")
    rec.episode_started("development", 2)

    root = tmp_path / "dbg" / "workspace"
    assert (root / "ep0001" / "skills.py").read_text() == "v1"
    assert (root / "ep0002" / "skills.py").read_text() == "v2"


def test_an_empty_workspace_leaves_no_empty_directories(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, workspace=ws)
    rec.episode_started("development", 1)
    assert not list((tmp_path / "dbg" / "workspace").iterdir())


def test_files_in_subdirectories_are_snapshotted(tmp_path):
    """The snapshot must recurse: an agent that organises its work into `dev/` would
    otherwise produce an empty tree, which reads as "wrote no code at all". Layout is
    mirrored so `dev/push.py` and `push.py` stay distinct."""
    ws = tmp_path / "ws"; (ws / "dev").mkdir(parents=True)
    (ws / "top.py").write_text("top")
    (ws / "dev" / "push.py").write_text("nested")
    (ws / "dev" / "top.py").write_text("different file, same name")
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, workspace=ws)

    rec.episode_started("development", 1)

    ep = tmp_path / "dbg" / "workspace" / "ep0001"
    assert (ep / "top.py").read_text() == "top"
    assert (ep / "dev" / "push.py").read_text() == "nested"
    assert (ep / "dev" / "top.py").read_text() == "different file, same name"


def test_heavy_directories_are_pruned_not_copied(tmp_path):
    """This runs once per EPISODE. An agent that creates a venv or clones a repo in
    its workspace would otherwise have tens of thousands of files copied ~100 times."""
    ws = tmp_path / "ws"
    (ws / ".venv" / "lib").mkdir(parents=True)
    (ws / "__pycache__").mkdir(parents=True)
    (ws / "node_modules").mkdir(parents=True)
    (ws / ".venv" / "lib" / "huge.py").write_text("x")
    (ws / "__pycache__" / "cached.py").write_text("x")
    (ws / "node_modules" / "dep.py").write_text("x")
    (ws / "mine.py").write_text("mine")
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, workspace=ws)

    rec.episode_started("development", 1)

    ep = tmp_path / "dbg" / "workspace" / "ep0001"
    assert [p.name for p in sorted(ep.rglob("*.py"))] == ["mine.py"]


def test_a_huge_workspace_is_capped_and_says_so(tmp_path, capsys, monkeypatch):
    """Silent truncation would read as the agent having written only these files."""
    monkeypatch.setattr(D.DebugRecorder, "_MAX_SNAPSHOT_FILES", 5)
    ws = tmp_path / "ws"; ws.mkdir()
    for i in range(12):
        (ws / f"f{i:02d}.py").write_text("x")
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, workspace=ws)

    rec.episode_started("development", 1)

    ep = tmp_path / "dbg" / "workspace" / "ep0001"
    assert len(list(ep.rglob("*.py"))) == 5
    assert "capped at 5" in capsys.readouterr().out


# -- it must never break a run ----------------------------------------------

def test_a_failing_recorder_never_propagates(tmp_path):
    """Debug output is a convenience. A bug in it must not end somebody's run."""
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True)

    class Explodes:
        def __getattr__(self, _):
            raise RuntimeError("boom")

    rec.frame({"robot0_agentview_left_image": Explodes()})
    rec.diagnosis(Explodes(), fake_obs())
    rec.event("x", detail=Explodes())
    rec.status({"phase": Explodes()})
    rec.close()          # no exception is the assertion


def test_the_recorder_is_wired_into_a_real_session(tmp_path):
    """End to end through MeteredSession: development frames and events are captured
    without the session knowing whether anyone is watching."""
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=1,
                          workspace=tmp_path / "absent")
    sess = MeteredSession(
        task="T", budgets=Budgets(interaction_steps=100, submissions=1),
        ledger=L.Ledger(tmp_path / "c.jsonl"),
        env_factory=lambda task, split="pretrain", scene=None: FakeEnv(success_after=-1),
        trial_seeds=[1], max_episode_steps=10, recorder=rec)

    sess.reset()
    sess.step([[0.0] * 12] * 2)
    sess.reset()                       # boundary -> would flush episode 1
    sess.close()

    # `segment` records come from the session itself; the per-op log is emitted one
    # layer up, at the Service dispatch chokepoint, so driving the session directly
    # yields only the segment.
    events = [json.loads(l)
              for l in (tmp_path / "dbg" / "events.jsonl").read_text().splitlines()]
    assert [e["op"] for e in events] == ["segment"]
    assert events[0]["reason"] == "step_complete" and events[0]["steps"] == 2
    assert events[0]["phase"] == "development" and events[0]["episode"] == 1

    # FakeEnv has no camera keys, so no video is expected. What is being checked is
    # that the session drove the recorder without raising and cut episodes at reset.
    assert rec._episode == 2
    assert not list((tmp_path / "dbg" / "episodes").glob("*.mp4"))


def test_trial_ends_are_recorded_as_events(tmp_path):
    """A trial that dies at the step ceiling or is forfeited by reset must leave a
    terminal event, not just 1000 step_complete segments."""
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=1,
                          workspace=tmp_path / "absent")
    sess = MeteredSession(
        task="T", budgets=Budgets(interaction_steps=100, submissions=1),
        ledger=L.Ledger(tmp_path / "c.jsonl"),
        env_factory=lambda task, split="pretrain", scene=None: FakeEnv(success_after=-1),
        trial_seeds=[1], max_episode_steps=10, recorder=rec,
        eval_plan_fn=lambda task: [(0, 1, 1, 11), (0, 1, 1, 12)])
    sess.begin_evaluation()
    sess.step([[0.0] * 12] * 50)       # runs into the 10-step trial ceiling
    sess.reset()                       # advance to trial 2
    sess.step([[0.0] * 12] * 3)
    sess.reset()                       # forfeit the live trial
    sess.close()

    events = [json.loads(l)
              for l in (tmp_path / "dbg" / "events.jsonl").read_text().splitlines()]
    ends = [e for e in events if e["op"] == "trial_end"]
    assert [e["ended_by"] for e in ends] == ["max_steps", "agent_reset"]
    assert [e["trial"] for e in ends] == [0, 1]
    assert all(e["success"] is False for e in ends)


@needs_imageio
def test_long_episode_segments_are_readable_before_close(tmp_path, monkeypatch):
    import imageio.v2 as imageio
    monkeypatch.setattr(D, "SEGMENT_SECONDS", 1)
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=2, fps=2)
    obs = {"robot0_agentview_left_image": np.zeros((16, 16, 3), dtype=np.uint8)}
    rec.episode_started("evaluation", 0, obs)
    for i in range(1, 9):
        rec.frame({"robot0_agentview_left_image": np.full((16, 16, 3), i * 20, dtype=np.uint8)})
    files = sorted((tmp_path / "dbg/episodes").glob("*.mp4"))
    assert len(files) == 2  # Durable before close, even if the daemon now dies.
    assert len(rec._frames) == 1
    rec.close()
    rec.close()
    files = sorted((tmp_path / "dbg/episodes").glob("*.mp4"))
    assert [p.name for p in files] == ["evaluation_ep0000.mp4",
                                     "evaluation_ep0000_part0001.mp4",
                                     "evaluation_ep0000_part0002.mp4"]
    decoded = [frame for path in files for frame in imageio.mimread(path)]
    assert len(decoded) == 5
    np.testing.assert_allclose([f.mean() for f in decoded], [0, 40, 80, 120, 160], atol=3)
    rec.episode_started("evaluation", 1, obs)
    rec.close()
    assert (tmp_path / "dbg/episodes/evaluation_ep0001.mp4").exists()


def test_segment_byte_limit_and_encoder_failure_release_memory(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    obs = {"robot0_agentview_left_image": np.zeros((16, 16, 3), dtype=np.uint8)}
    frame_bytes = D.tile_frame(obs).nbytes
    monkeypatch.setattr(D, "MAX_BUFFER_BYTES", frame_bytes * 2 + 1)
    batches = []
    def fail(path, frames, **kwargs):
        batches.append(len(frames))
        raise OSError("disk full")
    imageio = ModuleType("imageio")
    imageio.mimwrite = fail
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    rec = D.DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=1)
    for _ in range(21):
        rec.frame(obs)
        assert rec._buffer_bytes <= D.MAX_BUFFER_BYTES
        assert len(rec._frames) <= 2
    rec.close()
    assert batches == [2] * 10 + [1]
    assert not rec._frames and rec._buffer_bytes == 0
