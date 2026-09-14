"""Which way up the agent's camera frames are.

This exists because they were upside down, in production, for every run the task has
ever had. MuJoCo reads pixels bottom-up and robosuite's default IMAGE_CONVENTION
("opengl") passes that through untouched, so the array handed to the agent was inverted
-- while the frames this harness saved for humans were flipped on the way out and looked
fine. Nothing raised. It just made a perception-bottlenecked task much harder than
designed, against models trained on upright pictures.

The fix sets the convention once at the source (`env._use_upright_images`, called from
`env._before_construction`), so every consumer receives upright frames and none of them
flips again. Two things have to hold for that to stay true, and both are tested here:

  * the on-demand render (`render_frames`) uses robosuite's OWN convention value rather
    than a literal, so it cannot drift from the streamed observation;
  * the human-facing records (transcript PNGs, debug video) do NOT flip, because their
    input is already upright -- flipping would both invert them and stop them showing
    what the agent actually saw.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from harness import env as ENV


# A frame whose top and bottom differ, so an accidental flip cannot go unnoticed.
def asymmetric(h: int = 8, w: int = 4) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0, :, :] = 255          # top row white
    frame[-1, :, :] = 64          # bottom row grey
    return frame


class FakeSim:
    """Stands in for robosuite's MjSim: renders bottom-up, like MuJoCo does."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def render(self, camera_name=None, width=None, height=None, depth=False):
        self.calls.append((camera_name, width, height))
        return self.frame


class FakeCameraEnv:
    def __init__(self, frame):
        self.sim = FakeSim(frame)


# set_convention patches the REAL robosuite.macros: these tests prove the harness
# reads robosuite's value instead of hardcoding one, so a fake would be circular.
needs_robosuite = pytest.mark.skipif(
    importlib.util.find_spec("robosuite") is None,
    reason="robosuite not installed (make sim-robocasa)")


def set_convention(monkeypatch, value: str) -> None:
    import robosuite.macros as macros

    monkeypatch.setattr(macros, "IMAGE_CONVENTION", value)


# -- the convention is read, never hardcoded ---------------------------------

@needs_robosuite
def test_render_frames_follows_robosuite_s_convention(monkeypatch):
    """The whole point: one value decides orientation for both paths.

    If this were a literal `[::-1]`, an on-demand render and a streamed observation
    would agree only by coincidence -- which is exactly how they came to disagree.
    """
    raw = asymmetric()
    env = FakeCameraEnv(raw)

    set_convention(monkeypatch, "opengl")            # +1: pass through
    out = ENV.render_frames(env, ["cam"], 4, 8)
    assert np.array_equal(out["cam_image"], raw)

    set_convention(monkeypatch, "opencv")            # -1: flip
    out = ENV.render_frames(env, ["cam"], 4, 8)
    assert np.array_equal(out["cam_image"], raw[::-1])


# -- upright is what the simulator is asked for ------------------------------

@needs_robosuite
def test_the_harness_asks_for_upright_frames(monkeypatch):
    """`opengl` is robosuite's default and it means BOTTOM-UP. The harness must
    override it, or the agent gets inverted pictures."""
    set_convention(monkeypatch, "opengl")
    assert ENV.image_convention() == 1, "precondition: the default is not upright"

    ENV._use_upright_images()

    assert ENV.image_convention() == -1


@needs_robosuite
def test_env_construction_cannot_skip_the_orientation_setting(monkeypatch):
    """Both make_env branches go through _before_construction. The asset patch was
    already forgotten at a call site twice; orientation fails just as silently, so it
    is bundled into the same call rather than repeated per branch."""
    set_convention(monkeypatch, "opengl")
    called = []
    monkeypatch.setattr(ENV, "install_ro_assets_patch", lambda: called.append("assets"))

    ENV._before_construction()

    assert called == ["assets"]
    assert ENV.image_convention() == -1


# -- the human-facing records must not flip again ----------------------------

def test_the_transcript_saves_exactly_what_the_agent_saw(tmp_path):
    """Two properties at once: no double flip, and the saved evidence is byte-identical
    to the observation, so a frame in the artifacts is admissible as what the agent had.
    """
    Image = pytest.importorskip("PIL.Image")
    from harness import transcript as T

    raw = asymmetric(h=6, w=5)
    written = T.save_frames({"robot0_agentview_left_image": raw}, tmp_path, "t0")
    assert written, "no frame was written"

    saved = np.asarray(Image.open(tmp_path / written[0]))
    assert np.array_equal(saved, raw)


def test_the_debug_tile_does_not_flip(monkeypatch):
    from harness import debug as D

    raw = asymmetric(h=6, w=5)
    obs = {key: raw for key in D.TILED_CAMERAS}
    tile = D.tile_frame(obs)
    assert tile is not None
    # Every tile is the frame unchanged, so the top row stays the top row.
    assert np.array_equal(tile[0, : raw.shape[1]], raw[0])
    assert np.array_equal(tile[raw.shape[0] - 1, : raw.shape[1]], raw[-1])


# -- provenance --------------------------------------------------------------

@needs_robosuite
def test_the_ledger_records_which_orientation_a_run_was_scored_under(tmp_path,
                                                                    monkeypatch):
    """Changing the convention changes the pixels, so runs either side of it are not
    comparable. The ledger has to say which side it is on."""
    from harness import ledger as L
    from harness.session import Budgets, MeteredSession
    from test_speedrun_session import FakeEnv

    set_convention(monkeypatch, "opencv")
    sess = MeteredSession(
        task="T",
        budgets=Budgets(interaction_steps=10, submissions=1),
        ledger=L.Ledger(tmp_path / "conv.jsonl"),
        env_factory=lambda task, split="pretrain", scene=None: FakeEnv(split=split),
        trial_seeds=[1],
    )
    start = next(r for r in L.read_records(sess.ledger.path)
                 if r.kind == L.KIND_START)
    assert start.payload["image_convention"] == -1
