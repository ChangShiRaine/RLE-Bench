"""The on-demand render, and what it must never hand back.

Depth is the sensor the whole task turns on: there are no object poses, so a pixel is a
direction and nothing else until its depth makes it a point. Every agent run inspected
built its perception layer on `pixel + depth -> base frame`. That makes two failure modes
in `render_frames` worth pinning:

  * A camera that renders in colour but not in depth must be RETRIED, not silently
    downgraded. Seen on the first observation of a freshly reset episode and on every
    observation of a finished one.
  * A colour frame must never be UNPACKED as if it were `(rgb, depth)`. Destructuring an
    HxWx3 array binds two rows of pixels, which produced a two-pixel "image" and a
    "depth map" made of colour -- both plausible, neither real, and only when depth was
    already broken.

No MuJoCo here: `render_frames` takes whatever `env.sim.render` returns, so a fake sim
covers the branch structure exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import env as ENV

CAM = "robot0_eye_in_hand"


class FakeSim:
    """Renders `mode` for the first `heal_after` calls, then renders correctly.

    `heal_after=None` never heals, which is how a persistent failure is expressed.
    """

    def __init__(self, mode: str = "ok", heal_after: int | None = None):
        self.mode = mode
        self.heal_after = heal_after
        self.calls = 0

    def render(self, camera_name=None, width=None, height=None, depth=False):
        self.calls += 1
        mode = self.mode
        if self.heal_after is not None and self.calls > self.heal_after:
            mode = "ok"
        rgb = np.full((int(height), int(width), 3), 7, dtype=np.uint8)
        if mode == "raise":
            raise RuntimeError("no offscreen context")
        if not depth:
            return rgb
        if mode == "rgb_only":                       # the flag was ignored
            return rgb
        return rgb, np.full((int(height), int(width)), 0.5, dtype=np.float32)


class FakeEnv:
    def __init__(self, sim):
        self.sim = sim


@pytest.fixture(autouse=True)
def metres(monkeypatch):
    """`metric_depth` needs a real `sim` for the near/far planes; the conversion has its
    own test. Here the buffer is passed through so the keys are what is under test."""
    monkeypatch.setattr(ENV, "metric_depth", lambda sim, buffer: np.asarray(buffer))


def render(sim, depth=True):
    return ENV.render_frames(FakeEnv(sim), [CAM], 4, 4, depth=depth)


def test_a_working_depth_render_costs_one_render():
    """The retry must be on the failing path only -- every step pays this one."""
    sim = FakeSim()
    out = render(sim)
    assert set(out) == {f"{CAM}_image", f"{CAM}_depth"}
    assert sim.calls == 1


def test_depth_missing_is_retried_once_and_recovers():
    """The case that cost an agent run a calibration: the second read finds the context
    warm and returns what was asked for."""
    sim = FakeSim("rgb_only", heal_after=1)
    out = render(sim)
    assert set(out) == {f"{CAM}_image", f"{CAM}_depth"}
    assert sim.calls == 2


def test_a_colour_frame_is_never_unpacked_as_rgb_and_depth():
    """The silent corruption. Unpacking an HxWx3 frame SUCCEEDS -- it binds two rows --
    so the caller got a 2-row image and a depth map made of pixel values, with no error
    anywhere. Better to hand back a whole picture and no depth."""
    sim = FakeSim("rgb_only")
    out = render(sim)
    assert set(out) == {f"{CAM}_image"}
    assert out[f"{CAM}_image"].shape == (4, 4, 3)
    assert sim.calls == 2                            # tried twice, then gave up


def test_a_camera_that_cannot_render_at_all_is_omitted():
    """A missing frame is recoverable; a raised observation is not."""
    assert render(FakeSim("raise")) == {}


def test_asking_for_no_depth_never_pays_the_retry():
    sim = FakeSim("ok")
    out = render(sim, depth=False)
    assert set(out) == {f"{CAM}_image"}
    assert sim.calls == 1
