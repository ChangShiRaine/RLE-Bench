"""The observation-resolution contract, and the one thing it must never do.

Growing MuJoCo's offscreen framebuffer mid-run is not a resize: robosuite frees the LIVE
GL context and builds a new one (`binding_utils.update_offscreen_size`). That rebuild is
the only operation in this harness that has ever logged `OpenGL error 0x501 in or before
mjr_makeContext`, and a context that came back broken has both killed the daemon inside
the render and wedged a worker thread at 100% CPU until the trial was lost.

So the cameras are baked at the ceiling and every delivery is smaller: resampled from the
frame the step pipeline already rendered, or rendered at a size the buffer already holds.
What is pinned here is that no path asks for more -- with the sizes recorded, because the
failure was invisible in the reply and showed up only in a log nobody collects.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from harness import config as C
from harness import env as ENV
from harness.controller import ObsSpec

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


class RecordingSim:
    """A sim that answers any render and remembers what it was asked for."""

    def __init__(self):
        self.asked: list[tuple[int, int, bool]] = []

    def render(self, camera_name=None, width=None, height=None, depth=False):
        self.asked.append((int(width), int(height), bool(depth)))
        rgb = np.full((int(height), int(width), 3), 7, dtype=np.uint8)
        if not depth:
            return rgb
        return rgb, np.full((int(height), int(width)), 0.5, dtype=np.float32)


class FakeEnv:
    def __init__(self, sim):
        self.sim = sim


@pytest.fixture(autouse=True)
def metres(monkeypatch):
    monkeypatch.setattr(ENV, "metric_depth", lambda sim, buffer: np.asarray(buffer))


def baked_obs(size: int | None = None) -> dict:
    """What the step pipeline hands over: every camera at the baked size."""
    size = C.RENDER_RESOLUTION if size is None else size
    obs = {"robot0_base_to_eef_pos": np.zeros(3)}
    for i, cam in enumerate(CAMERAS):
        obs[f"{cam}_image"] = np.full((size, size, 3), i + 1, dtype=np.uint8)
    return obs


def apply(spec, obs=None):
    sim = RecordingSim()
    out, resolution = ENV.apply_obs_spec(FakeEnv(sim), baked_obs() if obs is None else obs,
                                         spec)
    return out, resolution, sim


# -- the invariant -------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    None,
    ObsSpec(),
    ObsSpec(width=128),
    ObsSpec(width=C.OBS_MAX_RESOLUTION, depth=True),
    ObsSpec(width=4096),                                  # above the ceiling
    ObsSpec(width=4096, depth=True),
    ObsSpec(width=1024, height=768),                      # oversized and non-square
    ObsSpec(width=300, height=200, depth=True),
    ObsSpec(cameras=(CAMERAS[0],), width=99999),
])
def test_no_spec_can_make_the_harness_ask_for_more_than_is_baked(spec):
    """THE regression guard. Every reachable spec, and not one render above the buffer."""
    _, _, sim = apply(spec)
    assert all(w <= C.RENDER_RESOLUTION and h <= C.RENDER_RESOLUTION
               for w, h, _ in sim.asked), sim.asked


def test_render_frames_clamps_even_when_called_directly():
    """The second lock: `apply_obs_spec` is not the only caller, and a future one must
    not be able to reintroduce the rebuild by passing a bigger number."""
    sim = RecordingSim()
    ENV.render_frames(FakeEnv(sim), [CAMERAS[0]], 4096, 4096, depth=True)
    assert sim.asked == [(C.RENDER_RESOLUTION, C.RENDER_RESOLUTION, True)]


def test_the_cameras_are_baked_at_the_ceiling():
    """If these drift apart the ceiling becomes a resize again."""
    defaults = inspect.signature(ENV.make_env).parameters
    assert defaults["camera_width"].default == C.RENDER_RESOLUTION
    assert defaults["camera_height"].default == C.RENDER_RESOLUTION
    assert C.RENDER_RESOLUTION == C.OBS_MAX_RESOLUTION


# -- what each spec delivers ---------------------------------------------------

def test_an_unsized_observation_delivers_the_default_and_renders_nothing():
    """The hot path: the frame is already in hand, so a smaller one is arithmetic."""
    out, resolution, sim = apply(ObsSpec())
    assert sim.asked == []
    assert resolution == C.OBS_RESOLUTION      # a bare number when none was asked for
    for cam in CAMERAS:
        assert out[f"{cam}_image"].shape == (C.OBS_RESOLUTION, C.OBS_RESOLUTION, 3)
    assert "robot0_base_to_eef_pos" in out


def test_a_smaller_square_colour_frame_is_resampled_not_rendered():
    out, resolution, sim = apply(ObsSpec(width=128))
    assert sim.asked == []
    assert resolution == [128, 128]
    assert out[f"{CAMERAS[0]}_image"].shape == (128, 128, 3)


def test_depth_is_rendered_at_the_size_asked_for():
    """Depth is not baked into the observation, so it costs a render -- one, at the
    requested size, which the buffer already holds."""
    out, _, sim = apply(ObsSpec(width=256, depth=True))
    assert sim.asked == [(256, 256, True)] * len(CAMERAS)
    assert out[f"{CAMERAS[0]}_depth"].shape == (256, 256)


def test_an_oversized_request_is_clamped_rather_than_refused():
    out, resolution, _ = apply(ObsSpec(width=4096))
    assert resolution == [C.OBS_MAX_RESOLUTION, C.OBS_MAX_RESOLUTION]
    assert out[f"{CAMERAS[0]}_image"].shape[0] == C.OBS_MAX_RESOLUTION


def test_a_non_square_frame_is_rendered_rather_than_stretched():
    """A 300x200 frame is a different projection from the square one the pipeline
    renders -- MuJoCo's fovy is vertical and the horizontal field follows the aspect.
    Resampling the square frame would hand back the same field with the wrong pixel
    aspect, and silently break a deprojection that trusts f = (H/2)/tan(fovy/2)."""
    out, _, sim = apply(ObsSpec(width=300, height=200))
    assert sim.asked == [(300, 200, False)] * len(CAMERAS)
    assert out[f"{CAMERAS[0]}_image"].shape[:2] == (200, 300)


def test_an_empty_camera_set_still_costs_nothing():
    out, resolution, sim = apply(ObsSpec(cameras=()))
    assert sim.asked == [] and resolution is None
    assert not [k for k in out if k.endswith("_image") or k.endswith("_depth")]


def test_a_camera_missing_from_the_observation_is_omitted_not_rendered():
    """The published camera set IS the baked set, so a frame that is not there means a
    finished episode -- nothing to render against, and a render would be work spent on
    a dead env."""
    obs = {k: v for k, v in baked_obs().items() if not k.startswith(CAMERAS[0])}
    sim = RecordingSim()
    out, _ = ENV.apply_obs_spec(FakeEnv(sim), obs, ObsSpec(width=128))
    assert f"{CAMERAS[0]}_image" not in out
    assert out[f"{CAMERAS[1]}_image"].shape == (128, 128, 3)
    assert sim.asked == []


# -- resampling ----------------------------------------------------------------

def test_colour_halves_by_averaging_blocks():
    src = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    out = ENV.resample(src, 2, 2)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    assert out[0, 0, 0] == round(float(src[0:2, 0:2, 0].mean()))


def test_depth_is_point_sampled_so_no_range_is_invented():
    """Averaging across a depth edge puts a surface at a distance nothing occupies --
    a phantom the agent then deprojects into a point in mid-air."""
    src = np.array([[1.0, 1.0, 9.0, 9.0]] * 4, dtype=np.float32)
    out = ENV.resample(src, 2, 2, point=True)
    assert set(np.unique(out)) <= {1.0, 9.0}


def test_resampling_to_the_same_size_is_the_frame_itself():
    src = np.zeros((8, 8, 3), dtype=np.uint8)
    assert ENV.resample(src, 8, 8) is src


def test_a_ratio_that_does_not_divide_still_resamples():
    src = np.arange(9 * 9, dtype=np.uint8).reshape(9, 9)
    out = ENV.resample(src, 4, 4)
    assert out.shape == (4, 4) and out.dtype == np.uint8
