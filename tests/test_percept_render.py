"""Stage B evidence videos: the estimator's input with truth and estimate drawn."""
import os
import shutil
import textwrap

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest

from harness import battery, render, scorer, spec
from rlebench.core.media import Media

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def mini_battery():
    return battery.build_battery(seeds=[11], stage_a_count=1, episode_max_frames=6)


def test_outline_skips_the_seams_between_boxes():
    t_shape = render.outline_segments(spec.BLOCK_COLLISION_BOXES["tshape"])
    assert len(t_shape) == 8                       # the T's eight outer edges
    lengths = sorted(round(np.hypot(b[0] - a[0], b[1] - a[1]), 3) for a, b in t_shape)
    assert lengths == [0.05, 0.05, 0.05, 0.075, 0.075, 0.15, 0.15, 0.2]
    for shape in spec.BLOCK_SHAPES:
        assert render.outline_segments(spec.BLOCK_COLLISION_BOXES[shape])


def test_project_inverts_backproject():
    from harness import sensor
    depth = np.full((spec.IMG_H, spec.IMG_W), 0.9, dtype=np.float32)
    pts = sensor.backproject(depth)
    uv = render.project(pts[[100, 400], [50, 600]])
    assert np.allclose(uv, [[50, 100], [600, 400]], atol=1e-6)


def test_compose_frame_has_a_depth_panel_only_when_depth_is_an_input(mini_battery):
    frame = mini_battery.episodes[0].frames[0]
    rgb_only = render.compose_frame(frame, frame.gt, ("rgb",), "tshape", "label", 0, 6)
    assert rgb_only.shape == (spec.IMG_H, spec.IMG_W, 3)
    both = render.compose_frame(frame, [np.nan] * 3, ("rgb", "depth"), "tshape", "label", 0, 6)
    assert both.shape == (spec.IMG_H, 2 * spec.IMG_W, 3)
    # the overlay changed the picture, and the depth panel is not black
    assert not np.array_equal(rgb_only, frame.obs["rgb"])
    assert both[:, spec.IMG_W:].mean() > 30


@needs_ffmpeg
def test_scoring_with_media_writes_one_video_per_episode(mini_battery, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "estimator.py").write_text(textwrap.dedent("""
        class E:
            def reset(self): pass
            def update(self, **obs): return (0.05, -0.02, 0.3, 0)
        def make_estimator(): return E()
    """))
    media = Media(tmp_path / "media")
    report = scorer.score_submission(str(sub), "d", battery=mini_battery, media=media)
    index = media.close()
    assert "reward" in report
    assert index["files"] == ["ep0_seed11.mp4"] and not index["skipped"]
    assert (tmp_path / "media" / "ep0_seed11.mp4").stat().st_size > 1000
