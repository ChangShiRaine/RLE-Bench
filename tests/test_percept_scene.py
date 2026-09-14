"""Dev tests for the task06 shared perception core (scene/sensor/episodes).

Golden-first: every property the scoring battery relies on is locked here —
pixel determinism, ground-truth/render consistency, the depth-noise model,
and that push episodes are real contact physics with occluded-but-moving
stretches.
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import pytest

from harness import episodes, scene, sensor, spec

SEED = 11


@pytest.fixture(scope="module")
def frame11():
    return episodes.single_frame(SEED)


@pytest.fixture(scope="module")
def episode23():
    return episodes.push_episode(23, render=False)


@pytest.fixture(scope="module")
def episode51():
    return episodes.push_episode(51, render=False)


# ---------------------------------------------------------------------------
# Determinism (invariant #3)
# ---------------------------------------------------------------------------
def test_pixels_and_noise_deterministic(frame11):
    again = episodes.single_frame(SEED)
    assert np.array_equal(frame11.obs["rgb"], again.obs["rgb"])
    assert np.array_equal(np.isnan(frame11.obs["depth"]),
                          np.isnan(again.obs["depth"]))
    assert np.array_equal(np.nan_to_num(frame11.obs["depth"]),
                          np.nan_to_num(again.obs["depth"]))
    assert frame11.gt == again.gt


def test_seeds_draw_different_instances(frame11):
    other = episodes.single_frame(23)
    assert not np.allclose(frame11.gt, other.gt, atol=1e-3)
    assert not np.array_equal(frame11.obs["rgb"], other.obs["rgb"])


def test_noise_frame_indexed(frame11):
    model, data = scene.build_scene(seed=SEED)
    scene.set_arm(model, data, spec.ARM_PARKED)
    scene.settle(model, data, 0.4)
    with sensor.CameraRig(model) as rig:
        clean = rig.depth(data)
    n0 = sensor.apply_depth_noise(clean, SEED, frame_idx=0)
    n1 = sensor.apply_depth_noise(clean, SEED, frame_idx=1)
    assert not np.array_equal(np.nan_to_num(n0), np.nan_to_num(n1))


# ---------------------------------------------------------------------------
# Ground truth vs rendered image
# ---------------------------------------------------------------------------
def _project(p_world):
    K = spec.intrinsics()
    R = spec.camera_rotation()
    pc = R.T @ (np.asarray(p_world) - np.asarray(spec.CAM_POS))
    u = K[0, 2] + K[0, 0] * pc[0] / -pc[2]
    v = K[1, 2] - K[1, 1] * pc[1] / -pc[2]
    return int(round(u)), int(round(v))


def _red_mask(rgb):
    r = rgb.astype(float)
    return (r[..., 0] > 90) & (r[..., 0] > 1.5 * r[..., 1]) \
        & (r[..., 0] > 1.5 * r[..., 2])


def test_gt_projects_onto_block(frame11):
    x, y, th = frame11.gt
    top_center = np.array([x, y, spec.BLOCK_HEIGHT])
    u, v = _project(top_center)
    mask = _red_mask(frame11.obs["rgb"])
    window = mask[max(v - 6, 0):v + 7, max(u - 6, 0):u + 7]
    assert window.any(), "projected GT block center missed the rendered block"


def test_backprojection_recovers_table_plane(frame11):
    depth = frame11.obs["depth"]
    pts = sensor.backproject(np.nan_to_num(depth, nan=np.inf))
    finite = np.isfinite(pts).all(axis=-1) & ~np.isnan(depth)
    # table pixels: within the tabletop, below block height
    on_table = finite & (np.abs(pts[..., 0]) < 0.4) \
        & (np.abs(pts[..., 1]) < 0.3) & (pts[..., 2] < 0.015)
    z = pts[..., 2][on_table]
    assert len(z) > 20000
    assert abs(np.median(z)) < 0.005, "table plane should sit near z=0"


# ---------------------------------------------------------------------------
# Depth-noise model
# ---------------------------------------------------------------------------
def test_noise_sigma_matches_spec():
    model, data = scene.build_scene(seed=SEED)
    scene.set_arm(model, data, spec.ARM_PARKED)
    scene.settle(model, data, 0.4)
    with sensor.CameraRig(model) as rig:
        clean = rig.depth(data)
    cos_i = sensor.incidence_cos(clean)
    # accumulate residuals over several noise realizations
    sel = (cos_i > 0.75) & (clean < spec.DEPTH_MAX_RANGE)
    resid = []
    for k in range(5):
        noisy = sensor.apply_depth_noise(clean, SEED, frame_idx=k)
        ok = sel & ~np.isnan(noisy)
        resid.append((noisy - clean)[ok] / sensor.depth_sigma(cos_i)[ok])
    resid = np.concatenate(resid)
    assert abs(np.std(resid) - 1.0) < 0.05, \
        "normalized residual std should be ~1 if sigma(theta) is honored"


def test_grazing_pixels_drop_out():
    model, data = scene.build_scene(seed=SEED)
    scene.set_arm(model, data, spec.ARM_PARKED)
    scene.settle(model, data, 0.4)
    with sensor.CameraRig(model) as rig:
        clean = rig.depth(data)
    cos_i = sensor.incidence_cos(clean)
    noisy = sensor.apply_depth_noise(clean, SEED)
    grazing = (cos_i < spec.DEPTH_DROP_COS) & (clean < spec.DEPTH_MAX_RANGE)
    if grazing.sum() > 50:
        frac = np.mean(np.isnan(noisy[grazing]))
        assert frac > 0.5 * spec.DEPTH_DROP_P
    frontal = cos_i > 0.8
    assert np.mean(np.isnan(noisy[frontal])) < 0.05


# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------
def test_ik_tracks_stroke_targets():
    model, data = scene.build_scene(seed=SEED)
    ik_data = mujoco.MjData(model)
    for tgt in [(-0.18, -0.12, spec.PUSH_TIP_HEIGHT),
                (0.11, -0.03, spec.PUSH_TIP_HEIGHT),
                (0.2, 0.15, spec.PUSH_TIP_HEIGHT),
                (0.3, -0.2, spec.PUSH_TIP_HEIGHT)]:
        _, res = episodes.ik_to(model, ik_data, tgt,
                                init7=episodes.ARM_REACH_INIT)
        assert res < 1e-3, f"IK residual {res*1000:.1f} mm at {tgt}"


def test_sampled_strokes_are_reachable():
    """Every stroke endpoint the sampler can produce must be IK-feasible."""
    model, data = scene.build_scene(seed=SEED)
    ik_data = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(40):
        bx = rng.uniform(-spec.SPAWN_HALF[0], spec.SPAWN_HALF[0])
        by = rng.uniform(-spec.SPAWN_HALF[1], spec.SPAWN_HALF[1])
        theta = rng.uniform(-np.pi, np.pi)
        shape = spec.BLOCK_SHAPES[int(rng.integers(len(spec.BLOCK_SHAPES)))]
        start, end = episodes.sample_stroke(rng, (bx, by, theta),
                                            toward_camera=rng.random() < 0.5,
                                            shape_name=shape)
        for tgt in (start, end):
            assert np.linalg.norm(tgt[:2] - episodes._BASE_XY) \
                >= episodes.MIN_BASE_DIST - 1e-9
            assert np.linalg.norm(tgt[:2] - episodes._BASE_XY) \
                <= episodes.MAX_BASE_DIST + 1e-9
            _, res = episodes.ik_to(model, ik_data, tgt,
                                    init7=episodes.ARM_REACH_INIT, iters=200)
            worst = max(worst, res)
    assert worst < 2e-3, f"worst sampled-stroke IK residual {worst*1000:.1f} mm"


# ---------------------------------------------------------------------------
# Episodes: real pushing, occlusion by construction
# ---------------------------------------------------------------------------
@pytest.mark.heavy  # simulates a full push episode
def test_episode_is_deterministic(episode23):
    again = episodes.push_episode(23, render=False)
    assert len(again.frames) == len(episode23.frames)
    assert np.allclose(np.array(again.gts), np.array(episode23.gts))
    assert again.occluded_flags == episode23.occluded_flags


@pytest.mark.heavy  # simulates a full push episode
def test_push_actually_displaces_block(episode23, episode51):
    for ep in (episode23, episode51):
        gts = np.array(ep.gts)
        disp = np.linalg.norm(gts[-1][:2] - gts[0][:2])
        assert disp > 0.05, "push episode barely moved the block"
        assert np.all(np.abs(gts[:, 0]) < spec.POSE_XY_BOUND)
        assert np.all(np.abs(gts[:, 1]) < spec.POSE_XY_BOUND)


@pytest.mark.heavy  # simulates a full push episode
def test_episode_has_occluded_frames(episode23, episode51):
    for ep in (episode23, episode51):
        flags = np.array(ep.occluded_flags)
        assert flags.mean() >= 0.10, "too few occluded frames"
        assert flags.sum() >= 8


@pytest.mark.heavy  # simulates a full push episode
def test_block_moves_while_occluded(episode23):
    gts = np.array(episode23.gts)
    flags = np.array(episode23.occluded_flags)
    idx = np.where(flags)[0]
    moved = np.linalg.norm(gts[idx[-1]][:2] - gts[idx[0]][:2])
    assert moved > 0.01, \
        "block should keep moving during occlusion (anti last-pose-hold)"


@pytest.mark.heavy  # simulates a full push episode
def test_frames_at_expected_rate(episode23):
    ts = [f.t for f in episode23.frames]
    dts = np.diff(ts)
    assert np.allclose(dts, 1.0 / spec.FRAME_HZ, atol=1e-6)


@pytest.mark.heavy
@pytest.mark.parametrize("shape", spec.BLOCK_SHAPES)
def test_tracking_window_contains_rotation(shape):
    from dev.audit_motion import (MIN_MEDIAN_ROTATION_DEG,
                                  MIN_ROTATING_FRACTION, motion_metrics)
    from harness import battery

    metrics = []
    for seed in spec.DESIGN_SEEDS:
        episode = episodes.push_episode(seed, render=False, shapes=(shape,),
                                        max_frames=battery.EPISODE_MAX_FRAMES)
        assert len(episode.frames) == battery.EPISODE_MAX_FRAMES
        assert episode.max_height_m < .02
        assert episode.max_tilt_deg < 15
        motion = episodes.occluded_motion(episode)
        assert motion["rotation_deg"] >= episodes.OCCLUDED_ROTATION_DEG
        assert motion["rotating_steps"] >= episodes.OCCLUDED_ROTATING_STEPS
        metrics.append(motion_metrics(episode))
    assert np.median([m["rotation_range_deg"] for m in metrics]) >= MIN_MEDIAN_ROTATION_DEG
    assert np.mean([m["rotating_fraction"] for m in metrics]) >= MIN_ROTATING_FRACTION
    assert np.mean([m["occluded_fraction"] for m in metrics]) >= 0.10


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def test_occluded_rotation_requires_both_endpoints():
    ep = episodes.Episode(seed=0, frames=[
        episodes.Frame(None, (0, 0, np.deg2rad(angle)), hidden, i/10)
        for i, (angle, hidden) in enumerate(
            [(0, False), (30, True), (40, True), (90, False)])
    ])
    motion = episodes.occluded_motion(ep)
    assert motion["rotation_deg"] == pytest.approx(10)
    assert motion["rotating_steps"] == 1


def test_wrap_angle():
    assert spec.wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert spec.wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert spec.wrap_angle(0.3) == pytest.approx(0.3)


def test_intrinsics_extrinsics_shapes():
    K = spec.intrinsics()
    T = spec.extrinsics()
    assert K.shape == (3, 3) and T.shape == (4, 4)
    assert np.allclose(T[:3, 3], spec.CAM_POS)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
