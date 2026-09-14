"""The camera extrinsics, against the simulator they were measured from.

The library turns a pixel into a base-frame target, and a wrong extrinsic does that
silently: the arm drives confidently to the wrong place and nothing raises. So these
tests build a real RoboCasa env, render real depth, and check the deprojected cloud
against geometry the simulator already knows.

They need a GPU and the asset tree, so they SKIP where those are absent -- which is why
tests/test_speedrun_skills.py covers the same functions analytically as well. The
constants in camera.py came from this measurement; this is what re-checks them.
"""

from __future__ import annotations

import os

# BEFORE mujoco is imported, which pytest does while collecting anything that touches the
# harness. Setting it inside the fixture is too late: rendering silently falls back to the
# CPU, whose depth buffer is not the normalised one robosuite's converter asserts on, and
# the whole module errors in setup for a reason that looks nothing like the cause.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from harness.skills import camera as C  # noqa: E402

TASK = "PickPlaceCounterToSink"

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROBOCASA_ASSET_DIR"),
    reason="needs a GPU and ROBOCASA_ASSET_DIR (make sim-robocasa)")


@pytest.fixture(scope="module")
def scene():
    """One env, one observation with depth, and the ground truth beside it."""
    from harness.env import make_env

    env = make_env(TASK, split="pretrain", seed=0, camera_height=256,
                   camera_width=256, camera_depths=True)
    raw = env.reset()
    sim = env.sim

    import robosuite.utils.camera_utils as CU

    obs = dict(raw)
    for cam in C.CAMERAS:
        obs[f"{cam}_depth"] = np.asarray(
            CU.get_real_depth_map(sim, raw[f"{cam}_depth"])).squeeze()

    site = env.robots[0].robot_model.base.correct_naming("center")
    sid = sim.model.site_name2id(site)
    base_pos = sim.data.site_xpos[sid].copy()
    base_rot = sim.data.site_xmat[sid].reshape(3, 3).copy()

    bid = sim.model.body_name2id(env.objects["obj"].root_body)
    target = base_rot.T @ (sim.data.body_xpos[bid] - base_pos)

    yield {"obs": obs, "target_in_base": target, "env": env}
    env.close()


def test_every_camera_puts_the_counter_at_the_same_height(scene):
    """Three cameras, three extrinsics, one countertop.

    Each sees it from a different pose, so a wrong extrinsic would put it somewhere else.
    Agreeing to a centimetre is a cross-check no single camera could give.
    """
    peaks = []
    for cam in C.CAMERAS:
        cloud = C.cloud_in_base(scene["obs"], cam, stride=2)
        assert len(cloud) > 1000
        counts, edges = np.histogram(cloud[:, 2], bins=200)
        peak = int(np.argmax(counts))
        peaks.append((edges[peak] + edges[peak + 1]) / 2)
    assert 0.15 < min(peaks) and max(peaks) < 0.30, f"counter at {peaks}"
    assert max(peaks) - min(peaks) < 0.02, f"cameras disagree: {peaks}"


def test_the_cloud_lands_on_the_object_the_simulator_knows_about(scene):
    """The end-to-end claim: a pixel of the target deprojects to where the target is."""
    cloud = C.cloud_in_base(scene["obs"], "robot0_agentview_right", stride=1)
    gap = np.linalg.norm(cloud - scene["target_in_base"], axis=1).min()
    # Surface, not centroid: a few centimetres is the object's own radius.
    assert gap < 0.06, f"nearest cloud point is {gap*100:.1f} cm from the object"


def test_the_mujoco_to_cv_flip_is_load_bearing(scene):
    """Guards the one thing that would silently ruin every target.

    MuJoCo cameras look down -z with +y up; the deprojection here uses +z forward and +y
    down. Skipping that half turn leaves a plausible-looking cloud in the wrong place.
    """
    obs = scene["obs"]
    cam = "robot0_agentview_right"
    rot, off = C.camera_pose_in_base(cam)
    depth = obs[f"{cam}_depth"]
    mat = C.intrinsics(cam, depth.shape[0], depth.shape[1])
    pts = C.deproject(depth, mat).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]

    flipped = rot @ np.diag([1.0, -1.0, -1.0])          # undo the conversion
    wrong = pts @ flipped.T + off
    gap = np.linalg.norm(wrong - scene["target_in_base"], axis=1).min()
    assert gap > 0.5, "the convention flip made no difference; something else is wrong"


def test_the_wrist_camera_needs_an_observation_and_the_others_do_not(scene):
    for cam in C._AGENTVIEW_POSE:
        rot, off = C.camera_pose_in_base(cam)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    with pytest.raises(ValueError, match="moves"):
        C.camera_pose_in_base(C.WRIST)
    rot, _ = C.camera_pose_in_base(C.WRIST, scene["obs"])
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-6)


def test_the_agentview_extrinsics_survive_torso_and_arm_motion(scene):
    """They hang off the mobile base, which the torso moves UNDER rather than with. If
    that were not so they could not be constants at all."""
    env = scene["env"]
    site = env.robots[0].robot_model.base.correct_naming("center")

    def measured(cam):
        sim = env.sim
        sid = sim.model.site_name2id(site)
        p_b = sim.data.site_xpos[sid].copy()
        r_b = sim.data.site_xmat[sid].reshape(3, 3).copy()
        cid = sim.model.camera_name2id(cam)
        t = r_b.T @ (sim.data.cam_xpos[cid] - p_b)
        r = r_b.T @ sim.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        return r, t

    action = np.zeros(12)
    action[11] = -1.0
    action[10] = 1.0        # torso
    action[0] = 0.5         # arm
    for _ in range(30):
        env.step(action)

    for cam in C._AGENTVIEW_POSE:
        rot, off = C.camera_pose_in_base(cam)
        got_rot, got_off = measured(cam)
        np.testing.assert_allclose(got_off, off, atol=1e-9)
        np.testing.assert_allclose(got_rot, rot, atol=1e-8)


def test_base_to_pixel_inverts_pixel_to_base_point(scene):
    """Round-trip through the extrinsic, on a pixel with real depth behind it."""
    obs = scene["obs"]
    cam = "robot0_agentview_left"
    depth = obs[f"{cam}_depth"]
    v, u = np.argwhere(np.isfinite(depth) & (depth > 0.3))[len(depth) // 2]
    point = C.pixel_to_base_point(obs, cam, u, v)
    assert point is not None
    back_u, back_v, z = C.base_to_pixel(obs, cam, point)
    assert z > 0
    assert abs(back_u - u) < 1.0 and abs(back_v - v) < 1.0
