"""The L2/L3 library's pure tiers, against analytic ground truth.

These functions are the agent's only tools for turning pixels into a base-frame target,
so a quiet error here is a wrong target in every trial rather than a crash. Everything
below has a closed-form answer -- a known pinhole, a box of known size, a known base
yaw -- so the tests check the value, not merely the shape.

No simulator, no GPU and no model weights: the perception tier is a socket client and is
covered separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.skills import camera as C
from harness.skills import geometry as G
from harness.skills import transforms as T


# -- transforms ----------------------------------------------------------------

def test_quat_and_mat_round_trip():
    for quat in ([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0],
                 [0.5, 0.5, 0.5, 0.5], [0.0, 0.3826834, 0.0, 0.9238795]):
        mat = T.quat_to_mat(quat)
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(T.quat_to_mat(T.mat_to_quat(mat)), mat, atol=1e-9)


def test_quat_convention_is_xyzw():
    """xyzw, like every observation key. Read as wxyz these are different rotations."""
    # A quarter turn about z: xyzw (0, 0, sin45, cos45) takes +x to +y.
    quat = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    np.testing.assert_allclose(T.quat_to_mat(quat) @ [1.0, 0, 0], [0, 1, 0], atol=1e-9)


def test_invert_transform_is_the_inverse():
    mat = T.make_transform([1.0, -2.0, 0.5], [0.0, 0.0, np.sin(0.3), np.cos(0.3)])
    np.testing.assert_allclose(mat @ T.invert_transform(mat), np.eye(4), atol=1e-9)


def test_decompose_transform_recovers_what_make_transform_built():
    pos, quat = [0.1, 0.2, 0.3], [0.0, np.sin(0.2), 0.0, np.cos(0.2)]
    back_pos, back_quat = T.decompose_transform(T.make_transform(pos, quat))
    np.testing.assert_allclose(back_pos, pos, atol=1e-9)
    np.testing.assert_allclose(T.quat_to_mat(back_quat), T.quat_to_mat(quat), atol=1e-9)


def test_transform_points_takes_one_point_or_many():
    mat = T.make_transform([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(T.transform_points(mat, [0.0, 0.0, 0.0]), [1, 0, 0])
    out = T.transform_points(mat, [[0.0, 0, 0], [0.0, 1, 0]])
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[1, 0, 0], [1, 1, 0]])


def test_world_to_base_undoes_a_yawed_base():
    """The gotcha the whole task warns about, in one assertion.

    A base yawed 90 degrees and standing at (2, 1, 0): a world point one metre in front
    of it along +y is one metre along the base's own +x.
    """
    obs = {"robot0_base_pos": [2.0, 1.0, 0.0],
           "robot0_base_quat": [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]}
    np.testing.assert_allclose(T.world_to_base(obs, [2.0, 2.0, 0.0]), [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(
        T.base_to_world(obs, T.world_to_base(obs, [0.3, -0.4, 0.9])),
        [0.3, -0.4, 0.9], atol=1e-9)


def test_mat_to_axisangle_recovers_axis_times_angle():
    np.testing.assert_allclose(
        T.mat_to_axisangle(T.quat_to_mat([0.0, 0.0, np.sin(0.3), np.cos(0.3)])),
        [0, 0, 0.6], atol=1e-9)
    np.testing.assert_allclose(T.mat_to_axisangle(np.eye(3)), [0, 0, 0])
    # A half turn, where the trace formula degenerates.
    np.testing.assert_allclose(
        T.mat_to_axisangle(np.diag([-1.0, -1.0, 1.0])), [0, 0, np.pi], atol=1e-9)


def test_mat_to_axisangle_takes_the_short_way_round():
    """A 5.8 rad turn about z is commanded as -0.48 rad, not as most of a full circle."""
    big = T.quat_to_mat([0.0, 0.0, np.sin(2.9), np.cos(2.9)])
    np.testing.assert_allclose(
        T.mat_to_axisangle(big), [0, 0, 5.8 - 2 * np.pi], atol=1e-9)


def test_normalize_leaves_a_zero_vector_alone():
    np.testing.assert_allclose(T.normalize([0.0, 0.0, 0.0]), [0, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(T.normalize([3.0, 4.0, 0.0])), 1.0)


# -- camera --------------------------------------------------------------------

def test_intrinsics_match_the_published_fovy():
    """The instruction publishes f = (H/2) / tan(fovy/2); this must be that number."""
    k = C.intrinsics("robot0_agentview_left", 128)
    assert k[0, 0] == pytest.approx((128 / 2) / np.tan(np.radians(60) / 2))
    assert C.intrinsics("robot0_eye_in_hand", 128)[0, 0] == pytest.approx(
        (128 / 2) / np.tan(np.radians(75) / 2))


def test_focal_length_scales_with_the_size_you_rendered():
    small = C.intrinsics("robot0_agentview_left", 128)[0, 0]
    big = C.intrinsics("robot0_agentview_left", 512)[0, 0]
    assert big == pytest.approx(4 * small)


def test_intrinsics_accepts_an_observation_key_name():
    a = C.intrinsics("robot0_eye_in_hand_image", 256)
    b = C.intrinsics("robot0_eye_in_hand_depth", 256)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, C.intrinsics("robot0_eye_in_hand", 256))


def test_unknown_camera_is_refused():
    with pytest.raises(KeyError):
        C.intrinsics("robot0_frontview", 128)


def test_deproject_inverts_a_known_projection():
    """Project a point by hand, then deproject the pixel it landed on."""
    k = C.intrinsics("robot0_agentview_left", 64)
    fx, cx, cy = k[0, 0], k[0, 2], k[1, 2]
    point = np.array([0.05, -0.03, 0.8])
    u = int(round(point[0] * fx / point[2] + cx))
    v = int(round(point[1] * fx / point[2] + cy))
    depth = np.full((64, 64), np.nan)
    depth[v, u] = point[2]
    np.testing.assert_allclose(C.deproject(depth, k)[v, u], point, atol=2e-2)


def test_deproject_marks_unusable_depth_rather_than_placing_it_at_the_origin():
    k = C.intrinsics("robot0_agentview_left", 8)
    depth = np.zeros((8, 8))
    depth[0, 0] = np.inf
    assert np.isnan(C.deproject(depth, k)).all()


# -- geometry ------------------------------------------------------------------

def _slab(z, n=40, extent=0.5):
    g = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def test_remove_support_plane_takes_the_counter_and_leaves_the_object():
    counter = _slab(0.90)
    obj = np.random.default_rng(0).uniform([-0.02, -0.02, 0.95],
                                           [0.02, 0.02, 1.00], size=(200, 3))
    kept = G.remove_support_plane(np.vstack([counter, obj]))
    assert len(kept) == len(obj)
    assert kept[:, 2].min() > 0.92


def test_remove_support_plane_leaves_a_cloud_with_no_surface_alone():
    """No band holds enough of the cloud, so nothing is a surface and nothing goes."""
    blob = np.random.default_rng(1).uniform(-0.5, 0.5, size=(500, 3))
    assert len(G.remove_support_plane(blob)) == len(blob)


def test_cluster_points_separates_two_objects():
    rng = np.random.default_rng(2)
    a = rng.normal([0.0, 0.0, 1.0], 0.01, size=(120, 3))
    b = rng.normal([0.5, 0.0, 1.0], 0.01, size=(60, 3))
    groups = G.cluster_points(np.vstack([a, b]), radius=0.03, min_size=20)
    assert len(groups) == 2
    assert len(groups[0]) > len(groups[1])          # largest first
    np.testing.assert_allclose(groups[0].mean(axis=0), [0, 0, 1], atol=0.02)


def test_cluster_points_drops_specks_and_survives_an_empty_cloud():
    assert G.cluster_points(np.zeros((0, 3))) == []
    speck = np.random.default_rng(3).normal(0, 0.001, size=(5, 3))
    assert G.cluster_points(speck, radius=0.03, min_size=20) == []


def test_oriented_bbox_recovers_a_rotated_box():
    rng = np.random.default_rng(4)
    half = np.array([0.10, 0.04, 0.03])
    local = rng.uniform(-half, half, size=(4000, 3))
    yaw = 0.6
    rot = T.quat_to_mat([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    cloud = local @ rot.T + np.array([1.0, 2.0, 0.9])

    box = G.oriented_bbox(cloud)
    np.testing.assert_allclose(box["center"], [1.0, 2.0, 0.9], atol=0.01)
    np.testing.assert_allclose(np.sort(box["extent"])[::-1], np.sort(2 * half)[::-1],
                               atol=0.02)
    # The box's major axis is the box's long side, up to sign.
    axes = T.quat_to_mat(box["quat"])
    assert abs(float(axes[:, 0] @ (rot @ [1.0, 0, 0]))) > 0.99


def test_oriented_bbox_is_gravity_aligned():
    """Its z axis is up whatever the cloud looks like -- objects sit on surfaces."""
    tilted = np.random.default_rng(5).uniform(-1, 1, size=(500, 3)) * [1.0, 0.2, 0.6]
    axes = T.quat_to_mat(G.oriented_bbox(tilted)["quat"])
    np.testing.assert_allclose(axes[:, 2], [0, 0, 1], atol=1e-9)


def test_oriented_bbox_needs_a_point():
    with pytest.raises(ValueError):
        G.oriented_bbox(np.zeros((0, 3)))


def _grasp(approach, score_pos=(0.0, 0.0, 1.0)):
    mat = np.eye(4)
    z = T.normalize(approach)
    x = T.normalize(np.cross([0.0, 1.0, 0.0], z)) if abs(z[1]) < 0.9 else np.array([1.0, 0, 0])
    mat[:3, 0], mat[:3, 1], mat[:3, 2] = x, np.cross(z, x), z
    mat[:3, 3] = score_pos
    return mat


def test_select_top_down_grasp_prefers_the_downward_one():
    grasps = np.stack([_grasp([1.0, 0.0, 0.0]), _grasp([0.0, 0.0, -1.0]),
                       _grasp([0.3, 0.0, -1.0])])
    assert G.select_top_down_grasp(grasps) == 1


def test_select_top_down_grasp_breaks_ties_on_score():
    grasps = np.stack([_grasp([0.0, 0.0, -1.0]), _grasp([0.0, 0.0, -1.0])])
    assert G.select_top_down_grasp(grasps, scores=[0.1, 0.9]) == 1


def test_select_top_down_grasp_returns_none_when_nothing_is_top_down():
    """A real answer: approaching from above will not work here."""
    grasps = np.stack([_grasp([1.0, 0.0, 0.0]), _grasp([0.0, 1.0, 0.0])])
    assert G.select_top_down_grasp(grasps, max_tilt_deg=45.0) is None
    assert G.select_top_down_grasp(grasps, max_tilt_deg=None) is not None


def test_select_top_down_grasp_refuses_a_wrong_shape():
    with pytest.raises(ValueError):
        G.select_top_down_grasp(np.zeros((3, 3)))
    assert G.select_top_down_grasp(np.zeros((0, 4, 4))) is None


# -- primitives ----------------------------------------------------------------
#
# Driven against a first-order arm rather than a mock: `commanded is not achieved` is the
# property these functions exist to cope with, so a fake that teleports to the command
# would pass every one of them while proving nothing.

from test_speedrun_service import FakeEnv, Harness           # noqa: E402
from harness.skills import primitives as P        # noqa: E402

_COMMAND_SCALE = 0.05        # an action of +-1 commands +-5 cm
_ROT_SCALE = 0.5             # ... or +-0.5 rad
_LAG = 0.35                  # fraction of the commanded delta achieved per step
_FINGER_OPEN = 0.04


def _rodrigues(w):
    """Axis-angle vector -> rotation matrix, for integrating the commanded twist."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, dtype=float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class KinematicEnv(FakeEnv):
    """An arm, a gripper and a mobile base that move toward what was commanded.

    Positions are the base frame for the end-effector and the world for the base, exactly
    as the real observation reports them. Nothing here is RoboCasa; it is the smallest
    thing that makes a servo loop meaningful.
    """

    # Where the base starts. A class attribute rather than an argument because the
    # harness's env factory builds these itself -- and setting it on a live env would be
    # reaching behind the daemon, which serves `observe` from the last observation it
    # took and would hand the skill a stale pose.
    INITIAL_YAW = 0.0

    def __init__(self, eef=(0.4, 0.0, 1.0), fingers=_FINGER_OPEN, **kw):
        super().__init__(**kw)
        self.eef = np.array(eef, dtype=float)
        self.eef_rot = np.eye(3)
        self.fingers = float(fingers)
        self.base_xy = np.zeros(2)
        self.base_yaw = float(type(self).INITIAL_YAW)
        self.vel = 0.0
        self.blocked_axis = None          # set to an index to jam that axis
        self.bias = np.zeros(3)           # per-step sag the arm must be held against

    def _obs(self):
        yaw = self.base_yaw
        quat = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
        return {
            "robot0_base_to_eef_pos": self.eef.copy(),
            "robot0_base_to_eef_quat": T.mat_to_quat(self.eef_rot),
            "robot0_base_pos": [self.base_xy[0], self.base_xy[1], 0.0],
            "robot0_base_quat": quat,
            "robot0_gripper_qpos": [self.fingers, -self.fingers],
            "robot0_joint_vel": [self.vel] * 7,
            "robot0_proprio-state": [0.0, 1.0],
            "robot0_eef_pos": [0.0, 0.0, 0.0],
            "step": self.steps,
            "object-state": [0.0] * 42,
            **self._images(),
        }

    def reset(self, seed=None):
        self.steps = 0
        self._success = False
        self.base_xy = np.zeros(2)
        self.base_yaw = float(type(self).INITIAL_YAW)
        self.draws.append(int(self.rng.integers(0, 2**31 - 1)))
        return self._obs()

    def step(self, action):
        assert len(action) == self.dim
        self.steps += 1
        a = np.asarray(action, dtype=float)

        delta = a[0:3] * _COMMAND_SCALE * _LAG
        if self.blocked_axis is not None:
            delta[self.blocked_axis] = 0.0
        self.eef = self.eef + delta - self.bias
        self.eef_rot = _rodrigues(a[3:6] * _ROT_SCALE * _LAG) @ self.eef_rot
        self.vel = float(np.abs(delta).max()) * 10.0

        target = 0.0 if a[6] > 0 else _FINGER_OPEN
        self.fingers += (target - self.fingers) * 0.5

        # Applied regardless of a[11]: the real HybridMobileBase's flag only selects
        # the arm's goal-update mode, the base velocity slice acts either way.
        # The slides sit BELOW the yaw joint, so they act in the reset-yaw frame,
        # not the body's current heading.
        yaw0 = float(type(self).INITIAL_YAW)
        rot = np.array([[np.cos(yaw0), -np.sin(yaw0)],
                        [np.sin(yaw0), np.cos(yaw0)]])
        self.base_xy = self.base_xy + rot @ (a[7:9] * _COMMAND_SCALE * _LAG)
        self.base_yaw += a[9] * 0.05 * _LAG

        if self.success_after >= 0 and self.steps >= self.success_after:
            self._success = True
        return self._obs(), 0.0, False, {}


def _sim(tmp_path, **kw):
    return Harness(tmp_path, env_cls=KinematicEnv, steps=5000,
                   max_episode_steps=4000, success_after=-1, **kw)


def test_reach_converges_and_reports_what_it_spent(tmp_path):
    h = _sim(tmp_path)
    try:
        h.client.reset()
        res = P.reach(h.client, [0.5, 0.1, 1.05], tol=0.01, gripper=P._OPEN)
        assert res["outcome"] == P.REACHED
        assert res["steps"] >= 1
        got = h.client.observe(P.PROPRIO_ONLY)["obs"]["robot0_base_to_eef_pos"]
        assert np.linalg.norm(np.asarray(got) - [0.5, 0.1, 1.05]) <= 0.01
    finally:
        h.stop()


def test_reach_costs_nothing_when_already_there(tmp_path):
    """No step is taken, so none is charged -- and the reply says so rather than lying."""
    h = _sim(tmp_path)
    try:
        h.client.reset()
        before = h.client.status()["steps_used"]
        res = P.reach(h.client, [0.4, 0.0, 1.0], tol=0.05, gripper=P._OPEN)
        assert res["outcome"] == P.REACHED
        assert res["steps"] == 0
        assert h.client.status()["steps_used"] == before
    finally:
        h.stop()


def test_reach_gives_up_on_an_unreachable_target_instead_of_burning_the_budget(tmp_path):
    """The stall detector is a budget device: max_steps is a ceiling, not a plan."""
    h = _sim(tmp_path)
    try:
        h.client.reset()
        h.envs[-1].blocked_axis = 0
        res = P.reach(h.client, [9.0, 0.0, 1.0], tol=0.01, max_steps=400, gripper=P._OPEN)
        assert res["outcome"] == P.GAVE_UP
        assert res["steps"] < 400
    finally:
        h.stop()


def test_move_eef_is_relative(tmp_path):
    h = _sim(tmp_path)
    try:
        h.client.reset()
        assert P.move_eef(h.client, [0.0, 0.0, 0.05], tol=0.01, gripper=P._OPEN)["outcome"] == P.REACHED
        got = h.client.observe(P.PROPRIO_ONLY)["obs"]["robot0_base_to_eef_pos"]
        assert got[2] == pytest.approx(1.05, abs=0.01)
    finally:
        h.stop()


def test_grasp_closes_and_release_opens(tmp_path):
    h = _sim(tmp_path)
    try:
        h.client.reset()
        assert P.grasp(h.client)["outcome"] == P.REACHED
        assert abs(h.envs[-1].fingers) < 0.005
        assert P.release(h.client)["outcome"] == P.REACHED
        assert h.envs[-1].fingers > 0.03
    finally:
        h.stop()


def test_lift_holds_the_grip_while_it_rises(tmp_path):
    """A lift that opened the hand would drop what the grasp just caught."""
    h = _sim(tmp_path)
    try:
        h.client.reset()
        P.grasp(h.client)
        assert P.lift(h.client, 0.10)["outcome"] == P.REACHED
        assert abs(h.envs[-1].fingers) < 0.005
        assert h.envs[-1].eef[2] == pytest.approx(1.10, abs=0.01)
    finally:
        h.stop()


def test_a_reach_after_a_grasp_keeps_the_hand_shut(tmp_path):
    h = _sim(tmp_path)
    try:
        h.client.reset()
        P.grasp(h.client)
        P.reach(h.client, [0.5, 0.1, 1.05], gripper=P._CLOSED)
        assert abs(h.envs[-1].fingers) < 0.005
    finally:
        h.stop()


def test_settle_waits_for_the_arm_to_stop(tmp_path):
    h = _sim(tmp_path)
    try:
        h.client.reset()
        P.reach(h.client, [0.6, 0.0, 1.0], tol=0.01, gripper=P._OPEN)
        assert P.settle(h.client, tol=1e-3, gripper=P._OPEN)["outcome"] == P.REACHED
        assert h.envs[-1].vel <= 1e-3
    finally:
        h.stop()


class YawedEnv(KinematicEnv):
    """The same robot, parked facing world +y. RoboCasa yaws the base every episode."""

    INITIAL_YAW = np.pi / 2


def test_move_base_drives_in_the_bases_own_frame(tmp_path):
    """Half a metre forward is forward for the robot, not +x in the world."""
    h = Harness(tmp_path, env_cls=YawedEnv, steps=5000, max_episode_steps=4000,
                success_after=-1)
    try:
        h.client.reset()
        res = P.move_base(h.client, dx=0.5, tol=0.03, gripper=P._OPEN)
        assert res["outcome"] == P.REACHED
        assert h.envs[-1].base_xy[1] == pytest.approx(0.5, abs=0.03)
        assert abs(h.envs[-1].base_xy[0]) < 0.03
    finally:
        h.stop()


def test_move_base_turns(tmp_path):
    h = _sim(tmp_path)
    try:
        h.client.reset()
        assert P.move_base(h.client, dyaw=0.6, gripper=P._OPEN)["outcome"] == P.REACHED
        assert h.envs[-1].base_yaw == pytest.approx(0.6, abs=0.05)
    finally:
        h.stop()


def test_move_base_translates_correctly_after_a_turn(tmp_path):
    """dx is forward for the CURRENT heading even though the slides never turn with it."""
    h = _sim(tmp_path)
    try:
        h.client.reset()
        assert P.move_base(h.client, dyaw=0.6, gripper=P._OPEN)["outcome"] == P.REACHED
        yaw = h.envs[-1].base_yaw
        res = P.move_base(h.client, dx=0.5, tol=0.03, gripper=P._OPEN)
        assert res["outcome"] == P.REACHED
        np.testing.assert_allclose(
            h.envs[-1].base_xy, [0.5 * np.cos(yaw), 0.5 * np.sin(yaw)], atol=0.04)
    finally:
        h.stop()


def test_reach_takes_an_orientation_target(tmp_path):
    """How the hand arrives, not just where: a quarter turn about z, held to rot_tol."""
    want = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    h = _sim(tmp_path)
    try:
        h.client.reset()
        res = P.reach(h.client, [0.5, 0.1, 1.05], tol=0.01, rot=want, rot_tol=0.05,
                      gripper=P._OPEN)
        assert res["outcome"] == P.REACHED
        assert res["pos_err"] <= 0.01 and res["rot_err"] <= 0.05
        np.testing.assert_allclose(h.envs[-1].eef_rot, want, atol=0.06)
        # ... and as an xyzw quaternion, which is how observations spell it.
        res = P.reach(h.client, [0.5, 0.1, 1.05], rot=T.mat_to_quat(np.eye(3)),
                      gripper=P._OPEN)
        assert res["outcome"] == P.REACHED
    finally:
        h.stop()


def test_a_reply_carries_the_final_errors(tmp_path):
    """A gave_up at 1 cm and a gave_up at 30 cm are different situations, and telling
    them apart must not cost another observation."""
    h = _sim(tmp_path)
    try:
        h.client.reset()
        h.envs[-1].blocked_axis = 0
        res = P.reach(h.client, [1.4, 0.0, 1.0], tol=0.01, max_steps=400, gripper=P._OPEN)
        assert res["outcome"] == P.GAVE_UP
        assert res["pos_err"] == pytest.approx(1.0, abs=0.05)
    finally:
        h.stop()


def test_reach_converges_under_steady_load(tmp_path):
    """A proportional command alone parks short where holding position needs a standing
    command; the integral term must walk that residual off."""
    h = _sim(tmp_path)
    try:
        h.client.reset()
        h.envs[-1].bias = np.array([0.003, 0.0, 0.0])
        res = P.reach(h.client, [0.5, 0.1, 1.05], tol=0.01, gripper=P._OPEN)
        assert res["outcome"] == P.REACHED
        assert res["pos_err"] <= 0.01
    finally:
        h.stop()


def test_gripper_is_required():
    with pytest.raises(TypeError):
        P.reach(None, [0.5, 0.1, 1.05])
    with pytest.raises(TypeError):
        P.settle(None)
    with pytest.raises(TypeError):
        P.move_base(None, dx=0.1)


def test_grasp_and_release_report_the_gap(tmp_path):
    h = _sim(tmp_path)
    try:
        h.client.reset()
        assert P.grasp(h.client)["gap"] == pytest.approx(0.0, abs=0.01)
        assert P.release(h.client)["gap"] == pytest.approx(2 * _FINGER_OPEN, abs=0.01)
    finally:
        h.stop()


def test_move_base_hold_eef_pins_the_hand_in_the_world(tmp_path):
    """The base drives away; the hand, holding, stays put in the world."""
    h = _sim(tmp_path)
    try:
        h.client.reset()
        P.grasp(h.client)
        env = h.envs[-1]
        world_before = env.base_xy + [0, 0] + env.eef[:2]      # yaw is zero here
        res = P.move_base(h.client, dx=-0.3, tol=0.03, hold_eef=True, gripper=P._CLOSED)
        assert res["outcome"] == P.REACHED
        assert env.base_xy[0] == pytest.approx(-0.3, abs=0.03)
        world_after = env.base_xy + env.eef[:2]
        np.testing.assert_allclose(world_after, world_before, atol=0.04)
        assert res["eef_drift"] < 0.04
        assert abs(env.fingers) < 0.005                        # still holding
    finally:
        h.stop()


def test_a_primitive_stops_the_moment_the_episode_ends(tmp_path):
    """It must never step a dead episode -- that is a RemoteError, not a verdict."""
    h = Harness(tmp_path, env_cls=KinematicEnv, steps=5000, max_episode_steps=5,
                success_after=-1)
    try:
        h.client.reset()
        res = P.reach(h.client, [3.0, 0.0, 1.0], tol=0.001, max_steps=200, gripper=P._OPEN)
        assert res["episode_over"] is True
        assert res["outcome"] in (P.REACHED, P.GAVE_UP)
    finally:
        h.stop()


def test_a_primitive_started_on_a_dead_episode_is_blocked_not_an_exception(tmp_path):
    h = Harness(tmp_path, env_cls=KinematicEnv, steps=5000, max_episode_steps=2,
                success_after=-1)
    try:
        h.client.reset()
        P.reach(h.client, [3.0, 0.0, 1.0], tol=0.001, max_steps=50, gripper=P._OPEN)
        res = P.reach(h.client, [3.0, 0.0, 1.0], tol=0.001, max_steps=50,
                      gripper=P._OPEN)
        assert res["outcome"] == P.BLOCKED
        assert res["steps"] == 0
    finally:
        h.stop()


# -- perception ----------------------------------------------------------------
#
# The client, against a fake service speaking the same wire. The models themselves need a
# GPU and gated weights, so what is pinned here is the CONTRACT: the request the agent
# sends, the shape it gets back, and what happens when nothing matched or nothing is
# listening.

import socket as _socket                                    # noqa: E402
import threading as _threading                              # noqa: E402

from harness import protocol as _P               # noqa: E402
from harness.skills import perception as PER     # noqa: E402


class FakePerception:
    """A perception service that records what it was asked and replies with `answer`."""

    def __init__(self, tmp_path, answer=None, name="perception.sock"):
        self.path = str(tmp_path / name)
        self.answer = answer if answer is not None else {"ok": True, "results": []}
        self.seen = []
        self._srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self._srv.bind(self.path)
        self._srv.listen(4)
        self._stop = False
        self._thread = _threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            try:
                msg = _P.LineReader(conn).read()
                if msg is not None:
                    self.seen.append(msg)
                    _P.send(conn, self.answer)
            finally:
                conn.close()

    def stop(self):
        self._stop = True
        self._srv.close()


@pytest.fixture
def fake_perception(tmp_path, monkeypatch):
    svc = FakePerception(tmp_path)
    monkeypatch.setattr(PER, "DEFAULT_SOCKET", svc.path)
    yield svc
    svc.stop()


def test_segment_by_text_sends_the_frame_and_the_words(fake_perception):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[2, 3] = [255, 0, 0]
    PER.segment_by_text(rgb, "coffee mug", threshold=0.4)
    sent = fake_perception.seen[-1]
    assert sent["op"] == "segment_text"
    assert sent["text"] == "coffee mug"
    assert sent["threshold"] == pytest.approx(0.4)
    np.testing.assert_array_equal(sent["rgb"], rgb)     # arrays survive the wire intact


def test_nothing_matched_is_an_empty_list_not_an_error(fake_perception):
    assert PER.segment_by_text(np.zeros((4, 4, 3), np.uint8), "unicorn") == []


def test_results_come_back_with_masks_as_arrays(tmp_path, monkeypatch):
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    svc = FakePerception(tmp_path, answer={"ok": True, "results": [
        {"mask": mask, "box": [1.0, 1.0, 3.0, 3.0], "score": 0.9}]})
    monkeypatch.setattr(PER, "DEFAULT_SOCKET", svc.path)
    try:
        hits = PER.segment_by_text(np.zeros((4, 4, 3), np.uint8), "mug")
        assert len(hits) == 1
        np.testing.assert_array_equal(hits[0]["mask"], mask)
        assert hits[0]["score"] == pytest.approx(0.9)
    finally:
        svc.stop()


def test_segment_by_boxes_sends_pixel_corners(fake_perception):
    """Pixels, the same format results come back in, so one call feeds the next."""
    PER.segment_by_boxes(np.zeros((8, 8, 3), np.uint8), [[1, 2, 5, 6]], labels=[True])
    sent = fake_perception.seen[-1]
    assert sent["op"] == "segment_boxes"
    assert sent["boxes"] == [[1.0, 2.0, 5.0, 6.0]]
    assert sent["labels"] == [True]


def test_plan_grasp_returns_poses_and_scores(tmp_path, monkeypatch):
    grasps = np.tile(np.eye(4), (3, 1, 1))
    svc = FakePerception(tmp_path, answer={"ok": True, "grasps": grasps,
                                           "scores": np.array([0.3, 0.9, 0.5])})
    monkeypatch.setattr(PER, "DEFAULT_SOCKET", svc.path)
    try:
        g, s = PER.plan_grasp(np.random.default_rng(0).normal(size=(200, 3)))
        assert g.shape == (3, 4, 4)
        assert s.shape == (3,)
    finally:
        svc.stop()


def test_no_candidates_is_an_empty_stack(tmp_path, monkeypatch):
    svc = FakePerception(tmp_path, answer={"ok": True})
    monkeypatch.setattr(PER, "DEFAULT_SOCKET", svc.path)
    try:
        g, s = PER.plan_grasp(np.zeros((100, 3)))
        assert g.shape == (0, 4, 4) and s.shape == (0,)
    finally:
        svc.stop()


def test_plan_grasp_survives_sam3s_leaked_bf16_autocast(monkeypatch):
    """SAM3's tracking predictor enters a bf16 autocast at build time and never exits;
    the service must shield CGN's forward and numpy conversion from it."""
    torch = pytest.importorskip("torch")
    from harness import perception_service as PS

    class FakeCGN:
        def predict_scene_grasps(self, cloud, pc_segments, local_regions,
                                 filter_grasps, forward_passes):
            pts = torch.from_numpy(np.asarray(cloud)).float()
            out = (pts @ torch.eye(3)).detach().cpu().numpy()   # vendor-style
            grasps = np.tile(np.eye(4, dtype=out.dtype), (2, 1, 1))
            return {0: grasps}, {0: np.array([0.7, 0.2])}, None, None

    monkeypatch.setitem(PS._MODELS, "cgn", FakeCGN())
    monkeypatch.setitem(PS._MODELS, "device", "cpu")
    ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    ctx.__enter__()
    try:
        grasps, scores = PS._plan_grasp(
            np.random.default_rng(0).normal(size=(64, 3)), None, 10)
    finally:
        ctx.__exit__(None, None, None)
    assert grasps.shape == (2, 4, 4)
    assert scores.tolist() == [0.7, 0.2]


def test_a_refusal_becomes_a_perception_error(tmp_path, monkeypatch):
    svc = FakePerception(tmp_path, answer={"ok": False, "error": "unknown op"})
    monkeypatch.setattr(PER, "DEFAULT_SOCKET", svc.path)
    try:
        with pytest.raises(PER.PerceptionError, match="unknown op"):
            PER.segment_by_text(np.zeros((4, 4, 3), np.uint8), "mug")
    finally:
        svc.stop()


def test_no_service_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(PER, "DEFAULT_SOCKET", str(tmp_path / "absent.sock"))
    with pytest.raises(PER.PerceptionError, match="no perception service"):
        PER.segment_by_text(np.zeros((4, 4, 3), np.uint8), "mug")


def test_perception_never_touches_the_metered_socket():
    """It is a different socket and a different service on purpose: perception advances
    no episode, so there is nothing to charge -- and nothing here may be able to."""
    import inspect

    src = inspect.getsource(PER)
    assert "speedrun.sock" not in src
    assert "SpeedrunClient" not in src


# -- extrinsics, analytically ---------------------------------------------------
#
# tests/test_speedrun_extrinsics.py checks these against the simulator they were measured
# from; that needs a GPU. What is checked here is the algebra around them, which does not.

def test_agentview_extrinsics_are_proper_rotations():
    for cam in C._AGENTVIEW_POSE:
        rot, off = C.camera_pose_in_base(cam)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)
        assert off.shape == (3,)


def test_the_two_agentviews_are_mirrored_across_the_robot():
    left, right = (C.camera_pose_in_base(c)[1] for c in C._AGENTVIEW_POSE)
    np.testing.assert_allclose(left, [-0.5, 0.35, 1.05])
    np.testing.assert_allclose(right, [-0.5, -0.35, 1.05])


def test_the_wrist_extrinsic_follows_the_hand():
    """It moves with the end-effector, so the same camera has a different base pose for
    a different arm configuration -- and the offset in the HAND frame stays put."""
    from harness.skills import transforms as TR

    a = {"robot0_base_to_eef_pos": [0.4, 0.0, 1.0],
         "robot0_base_to_eef_quat": [0.0, 0.0, 0.0, 1.0]}
    b = {"robot0_base_to_eef_pos": [0.4, 0.0, 1.0],
         "robot0_base_to_eef_quat": [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]}
    rot_a, off_a = C.camera_pose_in_base(C.WRIST, a)
    rot_b, off_b = C.camera_pose_in_base(C.WRIST, b)
    assert not np.allclose(rot_a, rot_b)

    # Back out the hand-frame offset from each: it is the same constant both times.
    for obs, rot, off in ((a, rot_a, off_a), (b, rot_b, off_b)):
        hand = TR.quat_to_mat(obs["robot0_base_to_eef_quat"])
        np.testing.assert_allclose(
            hand.T @ (off - np.asarray(obs["robot0_base_to_eef_pos"], dtype=float)),
            [0.05, 0.0, -0.097], atol=1e-9)


def test_to_base_is_the_inverse_of_the_camera_pose():
    rot, off = C.camera_pose_in_base("robot0_agentview_left")
    pts = np.random.default_rng(0).normal(size=(20, 3))
    back = (C.to_base(pts, "robot0_agentview_left") - off) @ rot
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_pixel_to_base_point_is_none_where_there_is_no_depth():
    obs = {"robot0_agentview_left_depth": np.zeros((16, 16)),
           "robot0_agentview_left_image": np.zeros((16, 16, 3), np.uint8)}
    assert C.pixel_to_base_point(obs, "robot0_agentview_left", 8, 8) is None


def test_pixel_to_base_point_medians_over_a_patch():
    """One pixel on a silhouette can land on background metres behind; the median is what
    stops a target jumping there."""
    depth = np.full((16, 16), 1.0)
    depth[8, 8] = 9.0                       # a single outlier at the pixel asked for
    obs = {"robot0_agentview_left_depth": depth}
    point = C.pixel_to_base_point(obs, "robot0_agentview_left", 8, 8, patch=2)
    rot, off = C.camera_pose_in_base("robot0_agentview_left")
    assert (rot.T @ (point - off))[2] == pytest.approx(1.0)


def test_cloud_in_base_honours_mask_stride_and_range():
    depth = np.full((32, 32), 1.0)
    depth[:, 16:] = 9.0
    obs = {"robot0_agentview_left_depth": depth}
    near = C.cloud_in_base(obs, "robot0_agentview_left", stride=1, max_range=4.0)
    assert len(near) == 32 * 16                       # the far half is dropped
    assert len(C.cloud_in_base(obs, "robot0_agentview_left", stride=2,
                               max_range=4.0)) == 16 * 8
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:4, 0:4] = True
    assert len(C.cloud_in_base(obs, "robot0_agentview_left", mask=mask, stride=1)) == 16
