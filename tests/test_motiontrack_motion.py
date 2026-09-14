"""Motion pipeline: quaternion math, resampling, and the FK velocity conventions.

The conventions are the fragile part. A wrong frame for the free joint's angular
velocity still produces plausible-looking numbers, so the load-bearing test
compares FK velocities against the numerical derivative of FK positions and
pins the margin over the wrong convention.
"""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from harness import motion  # noqa: E402
from harness import rotations as R  # noqa: E402

G1_XML = "third_party/motiontrack/unitree_g1/g1.xml"
CLIP = "third_party/motiontrack/lafan1/dance1_subject2.csv"
FRAMES, INPUT_FPS, FPS = (122, 722), 30, 50

pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(G1_XML),
    reason="run `make sim-motiontrack` first",
)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(G1_XML)


@pytest.fixture(scope="module")
def clip(model):
    return motion.Motion(motion.convert(CLIP, model, FRAMES, INPUT_FPS, FPS))


def rand_quats(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_quat_mul_matches_mujoco():
    a, b = rand_quats(16, 1), rand_quats(16, 2)
    want = np.empty((16, 4))
    for i in range(16):
        mujoco.mju_mulQuat(want[i], a[i], b[i])
    assert np.allclose(R.quat_mul(a, b), want, atol=1e-12)


def test_quat_rotate_matches_mujoco():
    q, v = rand_quats(16, 3), np.random.default_rng(4).normal(size=(16, 3))
    want = np.empty((16, 3))
    for i in range(16):
        mujoco.mju_rotVecQuat(want[i], v[i], q[i])
    assert np.allclose(R.quat_rotate(q, v), want, atol=1e-12)


def test_axis_angle_takes_the_shortest_path():
    # 350 degrees about z is -10 degrees, not +350.
    a = np.deg2rad(350.0)
    q = np.array([[np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]])
    assert np.allclose(R.axis_angle(q), [[0.0, 0.0, np.deg2rad(-10.0)]], atol=1e-9)


def test_axis_angle_of_identity_is_zero():
    assert np.allclose(R.axis_angle(np.array([[1.0, 0.0, 0.0, 0.0]])), 0.0)


def test_quat_error_is_zero_against_self_and_symmetric():
    q, p = rand_quats(8, 5), rand_quats(8, 6)
    assert np.allclose(R.quat_error(q, q), 0.0, atol=1e-7)
    assert np.allclose(R.quat_error(q, p), R.quat_error(p, q), atol=1e-9)


def test_slerp_endpoints_and_midpoint():
    a, b = rand_quats(8, 7), rand_quats(8, 8)
    t0, t1 = np.zeros((8, 1)), np.ones((8, 1))
    assert np.allclose(np.abs(np.sum(R.slerp(a, b, t0) * a, -1)), 1.0, atol=1e-9)
    assert np.allclose(np.abs(np.sum(R.slerp(a, b, t1) * b, -1)), 1.0, atol=1e-9)
    # The midpoint is equidistant from both ends.
    mid = R.slerp(a, b, np.full((8, 1), 0.5))
    assert np.allclose(R.quat_error(mid, a), R.quat_error(mid, b), atol=1e-7)


def test_yaw_quat_strips_roll_and_pitch():
    q = R.quat_mul(
        np.array([[np.cos(0.3), 0.0, 0.0, np.sin(0.3)]]),          # yaw 0.6
        np.array([[np.cos(0.2), np.sin(0.2), 0.0, 0.0]]),          # roll 0.4
    )
    assert np.allclose(R.yaw_quat(q), [[np.cos(0.3), 0, 0, np.sin(0.3)]], atol=1e-9)


def test_resample_frame_count_and_endpoints():
    rp, rq, jp = motion.load_csv(CLIP, FRAMES)
    out_p, out_q, out_j = motion.resample(rp, rq, jp, INPUT_FPS, FPS)
    n_in = FRAMES[1] - FRAMES[0] + 1
    assert len(out_p) == round((n_in - 1) / INPUT_FPS * FPS)
    assert np.allclose(out_p[0], rp[0]) and np.allclose(out_j[0], jp[0])
    assert np.allclose(np.abs(np.sum(out_q[0] * rq[0])), 1.0)


def test_resample_preserves_a_constant_pose():
    rp = np.tile([1.0, 2.0, 0.8], (10, 1))
    rq = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
    jp = np.tile(np.linspace(0, 1, motion.N_JOINTS), (10, 1))
    out_p, out_q, out_j = motion.resample(rp, rq, jp, 30, 50)
    assert np.allclose(out_p, rp[0]) and np.allclose(out_j, jp[0])
    assert np.allclose(out_q, rq[0])


def test_differentiate_recovers_a_constant_spin():
    fps, n, omega = 50, 100, 1.3
    t = np.arange(n) / fps
    rq = np.stack([np.cos(omega * t / 2), np.zeros(n), np.zeros(n),
                   np.sin(omega * t / 2)], axis=-1)
    rp = np.stack([2.0 * t, np.zeros(n), np.full(n, 0.8)], axis=-1)
    jp = np.zeros((n, motion.N_JOINTS))
    lin, ang, jvel = motion.differentiate(rp, rq, jp, fps)
    assert np.allclose(lin[2:-2, 0], 2.0, atol=1e-9)
    assert np.allclose(ang[2:-2, 2], omega, atol=1e-6)
    assert np.allclose(jvel, 0.0)


def test_schema_matches_the_reference(clip):
    assert clip.fps == FPS and clip.n_frames == 1000
    assert clip.duration == pytest.approx(20.0)
    assert clip.joint_pos.shape == (1000, motion.N_JOINTS)
    assert clip.body_pos_w.shape[1:] == (30, 3)
    assert clip.body_quat_w.shape[1:] == (30, 4)
    for f in motion.FIELDS:
        arr = getattr(clip, f)
        assert arr.dtype == np.float32 and np.isfinite(arr).all()


def test_body_and_joint_names_resolve(clip):
    assert clip.joint_names[:2] == ["left_hip_pitch_joint", "left_hip_roll_joint"]
    idx = clip.body_index(["pelvis", "torso_link"])
    assert clip.body_names[idx[0]] == "pelvis"


def test_quaternions_stay_unit(clip):
    assert np.allclose(np.linalg.norm(clip.body_quat_w, axis=-1), 1.0, atol=1e-5)


def test_fk_velocities_match_finite_differences(clip):
    """The conventions check. Body velocities must equal d(body position)/dt."""
    num = np.gradient(clip.body_pos_w, 1.0 / clip.fps, axis=0)
    err = np.linalg.norm(num - clip.body_lin_vel_w, axis=-1)
    assert err.mean() < 0.02

    dq = R.quat_mul(clip.body_quat_w[2:], R.quat_conj(clip.body_quat_w[:-2]))
    werr = np.linalg.norm(R.axis_angle(dq) * (clip.fps / 2.0)
                          - clip.body_ang_vel_w[1:-1], axis=-1)
    assert werr.mean() < 0.1


def test_world_frame_omega_would_be_much_worse(model):
    """Guards the fix, not just the symptom: MuJoCo's free joint wants omega in
    the BODY frame. Feeding the clip's world-frame omega straight through is the
    natural mistake and this pins how much worse it is."""
    rp, rq, jp = motion.resample(*motion.load_csv(CLIP, FRAMES), INPUT_FPS, FPS)
    lin_vel, ang_vel, joint_vel = motion.differentiate(rp, rq, jp, FPS)

    def mean_err(omega_in_body):
        data = mujoco.MjData(model)
        pos = np.empty((len(rp), model.nbody - 1, 3))
        vel = np.empty_like(pos)
        v6 = np.empty(6)
        for t in range(len(rp)):
            data.qpos[:3], data.qpos[3:7], data.qpos[7:] = rp[t], rq[t], jp[t]
            data.qvel[:3] = lin_vel[t]
            data.qvel[3:6] = (R.quat_rotate_inverse(rq[t], ang_vel[t])
                              if omega_in_body else ang_vel[t])
            data.qvel[6:] = joint_vel[t]
            mujoco.mj_forward(model, data)
            pos[t] = data.xpos[1:]
            for i in range(1, model.nbody):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                         i, v6, 0)
                vel[t, i - 1] = v6[3:]
        return np.linalg.norm(np.gradient(pos, 1.0 / FPS, axis=0) - vel, axis=-1).mean()

    assert mean_err(True) * 10 < mean_err(False)


def test_reference_clip_stays_near_the_floor(model, clip):
    """Retargeted clips are not ground-aligned to this model, and we do not shift
    them: the median contact sits within a few mm of the floor across walk/run/
    jump/dance, so a global offset would make stance worse, not better. This
    bounds the residual so a future model or clip swap cannot slip a large
    offset past unnoticed."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(model.nbody)]
    feet = [g for g in range(model.ngeom)
            if "ankle_roll" in (names[model.geom_bodyid[g]] or "")
            and model.geom_contype[g]]
    data = mujoco.MjData(model)
    lowest = np.empty(clip.n_frames)
    for t in range(clip.n_frames):
        data.qpos[7:] = clip.joint_pos[t]
        data.qpos[:3] = clip.body_pos_w[t, clip.body_index(["pelvis"])[0]]
        data.qpos[3:7] = clip.body_quat_w[t, clip.body_index(["pelvis"])[0]]
        mujoco.mj_forward(model, data)
        lowest[t] = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in feet)
    assert lowest.min() > -0.03
    assert abs(np.median(lowest)) < 0.01
