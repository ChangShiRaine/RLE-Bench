"""Dev tests for the task07 sim core (scene / parts / episodes / sensor).

Locks the properties the scoring battery relies on: deterministic piles and
pixels, parts settled inside the bin, the welded-bin invariant, ctrl
clamping, and the published cell model containing no parts.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import pytest

from harness import episodes, scene, sensor, spec

SEED = spec.DESIGN_SEEDS[0]


@pytest.fixture(scope="module")
def pile():
    return episodes.pile_poses(SEED)


@pytest.fixture(scope="module")
def cell(pile):
    model, data = scene.build_scene(part_poses=pile, seed=SEED)
    scene.set_arm(model, data, spec.ARM_HOME)
    return model, data


# ---------------------------------------------------------------------------
# Piles (determinism, invariant #3)
# ---------------------------------------------------------------------------
def test_pile_deterministic(pile):
    again = episodes.pile_poses(SEED)
    assert len(pile) == len(again)
    for (p1, q1), (p2, q2) in zip(pile, again):
        assert np.array_equal(p1, p2) and np.array_equal(q1, q2)


def test_pile_seeds_differ(pile):
    other = episodes.pile_poses(spec.DESIGN_SEEDS[1])
    assert len(pile) != len(other) or not np.allclose(
        np.stack([p for p, _ in pile]), np.stack([p for p, _ in other]))


def test_design_seed_battery_is_stratified_and_hidden_disjoint():
    assert spec.DESIGN_SEEDS == (35, 16, 307, 211, 101)
    assert tuple(len(episodes.pile_poses(seed))
                 for seed in spec.DESIGN_SEEDS) == (12, 13, 14, 15, 16)

    seed_path = Path(episodes.__file__).with_name("eval_seeds.json")
    hidden = json.loads(seed_path.read_text())
    private = {int(hidden["smoke"]), *(int(s) for s in hidden["eval"])}
    assert set(spec.DESIGN_SEEDS).isdisjoint(private)


def test_pile_count_in_range_and_inside_bin(pile):
    assert spec.N_PARTS_RANGE[0] - 4 <= len(pile) <= spec.N_PARTS_RANGE[1]
    for pos, _ in pile:
        assert episodes._inside_bin(pos), pos


def test_pile_is_settled(pile):
    model, data = scene.build_scene(part_poses=pile, with_arm=False)
    scene.settle(model, data, 0.3)
    n = len(pile)
    assert scene.part_speeds(data, n).max() < 0.05
    drift = np.linalg.norm(
        scene.part_positions(data, n) - np.stack([p for p, _ in pile]),
        axis=1)
    assert drift.max() < 0.01


# ---------------------------------------------------------------------------
# Scene / model structure
# ---------------------------------------------------------------------------
def test_cell_structure(cell, pile):
    model, _ = cell
    # 7 joint actuators; ctrl slot 7 is the magnet command (not an actuator)
    assert model.nu == 7 and spec.N_CTRL == 8
    assert scene.n_parts(model) == len(pile)
    # bin and conveyor are static (welded): world geoms, no free joints
    for g in ("bin_bottom", "bin_wall_nx", "belt", "bin_stand"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert gid >= 0 and model.geom_bodyid[gid] == 0
    # every part is a free body of GEOMS_PER_PART geoms
    assert model.nq == len(pile) * 7 + 7


def test_magnet_welds_compiled_inactive(cell, pile):
    model, data = cell
    assert model.neq == len(pile)
    for i in range(len(pile)):
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                              f"mag_weld_{i}")
        assert e >= 0 and not data.eq_active[e]
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             "magnet_head") == -1  # prefixed
    assert model.geom("arm_magnet_head") is not None


def test_ctrl_clamped(cell):
    model, data = cell
    out = scene.apply_ctrl(model, data, np.full(spec.N_CTRL, 1e6))
    for i in range(7):
        a = model.actuator(f"arm_actuator{i + 1}")
        assert out[i] == a.ctrlrange[1]
    assert out[7] == spec.MAG_CTRL_RANGE[1]


def test_arm_helpers(cell):
    model, data = cell
    scene.set_arm(model, data, spec.ARM_HOME)
    assert np.allclose(scene.get_arm(data), spec.ARM_HOME)
    assert scene.arm_tau(data).shape == (7,)


def test_wrist_camera_moves_with_arm(cell):
    model, data = cell
    scene.set_arm(model, data, spec.ARM_HOME)
    T1 = sensor.camera_pose(model, data, "wrist")
    moved = list(spec.ARM_HOME)
    moved[0] += 0.5
    scene.set_arm(model, data, moved)
    T2 = sensor.camera_pose(model, data, "wrist")
    assert not np.allclose(T1[:3, 3], T2[:3, 3])
    scene.set_arm(model, data, spec.ARM_HOME)
    T3 = sensor.camera_pose(model, data, "overhead")
    assert np.allclose(T3, spec.overhead_extrinsics())


# ---------------------------------------------------------------------------
# Sensor (pixel + noise determinism, intrinsics consistency)
# ---------------------------------------------------------------------------
def test_pixels_and_noise_deterministic(pile):
    frames = []
    for _ in range(2):
        model, data = scene.build_scene(part_poses=pile, seed=SEED)
        scene.set_arm(model, data, spec.ARM_HOME)
        with sensor.CameraRig(model) as rig:
            frames.append(sensor.frame_obs(rig, model, data, SEED, 0))
    a, b = frames
    for cam in ("overhead", "wrist"):
        assert np.array_equal(a[f"{cam}_rgb"], b[f"{cam}_rgb"])
        assert np.array_equal(np.isnan(a[f"{cam}_depth"]),
                              np.isnan(b[f"{cam}_depth"]))
        assert np.array_equal(np.nan_to_num(a[f"{cam}_depth"]),
                              np.nan_to_num(b[f"{cam}_depth"]))


def test_noise_varies_by_frame_and_camera(cell):
    model, data = cell
    with sensor.CameraRig(model) as rig:
        d0 = rig.depth(data, "overhead")
        K = spec.overhead_intrinsics()
        n0 = sensor.apply_depth_noise(d0, K, SEED, "overhead", 0)
        n1 = sensor.apply_depth_noise(d0, K, SEED, "overhead", 1)
    assert not np.array_equal(np.nan_to_num(n0), np.nan_to_num(n1))


def test_backprojection_consistency(cell):
    """Backprojected overhead depth must land on real world geometry:
    the belt top and the bin walls at their known heights."""
    model, data = cell
    with sensor.CameraRig(model) as rig:
        depth = rig.depth(data, "overhead")
    pts_cam = sensor.backproject_cam(depth, spec.overhead_intrinsics())
    T = spec.overhead_extrinsics()
    pts = pts_cam @ T[:3, :3].T + T[:3, 3]
    # center pixel looks straight down into the pile: z in the bin's range
    cz = pts[depth.shape[0] // 2, depth.shape[1] // 2, 2]
    assert spec.STAND_H < cz < spec.STAND_H + spec.BIN_INNER[2] + 0.15
    # floor pixels reconstruct near z=0
    corner = pts[5, 5, 2]
    assert abs(corner) < 0.02


# ---------------------------------------------------------------------------
# Published cell model (agent-side planning artifact)
# ---------------------------------------------------------------------------
def test_cell_mjb_has_no_parts(tmp_path):
    p = str(tmp_path / "cell.mjb")
    scene.save_cell_mjb(p)
    m = mujoco.MjModel.from_binary_path(p)
    assert m.nu == 7
    assert scene.n_parts(m) == 0
    assert m.nq == 7   # the arm and nothing else


def test_part_pose_helpers(cell):
    model, data = cell
    scene.set_part_pose(data, 0, spec.GRAVEYARD, [1, 0, 0, 0])
    pos, quat = scene.part_pose(data, 0)
    assert np.allclose(pos, spec.GRAVEYARD)
    assert np.allclose(quat, [1, 0, 0, 0])


# ---------------------------------------------------------------------------
# Wrist force-torque sensor
# ---------------------------------------------------------------------------
def test_ft_reads_tool_weight(cell):
    """Static at home, the F/T sensor carries the magnet assembly weight."""
    model, data = cell
    scene.set_arm(model, data, spec.ARM_HOME)
    scene.settle(model, data, 0.5)
    f = scene.ft_reading(model, data)[:3]
    tool_mass = model.body("arm_magnet").mass[0]
    assert abs(np.linalg.norm(f) - tool_mass * 9.81) < 1.0


def test_ft_senses_payload():
    """A welded part adds its weight to the wrist wrench."""
    from harness.runtime import MagnetEngine
    model, data = scene.build_scene(part_poses=[
        (np.array([0.4, -0.2, 0.5]), np.array([1.0, 0, 0, 0]))])
    scene.set_arm(model, data, spec.ARM_HOME)
    scene.apply_ctrl(model, data, [*spec.ARM_HOME, 255.0])
    scene.settle(model, data, 0.5)
    base = np.linalg.norm(scene.ft_reading(model, data)[:3])
    mag = MagnetEngine(model)
    # hang the part in mid-air below the tool, then weld it there
    tcp = data.site("arm_grip").xpos
    scene.set_part_pose(data, 0, tcp + [0, 0, -0.05], [1, 0, 0, 0])
    mujoco.mj_forward(model, data)
    mag._weld(data, 0)
    scene.settle(model, data, 0.5)
    loaded = np.linalg.norm(scene.ft_reading(model, data)[:3])
    part_mass = model.body("part0").mass[0]
    assert loaded - base > 0.7 * part_mass * 9.81


def test_ft_noise_seeded_and_deterministic():
    from harness import runtime

    def capture():
        fts = []

        class Probe(runtime.NullPolicy):
            def act(self, obs):
                fts.append(obs["ft"].copy())
                return super().act(obs)

        runtime.run_episode(Probe(), SEED, budget_t=0.5, render=False)
        return np.stack(fts)

    a, b = capture(), capture()
    assert np.array_equal(a, b)                    # invariant #3
    assert not np.array_equal(a[0], a[1])          # noise varies per tick
