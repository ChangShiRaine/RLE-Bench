"""Analytic and simulator regressions for the isolated physical 2x2."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from harness.tabletop.pocket import metrics as M

SOLVED = np.tile(np.eye(3), (8, 1, 1))


def test_all_faces_turns_inverses_and_four_turns():
    for axis in range(3):
        for sign in (-1, 1):
            for turns in (-1, 1, 2):
                move = (axis, sign, turns)
                state = M.apply_moves(SOLVED, [move])
                assert M.inspect_cube(state)["legal"]
                assert not M.inspect_cube(state)["solved"]
                np.testing.assert_array_equal(M.apply_moves(state, [(axis, sign, -turns)]), SOLVED)
            np.testing.assert_array_equal(M.apply_moves(SOLVED, [(axis, sign, 1)]*4), SOLVED)


def test_solved_is_global_rotation_invariant_and_odd_permutations_are_legal():
    for rotation in M.ROTATIONS:
        assert M.inspect_cube(rotation @ SOLVED)["solved"]
    state = M.apply_moves(SOLVED, [(0, 1, 1)])
    positions = np.einsum("bij,bj->bi", state, M.POSITIONS)
    perm = [next(i for i, p in enumerate(M.POSITIONS) if np.array_equal(p, q)) for q in positions]
    assert sum(a > b for i, a in enumerate(perm) for b in perm[i+1:]) % 2 == 1
    assert M.inspect_cube(state)["legal"]


def test_scramble_inverse_and_independent_rotation_reference():
    for seed in range(30):
        moves = M.scramble(seed, 20)
        state = M.apply_moves(SOLVED, moves)
        reference = SOLVED.copy()
        for axis, sign, turns in moves:
            turn = Rotation.from_rotvec(np.eye(3)[axis] * sign * turns * np.pi / 2).as_matrix()
            for i, position in enumerate(M.POSITIONS):
                if (reference[i] @ position)[axis] * sign > .5:
                    reference[i] = turn @ reference[i]
        np.testing.assert_allclose(state, reference, atol=1e-12)
        assert M.inspect_cube(state)["legal"]
        inverse = [(a, s, -t) for a, s, t in reversed(moves)]
        assert M.inspect_cube(M.apply_moves(state, inverse))["solved"]


def test_impossible_twist_overlap_mirror_and_misalignment_rejected():
    state = SOLVED.copy()
    state[0] = Rotation.from_rotvec(M.POSITIONS[0]/np.sqrt(3) * 2*np.pi/3).as_matrix()
    assert not M.inspect_cube(state)["legal"]
    state = SOLVED.copy()
    state[0] = next(r for r in M.ROTATIONS if np.array_equal(r @ M.POSITIONS[0], M.POSITIONS[1]))
    assert not M.inspect_cube(state)["legal"]
    state = SOLVED.copy()
    state[0, 0] *= -1
    assert not M.inspect_cube(state)["legal"]
    state = SOLVED.copy()
    state[0] = Rotation.from_rotvec([.1, 0, 0]).as_matrix()
    assert not M.inspect_cube(state)["legal"]
    assert not M.inspect_cube(np.full((8, 3, 3), np.nan))["legal"]


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_eight_corner_scene_settles_legally_without_cube_motors(seed):
    pytest.importorskip("robosuite")
    from harness.tabletop.pocket.scene import PocketCube
    env = PocketCube(seed=seed, use_camera_obs=False, has_offscreen_renderer=False)
    try:
        env.reset()
        model, data = env.sim.model._model, env.sim.data._data
        core = model.body("cube_core").id
        assert np.count_nonzero(model.body_parentid == core) == 8
        assert model.nu == 18 and env.action_dim == 14
        assert all(not model.joint(int(j)).name.startswith("cube_") for j in model.actuator_trnid[:, 0])
        assert env.cube_state()["legal"] and not env._check_success()
        for _ in range(20):
            env.step(np.zeros(14))
        assert env.cube_state()["legal"] and not any(w.number for w in data.warning)
        # The free core is unobservable: verifier orientation follows an anchor corner.
        anchor = data.xmat[env._cubie_ids[0]].reshape(3, 3)
        assert np.isfinite(anchor).all()
    finally:
        env.close()


def test_reset_and_rollout_are_deterministic():
    pytest.importorskip("robosuite")
    from harness.tabletop.pocket.scene import PocketCube
    states = []
    for _ in range(2):
        env = PocketCube(seed=17, use_camera_obs=False, has_offscreen_renderer=False)
        try:
            env.reset()
            for _ in range(10):
                env.step(np.zeros(14))
            states.append(env.sim.data.qpos.copy())
        finally:
            env.close()
    np.testing.assert_array_equal(*states)


def test_solved_score_ignores_core_gauge_but_requires_above_table():
    pytest.importorskip("robosuite")
    from harness.tabletop.pocket.scene import PocketCube
    from harness.tabletop.pocket.scene import SUCCESS_HOLD_STEPS
    env = PocketCube(seed=17, use_camera_obs=False, has_offscreen_renderer=False)
    try:
        env.reset()
        model, data = env.sim.model._model, env.sim.data._data
        free = model.joint("cube_free").qposadr[0]
        core = Rotation.from_rotvec([.37, -.21, .49])
        world = Rotation.from_rotvec([-.18, .23, .17])
        data.qpos[free+3:free+7] = core.as_quat(scalar_first=True)
        for body in env._cubie_ids:
            adr = model.jnt_qposadr[model.body_jntadr[body]]
            data.qpos[adr:adr+4] = (core.inv()*world).as_quat(scalar_first=True)
        data.qvel[:] = 0
        env.sim.forward()
        assert env.cube_state()["solved"]
        for _ in range(SUCCESS_HOLD_STEPS):
            env._post_action(np.zeros(14))
        assert env._check_success()
        data.qpos[free+2] = .785
        env.sim.forward()
        env._post_action(np.zeros(14))
        assert not env._check_success()
    finally:
        env.close()


def test_robot_restores_one_move_scramble_and_releases(tmp_path):
    pytest.importorskip("robosuite")
    from dev.pocket.probe import run
    records = run(tmp_path, render=False, one_move=True)
    assert not records[0]["cube"]["solved"]
    assert records[-1]["cube"]["legal"] and records[-1]["cube"]["matched"] == 24
    assert records[-1]["success"] and records[-1]["expected_turn_error_deg"] < 2
    assert all(not any(row["warnings"]) for row in records)


def test_cameras_survive_hard_reset_and_delayed_context_collection(tmp_path):
    pytest.importorskip("robosuite")
    from dev.pocket.camera_check import DEPTH_TOLERANCE_M, run
    result = run(tmp_path, resets=3)
    assert result["mismatches"] == 0 and result["max_rgb_mae"] < 1e-4
    assert all(row.get("depth_finite_positive", True) for row in result["rows"])
    assert all(row.get("depth_max_error_m", 0) < 1e-6 for row in result["rows"])
    for name, values in result["depth_alignment"].items():
        assert values["samples"] >= 20 and values["max_error_m"] < DEPTH_TOLERANCE_M[name]


def test_renderer_releases_retired_context_before_new_one_is_created():
    pytest.importorskip("robosuite")
    from harness.tabletop.pocket.scene import PocketCube
    env = PocketCube(seed=139)
    try:
        retired = env.sim._render_context_offscreen
        env.reset()
        assert retired._closed
        retired.close()
        current = env.sim._render_context_offscreen
        assert not current._closed
    finally:
        env.close()
    assert current._closed


def test_renderer_moves_between_serialized_client_threads():
    pytest.importorskip("robosuite")
    from concurrent.futures import ThreadPoolExecutor
    from harness import env as camera
    from harness.tabletop.pocket.scene import PocketCube
    from harness.tabletop.pocket.scene import CAMERAS
    camera._use_upright_images()
    env = PocketCube(seed=139)
    try:
        for _ in range(3):
            with ThreadPoolExecutor(max_workers=1) as worker:
                obs = worker.submit(env.reset).result()
                frames = worker.submit(camera.render_frames, env, CAMERAS, 512, 512, True).result()
            for name in CAMERAS:
                np.testing.assert_array_equal(obs[name+"_image"], frames[name+"_image"])
            # Read from the owner thread again after the connection worker exits.
            frames = camera.render_frames(env, CAMERAS, 512, 512, depth=True)
            for name in CAMERAS:
                np.testing.assert_array_equal(obs[name+"_image"], frames[name+"_image"])
    finally:
        env.close()


def test_torsional_pad_configuration_preserves_geometry_and_force_limits():
    import xml.etree.ElementTree as ET
    from harness.tabletop.pocket.model import configure_grippers
    root = ET.fromstring('''<mujoco><option/><worldbody><body>
      <geom name="finger_pad_collision" size=".008 .004 .008" friction="2 .05 .0001"/>
      <geom name="cube_pocket_0" condim="1" friction="0 0 0"/>
    </body></worldbody><actuator><position forcerange="-20 20"/></actuator></mujoco>''')
    cube = ET.tostring(root.find(".//geom[@name='cube_pocket_0']"))
    actuator = ET.tostring(root.find("actuator"))
    configure_grippers(root, "torsion")
    pad = root.find(".//geom[@name='finger_pad_collision']")
    assert pad.get("condim") == "4" and pad.get("size") == ".008 .004 .008"
    sliding, torsional, _ = np.fromstring(pad.get("friction"), sep=" ")
    assert sliding == 2 and 0 < torsional < .008
    assert cube == ET.tostring(root.find(".//geom[@name='cube_pocket_0']"))
    assert actuator == ET.tostring(root.find("actuator"))


@pytest.mark.parametrize("direction", [1, -1])
def test_torsional_pads_turn_a_scrambled_cube_without_dropping(tmp_path, direction):
    pytest.importorskip("robosuite")
    from dev.pocket.probe import run
    records = run(tmp_path, render=False, seed=0, scrambled=True,
                  direction=direction, grip_profile="torsion")
    final = records[-1]
    assert final["cube"]["legal"] and final["expected_turn_error_deg"] < 3
    assert final["core_position"][2] > 1.05
    assert any(c["dim"] == 4 for row in records for c in row["pad_contacts"])
    assert all(not any(row["warnings"]) for row in records)
