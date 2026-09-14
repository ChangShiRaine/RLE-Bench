"""Analytic turn-count and state-preserving rescue checks; no agent ground truth."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
pytest.importorskip("robosuite")
from harness.tabletop.pocket.recovery import RecoveryPocketCube, canonical, transition_qtm
from harness.tabletop.pocket import metrics

def test_canonical_rigid_rotation_and_turn_costs():
    state = metrics.apply_moves(np.tile(np.eye(3),(8,1,1)),[(0,1,1),(1,-1,1)])
    before = canonical(state)
    for rotation in (Rotation.from_rotvec([.3,-.8,.4]).as_matrix(), np.eye(3)):
        np.testing.assert_array_equal(canonical(rotation@state),before)
    for axis in range(3):
        for sign in (-1,1):
            for turns,cost in ((1,1),(-1,1),(2,2)):
                after = canonical(metrics.apply_moves(state,[(axis,sign,turns)]))
                assert transition_qtm(before,after)==cost
    assert transition_qtm(before,before)==0

def test_drop_only_preserves_changed_puzzle_and_time():
    env = RecoveryPocketCube(seed=17,use_camera_obs=False,has_offscreen_renderer=False)
    try:
        with pytest.raises(ValueError): env.recover_drop()
        model,data = env.sim.model._model,env.sim.data._data
        initial = env._checkpoint.copy()
        changed = metrics.apply_moves(initial,[(0,1,1)])
        for body,rot in zip(env._cubie_ids,changed):
            adr=model.jnt_qposadr[model.body_jntadr[body]]
            data.qpos[adr:adr+4]=Rotation.from_matrix(rot).as_quat(scalar_first=True)
        env.sim.forward();env._post_action(np.zeros(14))
        expected=canonical(changed)
        assert env._face_qtm==1
        adr=model.joint('cube_free').qposadr[0]
        data.qpos[adr+2]=.79
        env.sim.forward()
        time=data.time
        env.recover_drop()
        assert data.time==time
        np.testing.assert_array_equal(canonical(data.xmat[env._cubie_ids].reshape(8,3,3)),expected)
        assert env._face_qtm==1 and env._recoveries==1
        assert not np.array_equal(expected,initial)
        assert env._solved_steps==0
        assert np.max(np.abs(data.qvel))==0
        with pytest.raises(ValueError):env.recover_drop()
    finally:env.close()


def test_exact_optimal_qtm():
    from harness.tabletop.pocket.optimal import optimal_qtm
    solved = np.tile(np.eye(3), (8, 1, 1))
    assert optimal_qtm(solved) == 0
    assert optimal_qtm(metrics.apply_moves(solved, [(0, 1, 1)])) == 1
    assert optimal_qtm(metrics.apply_moves(solved, [(0, 1, 2)])) == 2
    assert optimal_qtm(metrics.apply_moves(solved, [(0, 1, 1), (1, 1, 1)])) == 2
