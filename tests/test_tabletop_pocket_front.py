"""Regression checks for the isolated single-camera layout."""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import pytest
pytest.importorskip("robosuite")
from harness.tabletop.pocket.front import FrontPocketCube, BASE_POSITIONS, BASE_ROTATIONS


def test_mount_geometry():
    for i, rotation in enumerate(BASE_ROTATIONS):
        axis=rotation.apply([0,0,1])
        np.testing.assert_allclose(axis, [(-1 if i==0 else 1)/np.sqrt(2),0,1/np.sqrt(2)],atol=1e-12)
        delta=np.array([.1,-.2,.3])
        np.testing.assert_allclose(rotation.apply(rotation.inv().apply(delta)),delta,atol=1e-12)


def test_single_camera_pinned_reset_and_hold():
    env=FrontPocketCube(seed=17,use_camera_obs=False,has_offscreen_renderer=False)
    try:
        model,data=env.sim.model._model,env.sim.data._data
        assert model.ncam==1 and model.camera(0).name=='front'
        for robot,pos,rot in zip(env.robots,BASE_POSITIONS,BASE_ROTATIONS):
            body=model.body(robot.robot_model.root_body).id
            np.testing.assert_allclose(data.xpos[body],pos,atol=1e-12)
            np.testing.assert_allclose(data.xmat[body].reshape(3,3),rot.as_matrix(),atol=1e-12)
            limits=model.jnt_range[[model.joint(j).id for j in robot.robot_model.joints]]
            q=data.qpos[robot._ref_joint_pos_indexes]
            assert np.minimum(q-limits[:,0],limits[:,1]-q).min()>.35
        positions=np.array([data.site_xpos[r.eef_site_id['right']].copy() for r in env.robots])
        for _ in range(100):
            action=np.zeros(14);action[[6,13]]=-1
            env.step(action)
        final=np.array([data.site_xpos[r.eef_site_id['right']] for r in env.robots])
        assert np.max(np.linalg.norm(final-positions,axis=1))<.005
        assert not any(w.number for w in data.warning)
        assert env.cube_state()['legal']
        assert not [c for c in data.contact if c.dist<-.001]
        env.reset()
        data=env.sim.data._data
        np.testing.assert_allclose([data.site_xpos[r.eef_site_id['right']] for r in env.robots],positions,atol=1e-6)
    finally:env.close()


def test_raised_grasp_and_both_quarter_turn_ik_endpoints():
    env=FrontPocketCube(use_camera_obs=False,has_offscreen_renderer=False)
    try:
        model,data=env.sim.model._model,env.sim.data._data
        for i,robot in enumerate(env.robots):
            ids=robot._ref_joint_pos_indexes
            site=robot.eef_site_id['right']
            limits=model.jnt_range[[model.joint(j).id for j in robot.robot_model.joints]]
            q0=data.qpos[ids].copy()
            original=data.site_xmat[site].reshape(3,3).copy()
            for angle in (0,90,-90):
                target=np.array([-.016 if i==0 else .016,0,1.075])
                rotation=Rotation.from_euler('x',angle,degrees=True).as_matrix()@original
                def residual(q):
                    data.qpos[ids]=q;env.sim.forward()
                    delta=Rotation.from_matrix(rotation@data.site_xmat[site].reshape(3,3).T).as_rotvec()
                    return np.r_[data.site_xpos[site]-target,.25*delta]
                rng=np.random.default_rng(5)
                best=1.
                for guess in [q0,*rng.uniform(limits[:,0]+.25,limits[:,1]-.25,(12,7))]:
                    fit=least_squares(residual,guess,bounds=(limits[:,0]+.2,limits[:,1]-.2),
                                      max_nfev=220,gtol=1e-10,ftol=1e-10,xtol=1e-10)
                    best=min(best,np.linalg.norm(fit.fun))
                    if best<1e-5:break
                assert best<1e-5,(i,angle,best)
    finally:env.close()
