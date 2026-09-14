"""Opt-in drop rescue; puzzle snapshots and rotation counters remain private."""
import numpy as np
from scipy.spatial.transform import Rotation
from .front import FrontPocketCube
from . import metrics
from .optimal import optimal_qtm
from . import metrics as cube

DROP_HEIGHT = .90

def canonical(rotations):
    relative = rotations[0].T @ rotations
    if not metrics.inspect_cube(relative)['legal']:
        return None
    return cube.ROTATIONS[np.einsum('bij,kij->bk', relative, cube.ROTATIONS).argmax(axis=1)]

def transition_qtm(before, after):
    if np.array_equal(before, after):
        return 0
    for turns, cost in ((1,1),(-1,1),(2,2)):
        for axis in range(3):
            for sign in (-1,1):
                moved = canonical(metrics.apply_moves(before, [(axis,sign,turns)]))
                if np.array_equal(moved, after):
                    return cost
    return None

class RecoveryPocketCube(FrontPocketCube):
    def _reset_internal(self):
        super()._reset_internal()
        self._checkpoint = canonical(self.sim.data._data.xmat[self._cubie_ids].reshape(8,3,3))
        self._optimal_qtm = optimal_qtm(self._checkpoint)
        self._last_aligned = self._checkpoint.copy()
        self._recoveries = self._face_qtm = self._unclassified = 0

    def private_counters(self):
        return dict(optimal_qtm=self._optimal_qtm, actual_qtm=self._face_qtm,
                    recoveries=self._recoveries, recognized_face_qtm=self._face_qtm,
                    unclassified_transitions=self._unclassified)

    def _post_action(self, action):
        out = super()._post_action(action)
        model, data = self.sim.model._model, self.sim.data._data
        aligned = canonical(data.xmat[self._cubie_ids].reshape(8,3,3))
        if aligned is not None:
            cost = transition_qtm(self._last_aligned, aligned)
            if cost is None:
                self._unclassified += 1
            else:
                self._face_qtm += cost
            self._last_aligned = aligned.copy()
            if data.xpos[model.body('cube_core').id,2] >= DROP_HEIGHT:
                self._checkpoint = aligned.copy()
        return out

    def require_dropped(self):
        model, data = self.sim.model._model, self.sim.data._data
        if data.xpos[model.body('cube_core').id,2] >= DROP_HEIGHT:
            raise ValueError('Recovery is available only after the cube falls below 0.90 m.')

    def recover_drop(self):
        self.require_dropped()
        model, data = self.sim.model._model, self.sim.data._data
        adr = model.joint('cube_free').qposadr[0]
        data.qpos[adr:adr+7] = [0,0,1.04,1,0,0,0]
        for body, rotation in zip(self._cubie_ids,self._checkpoint):
            adr = model.jnt_qposadr[model.body_jntadr[body]]
            data.qpos[adr:adr+4] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        for robot,q in zip(self.robots,self.approach_qpos):
            data.qpos[robot._ref_joint_pos_indexes] = q
            for arm in robot.arms:
                data.qpos[robot._ref_gripper_joint_pos_indexes[arm]] = robot.gripper[arm].init_qpos
        data.qvel[:] = 0
        data.qacc_warmstart[:] = 0
        data.ctrl[:] = 0
        self.sim.forward()
        for robot in self.robots:
            robot.reset(deterministic=True,rng=self.rng)
        self._solved_steps = 0
        self._last_aligned = self._checkpoint.copy()
        self._recoveries += 1
