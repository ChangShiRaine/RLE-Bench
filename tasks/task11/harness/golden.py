"""Privileged golden packer for task11 (dev/verifier side — NEVER shipped).

The baseline FSM with perception replaced by simulator ground truth: true
box poses and dims on the belt and the verifier's own heightmap of the
tote. It anchors the calibration of the force limits and reward caps and
must log zero floor drops, damage events and tote strikes.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import packing, scene, spec
from .baseline import PLAN_RES, Packer


class GoldenPacker(Packer):
    privileged = True
    place_clearance = 0.005     # exact state affords a tighter pack

    def bind(self, cell) -> None:
        self.sim = cell

    def _load_model(self) -> None:
        self.model = self.sim.model
        self.ik_data = mujoco.MjData(self.model)

    def _held(self):
        d = self.sim.data
        return next((i for i in range(self.sim.n) if d.eq_active[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                              f"suction_weld_{i}")]), None)

    def boxes(self, obs) -> dict[int, dict]:
        out = {}
        for i in self.sim.active:
            p = self.sim.pos(i)
            if i in self.sim.grasped or p[2] < spec.BELT_TOP - 0.02:
                continue
            R = scene.box_rot(self.sim.data, i)
            dims = np.array(self.sim.stream.boxes[i].dims)
            out[i] = {"pos": p + [0, 0, dims[2] / 2],
                      "yaw": float(np.arctan2(R[1, 0], R[0, 0])),
                      "dims": dims}
        return out

    def grasp_offset(self, obs):
        i = self.target["id"]
        tcp, yaw = self._tcp(obs["qpos"]), self.tool_yaw(obs["qpos"])
        R = scene.box_rot(self.sim.data, i)
        c, s = np.cos(-yaw), np.sin(-yaw)
        rel = np.array([[c, -s], [s, c]]) @ (self.sim.pos(i)[:2] - tcp[:2])
        dyaw = float(np.arctan2(R[1, 0], R[0, 0])) - yaw
        return rel, dyaw - np.pi * np.round(dyaw / np.pi)

    def heightmap(self, obs) -> np.ndarray:
        held = self._held()
        return packing.heightmap(
            [scene.box_corners(self.model, self.sim.data, i)
             for i in self.sim.active if i != held
             and self.sim.in_bin_xy(self.sim.pos(i), 0.05)
             and self.sim.pos(i)[2] < spec.BIN_RIM_Z + 0.3], PLAN_RES)
