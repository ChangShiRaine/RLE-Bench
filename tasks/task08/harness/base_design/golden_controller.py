"""Verifier-side base control and the IK helpers the Oracle shelf policy
bundles into its standalone submission."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..sim.mecanum import make_base_controller

__all__ = ["ArmPlan", "GoldenArmController", "make_base_controller"]


@dataclass(frozen=True)
class ArmPlan:
    """Joint-space realization of stow -> approach -> pickup target."""

    transit_knots: tuple[np.ndarray, ...]
    reach_knots: tuple[np.ndarray, ...]
    approach_error_m: float
    target_error_m: float


class GoldenArmController:
    """Deterministic Cartesian IK for one canonical arm.

    Joint and actuator identities come from the trusted ArmSpec.
    """

    def __init__(self, model, qpos0: np.ndarray, ik_tolerance_m: float, arm):
        self.model = model
        self.qpos0 = np.asarray(qpos0, dtype=float).copy()
        self.ik_tolerance_m = float(ik_tolerance_m)
        self.arm = arm
        count = len(arm.joint_names)
        joint_names = [arm.joint(i) for i in range(count)]
        self.stow = np.asarray(arm.stow, dtype=float)
        self.site_id = model.site(arm.ee_site).id
        self.qpos_adrs = [model.joint(name).qposadr[0] for name in joint_names]
        self.dof_adrs = [model.joint(name).dofadr[0] for name in joint_names]
        self.actuator_ids = [model.actuator(arm.actuator(i)).id
                             for i in range(count)]
        ranges = np.asarray([model.joint(name).range for name in joint_names])
        self.lower, self.upper = ranges[:, 0], ranges[:, 1]
        self.ik_starts = arm.controller_seeds()
        if arm.name == "ur5e":
            # Fixed coverage seeds for the 6-DoF arm's disconnected shelf
            # branches: deterministic verifier calibration, not submission
            # state.
            rng = np.random.default_rng(9011)
            coverage = tuple(
                rng.uniform(self.lower, self.upper) for _ in range(48))
            self.ik_starts = coverage + self.ik_starts

    def _ik(self, data, target: np.ndarray, seeds) -> tuple[np.ndarray, float]:
        jacp = np.zeros((3, self.model.nv))

        def error(q):
            for adr, value in zip(self.qpos_adrs, q):
                data.qpos[adr] = value
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            return target - data.site_xpos[self.site_id]

        best_q = np.asarray(seeds[0], dtype=float).copy()
        best_error = float("inf")
        best_key = (1, float("inf"))
        for start in seeds:
            q = np.clip(np.asarray(start, dtype=float), self.lower, self.upper)
            for _ in range(200):
                err = error(q)
                if np.linalg.norm(err) < 0.5 * self.ik_tolerance_m:
                    break
                mujoco.mj_jacSite(self.model, data, jacp, None, self.site_id)
                jac = jacp[:, self.dof_adrs]
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + 1e-4 * np.eye(3), err)
                step = np.linalg.norm(dq)
                if step > 0.25:
                    dq *= 0.25 / step
                q = np.clip(q + dq, self.lower, self.upper)
            err_norm = float(np.linalg.norm(error(q)))
            # Position-only IK has many branches. Prefer one that does not
            # already intersect the shelf at the knot; otherwise an arm can be
            # graded on a solution that is accurate but physically invalid.
            mujoco.mj_forward(self.model, data)
            touches_shelf = False
            for contact in data.contact:
                names = (self.model.geom(int(contact.geom1)).name or "",
                         self.model.geom(int(contact.geom2)).name or "")
                if any(name.startswith("pod_") for name in names):
                    touches_shelf = True
                    break
            key = (1 if touches_shelf else 0, err_norm)
            if key < best_key:
                best_key = key
                best_q, best_error = q.copy(), err_norm
            if not touches_shelf and err_norm < 0.5 * self.ik_tolerance_m:
                break
        return best_q, best_error

    @staticmethod
    def _segment(start: np.ndarray, end: np.ndarray, spacing_m: float):
        count = max(int(np.ceil(np.linalg.norm(end - start) / spacing_m)), 1)
        return [start + (end - start) * (index + 1) / count
                for index in range(count)]

    def _touches_shelf(self, data, q: np.ndarray) -> bool:
        """Whether an arm configuration intersects shelf geometry."""
        for adr, value in zip(self.qpos_adrs, q):
            data.qpos[adr] = value
        mujoco.mj_forward(self.model, data)
        for contact in data.contact:
            names = (self.model.geom(int(contact.geom1)).name or "",
                     self.model.geom(int(contact.geom2)).name or "")
            if any(name.startswith("pod_") for name in names):
                return True
        return False

    def _edge_is_clear(self, data, start: np.ndarray,
                       end: np.ndarray, max_step: float = 0.08) -> bool:
        count = max(int(np.ceil(
            np.max(np.abs(np.asarray(end) - np.asarray(start))) / max_step)), 1)
        return all(not self._touches_shelf(
            data, start + (end - start) * index / count)
                   for index in range(count + 1))

    def _ik_connected(self, data, target: np.ndarray,
                      seed: np.ndarray, starts) -> tuple[np.ndarray, float]:
        """Choose a collision-free IK solution connected to the prior knot.

        Endpoint collision checks are insufficient because interpolation
        between two valid solutions may still cross a shelf panel. Solve each
        branch independently, then prefer the smallest joint displacement
        whose full edge is clear.
        """
        def connected(q, error):
            return (error <= self.ik_tolerance_m
                    and not self._touches_shelf(data, q)
                    and self._edge_is_clear(
                        data, seed, q, max_step=0.05))

        # Cartesian continuation normally succeeds from the previous knot.
        q, error = self._ik(data, target, (seed,))
        if connected(q, error):
            return q.copy(), float(error)

        best = None
        for start in starts:
            q, error = self._ik(data, target, (start,))
            if not connected(q, error):
                continue
            candidate = (float(np.linalg.norm(q - seed)), error, q.copy())
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            # A local branch change this small is already dynamically gentle.
            if candidate[0] < 0.35:
                break
        if best is None:
            return np.asarray(seed, dtype=float).copy(), float("inf")
        _, error, q = best
        return q, float(error)

    @staticmethod
    def _interpolate(knots: tuple[np.ndarray, ...], alpha: float) -> np.ndarray:
        if len(knots) == 1:
            return knots[0]
        scaled = float(np.clip(alpha, 0.0, 1.0)) * (len(knots) - 1)
        index = min(int(scaled), len(knots) - 2)
        return knots[index] + (knots[index + 1] - knots[index]) * (scaled - index)
