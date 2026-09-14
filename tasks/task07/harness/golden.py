"""Privileged golden picker for task07 (dev/verifier side — NEVER shipped).

Reads ground-truth part poses straight from the simulator (via the
runtime's privileged bind) and clears the bin with the contact-magnet
end-effector. FSM per pick cycle:

  seek    pick the highest settled part in the bin, IK a face-down
          approach above its topmost surface point
  servo   closed-loop descent, head energized: every tick re-IK against
          the part's LIVE top point until the face touches and welds
  verify  still holding after the lift? (privileged: read straight from
          the weld constraints, a channel no policy is given)
  place   carry over the drop zone, de-energize, return

It sets the calibration ceiling for clearance/speed and must log ZERO
damage events, pinning DAMAGE_FORCE_N above ordinary handling forces.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import parts, scene, spec

MAG_ON = 255.0
MAG_OFF = 0.0

PRE_GRASP_DZ = 0.10
RIM_TOP = spec.STAND_H + spec.BIN_WALL + spec.BIN_INNER[2]
PRE_MIN_Z = RIM_TOP + 0.03   # transits stay above the rim: a pre-grasp
                             # below it sweeps the tool INTO the bin wall
PUSH_IN = 0.002           # aim the face this far past the top point
SERVO_TICKS = 60          # descent budget (3 s) before giving up
SEEK_SPEED = 2.2          # rad/s joint-space, transits
CARRY_SPEED = 1.4         # rad/s while holding parts (welded: can be brisk)
DESCEND_SPEED = 0.9       # rad/s far-phase descent
TOUCH_SPEED = 0.4         # rad/s final 25 mm onto the pile
MAX_PART_SPEED = 0.05     # m/s: don't chase still-moving parts
MAX_FAILS = 3
CARRY_Z = spec.STAND_H + spec.BIN_INNER[2] + 0.18
DROP_Z = spec.BELT_TOP + 0.10
# deliveries go to the NEAR corner of the zone — close enough that a
# stuck part can be picked back off the belt (zone recovery)
DROP_POINT = (0.45, 0.38)   # nest center
# dead-belt spot OUTSIDE the zone: doubles get dropped here (the impact
# splits the nested stack) and are re-picked singly
PRESTAGE = (0.16, 0.36)
RELEASE_TICKS = 26        # wait out the latch (~0.9 s) so the zone is
                          # empty again before the next delivery


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


class GoldenPicker:
    """Implements the Policy interface, plus the privileged bind hook."""

    privileged = True

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ------------------------------------------------------------ plumbing
    def bind(self, model, data) -> None:
        self.model = model
        self.data = data
        self.ik_data = mujoco.MjData(model)
        self.n = scene.n_parts(model)
        self.site_id = model.site("arm_grip").id
        self.dofs = [model.joint(f"arm_joint{i}").dofadr[0]
                     for i in range(1, 8)]
        self.q_lo = np.array([model.joint(f"arm_joint{i}").range[0]
                              for i in range(1, 8)])
        self.q_hi = np.array([model.joint(f"arm_joint{i}").range[1]
                              for i in range(1, 8)])
        self.eq = {i: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                        f"mag_weld_{i}") for i in range(self.n)}
        self.part_geoms = {}
        for i in range(self.n):
            body = model.body(parts.part_name(i)).id
            self.part_geoms[i] = [g for g in range(model.ngeom)
                                  if model.geom_bodyid[g] == body]

    def reset(self, cell_spec: dict, seed: int) -> None:
        self.queue: list[tuple[np.ndarray, float]] = []
        self.state = "seek"
        self.fails: dict[int, int] = {}
        self.black_at: dict[int, int] = {}   # drop count when blacklisted
        self.target = -1
        self.servo_left = 0
        self._servo_cmd = None
        self.drop_i = 0
        self.q_ready: np.ndarray | None = None

    # ------------------------------------------------------------------ IK
    def _ik(self, target_pos, R_des, q_init, iters: int = 100,
            tol: float = 0.004):
        model, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        q = np.asarray(q_init, dtype=float).copy()
        self._set_ik_arm(q)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        target_pos = np.asarray(target_pos, dtype=float)
        for _ in range(iters):
            e_pos = target_pos - d.site_xpos[self.site_id]
            R_cur = d.site_xmat[self.site_id].reshape(3, 3)
            e_rot = 0.5 * (np.cross(R_cur[:, 0], R_des[:, 0])
                           + np.cross(R_cur[:, 1], R_des[:, 1])
                           + np.cross(R_cur[:, 2], R_des[:, 2]))
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_rot) < 0.02:
                break
            mujoco.mj_jacSite(model, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dofs], 0.5 * jacr[:, self.dofs]])
            e = np.concatenate([e_pos, 0.5 * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + 0.7 * dq, self.q_lo, self.q_hi)
            self._set_ik_arm(q)
        res = float(np.linalg.norm(target_pos - d.site_xpos[self.site_id]))
        return q, res

    def _set_ik_arm(self, q7) -> None:
        for i, qv in enumerate(q7, start=1):
            self.ik_data.joint(f"arm_joint{i}").qpos[0] = qv
        mujoco.mj_forward(self.model, self.ik_data)

    # ------------------------------------------------------------ helpers
    def _q_now(self) -> np.ndarray:
        return np.array(scene.get_arm(self.data))

    def _held(self) -> set[int]:
        return {i for i, e in self.eq.items() if self.data.eq_active[e]}

    def _enqueue_traj(self, q_to, mag, speed, dwell: int = 0) -> None:
        q_from = self.queue[-1][0] if self.queue else self._q_now()
        q_to = np.asarray(q_to, dtype=float)
        dq = np.abs(q_to - q_from).max()
        n = max(int(np.ceil(dq / (speed / spec.CTRL_HZ) * 1.5)), 1)
        for k in range(1, n + 1):
            a = k / n
            s = a * a * (3 - 2 * a)
            self.queue.append((q_from + (q_to - q_from) * s, mag))
        for _ in range(dwell):
            self.queue.append((q_to, mag))

    def _bin_parts(self) -> list[int]:
        """Pickable parts: in the bin, or pre-staged on the dead belt
        (outside the zone — in-zone parts go through zone recovery)."""
        lo, hi = np.asarray(spec.BIN_LO), np.asarray(spec.BIN_HI)
        zlo, zhi = np.asarray(spec.ZONE_LO), np.asarray(spec.ZONE_HI)
        blo = np.array([0.05, spec.BELT_CENTER[1] - spec.BELT_HALF[1],
                        spec.BELT_TOP - 0.01])
        bhi = np.array([0.72, spec.BELT_CENTER[1] + spec.BELT_HALF[1],
                        spec.BELT_TOP + 0.25])
        speeds = scene.part_speeds(self.data, self.n)
        out = []
        for i in range(self.n):
            if self.fails.get(i, 0) >= MAX_FAILS:
                if self.drop_i > self.black_at.get(i, -1):
                    self.fails[i] = 0    # pile changed since: retry
                else:
                    continue
            p, _ = scene.part_pose(self.data, i)
            if speeds[i] >= MAX_PART_SPEED:
                continue
            in_bin = np.all(p >= lo) and np.all(p <= hi)
            in_zone = np.all(p >= zlo) and np.all(p <= zhi)
            on_belt = np.all(p >= blo) and np.all(p <= bhi) and not in_zone
            if in_bin or on_belt:
                out.append(i)
        return out

    def _fail(self, i: int) -> None:
        self.fails[i] = self.fails.get(i, 0) + 1
        if self.fails[i] >= MAX_FAILS:
            self.black_at[i] = self.drop_i

    def _top_point(self, i: int) -> np.ndarray:
        """Highest corner of the part's boxes — where the face lands."""
        best = None
        for g in self.part_geoms[i]:
            R = self.data.geom_xmat[g].reshape(3, 3)
            p = self.data.geom_xpos[g]
            s = self.model.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        c = p + R @ (s * np.array([sx, sy, sz]))
                        if best is None or c[2] > best[2]:
                            best = c
        return best.copy()

    def _face_R(self, yaw_dir=(0.0, 1.0)) -> np.ndarray:
        z = np.array([0.0, 0.0, -1.0])
        y = np.array([yaw_dir[0], yaw_dir[1], 0.0])
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        return np.column_stack([x, y, z])

    def _reachable(self, p: np.ndarray) -> bool:
        return 0.25 < np.hypot(p[0], p[1]) < 0.72

    def _plan_ready(self) -> np.ndarray:
        if self.q_ready is None:
            over_bin = np.array([spec.BIN_POS[0], spec.BIN_POS[1], CARRY_Z])
            self.q_ready, _ = self._ik(over_bin, self._face_R(),
                                       self._q_now())
        return self.q_ready

    # ----------------------------------------------------------------- FSM
    def act(self, obs: dict):
        if not self.queue and self.state == "servo":
            step = self._servo_step()
            if step is not None:
                return step
        if not self.queue:
            self._plan()
        if self.queue:
            q, mag = self.queue.pop(0)
            return np.array([*q, mag])
        return np.array([*self._q_now(), MAG_OFF])

    def _plan(self) -> None:
        if self.state == "verify":
            self._plan_verify()
        if self.state == "seek":
            self._plan_seek()

    def _zone_live(self, exclude=()) -> list[int]:
        """ANY live part inside the nest (settling or settled)."""
        lo, hi = np.asarray(spec.ZONE_LO), np.asarray(spec.ZONE_HI)
        out = []
        for i in range(self.n):
            if i in exclude:
                continue
            p, _ = scene.part_pose(self.data, i)
            if p[2] > -0.5 and np.all(p >= lo) and np.all(p <= hi):
                out.append(i)
        return out

    def _zone_stuck(self) -> list[int]:
        """Live, settled parts inside the zone. Two or more = the zone is
        blocked (singulated delivery) and needs recovery."""
        lo, hi = np.asarray(spec.ZONE_LO), np.asarray(spec.ZONE_HI)
        speeds = scene.part_speeds(self.data, self.n)
        out = []
        for i in range(self.n):
            p, _ = scene.part_pose(self.data, i)
            if p[2] > -0.5 and np.all(p >= lo) and np.all(p <= hi) \
                    and speeds[i] < MAX_PART_SPEED:
                out.append(i)
        return out

    def _plan_seek(self) -> None:
        stuck = self._zone_stuck()
        # blocked zone: lift one stuck part off the belt — holding it lets
        # the other latch, and the verify flow then re-delivers this one
        cand = stuck if len(stuck) >= 2 else self._bin_parts()
        positions = scene.part_positions(self.data, self.n)
        for i in sorted(cand, key=lambda i: -positions[i][2]):
            top = self._top_point(i)
            if not self._reachable(top):
                self._fail(i)
                continue
            for yaw in ((0.0, 1.0), (1.0, 0.0)):
                R = self._face_R(yaw)
                pre = np.array([top[0], top[1],
                                max(top[2] + PRE_GRASP_DZ, PRE_MIN_Z)])
                q_pre, r1 = self._ik(pre, R, self._q_now())
                if r1 > 0.012:
                    continue
                self.target = i
                self._face_yaw = yaw
                self.servo_left = SERVO_TICKS
                self._servo_cmd = None
                self._enqueue_traj(q_pre, MAG_OFF, SEEK_SPEED, dwell=2)
                self.state = "servo"
                return
            self._fail(i)
        self._enqueue_traj(self._plan_ready(), MAG_OFF, SEEK_SPEED,
                           dwell=10)

    def _servo_step(self):
        """One tick of energized descent onto the live top point."""
        i = self.target
        self.servo_left -= 1
        if self._held():
            pos, _ = scene.part_pose(self.data, i)
            q_lift, _ = self._ik([pos[0], pos[1], CARRY_Z], self._face_R(
                self._face_yaw), self._q_now())
            self._enqueue_traj(q_lift, MAG_ON, CARRY_SPEED, dwell=2)
            self.state = "verify"
            if self.verbose:
                print(f"[golden] grabbed {sorted(self._held())}")
            return None
        if self.servo_left <= 0:
            self._fail(i)
            self.state = "seek"
            if self.verbose:
                print(f"[golden] part {i} servo aborted")
            return None
        top = self._top_point(i)
        if not self._reachable(top):
            self._fail(i)
            self.state = "seek"
            return None
        target = top + [0, 0, -PUSH_IN]
        R = self._face_R(self._face_yaw)
        if self._servo_cmd is None:
            self._servo_cmd = self._q_now()
        q_des, _ = self._ik(target, R, self._q_now(), iters=40)
        tcp = self.data.site_xpos[self.site_id]
        speed = DESCEND_SPEED if np.linalg.norm(tcp - target) > 0.025 \
            else TOUCH_SPEED
        dq = np.clip(q_des - self._servo_cmd,
                     -speed / spec.CTRL_HZ,
                     speed / spec.CTRL_HZ)
        self._servo_cmd = np.clip(self._servo_cmd + dq,
                                  self._q_now() - 0.10,
                                  self._q_now() + 0.10)
        return np.array([*self._servo_cmd, MAG_ON])

    def _plan_verify(self) -> None:
        held = self._held()
        if not held:
            self._fail(self.target)
            if self.verbose:
                print(f"[golden] part {self.target} lost on lift")
            self.state = "seek"
            return
        divert = len(held) > 1 or self._zone_live(exclude=held)
        if divert:
            # singulated delivery: doubles go to the pre-stage spot on the
            # dead belt (the drop splits the nested stack), and singles
            # divert there too whenever the nest is NOT empty — never feed
            # a crowded nest; recovered/pre-staged parts are re-picked
            # singly later
            R = self._face_R(self._face_yaw)
            pos0, _ = scene.part_pose(self.data, self.target)
            q_up, _ = self._ik([pos0[0], pos0[1], CARRY_Z], R,
                               self._q_now())
            q_pre, _ = self._ik([PRESTAGE[0], PRESTAGE[1], DROP_Z], R, q_up)
            self._enqueue_traj(q_up, MAG_ON, CARRY_SPEED)
            self._enqueue_traj(q_pre, MAG_ON, CARRY_SPEED, dwell=2)
            self._enqueue_traj(q_pre, MAG_OFF, CARRY_SPEED, dwell=4)
            self.state = "seek"
            if self.verbose:
                why = "double" if len(held) > 1 else "nest occupied"
                print(f"[golden] {why}: {sorted(held)} -> prestage")
            return
        drop = np.array([DROP_POINT[0], DROP_POINT[1], DROP_Z])
        self.drop_i += 1
        R = self._face_R(self._face_yaw)
        q_mid, _ = self._ik([0.35, 0.12, CARRY_Z + 0.05], R, self._q_now())
        q_drop, res = self._ik(drop, R, q_mid)
        if res > 0.02:
            q_drop, res = self._ik(drop + [0, -0.06, 0.03], R, q_mid)
        self._enqueue_traj(q_mid, MAG_ON, CARRY_SPEED)
        self._enqueue_traj(q_drop, MAG_ON, CARRY_SPEED, dwell=3)
        self._enqueue_traj(q_drop, MAG_OFF, CARRY_SPEED,
                           dwell=RELEASE_TICKS)
        self.state = "seek"          # next seek plans straight from here
        if self.verbose:
            print(f"[golden] carrying {sorted(held)} -> drop {self.drop_i}")
