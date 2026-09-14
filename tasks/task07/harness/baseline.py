"""Unprivileged baseline policy for task07 — the Harbor Oracle payload.

Uses ONLY the public contract: overhead depth frames, the published camera
calibration, the compiled cell model for IK, and the wrist F/T sensor.
Look-then-move: park clear of the camera, take a fresh depth frame,
backproject the highest point in the bin, dive blind on it with the head
energized, lift clear and weigh the payload, then deliver a single part or
pre-stage a stack for one-by-one recovery.

Deliberately import-light (numpy + mujoco only, everything parameterized by
cell_spec) so the same file ships verbatim as ``policy/policy.py`` in
``solution/``. It also pins the reward mid-scale during calibration.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.ndimage import median_filter

MAG_ON = 255.0
MAG_OFF = 0.0
SCAN_WAIT = 4             # ticks parked before trusting a frame
PRE_DZ = 0.10
OVERSHOOT = 0.05          # descend this far past the depth estimate
SERVO_TICKS = 70
SEEK_SPEED = 2.0          # rad/s
CARRY_SPEED = 1.2
DESCEND_SPEED = 0.7
TOP_K = 40                # pixels pooled for the robust top-point estimate
PART_N = 3.16             # one bracket's weight at the wrist (~3 N, published)
WEIGH_TICKS = 8           # settle ticks before reading the payload
WEIGH_AVG = 4             # last N of those averaged
TARE_ALPHA = 0.3          # EMA on the parked tool-only reading
DIVE_EPS = 0.01           # commanded-vs-target convergence: dive is done
STALL_QVEL = 0.02         # rad/s: the arm has stopped descending
STALL_TICKS = 4           # consecutive stalled ticks that end a blind dive
TARE_FALLBACK = 6.6       # tool-only |f| if a dive somehow precedes a scan


class BaselinePolicy:
    # ------------------------------------------------------------ lifecycle
    def reset(self, cell: dict, seed: int) -> None:
        self.cell = cell
        self.model = mujoco.MjModel.from_binary_path(cell["model_file"])
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("arm_grip").id
        self.dofs = [self.model.joint(f"arm_joint{i}").dofadr[0]
                     for i in range(1, 8)]
        self.q_lo = np.array([self.model.joint(f"arm_joint{i}").range[0]
                              for i in range(1, 8)])
        self.q_hi = np.array([self.model.joint(f"arm_joint{i}").range[1]
                              for i in range(1, 8)])
        self.home = np.array(cell["arm_home"])
        self.K = np.array(cell["overhead_K"])
        self.T = np.array(cell["overhead_T_world_cam"])
        self.bin_lo = np.array(cell["bin_lo"])
        self.bin_hi = np.array(cell["bin_hi"])
        zone_lo, zone_hi = np.array(cell["zone_lo"]), np.array(cell["zone_hi"])
        # near corner of the zone (comfortably reachable), not the center
        self.drop = zone_lo + 0.15 * (zone_hi - zone_lo)
        self.drop[0] = zone_lo[0] + 0.06
        self.drop[1] = zone_lo[1] + 0.07
        self.drop[2] = cell["belt_top"] + 0.10
        self.bin_top = cell["bin_top_z"]
        self.transit_z = self.bin_top + 0.04    # tool tip clears the rim
        # dead-belt pre-stage spot for doubles (outside the zone)
        self.prestage = np.array([0.16, zone_lo[1] + 0.05,
                                  cell["belt_top"]])
        self.pending = 0        # parts waiting on the dead belt
        self.pre_tries = 0
        self.state = "park"
        self.wait = 0
        self.queue: list[tuple[np.ndarray, float]] = []
        self.depth = None
        self.fresh = False
        self.servo_left = 0
        self.cmd = None
        self.retry = 0
        self.tare = None            # tool-only |f| at the parked pose
        self.q_des = self.home.copy()
        self.stall = 0
        self.weigh_left = 0
        self.weigh_buf: list[float] = []
        self.weigh_ctx = "dive"     # which state asked for the weighing

    # ------------------------------------------------------------------- IK
    def _ik(self, target, q_init, iters=80):
        d = self.data
        q = np.asarray(q_init, dtype=float).copy()
        self._set(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        z_des = np.array([0.0, 0.0, -1.0])
        target = np.asarray(target, dtype=float)
        for _ in range(iters):
            e_pos = target - d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            e_rot = 0.5 * np.cross(R[:, 2], z_des)
            if np.linalg.norm(e_pos) < 0.004 and np.linalg.norm(e_rot) < 0.03:
                break
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dofs], 0.6 * jacr[:, self.dofs]])
            e = np.concatenate([e_pos, 0.6 * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + 0.7 * dq, self.q_lo, self.q_hi)
            self._set(q)
        return q, float(np.linalg.norm(target - d.site_xpos[self.site]))

    def _set(self, q):
        for i, qv in enumerate(q, start=1):
            self.data.joint(f"arm_joint{i}").qpos[0] = qv
        mujoco.mj_forward(self.model, self.data)

    def _fk(self, q):
        self._set(q)
        return self.data.site_xpos[self.site].copy()

    # ------------------------------------------------------------ perception
    def _top_point(self):
        """Robust highest pile point from the latest overhead depth."""
        d = self.depth
        if d is None:
            return None
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        h, w = d.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        with np.errstate(invalid="ignore"):
            xc = (u - cx) / fx * d
            yc = -(v - cy) / fy * d
            pts = np.stack([xc, yc, -d], axis=-1) @ self.T[:3, :3].T \
                + self.T[:3, 3]
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        m = (np.isfinite(z)
             & (x > self.bin_lo[0] + 0.025) & (x < self.bin_hi[0] - 0.025)
             & (y > self.bin_lo[1] + 0.025) & (y < self.bin_hi[1] - 0.025)
             & (z > self.bin_lo[2] + 0.008) & (z < self.bin_top - 0.012))
        if m.sum() < TOP_K:
            return None
        # median-filtered height map: single-pixel noise spikes can be
        # 30+ mm at grazing incidence and would dominate a raw arg-max
        zf = median_filter(np.where(m, z, -1.0), size=5)
        zf[~m] = -1.0
        r, c = np.unravel_index(np.argmax(zf), zf.shape)
        if zf[r, c] < self.bin_lo[2]:
            return None
        w0, w1 = max(r - 3, 0), r + 4
        h0, h1 = max(c - 3, 0), c + 4
        win = m[w0:w1, h0:h1]
        if win.sum() < 5:
            return None
        return np.array([np.median(x[w0:w1, h0:h1][win]),
                         np.median(y[w0:w1, h0:h1][win]),
                         np.median(z[w0:w1, h0:h1][win])])

    # ---------------------------------------------------------------- motion
    def _traj(self, q_to, mag, speed, dwell=0):
        q_from = self.queue[-1][0] if self.queue else self._q_obs
        q_to = np.asarray(q_to, dtype=float)
        n = max(int(np.ceil(np.abs(q_to - q_from).max()
                            / (speed / self.ctrl_hz) * 1.5)), 1)
        for k in range(1, n + 1):
            a = k / n
            s = a * a * (3 - 2 * a)
            self.queue.append((q_from + (q_to - q_from) * s, mag))
        for _ in range(dwell):
            self.queue.append((q_to, mag))

    def _dive_done(self, obs, q_target) -> bool:
        """A blind dive ends on reaching the target or stalling on contact."""
        if self.cmd is None:
            return False
        if np.max(np.abs(self.cmd - q_target)) < DIVE_EPS:
            return True
        # settled against the pile: the joints stop tracking the command
        if (self.servo_left < SERVO_TICKS - 3
                and np.max(np.abs(np.asarray(obs["qvel"], float))) < STALL_QVEL):
            self.stall += 1
        else:
            self.stall = 0
        return self.stall >= STALL_TICKS

    # --------------------------------------------------------------- weighing
    def _weigh(self, q_up, ctx, obs):
        """Hold at q_up with the head energized, then read the payload."""
        self.weigh_q = np.asarray(q_up, dtype=float)
        self.weigh_ctx = ctx
        self.weigh_left = WEIGH_TICKS
        self.weigh_buf = []
        self.state = "weigh"
        self.stall = 0
        return self.act(obs)

    def _deliver(self, n, obs):
        """Route the weighed payload: nothing, one part, or a stack."""
        if self.weigh_ctx == "recover":
            if n == 1:
                q_drop, _ = self._ik(self.drop, self.weigh_q)
                self._traj(q_drop, MAG_ON, CARRY_SPEED, dwell=3)
                self._traj(q_drop, MAG_OFF, CARRY_SPEED, dwell=26)
                self.pending -= 1
                self.state = "recover" if self.pending > 0 else "park"
            else:               # empty, or the whole stack again: put it back
                q_pre, _ = self._ik([self.prestage[0], self.prestage[1],
                                     self.prestage[2] + 0.10], self.weigh_q)
                self._traj(q_pre, MAG_ON, DESCEND_SPEED)
                self._traj(q_pre, MAG_OFF, DESCEND_SPEED, dwell=4)
                self.pre_tries += 1
                if self.pre_tries > 3:          # give up: leave them
                    self.pending = 0
                    self.state = "park"
                else:
                    self.state = "recover"
        elif n == 1:
            q_drop, _ = self._ik(self.drop, self.weigh_q)
            self._traj(q_drop, MAG_ON, CARRY_SPEED, dwell=3)
            # wait out the latch so the zone empties before rescanning
            self._traj(q_drop, MAG_OFF, CARRY_SPEED, dwell=26)
            self.state = "park"
            self.retry = 0
        elif n >= 2:
            # double pick: pre-stage the stack on the dead belt (the drop
            # separates it), then recover the parts one by one
            q_pre, _ = self._ik([self.prestage[0], self.prestage[1],
                                 self.prestage[2] + 0.10], self.weigh_q)
            self._traj(q_pre, MAG_ON, CARRY_SPEED, dwell=2)
            self._traj(q_pre, MAG_OFF, CARRY_SPEED, dwell=4)
            self.pending += n
            self.pre_tries = 0
            self.state = "recover"
        else:                   # came up empty
            self.retry += 1
            self.state = "park"
        self.cmd = None
        self.servo_left = SERVO_TICKS
        return self.act(obs)

    # ------------------------------------------------------------------- act
    def act(self, obs):
        self._q_obs = np.asarray(obs["qpos"], dtype=float)
        self.ctrl_hz = self.cell["ctrl_hz"]
        if "overhead_depth" in obs:
            self.depth = obs["overhead_depth"]
            self.fresh = True

        if self.queue:
            q, mag = self.queue.pop(0)
            return np.array([*q, mag])

        if self.state == "weigh":
            # the lift has drained from the queue: the arm is clear of the
            # pile and holding still, so |f| - tare reads the payload.
            self.weigh_left -= 1
            if self.weigh_left < WEIGH_AVG:
                self.weigh_buf.append(
                    float(np.linalg.norm(np.asarray(obs["ft"], float)[:3])))
            if self.weigh_left > 0:
                return np.array([*self.weigh_q, MAG_ON])
            tare = TARE_FALLBACK if self.tare is None else self.tare
            payload = float(np.mean(self.weigh_buf)) - tare
            self.weigh_buf = []
            return self._deliver(max(int(np.floor(payload / PART_N + 0.5)), 0),
                                 obs)

        if self.state == "recover":
            # blind single-pick from the pre-stage spot (we put the parts
            # there), then lift clear and weigh what actually came up
            self.servo_left -= 1
            if self.servo_left <= 0 or self._dive_done(obs, self.q_des):
                q_up, _ = self._ik([self.prestage[0], self.prestage[1],
                                    self.prestage[2] + 0.12], self._q_obs)
                self._traj(q_up, MAG_ON, DESCEND_SPEED)
                return self._weigh(q_up, "recover", obs)
            # dive at the pre-stage spot with per-try jitter
            js = ((0, 0), (0.03, 0), (-0.03, 0), (0, 0.03))[self.pre_tries % 4]
            tgt = [self.prestage[0] + js[0], self.prestage[1] + js[1],
                   self.prestage[2] - 0.01]
            self.q_des, _ = self._ik(tgt, self._q_obs, iters=40)
            if self.cmd is None:
                self.cmd = self._q_obs.copy()
            dq = np.clip(self.q_des - self.cmd, -DESCEND_SPEED / self.ctrl_hz,
                         DESCEND_SPEED / self.ctrl_hz)
            self.cmd = np.clip(self.cmd + dq, self._q_obs - 0.10,
                               self._q_obs + 0.10)
            return np.array([*self.cmd, MAG_ON])

        if self.state == "park":
            self._traj(self.home, MAG_OFF, SEEK_SPEED, dwell=1)
            self.state = "scan"
            self.wait = -1          # sentinel: arm not yet parked
            return self.act(obs)

        if self.state == "scan":
            if self.wait < 0:
                # first tick with the queue drained: the arm is parked NOW.
                # Frames taken during the transit show the arm over the bin —
                # only a frame captured from here on is trustworthy.
                self.wait = SCAN_WAIT
                self.fresh = False
            if self.wait > 0 or not self.fresh:
                # parked, head de-energized, arm still: tool-only tare
                f = float(np.linalg.norm(np.asarray(obs["ft"], float)[:3]))
                self.tare = f if self.tare is None else \
                    (1.0 - TARE_ALPHA) * self.tare + TARE_ALPHA * f
                self.wait -= 1
                return np.array([*self._q_obs, MAG_OFF])
            top = self._top_point()
            if top is None:
                self.wait = SCAN_WAIT       # nothing visible: rescan
                self.fresh = False
                return np.array([*self._q_obs, MAG_OFF])
            # deterministic retry jitter spreads repeat attempts around
            js = ((0, 0), (0.02, 0), (-0.02, 0), (0, 0.02))[self.retry % 4]
            self.target = top + np.array([js[0], js[1], 0.0])
            pre = np.array([self.target[0], self.target[1],
                            max(self.target[2] + PRE_DZ, self.transit_z)])
            q_pre, res = self._ik(pre, self._q_obs)
            if res > 0.015:
                self.retry += 1
                self.state = "park"
                return self.act(obs)
            self._traj(q_pre, MAG_OFF, SEEK_SPEED, dwell=2)
            q_dive, res2 = self._ik(self.target - [0, 0, OVERSHOOT], q_pre)
            if res2 > 0.02:
                self.retry += 1
                self.state = "park"
                return self.act(obs)
            self.q_dive = q_dive
            self.state = "dive"
            self.servo_left = SERVO_TICKS
            self.cmd = None
            return self.act(obs)

        if self.state == "dive":
            # descend blind onto the depth estimate; nothing here reports
            # whether anything stuck, so always lift clear and weigh
            self.servo_left -= 1
            if self.servo_left <= 0 or self._dive_done(obs, self.q_dive):
                q_up, _ = self._ik([self.target[0], self.target[1],
                                    self.transit_z + 0.05], self._q_obs)
                self._traj(q_up, MAG_ON, DESCEND_SPEED, dwell=2)
                return self._weigh(q_up, "dive", obs)
            if self.cmd is None:
                self.cmd = self._q_obs.copy()
            dq = np.clip(self.q_dive - self.cmd,
                         -DESCEND_SPEED / self.ctrl_hz,
                         DESCEND_SPEED / self.ctrl_hz)
            self.cmd = np.clip(self.cmd + dq,
                               self._q_obs - 0.10, self._q_obs + 0.10)
            return np.array([*self.cmd, MAG_ON])

        return np.array([*self._q_obs, MAG_OFF])


def make_policy():
    return BaselinePolicy()
