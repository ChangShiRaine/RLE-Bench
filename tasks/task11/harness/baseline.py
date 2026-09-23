"""Public-sensor baseline packer for task11 — the Harbor Oracle payload.

Uses ONLY the public contract: scanner rows and the belt speed to
dead-reckon boxes on the belt, the top-view depth camera for the tote
heightmap, the vacuum switch and the compiled cell model for IK. It ignores
the wrist camera. Per box:

  choose  the oldest reachable box with a feasible placement under the
          published fit rule (lowest base, then nearest the low corner)
  meet    joint-space move above the box's predicted position
  track   descend onto the moving box with the cup energized until sealed
  place   lift, carry above the slot, lower, release, retreat

Self-contained (numpy, scipy, mujoco) so it ships verbatim as policy/policy.py.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter

ON, OFF = 255.0, 0.0
MOVE_SPEED = 2.0          # rad/s joint-space transits
CARRY_SPEED = 1.5
WIN_X = (0.12, 0.58)      # belt x range the arm tracks comfortably
TRACK_S = 2.0             # belt time reserved for descend + seal
PRE_DZ = 0.05             # pre-grasp height above the box top
DESCEND_MPS = 0.12
PUSH = 0.006              # aim this far below the top face
LEAD_S = 0.12             # position-servo lag compensation
CLEAR_Z = 0.08            # transit clearance over the rim / belt boxes
RELEASE_DZ = 0.004        # release height above the support
PLAN_RES = 0.005          # planning grid, finer than the published rule
PLACE_CLEARANCE = 0.010   # per side, as the published rule
DEPTH_TCP_Y = 0.20        # refresh the tote depth map only with the tool this far out
WALL_BAND = 0.015
DEPTH_PERIOD_S = 1.0
FLAT_GRAD = 0.003          # max smoothed depth change per pixel, m
GUARD_N = 12.0            # tool-axis force change that ends the final approach
WAIT_XY = (0.40, 0.30)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rot2(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _smooth(n: int):
    a = np.arange(1, n + 1) / n
    return a * a * (3 - 2 * a)


class Packer:
    """Shared FSM. Subclasses override the perception hooks."""

    place_clearance = PLACE_CLEARANCE

    # ------------------------------------------------------------ lifecycle
    def reset(self, cell: dict, seed: int) -> None:
        self.cell = cell
        self.dt = 1.0 / cell["ctrl_hz"]
        self._load_model()
        self.site = self.model.site("arm_tcp").id
        self.dofs = [self.model.joint(f"arm_joint{i}").dofadr[0]
                     for i in range(1, 8)]
        self.qadr = [self.model.joint(f"arm_joint{i}").qposadr[0]
                     for i in range(1, 8)]
        self.q_lo = np.array([self.model.jnt_range[self.model.joint(
            f"arm_joint{i}").id][0] for i in range(1, 8)])
        self.q_hi = np.array([self.model.jnt_range[self.model.joint(
            f"arm_joint{i}").id][1] for i in range(1, 8)])
        self.lo = np.array(cell["bin_lo"])
        self.hi = np.array(cell["bin_hi"])
        self.res = PLAN_RES
        self.grid = (int(round(cell["bin_inner"][0] / self.res)),
                     int(round(cell["bin_inner"][1] / self.res)))
        self.placed_hmap = np.zeros(self.grid)
        self.depth_hmap: np.ndarray | None = None
        self.depth_t = -np.inf
        self.queue: list[tuple[np.ndarray, float]] = []
        self.mode = "idle"
        self.skip: set[int] = set()
        self.target = None
        self.track_left = 0
        self.q_cmd = None

    def _load_model(self) -> None:
        self.model = mujoco.MjModel.from_binary_path(self.cell["model_file"])
        self.ik_data = mujoco.MjData(self.model)

    # ------------------------------------------------------------ perception
    def boxes(self, obs) -> dict[int, dict]:
        """Box id -> {pos (top centre, now), yaw, dims} for boxes on the belt."""
        out = {}
        v = np.array(self.cell["belt_dir"]) * obs["belt_v"]
        for row in np.asarray(obs["scans"]).reshape(-1, 9):
            i = int(row[0])
            pos = np.array([row[2], row[3], self.cell["belt_top"] + row[7]])
            pos += v * (obs["t"] - row[1])
            if pos[0] > self.cell["belt_end_x"]:
                out[i] = {"pos": pos, "yaw": float(row[4]),
                          "dims": row[5:8].copy()}
        return out

    def heightmap(self, obs) -> np.ndarray:
        self._update_depth(obs)
        if self.depth_hmap is None:
            return self.placed_hmap
        return np.maximum(self.placed_hmap, self.depth_hmap)

    def _update_depth(self, obs) -> None:
        """Per-cell median height of the tote from the top-view depth, at most
        once per DEPTH_PERIOD_S and only with the arm clear of the tote."""
        if "top_depth" not in obs or obs["t"] < self.depth_t + DEPTH_PERIOD_S \
                or self._tcp(obs["qpos"])[1] < DEPTH_TCP_Y:
            return
        self.depth_t = obs["t"]
        K = np.array(self.cell["top_K"])
        T = np.array(self.cell["top_T_world_cam"])
        if not hasattr(self, "roi"):            # tote pixels (+ margin)
            corners = [(x, y) for x in (self.lo[0], self.hi[0])
                       for y in (self.lo[1], self.hi[1])]
            depth = T[2, 3] - self.cell["bin_floor_z"]
            uv = np.array([[K[0, 0] * (x - T[0, 3]) / depth + K[0, 2],
                            K[1, 2] - K[1, 1] * (y - T[1, 3]) / depth]
                           for x, y in corners])
            lo = np.floor(uv.min(0)).astype(int) - 8
            hi = np.ceil(uv.max(0)).astype(int) + 8
            self.roi = (slice(max(lo[1], 0), hi[1]), slice(max(lo[0], 0),
                                                          hi[0]))
        # keep only locally flat pixels: box tops and the tote floor; walls
        # and box sides seen obliquely are steep and noisy
        raw = np.asarray(obs["top_depth"], dtype=float)[self.roi]
        valid = np.isfinite(raw)
        cnt = uniform_filter(valid.astype(float), 3)
        d = uniform_filter(np.where(valid, raw, 0.0), 3) / np.maximum(cnt, 1e-9)
        gy, gx = np.gradient(d)
        d[(cnt < 0.99) | (np.hypot(gx, gy) > FLAT_GRAD)] = np.nan
        v, u = np.mgrid[self.roi[0], self.roi[1]]
        pts = np.stack([(u - K[0, 2]) / K[0, 0] * d,
                        -(v - K[1, 2]) / K[1, 1] * d, -d], -1).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(1)] @ T[:3, :3].T + T[:3, 3]
        # drop a band along the walls: their grazing, noisy faces project
        # into it (placement bookkeeping covers those cells)
        inside = ((pts[:, :2] > self.lo[:2] + WALL_BAND)
                  & (pts[:, :2] < self.hi[:2] - WALL_BAND)).all(1)
        pts = pts[inside]
        ij = np.floor((pts[:, :2] - self.lo[:2]) / self.res).astype(int)
        ok = (ij[:, 0] < self.grid[0]) & (ij[:, 1] < self.grid[1])
        z = pts[ok, 2] - self.cell["bin_floor_z"]
        flat = ij[ok, 0] * self.grid[1] + ij[ok, 1]
        order = np.lexsort((z, flat))
        flat, z = flat[order], z[order]
        starts = np.r_[0, np.nonzero(np.diff(flat))[0] + 1]
        counts = np.diff(np.r_[starts, len(flat)])
        hm = np.zeros(self.grid[0] * self.grid[1])
        hm[flat[starts]] = z[starts + counts // 2]     # per-cell median
        self.depth_hmap = np.clip(hm.reshape(self.grid), 0.0, None)

    # ------------------------------------------------------------ fit rule
    def placement(self, hmap, dims):
        """Lowest slot under the published support rule (on the planning
        grid, with ``place_clearance``), nearest the low corner:
        (centre xy, base, rotated) or None."""
        l, w, h = dims
        best = None
        clr = self.place_clearance
        ring = int(round(clr / self.res))
        for rot, (a, b) in ((False, (l, w)), (True, (w, l))):
            fx = int(np.ceil((a + 2 * clr) / self.res - 1e-9))
            fy = int(np.ceil((b + 2 * clr) / self.res - 1e-9))
            if fx > self.grid[0] or fy > self.grid[1]:
                continue
            win = np.lib.stride_tricks.sliding_window_view(hmap, (fx, fy))
            base = win.max(axis=(2, 3))
            sup = win >= base[..., None, None] - self.cell["support_tol"]
            cx = slice((fx - 1) // 2, fx // 2 + 1)
            cy = slice((fy - 1) // 2, fy // 2 + 1)
            ok = ((base + h <= self.cell["bin_inner"][2])
                  & (sup[..., ring:fx - ring, ring:fy - ring].mean(axis=(2, 3))
                     >= self.cell["support_frac"])
                  & sup[..., cx, cy].all(axis=(2, 3)))
            for i, j in zip(*np.nonzero(ok)):
                key = (round(float(base[i, j]), 3), i + j)
                if best is None or key < best[0]:
                    c = self.lo[:2] + (np.array([i + fx / 2, j + fy / 2])
                                       * self.res)
                    best = (key, c, float(base[i, j]), rot)
        return None if best is None else best[1:]

    # ------------------------------------------------------------ kinematics
    def _R(self, yaw: float) -> np.ndarray:
        """Tool frame pointing down with its x axis at ``yaw``."""
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])

    def _set(self, q) -> None:
        self.ik_data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.ik_data)
        mujoco.mj_comPos(self.model, self.ik_data)

    def _tcp(self, q) -> np.ndarray:
        self._set(q)
        return self.ik_data.site_xpos[self.site].copy()

    def posture(self, pos, yaw) -> np.ndarray:
        """Elbow-up seed facing ``pos`` with the tool at exactly ``yaw``
        (tool yaw = j1 - j7 + 5*pi/4 for this arm family)."""
        j1 = float(np.arctan2(pos[1], pos[0]))
        j7 = _wrap(j1 - yaw + 1.25 * np.pi)
        return np.array([j1, -0.3, 0.0, -2.2, 0.0, 1.9, j7])

    def tool_yaw_for(self, pos, yaw) -> float:
        """``yaw`` or ``yaw + pi``, whichever keeps joint 7 mid-range."""
        return yaw if abs(self.posture(pos, yaw)[6]) <= np.pi / 2 \
            else _wrap(yaw + np.pi)

    def tool_yaw(self, q) -> float:
        self._set(q)
        x = self.ik_data.site_xmat[self.site].reshape(3, 3)[:, 0]
        return float(np.arctan2(x[1], x[0]))

    def ik(self, pos, yaw, q0=None, iters: int = 80):
        """Damped least squares with a nullspace pull toward ``posture``.
        Returns (q, position error)."""
        R = self._R(yaw)
        ref = self.posture(pos, yaw)
        if q0 is not None:            # stay on the continuous branch
            ref[6] -= 2 * np.pi * np.round((ref[6] - q0[6]) / (2 * np.pi))
        q = ref.copy() if q0 is None else np.array(q0, dtype=float)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            self._set(q)
            ep = np.asarray(pos) - self.ik_data.site_xpos[self.site]
            Rc = self.ik_data.site_xmat[self.site].reshape(3, 3)
            er = 0.5 * sum(np.cross(Rc[:, k], R[:, k]) for k in range(3))
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 3e-3:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.site)
            J = np.vstack([jp[:, self.dofs], jr[:, self.dofs]])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            dq = Jp @ np.concatenate([ep, er]) \
                + (np.eye(7) - Jp @ J) @ (0.2 * (ref - q))
            q = np.clip(q + 0.8 * dq, self.q_lo, self.q_hi)
        self._set(q)
        return q, float(np.linalg.norm(
            np.asarray(pos) - self.ik_data.site_xpos[self.site]))

    # ------------------------------------------------------------ motion
    # Queue entries: ("q", q, suction, guarded) or ("settle", suction, tol, n).
    def _last(self) -> np.ndarray:
        for e in reversed(self.queue):
            if e[0] == "q":
                return e[1]
        return self.q_cmd

    def joint(self, q_to, suction: float, speed: float,
              ticks: int | None = None) -> None:
        q_from, q_to = self._last(), np.asarray(q_to, dtype=float)
        n = ticks or self.joint_ticks(q_to, speed)
        for a in _smooth(n):
            self.queue.append(("q", q_from + (q_to - q_from) * a, suction,
                               False))

    def joint_ticks(self, q_to, speed: float) -> int:
        dq = np.abs(np.asarray(q_to) - self._last()).max()
        return max(int(np.ceil(dq / (speed * self.dt) * 1.5)), 1)

    def line(self, p_to, yaw, suction: float, mps: float,
             guarded: bool = False) -> None:
        """Straight Cartesian segment from the queued end pose. A guarded
        segment is cut short as soon as the wrist feels contact."""
        q = self._last()
        p_from = self._tcp(q)
        n = max(int(np.ceil(np.linalg.norm(np.asarray(p_to) - p_from)
                            / (mps * self.dt) * 1.5)), 1)
        for a in _smooth(n):
            q, _ = self.ik(p_from + (np.asarray(p_to) - p_from) * a, yaw, q,
                           iters=30)
            self.queue.append(("q", q, suction, guarded))

    def settle(self, suction: float, tol: float, limit: int) -> None:
        """Hold the last target until the joints converge (or ``limit``)."""
        self.queue.append(("settle", suction, tol, limit))

    def hold(self, suction: float, ticks: int) -> None:
        self.queue += [("q", self._last(), suction, False)] * ticks

    # ------------------------------------------------------------ policy
    def act(self, obs: dict):
        if self.q_cmd is None:
            self.q_cmd = np.asarray(obs["qpos"], dtype=float).copy()
            self.suction_cmd = OFF
            self.f_ref = None
        while True:
            if not self.queue:
                getattr(self, f"_{self.mode}")(obs)
                if not self.queue:
                    break
            e = self.queue[0]
            if e[0] == "settle":
                _, suction, tol, left = e
                self.suction_cmd = suction
                if left > 0 and np.abs(obs["qpos"] - self.q_cmd).max() > tol:
                    self.queue[0] = ("settle", suction, tol, left - 1)
                    break
                self.queue.pop(0)
                continue
            _, q, suction, guarded = e
            if guarded:
                fz = float(obs["ft"][2])
                if self.f_ref is None:
                    self.f_ref, self.f_hits = fz, 0
                self.f_hits = self.f_hits + 1 \
                    if abs(fz - self.f_ref) > GUARD_N else 0
                if self.f_hits >= 2:
                    while self.queue and self.queue[0][0] == "q" \
                            and self.queue[0][3]:
                        self.queue.pop(0)       # contact: stop descending
                    self.f_ref = None
                    continue
            else:
                self.f_ref = None
            self.queue.pop(0)
            self.q_cmd, self.suction_cmd = q, suction
            break
        return np.array([*self.q_cmd, self.suction_cmd])

    def _idle(self, obs) -> None:
        hmap = self.heightmap(obs)
        for i, b in sorted(self.boxes(obs).items()):
            if i in self.skip:
                continue
            slot = self.placement(hmap, b["dims"])
            if slot is None:
                self.skip.add(i)          # does not fit: let it pass
                continue
            if self._plan_meet(i, b, obs, slot):
                return
        self._wait()

    def _wait(self) -> None:
        q, err = self.ik([*WAIT_XY, self.cell["belt_top"] + 0.25], 0.0)
        if err < 0.01 and np.abs(q - self.q_cmd).max() > 0.02:
            self.joint(q, OFF, MOVE_SPEED)
        else:
            self.hold(OFF, 1)

    def _plan_meet(self, i, b, obs, slot) -> bool:
        v = obs["belt_v"]
        top = b["pos"]
        dt_meet = max((top[0] - WIN_X[1]) / v, 0.0)
        yaw = self.tool_yaw_for(top, b["yaw"])
        for _ in range(3):
            meet = top + np.array([-v * dt_meet, 0.0, PRE_DZ])
            q_pre, err = self.ik(meet, yaw)
            dt_meet = max(dt_meet, self.joint_ticks(q_pre, MOVE_SPEED)
                          * self.dt)
        meet_x = top[0] - v * dt_meet
        if err > 0.005 or meet_x - v * TRACK_S < WIN_X[0]:
            self.skip.add(i)              # too late for this one
            return False
        self.joint(q_pre, OFF, MOVE_SPEED, ticks=int(round(dt_meet / self.dt)))
        self.target = {"id": i, "yaw": yaw, "slot": slot,
                       "dims": b["dims"], "z": top[2] + PRE_DZ}
        self.track_left = int(TRACK_S / self.dt)
        self.mode = "track"
        return True

    def _track(self, obs) -> None:
        tg = self.target
        if obs["seal"] > 0.5:
            tg["offset"] = self.grasp_offset(obs)
            self._plan_place()
            return
        b = self.boxes(obs).get(tg["id"])
        self.track_left -= 1
        if b is None or self.track_left <= 0 or b["pos"][0] < WIN_X[0] - 0.05:
            self.skip.add(tg["id"])
            self.mode = "idle"
            self.line(self._tcp(self.q_cmd) + [0, 0, 0.08], tg["yaw"], OFF,
                      0.3)
            return
        p = b["pos"] + np.array(self.cell["belt_dir"]) * obs["belt_v"] * LEAD_S
        tg["z"] = max(tg["z"] - DESCEND_MPS * self.dt, p[2] - PUSH)
        q, _ = self.ik([p[0], p[1], tg["z"]], tg["yaw"], self.q_cmd, iters=30)
        self.queue.append(("q", q, ON, False))

    def grasp_offset(self, obs):
        """(box centre in the planar tool frame, box yaw - tool yaw) at seal,
        from the dead-reckoned box pose."""
        b = self.boxes(obs).get(self.target["id"])
        tcp, yaw = self._tcp(obs["qpos"]), self.tool_yaw(obs["qpos"])
        if b is None:
            return np.zeros(2), 0.0
        rel = _rot2(-yaw) @ (b["pos"][:2] - tcp[:2])
        dyaw = b["yaw"] - yaw
        return rel, dyaw - np.pi * np.round(dyaw / np.pi)

    def _plan_place(self) -> None:
        tg = self.target
        center, base, rot = tg["slot"]
        h = tg["dims"][2]
        carry_z = max(self.cell["bin_rim_z"],
                      self.cell["belt_top"] + 0.13) + h + CLEAR_Z
        rel, dyaw = tg["offset"]
        yaw = self.tool_yaw_for(center, (np.pi / 2 if rot else 0.0) - dyaw)
        xy = center - _rot2(yaw) @ rel          # puts the box centre on slot
        z_rel = self.cell["bin_floor_z"] + base + h + RELEASE_DZ
        tcp = self._tcp(self.q_cmd)
        self.line([tcp[0] - 0.02, tcp[1], carry_z], tg["yaw"], ON, 0.5)
        q_over, _ = self.ik([*xy, carry_z], yaw)
        self.joint(q_over, ON, CARRY_SPEED)
        self.settle(ON, 0.01, 15)
        self.line([*xy, z_rel + 0.03], yaw, ON, 0.35)
        self.settle(ON, 0.005, 10)
        self.line([*xy, z_rel], yaw, ON, 0.06, guarded=True)
        self.settle(ON, 0.004, 6)
        self.hold(OFF, 3)
        self.line([*xy, carry_z], yaw, OFF, 0.5)
        self._record(center, base, rot, tg["dims"])
        self.mode = "idle"

    def _record(self, center, base, rot, dims) -> None:
        a, b = (dims[1], dims[0]) if rot else (dims[0], dims[1])
        lo = np.floor((center - self.lo[:2] - [a / 2, b / 2]) / self.res)
        hi = np.ceil((center - self.lo[:2] + [a / 2, b / 2]) / self.res)
        lo, hi = np.clip(lo, 0, self.grid).astype(int), \
            np.clip(hi, 0, self.grid).astype(int)
        r = self.placed_hmap[lo[0]:hi[0], lo[1]:hi[1]]
        np.maximum(r, base + dims[2], out=r)


def make_policy():
    return Packer()
