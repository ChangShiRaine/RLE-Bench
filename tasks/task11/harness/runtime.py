"""The task11 episode loop: 20 Hz policy control over 500 Hz physics.

Drives any object exposing the Policy interface (a sandboxed submission,
an in-process test policy or the privileged golden packer) through one
seeded box stream. The runtime spawns boxes onto the moving belt, simulates
the suction cup (SuctionEngine), removes boxes that pass the overflow line,
and measures packing, drops, damage and tote strikes from HARNESS state only.

A policy fault (exception, bad shape, non-finite values, sandbox death)
never crashes the episode: the last valid ctrl is held and the fault is
counted. Faults surface through the verifier gate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rlebench.core.media import MujocoCamera, Recorder

from . import boxes as boxes_mod
from . import config, metrics, packing, scene, sensor, spec

PRE_SETTLE_T = 0.2
SPAWN_CLEAR_X = 0.30      # a box waits while another sits this close to SPAWN_X
COMMIT_ABOVE_RIM = 0.15   # release this far above the rim still commits


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class SuctionEngine:
    """Single vacuum cup: seals one box riding the belt whose face the
    energized cup touches square-on with the whole cup on that face;
    releases on de-energize."""

    def __init__(self, model, n: int, belt_v: float):
        self.model = model
        self.belt_v = belt_v
        self.eq = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                     f"suction_weld_{i}") for i in range(n)]
        self.cup = model.geom("arm_tool_cup").id
        self.tool = model.body("arm_tool").id
        self.tcp = model.site("arm_tcp").id
        self.box_of_geom = {scene.box_geom(model, i): i for i in range(n)}
        self.cos_tilt = np.cos(np.deg2rad(spec.SEAL_MAX_TILT_DEG))

    def held(self, data) -> int | None:
        return next((i for i, e in enumerate(self.eq) if data.eq_active[e]),
                    None)

    def riding(self, data, i: int) -> bool:
        """On the belt surface, inside its width, moving with it."""
        g = scene.box_geom(self.model, i)
        bottom = scene.box_corners(self.model, data, i)[:, 2].min()
        vel = scene.box_velocity(data, i)
        return bool(abs(bottom - spec.BELT_TOP) <= spec.BELT_RIDE_DZ
                    and abs(data.geom_xpos[g][1] - spec.BELT_Y)
                    <= spec.BELT_HALF_W
                    and np.hypot(vel[0] + self.belt_v, vel[1])
                    <= spec.BELT_RIDE_DV)

    def seal_ok(self, data, i: int) -> bool:
        g = scene.box_geom(self.model, i)
        half = self.model.geom_size[g]
        if self.model.body_mass[self.model.geom_bodyid[g]] \
                > spec.SUCTION_MAX_KG:
            return False
        if not self.riding(data, i):
            return False
        R = data.geom_xmat[g].reshape(3, 3)
        axis = data.site_xmat[self.tcp].reshape(3, 3)[:, 2]  # flange -> cup
        d = R.T @ -axis                  # outward normal of the sealed face
        k = int(np.argmax(np.abs(d)))
        if abs(d[k]) < self.cos_tilt:
            return False
        p = R.T @ (data.site_xpos[self.tcp] - data.geom_xpos[g])
        if abs(np.sign(d[k]) * p[k] - half[k]) > 0.005:
            return False
        others = [j for j in range(3) if j != k]
        return all(abs(p[j]) <= half[j] - spec.SEAL_EDGE_M for j in others)

    def _weld(self, data, i: int) -> None:
        b1, b2 = self.tool, self.model.body(scene.box_name(i)).id
        R1 = data.xmat[b1].reshape(3, 3)
        q1n, rel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1n, data.xquat[b1])
        mujoco.mju_mulQuat(rel, q1n, data.xquat[b2])
        e = self.eq[i]
        self.model.eq_data[e, 0:3] = 0.0
        self.model.eq_data[e, 3:6] = R1.T @ (data.xpos[b2] - data.xpos[b1])
        self.model.eq_data[e, 6:10] = rel
        self.model.eq_data[e, 10] = 1.0
        data.eq_active[e] = 1

    def release(self, data) -> None:
        for e in self.eq:
            data.eq_active[e] = 0

    def update(self, data, on: bool, exclude: set[int]) -> int | None:
        if not on:
            self.release(data)
            return None
        h = self.held(data)
        if h is not None:
            if h not in exclude:
                return h
            self.release(data)
        for c in range(data.ncon):
            con = data.contact[c]
            if con.dist > spec.SEAL_TOUCH:
                continue
            g1, g2 = int(con.geom1), int(con.geom2)
            other = g2 if g1 == self.cup else g1 if g2 == self.cup else None
            i = self.box_of_geom.get(other)
            if i is None or i in exclude or not self.seal_ok(data, i):
                continue
            self._weld(data, i)
            return i
        return None


@dataclass
class EpisodeLog:
    seed: int
    belt_v: float = 0.0
    budget_t: float = spec.EPISODE_T
    spawned: int = 0
    attempts: int = 0
    grasps: int = 0
    commit_t: dict = field(default_factory=dict)
    packed: list = field(default_factory=list)
    packed_volume: float = 0.0
    full_t: float | None = None
    passed: int = 0
    missed_fitting: int = 0
    floor_drops: int = 0
    damage: dict = field(default_factory=dict)
    bin_hits: int = 0
    peak_sustained: float = 0.0
    peak_tool_bin: float = 0.0
    penalty_events: list = field(default_factory=list)
    ticks: int = 0
    policy_faults: int = 0
    fault_reasons: list = field(default_factory=list)
    aborted: str | None = None      # "sim_unstable" | "wall_budget" | None
    digest: str = ""

    def metrics(self) -> dict:
        dropped = {e["box"] for e in self.penalty_events
                   if e["type"] == "floor_drop"}
        m = metrics.episode_metrics(
            n_packed=len(self.packed), packed_volume=self.packed_volume,
            attempts=self.attempts, grasps=self.grasps,
            commit_times=[self.commit_t[i] for i in self.packed],
            full_t=self.full_t, floor_drops=self.floor_drops,
            damage_events=len(set(self.damage) - dropped),
            bin_hits=self.bin_hits, passed=self.passed,
            missed_fitting=self.missed_fitting, spawned=self.spawned,
            budget_t=self.budget_t)
        m.update(belt_v=round(self.belt_v, 4), ticks=self.ticks,
                 policy_faults=self.policy_faults,
                 fault_reasons=list(self.fault_reasons),
                 aborted=self.aborted,
                 peak_sustained_force=round(self.peak_sustained, 2),
                 peak_tool_bin_force=round(self.peak_tool_bin, 2),
                 penalty_events=list(self.penalty_events),
                 digest=self.digest)
        return m


class NullPolicy:
    """Holds the home pose with the suction off: the do-nothing floor."""

    def reset(self, cell_spec: dict, seed: int) -> None:
        pass

    def act(self, obs: dict):
        return np.array([*spec.ARM_HOME, 0.0])


def default_cell_spec() -> dict:
    cell = spec.cell_spec()
    cell.update(damage_force_n=config.DAMAGE_FORCE_N,
                damage_ticks=config.DMG_TICKS,
                drop_speed_limit=config.DROP_SPEED_LIMIT,
                bin_hit_force_n=config.BIN_HIT_FORCE_N)
    return cell


def _policy_wall_s(policy) -> float:
    return float(getattr(policy, "policy_wall_s", 0.0))


def _r(v, digits: int = 4) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(v).reshape(-1)]


class _Cell:
    """Harness-side episode state: stream, spawning, belt, box bookkeeping."""

    def __init__(self, seed: int):
        self.seed = seed
        self.stream = boxes_mod.stream(seed)
        self.model, self.data = scene.build_scene(self.stream.boxes, seed=seed)
        self.n = len(self.stream.boxes)
        self.v = self.stream.belt_v
        jid = [self.model.joint(scene.box_name(i)).id for i in range(self.n)]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in jid])
        self.active: list[int] = []      # spawned and still in the cell
        self.next = 0
        self.scans: list[list[float]] = []
        self.grasped: set[int] = set()

    def pos(self, i: int) -> np.ndarray:
        return self.data.qpos[self.qadr[i]:self.qadr[i] + 3].copy()

    def speed(self, i: int) -> float:
        return float(np.linalg.norm(
            self.data.qvel[self.vadr[i]:self.vadr[i] + 3]))

    def spawn_due(self, t: float) -> None:
        while self.next < self.n \
                and self.stream.boxes[self.next].arrival_t <= t + 1e-9:
            if any(abs(self.pos(j)[0] - spec.SPAWN_X) < SPAWN_CLEAR_X
                   and self.pos(j)[2] > spec.BELT_TOP - 0.02
                   for j in self.active):
                return                      # entry blocked: wait
            b = self.stream.boxes[self.next]
            scene.set_box_active(self.model, b.index, True)
            pos = [spec.SPAWN_X, b.y, spec.BELT_TOP + b.dims[2] / 2 + 0.001]
            scene.set_box_pose(self.data, b.index, pos, _quat_yaw(b.yaw),
                               vel=(-self.v, 0.0, 0.0))
            rng = np.random.default_rng([self.seed, spec.STREAM_SCAN, b.index])
            dims = np.asarray(b.dims) + rng.normal(0, spec.SCAN_NOISE_DIM, 3)
            yaw = np.arctan2(np.sin(2 * b.yaw), np.cos(2 * b.yaw)) / 2 \
                + np.deg2rad(rng.normal(0, spec.SCAN_NOISE_YAW_DEG))
            self.scans.append([
                b.index, t, pos[0] + rng.normal(0, spec.SCAN_NOISE_POS),
                b.y + rng.normal(0, spec.SCAN_NOISE_POS), yaw, *dims,
                b.mass * (1 + rng.normal(0, spec.SCAN_NOISE_MASS_FRAC))])
            self.active.append(b.index)
            self.next += 1

    def retire(self, i: int) -> None:
        self.active.remove(i)
        scene.park(self.model, self.data, i)

    def in_bin_xy(self, p, margin: float = 0.0) -> bool:
        return (spec.BIN_LO[0] - margin <= p[0] <= spec.BIN_HI[0] + margin
                and spec.BIN_LO[1] - margin <= p[1] <= spec.BIN_HI[1] + margin)

    def bin_heightmap(self, held: int | None) -> np.ndarray:
        corners = [scene.box_corners(self.model, self.data, i)
                   for i in self.active
                   if i != held and self.in_bin_xy(self.pos(i), 0.05)
                   and self.pos(i)[2] < spec.BIN_RIM_Z + 0.3]
        return packing.heightmap(corners)

    def is_packed(self, i: int) -> bool:
        c = scene.box_corners(self.model, self.data, i)
        lo = np.asarray(spec.BIN_LO) - spec.PACK_TOL
        hi = np.asarray(spec.BIN_HI) + spec.PACK_TOL
        return bool(np.all(c >= lo) and np.all(c <= hi)
                    and scene.box_tilt_deg(self.data, i)
                    <= spec.PACK_MAX_TILT_DEG
                    and self.speed(i) < spec.REST_SPEED)


def run_episode(policy, seed: int, budget_t: float = spec.EPISODE_T,
                render: bool = True, wall_budget_s: float | None = None,
                cell_spec: dict | None = None, policy_seed: int | None = None,
                video=None, video_every: int = 1,
                video_size: tuple = (640, 360)) -> EpisodeLog:
    """One seeded episode. ``seed`` is private environment randomness (box
    stream, belt speed, sensor noise); ``policy_seed`` is the independent RNG
    seed passed to ``policy.reset`` (zero when omitted, never ``seed``).

    ``render=False`` skips the cameras (frame keys absent from obs).
    ``wall_budget_s`` meters only parent-observed reset()/act() time.
    ``video`` records the overview camera to an mp4 path or VideoWriter; the
    recorder is a pure observer and never steps physics.
    """
    cell = _Cell(seed)
    model, data, n = cell.model, cell.data, cell.n
    log = EpisodeLog(seed=seed, belt_v=cell.v, budget_t=budget_t)
    scene.set_arm(model, data, spec.ARM_HOME)
    scene.apply_ctrl(model, data, [*spec.ARM_HOME, 0.0])
    scene.settle(model, data, PRE_SETTLE_T, cell.v)

    suction = SuctionEngine(model, n, cell.v)
    full = metrics.FullDetector()
    floor = metrics.FloorTracker()
    damage = metrics.DamageTracker(config.DAMAGE_FORCE_N, config.DMG_TICKS,
                                   config.DROP_SPEED_LIMIT)
    bin_hit = metrics.BinHitTracker(config.BIN_HIT_FORCE_N,
                                    config.BIN_HIT_TICKS,
                                    config.BIN_HIT_REARM_TICKS)
    if getattr(policy, "privileged", False):
        policy.bind(cell)
    policy_wall_start = _policy_wall_s(policy)
    policy.reset(cell_spec or default_cell_spec(),
                 0 if policy_seed is None else int(policy_seed))

    rig = sensor.CameraRig(model) if render else None
    view = recorder = None
    if video:
        try:
            view = MujocoCamera(model, "overview", size=video_size)
            recorder = Recorder(video, fps=spec.CTRL_HZ / video_every,
                                every=video_every)
        except Exception:  # noqa: BLE001 — no GL: skip the recording
            view = recorder = None

    held_ctrl = np.array([*spec.ARM_HOME, 0.0])
    held: int | None = None
    prev_on = grasped_this_attempt = False
    ft_sigma = np.array([spec.FT_NOISE_FORCE_N] * 3
                        + [spec.FT_NOISE_TORQUE_NM] * 3)
    frame_idx = 0

    def over_budget() -> bool:
        return (wall_budget_s is not None and
                _policy_wall_s(policy) - policy_wall_start > wall_budget_s)

    try:
        for tick in range(int(round(budget_t * spec.CTRL_HZ))):
            if over_budget():
                log.aborted = "wall_budget"
                break
            t = tick / spec.CTRL_HZ
            cell.spawn_due(t)
            ft_rng = np.random.default_rng([seed, spec.STREAM_FT, tick])
            obs = {
                "qpos": scene.get_arm(data), "qvel": scene.get_arm_vel(data),
                "tau": scene.arm_tau(data),
                "ft": scene.ft_reading(model, data)
                + ft_rng.normal(0.0, 1.0, 6) * ft_sigma,
                "suction_on": float(held_ctrl[7] >= spec.SUCTION_ON),
                "seal": float(held is not None),
                "t": t, "belt_v": cell.v,
                "scans": np.array(cell.scans, dtype=float).reshape(-1, 9),
            }
            if rig is not None and tick % spec.FRAME_EVERY_TICKS == 0:
                obs.update(sensor.frame_obs(rig, model, data, seed, frame_idx))
                frame_idx += 1

            ctrl, fault = None, None
            try:
                ctrl = policy.act(obs)
            except Exception as e:  # noqa: BLE001
                fault = f"act raised: {e!r}"
            if over_budget():
                log.aborted = "wall_budget"
                break
            if ctrl is not None:
                ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
                if ctrl.shape != (spec.N_CTRL,) or not np.isfinite(ctrl).all():
                    fault, ctrl = f"bad ctrl (shape {ctrl.shape})", None
            if ctrl is None:
                log.policy_faults += 1
                if fault and len(log.fault_reasons) < 10:
                    log.fault_reasons.append(f"t={t:.2f} {fault}")
                ctrl = held_ctrl
            held_ctrl = scene.apply_ctrl(model, data, ctrl)

            on = bool(held_ctrl[7] >= spec.SUCTION_ON)
            if on and not prev_on:
                log.attempts += 1
                grasped_this_attempt = False
            prev_on = on
            before = held
            gone = set(range(n)) - set(cell.active)
            held = suction.update(data, on, exclude=set(log.commit_t) | gone)
            if held is not None and held != before:
                cell.grasped.add(held)
                full.reset()
                if not grasped_this_attempt:
                    log.grasps += 1
                    grasped_this_attempt = True
            if before is not None and held != before:
                p = cell.pos(before)
                if cell.in_bin_xy(p) and p[2] < spec.BIN_RIM_Z \
                        + COMMIT_ABOVE_RIM:
                    log.commit_t[before] = t
                    full.reset()

            for _ in range(spec.CTRL_DECIMATION):
                scene.drive_belt(data, cell.v)
                mujoco.mj_step(model, data)
            log.ticks = tick + 1
            t_next = (tick + 1) / spec.CTRL_HZ
            if recorder is not None:
                recorder.capture(lambda: view.render(data))
            if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 \
                    or not np.isfinite(data.qpos).all():
                log.aborted = "sim_unstable"
                break

            act = list(cell.active)
            pos = {i: cell.pos(i) for i in act}
            for i in floor.update(pos):
                log.penalty_events.append({"type": "floor_drop", "box": i,
                                           "t": t_next, "pos": _r(pos[i])})
            forces_all = scene.box_contact_forces(model, data, n)
            contact_all = scene.box_in_contact(model, data, n)
            free = {i for i in act if i in cell.grasped and i != held}
            for i in damage.update(
                    {i: float(forces_all[i]) for i in act},
                    {i: cell.speed(i) for i in act},
                    {i: bool(contact_all[i]) for i in act}, free):
                log.penalty_events.append({
                    "type": damage.damaged[i], "box": i, "t": t_next,
                    "pos": _r(pos[i]), "force_n": round(float(forces_all[i]), 1),
                    "held": i == held})
            if bin_hit.update(scene.tool_bin_force(model, data)):
                log.penalty_events.append({"type": "bin_hit", "t": t_next})
            for i in act:
                if i != held and pos[i][0] < spec.BELT_END_X \
                        and pos[i][2] > spec.BELT_TOP - 0.05:
                    fitted = packing.fits(cell.bin_heightmap(held),
                                          cell.stream.boxes[i].dims)
                    cell.retire(i)
                    full.on_pass(t_next, fitted)
            if full.full:
                break

        for _ in range(int(round(spec.FINAL_SETTLE_T / spec.TIMESTEP))):
            scene.drive_belt(data, cell.v)
            mujoco.mj_step(model, data)
    finally:
        if rig is not None:
            rig.close()
        if recorder is not None:
            recorder.close()
            view.close()

    log.spawned = cell.next
    log.packed = sorted(i for i in log.commit_t
                        if i in cell.active and i != held and cell.is_packed(i))
    log.packed_volume = float(sum(cell.stream.boxes[i].volume
                                  for i in log.packed))
    log.full_t = full.full_t
    log.passed, log.missed_fitting = full.passed, full.missed_fitting
    log.floor_drops = len(floor.dropped)
    log.damage = dict(damage.damaged)
    log.bin_hits = bin_hit.hits
    log.peak_sustained = damage.peak_sustained
    log.peak_tool_bin = bin_hit.peak_sustained
    h = hashlib.sha256(np.round(np.asarray(data.qpos), 9).tobytes())
    h.update(repr(sorted(log.commit_t.items())).encode())
    h.update(f"{log.attempts}:{log.passed}:{log.full_t}".encode())
    log.digest = h.hexdigest()[:16]
    return log
