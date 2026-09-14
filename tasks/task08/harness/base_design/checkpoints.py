"""The checkpoints, each a pure function of harness results.

Checkpoints are computed from the harness's own instrumentation, never from
artifacts the agent-under-test reports.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .. import config
from ..sim.mecanum import WHEEL_ORDER, geometry_from_model
from rlebench.core.model import (SETTLE_DRIFT_EPS, SETTLE_PENETRATION_EPS,
                        SETTLE_SPEED_MAX, SETTLE_TILT_DEG_MAX)
from ..sim.arm_variants import (ArmSpec, canonical_arm_body_names,
                               canonical_reach)
from rlebench.core.scoring import Checkpoint, share_credit
from rlebench.core.scoring import constraint_credit as _constraint_credit


CHECKPOINTS = [
    Checkpoint("S1.load", "validity", 0.03, gate=True),
    Checkpoint("S1.inertia", "validity", 0.03),
    Checkpoint("S1.structure", "validity", 0.03, gate=True),
    Checkpoint("S1.mecanum", "validity", 0.03),
    Checkpoint("S1.equilibrium", "validity", 0.03),
    Checkpoint("S3.still_valid", "design", 0.03),
    Checkpoint("S3.resource_constraints", "design", 0.02),
    Checkpoint("S3.design_efficiency", "design", 0.22),
    Checkpoint("S3.reach_preserved", "design", 0.025),
    Checkpoint("S3.reach_beyond", "design", 0.025),
    Checkpoint("S3.battery_onboard", "design", 0.03),
    Checkpoint("S3.static_margin", "static", 0.15),
    Checkpoint("S5.testB", "dynamic", 0.04),
    Checkpoint("S5.testC", "dynamic", 0.04),
    Checkpoint("S5.testD1", "dynamic", 0.04),
    Checkpoint("S5.testD2", "dynamic", 0.04),
    Checkpoint("S5.testE", "dynamic", 0.04),
    Checkpoint("S7.pick_compat", "integration", 0.10),
    Checkpoint("S7.payload_margin", "integration", 0.05),
]


def saturating(value: float, sat: float) -> float:
    """Reward in [0, 1], linear up to `sat`, capped there: over-building past
    the calibrated level earns nothing extra."""
    if not math.isfinite(value) or value <= 0.0 or sat <= 0.0:
        return 0.0
    return min(value / sat, 1.0)


def constraint_credit(ratio: float | None,
                      band: float | None = None) -> float:
    """task08's calibrated decay band by default; the math lives in core."""
    return _constraint_credit(ratio, config.CONSTRAINT_BAND if band is None else band)


# ---------------------------------------------------------------------------
# Model inspection (the harness's own reading of the submitted model)
# ---------------------------------------------------------------------------

def _geom_aabb_xy(model, data, gid) -> tuple[float, float, float, float]:
    """World-frame AABB (xmin, xmax, ymin, ymax) of one primitive geom."""
    pos = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    ext = np.zeros(3)
    gtype = model.geom_type[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        ext[:] = size[0]
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        ext = np.abs(R) @ size
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ext = np.abs(R[:, 2]) * size[1] + size[0]
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = np.abs(R[:, 2])
        ext = a * size[1] + size[0] * np.sqrt(np.clip(1 - a ** 2, 0, 1))
    else:  # mesh/other: conservative bounding sphere
        ext[:] = model.geom_rbound[gid]
    return pos[0] - ext[0], pos[0] + ext[0], pos[1] - ext[1], pos[1] + ext[1]


def footprint_bounds(model, data, exclude_bodies: set[str]):
    """World-frame XY bounds (lo, hi) of everything except the canonical
    arm/payload, in the pose currently in `data`."""
    mujoco.mj_forward(model, data)
    lo = np.array([np.inf, np.inf])
    hi = -lo.copy()
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if int(model.body_rootid[bid]) == 0:
            continue  # terrain / world
        if (model.body(bid).name or "") in exclude_bodies:
            continue
        x0, x1, y0, y1 = _geom_aabb_xy(model, data, gid)
        lo = np.minimum(lo, (x0, y0))
        hi = np.maximum(hi, (x1, y1))
    return lo, hi


def base_footprint(model, data, exclude_bodies: set[str]) -> tuple[float, float]:
    """(x_extent, y_extent) of everything except the canonical arm/payload,
    in the pose currently in `data` (the scenario-start pose — never a
    submission-controlled keyframe)."""
    lo, hi = footprint_bounds(model, data, exclude_bodies)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1])


_PROFILE_HALF_SECTIONS_M = {
    "2020": (0.01, 0.01),
    "2040": (0.01, 0.02),
    "4040": (0.02, 0.02),
}


def stock_profile_inventory(model, exclude_bodies: set[str]) -> dict:
    """Measure stock-equivalent profile length from actual chassis boxes.

    Classification uses local geom dimensions, not submitted names. Canonical
    wheel/battery/arm component geoms are excluded by their fixed identities;
    any other non-world geom must match a 2020, 2040, or 4040 cross-section for
    the design to be eligible for profile-efficiency credit.
    """
    trusted = {"battery_geom"}
    for suffix in WHEEL_ORDER:
        trusted.add(f"hub_{suffix}")
        trusted.update(f"rollerg_{suffix}_{i}" for i in range(16))
    lengths = {name: 0.0 for name in _PROFILE_HALF_SECTIONS_M}
    counts = {name: 0 for name in _PROFILE_HALF_SECTIONS_M}
    non_profile = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if int(model.body_rootid[bid]) == 0:
            continue
        body_name = model.body(bid).name or ""
        geom_name = model.geom(gid).name or f"geom_{gid}"
        if body_name in exclude_bodies or geom_name in trusted:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            non_profile.append(geom_name)
            continue
        half = np.asarray(model.geom_size[gid], dtype=float)
        candidates = []
        for profile, section in _PROFILE_HALF_SECTIONS_M.items():
            for axis in range(3):
                cross = np.delete(half, axis)
                if np.allclose(sorted(cross), section,
                               rtol=0.02, atol=0.0005):
                    candidates.append((2.0 * float(half[axis]), profile))
        if not candidates:
            non_profile.append(geom_name)
            continue
        length, profile = max(candidates)
        lengths[profile] += length
        counts[profile] += 1
    total = float(sum(lengths.values()))
    count = int(sum(counts.values()))
    return dict(total_length_m=total, count=count,
                length_by_type_m=lengths, count_by_type=counts,
                non_profile_chassis_geoms=non_profile,
                eligible=bool(count and not non_profile))


def _remaining_budget_fraction(value: float, limit: float) -> float:
    """Continuous smaller-is-better reward: remaining fraction of budget."""
    if not math.isfinite(value) or value < 0.0 or limit <= 0.0:
        return 0.0
    return float(np.clip(1.0 - value / limit, 0.0, 1.0))


def _soft_upper_target_score(value: float, target: float) -> float:
    """Full credit through a preferred target, then a smooth inverse falloff."""
    if not math.isfinite(value) or value < 0.0 or target <= 0.0:
        return 0.0
    if value <= target:
        return 1.0
    return float(target / value)


def resource_constraint_score(model_info: dict) -> dict:
    """Soft resource-target score; no target violation invalidates a model.

    Values within the preferred envelope receive full constraint credit.
    Oversized or heavy designs retain partial credit that decreases smoothly.
    Unrecognized chassis geometry loses only the profile component.
    """
    if not model_info.get("ok"):
        return dict(score=0.0, components={})
    footprint = model_info.get("footprint", (float("inf"), float("inf")))
    inventory = model_info.get("profile_inventory", {})
    profile_score = (
        _soft_upper_target_score(
            float(inventory.get("total_length_m", float("inf"))),
            config.PROFILE_LENGTH_BUDGET_M)
        if inventory.get("eligible") else 0.0)
    components = {
        "footprint_x": _soft_upper_target_score(
            float(footprint[0]), config.FOOTPRINT_BUDGET_M),
        "footprint_y": _soft_upper_target_score(
            float(footprint[1]), config.FOOTPRINT_BUDGET_M),
        "assembled_mass": _soft_upper_target_score(
            float(model_info.get("total_mass", float("inf"))),
            config.MASS_BUDGET_KG),
        "profile_length": profile_score,
        "arm_integrity": 1.0 if model_info.get("arm", {}).get("ok") else 0.0,
        "actuators": 1.0 if model_info.get("actuators", {}).get("ok") else 0.0,
    }
    return dict(score=float(np.mean(list(components.values()))),
                components=components)


def design_efficiency(model_info: dict) -> dict:
    """Mass/profile/footprint reward from normalized remaining budgets.

    There is deliberately no full-credit plateau below the soft targets: every
    reduction in a valid design's resource use increases headline reward.
    """
    footprint = model_info.get("footprint", (float("inf"), float("inf")))
    assembled_mass = float(model_info.get("total_mass", float("inf")))
    mass = float(model_info.get("base_mass", assembled_mass))
    mass_limit = (config.BASE_MASS_BUDGET_KG
                  if "base_mass" in model_info else config.MASS_BUDGET_KG)
    inventory = model_info.get("profile_inventory", {})
    length = float(inventory.get("total_length_m", float("inf")))
    area = float(footprint[0] * footprint[1])
    area_limit = config.FOOTPRINT_BUDGET_M ** 2
    components = {
        "mass": _remaining_budget_fraction(mass, mass_limit),
        "profile_length": (_remaining_budget_fraction(
            length, config.PROFILE_LENGTH_BUDGET_M)
            if inventory.get("eligible") else 0.0),
        "footprint_area": _remaining_budget_fraction(area, area_limit),
    }
    return dict(
        score=float(np.mean(list(components.values()))),
        components=components,
        total_mass_kg=assembled_mass,
        base_mass_kg=(mass if "base_mass" in model_info else None),
        mass_budget_kg=mass_limit,
        mass_headroom_kg=mass_limit - mass,
        mass_headroom_fraction=components["mass"],
        profile_length_m=length,
        profile_length_headroom_m=config.PROFILE_LENGTH_BUDGET_M - length,
        profile_length_headroom_fraction=components["profile_length"],
        profile_count=int(inventory.get("count", 0)),
        profile_eligible=bool(inventory.get("eligible", False)),
        footprint_m=[float(footprint[0]), float(footprint[1])],
        footprint_axis_headroom_m=[
            config.FOOTPRINT_BUDGET_M - float(footprint[0]),
            config.FOOTPRINT_BUDGET_M - float(footprint[1])],
        footprint_area_m2=area,
        footprint_area_headroom_m2=area_limit - area,
        footprint_area_headroom_fraction=components["footprint_area"],
    )


def arm_integrity(model, arm: ArmSpec) -> dict:
    """Check canonical arm masses, limits and actuator gains."""
    ref = mujoco.MjModel.from_xml_path(arm.reference_xml)
    problems = []
    for i in range(1, ref.nbody):
        assembled = arm.body(ref.body(i).name)
        try:
            body = model.body(assembled)
        except KeyError:
            problems.append(f"missing arm body {assembled}")
            continue
        if not np.isclose(body.mass[0], ref.body(i).mass[0], rtol=0.01):
            problems.append(f"{assembled} mass changed")
        if not np.allclose(body.inertia, ref.body(i).inertia, rtol=0.02):
            problems.append(f"{assembled} inertia changed")
    for j in range(ref.njnt):
        assembled = arm.body(ref.joint(j).name)
        try:
            jnt = model.joint(assembled)
        except KeyError:
            problems.append(f"missing arm joint {assembled}")
            continue
        if not np.allclose(jnt.range, ref.joint(j).range, atol=1e-6):
            problems.append(f"{assembled} joint range changed")
    for a in range(ref.nu):
        assembled = arm.body(ref.actuator(a).name)
        try:
            act = model.actuator(assembled)
        except KeyError:
            problems.append(f"missing arm actuator {assembled}")
            continue
        if not np.allclose(act.gainprm[:3], ref.actuator(a).gainprm[:3], rtol=0.01):
            problems.append(f"{assembled} actuator gain changed")
    return dict(ok=not problems, problems=problems)


def wheel_actuator_caps(model) -> dict:
    """Anti-cheese: wheel motors within spec limits."""
    problems = []
    for w in WHEEL_ORDER:
        try:
            act = model.actuator(f"motor_{w}")
        except KeyError:
            problems.append(f"missing motor_{w}")
            continue
        if act.gainprm[0] > config.WHEEL_KV_MAX + 1e-9:
            problems.append(f"motor_{w} kv {act.gainprm[0]} > {config.WHEEL_KV_MAX}")
        if np.max(np.abs(act.forcerange)) > config.WHEEL_FORCERANGE_MAX + 1e-9:
            problems.append(f"motor_{w} forcerange exceeds cap")
        if np.max(np.abs(act.ctrlrange)) > config.WHEEL_CTRLRANGE_MAX + 1e-9:
            problems.append(f"motor_{w} ctrlrange exceeds cap")
    return dict(ok=not problems, problems=problems)


def measured_reach(model, data, arm: ArmSpec) -> float:
    """Horizontal reach of the flange from the arm root, canonical extended
    pose, on the submitted model."""
    mujoco.mj_resetData(model, data)
    for i, q in enumerate(arm.extended(0.0)):
        data.qpos[model.joint(arm.joint(i)).qposadr[0]] = q
    mujoco.mj_forward(model, data)
    site = data.site(arm.ee_site).xpos
    root = data.body(arm.assembled_root).xpos
    return float(np.hypot(site[0] - root[0], site[1] - root[1]))


def measured_reach_beyond(model, data, exclude_bodies: set[str],
                          arm: ArmSpec) -> float:
    """Worst-direction horizontal overhang of the flange past the base
    footprint boundary, swept over the arm's slewing range: the reach the
    robot can actually use to work OUTSIDE its own base. An arm buried in
    the middle of a maximal footprint preserves nominal reach yet can barely
    clear the chassis — this is the number that catches it."""
    mujoco.mj_resetData(model, data)
    lo, hi = footprint_bounds(model, data, exclude_bodies)
    site_id = model.site(arm.ee_site).id
    qadr = [model.joint(arm.joint(i)).qposadr[0]
            for i in range(len(arm.joint_names))]
    worst = float("inf")
    az_limit = min(np.pi, float(model.joint(arm.joint(0)).range[1]))
    for az in np.linspace(-az_limit, az_limit, 12):
        for adr, q in zip(qadr, arm.extended(az)):
            data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        x, y = data.site_xpos[site_id][:2]
        dx = max(lo[0] - x, 0.0, x - hi[0])
        dy = max(lo[1] - y, 0.0, y - hi[1])
        worst = min(worst, float(np.hypot(dx, dy)))
    return worst


def battery_onboard(model) -> dict:
    """The spec battery pack must ride on the base: right mass, right size,
    rigidly mounted (no joints between battery and base — a pack hanging off
    the arm or a wheel is not an installation)."""
    problems = []
    try:
        body = model.body("battery")
    except KeyError:
        return dict(ok=False, problems=["missing body 'battery'"])
    mass = float(body.mass[0])
    if not math.isclose(mass, config.BATTERY_MASS_KG,
                        rel_tol=config.BATTERY_MASS_RTOL):
        problems.append(f"battery mass {mass:.2f} kg != spec "
                        f"{config.BATTERY_MASS_KG} kg")
    try:
        geom = model.geom("battery_geom")
        if int(model.geom_bodyid[geom.id]) != body.id:
            problems.append("battery_geom is not on the battery body")
        if model.geom_type[geom.id] != mujoco.mjtGeom.mjGEOM_BOX:
            problems.append("battery_geom must be a box")
        else:
            got = sorted(float(s) for s in model.geom_size[geom.id])
            want = sorted(config.BATTERY_SIZE_HALF_M)
            if not all(math.isclose(a, b, rel_tol=config.BATTERY_SIZE_RTOL)
                       for a, b in zip(got, want)):
                problems.append(f"battery_geom size {got} != spec {want}")
    except KeyError:
        problems.append("missing geom 'battery_geom'")
    # rigid mount: no joint anywhere between the battery and the base
    try:
        base_id = model.body("base").id
        bid = body.id
        while bid not in (base_id, 0):
            if int(model.body_jntnum[bid]) > 0:
                problems.append("battery is not rigidly mounted to the base")
                break
            bid = int(model.body_parentid[bid])
        else:
            if bid == 0:
                problems.append("battery is not mounted under the base")
    except KeyError:
        problems.append("missing body 'base'")
    return dict(ok=not problems, problems=problems, mass=mass)


def analyze_model(robot_xml: str, arm: ArmSpec) -> dict:
    """Harness's own reading of the submitted model with one canonical arm
    (budgets, reach, caps)."""
    from ..sim.scenarios import Envelope, compose_scene, set_payload
    out: dict = dict(ok=False)
    try:
        model, data, _ = compose_scene(robot_xml, "A", 0, Envelope(), arm)
    except Exception as e:
        out["error"] = str(e)
        return out
    set_payload(model, data, 0.0, arm)
    # No keyframe reset: data holds compose_scene's scenario-start pose, so
    # the footprint is measured in the configuration that is actually
    # simulated (an agent-supplied keyframe is untrusted input).
    exclude = canonical_arm_body_names(arm)
    out["footprint"] = base_footprint(model, data, exclude)
    out["total_mass"] = float(mujoco.mj_getTotalmass(model))
    out["base_mass"] = float(sum(
        model.body_mass[i] for i in range(1, model.nbody)
        if (model.body(i).name or "") not in exclude))
    out["profile_inventory"] = stock_profile_inventory(model, exclude)
    out["design_efficiency"] = design_efficiency(out)
    out["arm"] = arm_integrity(model, arm)
    out["actuators"] = wheel_actuator_caps(model)
    out["ok"] = True
    out["resource_constraints"] = resource_constraint_score(out)
    out["battery"] = battery_onboard(model)
    out["reach"] = measured_reach(model, data, arm)
    # "Preserved" means keeping essentially all of the reach the arm has on
    # its own: one absolute bar cannot serve three arms whose own reaches
    # span 0.77-0.97 m.
    out["canonical_reach"] = canonical_reach(arm)
    out["reach_floor"] = config.REACH_PRESERVED_FRAC * out["canonical_reach"]
    out["reach_beyond"] = measured_reach_beyond(model, data, exclude, arm)
    lx, ly, _ = geometry_from_model(model)
    out["half_wheelbase"], out["half_track"] = lx, ly
    return out


# ---------------------------------------------------------------------------
# Per-scenario scores (hidden-seed battery)
# ---------------------------------------------------------------------------

def tracking_error(result: dict) -> float:
    """Weighted normalized RMS twist error over the scenario."""
    cmd = np.asarray(result["twist_cmd"], dtype=float)
    meas = np.asarray(result["twist_meas"], dtype=float)
    if cmd.size == 0:
        return float("inf")
    w = np.array([1.0, 1.0, 0.3])
    err = np.sqrt(np.mean(np.sum(w * (cmd - meas) ** 2, axis=1)))
    ref = np.sqrt(np.mean(np.sum(w * cmd ** 2, axis=1)))
    return float(err / (ref + 1e-9))


def score_scenario(result: dict) -> float:
    """One scenario run -> [0, 1]. Hard fails (tip, support loss, or not
    executing the commanded maneuver) zero it; continuous margins earn
    saturating partial credit.

    The tracking gate is what makes S5 unclaimable by a stationary
    controller: being stable while refusing to drive is scenario A, not the
    dynamic battery."""
    sid = result["scenario"]
    if result["tip"]:
        return 0.0
    if tracking_error(result) > config.S5_TRACKING_GATE:
        return 0.0
    events = result["liftoff_events"]
    fasm_min = result.get("min_fasm_robust", result["min_fasm"])
    if sid == "B":
        if events:
            return 0.0
        return saturating(fasm_min, config.FASM_SAT)
    if sid == "C":
        if events:
            return 0.0
        # full credit below 20% of the limit, linear to zero at the limit
        return float(np.clip((config.ROLL_LIMIT_DEG - result["max_tilt_deg"])
                             / (0.8 * config.ROLL_LIMIT_DEG), 0.0, 1.0))
    if sid == "D1":
        if events:
            return 0.0
        # any positive margin on the worst held-out slope saturates quickly
        return saturating(fasm_min, 1.0)
    if sid == "D2":
        # binary on the calibrated caps: event COUNT on a rigid frame is
        # terrain-seed luck; what distinguishes designs is tipping and
        # sustained multi-wheel support loss
        if len(events) > config.D2_MAX_LIFTOFF_EVENTS:
            return 0.0
        if any(e["duration"] > config.D2_MAX_LIFTOFF_DURATION_S for e in events):
            return 0.0
        return 1.0
    if sid == "E":
        if events:
            return 0.0
        return saturating(fasm_min, config.FASM_SAT)
    raise ValueError(sid)


def _settle_credit(settle: dict) -> float:
    """Continuous "stands still at rest": the worst of the five settle
    constraints. One property measured five ways, so the worst one defines
    it — a base that drifts twice the limit is not two-thirds settled."""
    if not settle:
        return 0.0
    ratios = (
        settle.get("drift_xy", float("inf")) / SETTLE_DRIFT_EPS,
        abs(settle.get("dz", float("inf"))) / SETTLE_DRIFT_EPS,
        settle.get("tilt_deg", float("inf")) / SETTLE_TILT_DEG_MAX,
        settle.get("end_speed", float("inf")) / SETTLE_SPEED_MAX,
        settle.get("max_penetration", float("inf")) / SETTLE_PENETRATION_EPS,
    )
    return min(constraint_credit(r) for r in ratios)


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate(hidden_results: dict, model_info: dict,
             pick_probe: dict | None = None) -> dict:
    """Return {checkpoint_id: value in [0,1]}, which feeds the reward.

    hidden_results: runner.run_all output on the hidden seeds.
    model_info: analyze_model() output.
    pick_probe: controlled_pick.evaluate_controlled_pick() output, or None
    if the shelf evaluation could not run.
    """
    v = hidden_results.get("validity", {})
    scores: dict[str, float] = {}

    # S1.load and S1.structure are the two hard gates: without a model that
    # compiles and carries the interface, nothing downstream is measurable.
    scores["S1.load"] = 1.0 if v.get("loads") else 0.0
    scores["S1.structure"] = 1.0 if v.get("structure", {}).get("ok") else 0.0
    # the rest score continuously — share of bodies with realizable inertia
    scores["S1.inertia"] = share_credit(
        1.0 if b.get("ok") else 0.0 for b in v.get("inertia", {}).values())
    # share of holonomic twists the base actually tracks
    scores["S1.mecanum"] = share_credit(
        1.0 if c.get("ok") else 0.0 for c in v.get("mecanum", {}).get("cases", []))
    scores["S1.equilibrium"] = _settle_credit(v.get("settle", {}))

    scores["S3.still_valid"] = 1.0 if v.get("ok") else 0.0
    constraints = model_info.get("resource_constraints", {})
    scores["S3.resource_constraints"] = float(
        constraints.get("score", 0.0))
    efficiency = model_info.get("design_efficiency", {})
    scores["S3.design_efficiency"] = float(efficiency.get("score", 0.0))
    scores["S3.reach_preserved"] = (1.0 if model_info.get("ok")
                                    and model_info["reach"] >= model_info["reach_floor"]
                                    else 0.0)
    scores["S3.reach_beyond"] = (
        1.0 if model_info.get("ok")
        and model_info["reach_beyond"] >= config.REACH_BEYOND_MIN_M else 0.0)
    scores["S3.battery_onboard"] = (
        1.0 if model_info.get("ok") and model_info["battery"]["ok"] else 0.0)

    a_mins = list(hidden_results.get("scenario_a_min_ssm", {}).values())
    if a_mins:
        span = config.SSM_SAT_M - config.SSM_THRESHOLD_M
        scores["S3.static_margin"] = float(np.mean(
            [saturating(m - config.SSM_THRESHOLD_M, span) for m in a_mins]))
    else:
        scores["S3.static_margin"] = 0.0

    for sid in ("B", "C", "D1", "D2", "E"):
        runs = list(hidden_results.get("scenarios", {}).get(sid, {}).values())
        scores[f"S5.test{sid}"] = float(np.mean(
            [score_scenario(r) for r in runs])) if runs else 0.0

    scores["S7.pick_compat"] = (float(pick_probe["score"])
                                if pick_probe else 0.0)
    scores["S7.payload_margin"] = (float(pick_probe["margin_score"])
                                  if pick_probe else 0.0)
    return scores
