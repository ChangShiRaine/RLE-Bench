"""Shelf entry under submitted control, measured only from verifier physics."""
from __future__ import annotations

from collections import deque
import hashlib
import math

import mujoco
import numpy as np

from .. import config
from ..metrics.fasm import force_angle_stability, net_com_force
from ..metrics.support_polygon import contact_normal_forces, support_polygon_3d, wheel_contacts
from ..sim.mecanum import WHEEL_ORDER
from ..sim.scenarios import (DECIMATE, LIFTOFF_FORCE_EPS, LIFTOFF_TAU,
                             METRIC_WINDOW, TIMESTEP, TIP_TILT_DEG)
from .controller_process import ControllerProcess
from . import pickscene as scene


def initialize_episode(model, data):
    """Set the public base pose; compose_pick_scene supplies the arm stow."""
    adr = model.joint("base_free").qposadr[0]
    data.qpos[adr:adr + 3] = config.PICK_START_POSITION
    data.qpos[adr + 3:adr + 7] = config.PICK_START_QUATERNION
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def controller_context(model, data, arm, target, name):
    return dict(
        arm=arm.name, target=target.tolist(), target_name=name,
        seed=config.PICK_CONTROLLER_SEED, control_dt=config.PICK_CONTROL_DT,
        duration=config.PICK_EPISODE_SECONDS,
        qpos=data.qpos.tolist(), qvel=data.qvel.tolist(), ctrl=data.ctrl.tolist(),
        actuator_names=[model.actuator(i).name for i in range(model.nu)],
        arm_joint_names=[arm.joint(i) for i in range(len(arm.joint_names))],
        arm_actuator_names=[arm.actuator(i) for i in range(len(arm.joint_names))],
        ee_site=arm.ee_site, base_joint="base_free",
        wheel_actuator_names=[f"motor_{w}" for w in WHEEL_ORDER])


def _step_physics(model, data):
    before = data.time
    mujoco.mj_step(model, data)
    # MuJoCo can reset unstable state internally. Such a reset must never
    # teleport a submission past the outside approach requirement.
    numerical = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                 mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
    if (data.time <= before or any(data.warning[k].number for k in numerical)
            or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()):
        raise ValueError("unstable physics or simulation reset")


def _measure(model, data, target, site, history, now):
    history.append((now, wheel_contacts(model, data), net_com_force(model, data)))
    while history and history[0][0] < now - METRIC_WINDOW:
        history.popleft()
    points = [row[1] for row in history if len(row[1])]
    contacts = np.vstack(points) if points else np.zeros((0, 3))
    force = np.median([row[2] for row in history], axis=0)
    margin = force_angle_stability(data.subtree_com[0], support_polygon_3d(contacts), force)
    error = float(np.linalg.norm(data.site_xpos[site] - target))
    return error, margin


def run_target(robot_xml, face_x, arm, target, name, payload_kg, source,
               process_factory=ControllerProcess):
    model, data, shelves = scene.compose_pick_scene(robot_xml, face_x, payload_kg, arm)
    shelf_ids = {model.geom(n).id for n in shelves}
    initialize_episode(model, data)
    context = controller_context(model, data, arm, target, name)
    context.update(payload_kg=float(payload_kg), shelf_face_x=float(face_x))
    out = dict(ok=False, reason="target hold not completed", payload_kg=float(payload_kg),
               target=target.tolist(), target_name=name,
               ee_err=None, min_fasm_hold=None, shelf_force_max=0.0,
               tip=False, liftoff_events=[], controller_fault=None)
    robot_bodies = set(range(1, model.nbody))
    history, hold = deque(), deque()
    degraded_since = None
    elapsed = 0.0
    digest = hashlib.sha256()
    site = model.site(arm.ee_site).id
    cf = np.zeros(6)
    period = round(config.PICK_CONTROL_DT / TIMESTEP)
    try:
        with process_factory(source, model, context) as controller:
            for step in range(round(config.PICK_EPISODE_SECONDS / TIMESTEP)):
                if step % period == 0:
                    observation = dict(time=step * TIMESTEP,
                                       qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                                       ctrl=data.ctrl.tolist())
                    action = np.asarray(controller.act(observation), dtype=float)
                    if action.shape != (model.nu,) or not np.isfinite(action).all():
                        raise ValueError("invalid controller action")
                    # MuJoCo applies trusted actuator force limits; clamp controls
                    # explicitly too. All other model/data fields stay parent-owned.
                    limited = model.actuator_ctrllimited.astype(bool)
                    action[limited] = np.clip(action[limited],
                                             model.actuator_ctrlrange[limited, 0],
                                             model.actuator_ctrlrange[limited, 1])
                    data.ctrl[:] = action
                    digest.update(action.astype("<f8").tobytes())
                _step_physics(model, data)
                elapsed = (step + 1) * TIMESTEP
                tilt = math.degrees(math.acos(float(np.clip(
                    data.body("base").xmat.reshape(3, 3)[2, 2], -1, 1))))
                if tilt > TIP_TILT_DEG:
                    out.update(reason="tipped over", tip=True)
                    break
                normal = contact_normal_forces(model, data)
                loaded = sum(normal.get(w, 0.0) >= LIFTOFF_FORCE_EPS for w in WHEEL_ORDER)
                if loaded < 3:
                    degraded_since = elapsed if degraded_since is None else degraded_since
                    if elapsed - degraded_since > LIFTOFF_TAU:
                        out.update(reason="lost wheel support",
                                   liftoff_events=[dict(t=degraded_since,
                                                       duration=elapsed-degraded_since)])
                        break
                else:
                    degraded_since = None
                force = scene._robot_shelf_force(model, data, shelf_ids, robot_bodies, cf)
                out["shelf_force_max"] = max(out["shelf_force_max"], force)
                if force > scene.SHELF_FORCE_MAX_N:
                    out["reason"] = "pressed into the shelf"
                    break
                if step % DECIMATE:
                    continue
                error, margin = _measure(model, data, target, site, history, elapsed)
                out.update(ee_err=error, min_fasm_hold=margin)
                if error <= scene.HOLD_POS_TOL_M and margin > 0:
                    hold.append((elapsed, error, margin))
                    if elapsed - hold[0][0] >= scene.HOLD_WINDOW_T:
                        out.update(ok=True, reason=None,
                                   ee_err=max(row[1] for row in hold),
                                   min_fasm_hold=min(row[2] for row in hold))
                        break
                else:
                    hold.clear()
    except Exception as error:
        out.update(ok=False, reason="controller fault", controller_fault=str(error))
    out.update(elapsed_s=elapsed, action_sha256=digest.hexdigest())
    return out


def evaluate_controlled_pick(robot_xml, arm_reference_xml, arm, source):
    front_x, face_x = scene.shelf_placement(robot_xml, arm_reference_xml, arm)
    schedule = scene.scored_target_schedule(face_x, arm)
    required = float(config.PICK_PAYLOAD_KG)
    margin = required + float(config.PICK_EXTRA_PAYLOAD_KG)
    bins, margin_bins = [], []
    for payload, rows in ((required, bins), (margin, margin_bins)):
        for index, name, target in schedule:
            result = run_target(robot_xml, face_x, arm, target, name, payload, source)
            result["target_index"] = index
            rows.append(result)
    # Same input must reproduce the action stream and outcome. Each episode
    # starts a fresh worker, so no state carries between targets or payloads.
    index, name, target = schedule[0]
    repeat = run_target(robot_xml, face_x, arm, target, name, required, source)
    deterministic = all(repeat[key] == bins[0][key] for key in ("action_sha256", "ok", "reason"))
    score = float(np.mean([r["ok"] for r in bins])) if deterministic else 0.0
    margin_score = float(np.mean([r["ok"] for r in margin_bins])) if deterministic else 0.0
    return dict(score=score, margin_score=margin_score, payload_kg=required,
                margin_payload_kg=margin, deterministic=deterministic,
                front_x=front_x, face_x=face_x,
                target_names=[n for _, n, _ in schedule],
                bins=bins, margin_bins=margin_bins)
