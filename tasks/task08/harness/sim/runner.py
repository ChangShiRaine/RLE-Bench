"""Deterministic scenario runner: validity checks + full battery over seeds.

This is the harness's own instrumentation: every number the scorer consumes
is computed here, from sim state, never from agent-reported artifacts.
"""
from __future__ import annotations

from .mecanum import verify_holonomic_motion
from .arm_variants import ArmSpec
from rlebench.core.model import (check_inertias_valid, check_inertias_valid_xml,
                    check_structure, settles_to_equilibrium,
                    total_mass_and_com)
from .scenarios import (SCENARIOS, Envelope, compose_scene, run_scenario,
                        scenario_a_min_ssm, set_payload)


def check_validity(robot_xml: str, env: Envelope | None = None,
                   arm: ArmSpec | None = None) -> dict:
    """Stage-1 gate: load, inertias, mass/CoM, equilibrium, mecanum motion.

    Never raises on a bad model — returns ok=False with diagnostics, since an
    unloadable model is an expected verifier input.
    """
    env = env or Envelope()
    out: dict = dict(loads=False)
    try:
        out["inertia_xml"] = check_inertias_valid_xml(robot_xml)
    except Exception as e:  # malformed XML entirely
        out["inertia_xml_error"] = str(e)
        out["inertia_xml"] = None
    try:
        model, data, info = compose_scene(robot_xml, "A", 0, env, arm)
    except Exception as e:
        out["load_error"] = str(e)
        return out
    out["loads"] = True
    set_payload(model, data, 0.0, arm)  # exclude the test load
    out["inertia"] = check_inertias_valid(model)
    out["inertia_ok"] = all(v["ok"] for v in out["inertia"].values())
    # connectivity, measured in the scenario-start pose (before settling)
    out["structure"] = check_structure(
        model, data,
        require_wheel_center=env.require_wheel_center_attachment)
    mass, com = total_mass_and_com(model, data)
    out["total_mass"] = mass
    out["com"] = com.tolist()
    # reset=False: measured from the scenario-start pose compose_scene set,
    # not from a submission-controlled keyframe
    out["settle"] = settles_to_equilibrium(model, data, reset=False)
    # fresh scene for the motion check (settle leaves state behind)
    model, data, _ = compose_scene(robot_xml, "A", 0, env, arm)
    set_payload(model, data, 0.0, arm)
    out["mecanum"] = verify_holonomic_motion(model, data)
    out["ok"] = bool(out["inertia_ok"] and out["structure"]["ok"]
                     and out["settle"]["ok"] and out["mecanum"]["ok"])
    return out


def run_all(robot_xml: str, controller, seeds, env: Envelope | None = None,
            scenarios=None, arm: ArmSpec | None = None) -> dict:
    """Run every scenario over every seed; return raw results for the scorer.

    Returns {"validity": {...}, "scenarios": {sid: {seed: result}},
             "scenario_a_min_ssm": {seed: float}}.
    """
    env = env or Envelope()
    scenarios = scenarios or SCENARIOS
    results: dict = dict(validity=check_validity(robot_xml, env, arm),
                         scenarios={}, scenario_a_min_ssm={})
    if not results["validity"]["loads"]:
        return results  # nothing else is runnable, scorer gates on this
    for sid in scenarios:
        results["scenarios"][sid] = {}
        for seed in seeds:
            res = run_scenario(robot_xml, sid, seed, env, controller, arm)
            results["scenarios"][sid][seed] = res
            if sid == "A":
                results["scenario_a_min_ssm"][seed] = scenario_a_min_ssm(res)
    return results
