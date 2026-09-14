"""Validate or calibrate the release reference across all arms and both seed sets.

By default, check the installed thresholds without changing files. ``--output``
validates and emits a candidate; ``--write`` explicitly installs it. Shelf control
is validated separately through the full Harbor Oracle in the root verifier.

Derivation rules:
  SSM_THRESHOLD_M = 0.6 * worst static margin; SSM_SAT_M = worst static margin
  FASM_SAT = 0.75 * worst B/E margin
  D2 caps = worst count + 1, worst duration * 1.5
  TRACKING_RMS_TOL = 1.5 * worst B/C/D1 error
  S5_TRACKING_GATE = 2 * worst dynamic tracking error
  REACH_BEYOND_MIN_M = 0.6 * worst arm's footprint overhang
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[1]
ARM_REFERENCE = REPO / "assets/robots/franka_emika_panda/panda_nohand.xml"
THRESHOLDS_PATH = TASK / "harness/thresholds.py"
THRESHOLD_NAMES = (
    "REACH_BEYOND_MIN_M", "SSM_THRESHOLD_M", "SSM_SAT_M", "FASM_SAT",
    "D2_MAX_LIFTOFF_EVENTS", "D2_MAX_LIFTOFF_DURATION_S", "TRACKING_RMS_TOL",
    "S5_TRACKING_GATE",
)
HEADER = '"""Calibrated thresholds; emitted by ``python -m dev.calibrate --write``."""\n\n'


def measure_golden() -> dict:
    from build_assets import ensure_reference_robot
    from harness import config
    from harness.base_design.checkpoints import analyze_model
    from harness.base_design.golden_controller import make_base_controller
    from harness.sim.arm_variants import trusted_arm_specs
    from harness.sim.runner import run_all
    from harness.sim.scenarios import Envelope

    ensure_reference_robot()
    robot = str(TASK / "reference/robot.xml")
    seeds = tuple(config.HIDDEN_SEEDS) + tuple(config.DESIGN_SEEDS)
    measurements = {}
    for name, arm in trusted_arm_specs(str(ARM_REFERENCE)).items():
        print(f"measuring {name}, seeds {seeds}", flush=True)
        info = analyze_model(robot, arm)
        battery = run_all(robot, controller=make_base_controller,
                          seeds=seeds, env=Envelope(), arm=arm)
        # Retain the verifier's scoring inputs, without unrelated render traces.
        keys = ("scenario", "tip", "liftoff_events", "min_fasm", "min_fasm_robust",
                "max_tilt_deg", "twist_cmd", "twist_meas")
        battery["scenarios"] = {
            sid: {seed: {key: result[key] for key in keys}
                  for seed, result in runs.items()}
            for sid, runs in battery["scenarios"].items()}
        measurements[name] = {"model_info": info, "battery": battery}
    return measurements


def validate_values(values: dict) -> None:
    if set(values) != set(THRESHOLD_NAMES):
        raise ValueError("candidate must contain exactly the calibrated thresholds")
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid threshold {name}: {value}")
        if name != "D2_MAX_LIFTOFF_DURATION_S" and value == 0:
            raise ValueError(f"threshold must be positive: {name}")
    if type(values["D2_MAX_LIFTOFF_EVENTS"]) is not int:
        raise ValueError("D2 event cap must be an integer")
    if values["SSM_THRESHOLD_M"] >= values["SSM_SAT_M"]:
        raise ValueError("static saturation must exceed the static threshold")
    if values["S5_TRACKING_GATE"] >= 1.0:
        raise ValueError("tracking gate must reject a stationary controller")
    if values["TRACKING_RMS_TOL"] > values["S5_TRACKING_GATE"]:
        raise ValueError("tracking gate must cover the tracking tolerance")


def derive(measurements: dict) -> dict:
    from harness.base_design.checkpoints import tracking_error

    margins, reach, fasm, events, durations, tracking, all_tracking = [], [], [], [], [0.0], [], []
    for record in measurements.values():
        reach.append(record["model_info"]["reach_beyond"])
        battery = record["battery"]
        margins.extend(battery["scenario_a_min_ssm"].values())
        for sid, runs in battery["scenarios"].items():
            for result in runs.values():
                if result["tip"]:
                    raise ValueError(f"reference tipped in {sid}")
                if sid in ("B", "E"):
                    fasm.append(result["min_fasm_robust"])
                if sid == "D2":
                    events.append(len(result["liftoff_events"]))
                    durations.extend(e["duration"] for e in result["liftoff_events"])
                if sid in ("B", "C", "D1"):
                    tracking.append(tracking_error(result))
                if sid != "A":
                    all_tracking.append(tracking_error(result))
    for samples in (margins, reach, fasm, durations, tracking, all_tracking):
        if not samples or not all(math.isfinite(x) for x in samples):
            raise ValueError("missing or non-finite calibration measurements")
    values = dict(
        REACH_BEYOND_MIN_M=round(0.6 * min(reach), 2),
        SSM_THRESHOLD_M=round(0.6 * min(margins), 3),
        SSM_SAT_M=round(min(margins), 3),
        FASM_SAT=round(0.75 * min(fasm), 1),
        D2_MAX_LIFTOFF_EVENTS=max(events) + 1,
        D2_MAX_LIFTOFF_DURATION_S=round(1.5 * max(durations), 1),
        TRACKING_RMS_TOL=round(1.5 * max(tracking), 3),
        S5_TRACKING_GATE=round(2.0 * max(all_tracking), 3),
    )
    validate_values(values)
    return values


def validate_measurements(measurements: dict) -> None:
    from harness import config
    from harness.base_design.checkpoints import CHECKPOINTS, evaluate, score_scenario
    from harness.sim.scenarios import SCENARIOS

    seeds = {str(seed) for seed in (*config.HIDDEN_SEEDS, *config.DESIGN_SEEDS)}
    if set(measurements) != {"panda", "ur5e", "xarm7"}:
        raise ValueError("calibration requires all three trusted arms")
    for name, record in measurements.items():
        battery, info = record["battery"], record["model_info"]
        if not battery["validity"].get("ok"):
            raise ValueError(f"{name}: reference fails validity")
        if not info.get("ok") or not all(info[key]["ok"] for key in ("battery", "arm", "actuators")):
            raise ValueError(f"{name}: reference fails model checks")
        if (set(battery["scenarios"]) != set(SCENARIOS)
                or set(map(str, battery["scenario_a_min_ssm"])) != seeds):
            raise ValueError(f"{name}: incomplete battery")
        for sid, runs in battery["scenarios"].items():
            if set(map(str, runs)) != seeds:
                raise ValueError(f"{name}/{sid}: incomplete seed coverage")
            for seed, result in runs.items():
                if result["scenario"] != sid or result["tip"] or (sid != "A" and score_scenario(result) < 0.99):
                    raise ValueError(f"{name}/{sid}/seed{seed}: reference fails scenario")
        scores = evaluate(battery, info)
        excluded = {"S3.design_efficiency", "S7.pick_compat", "S7.payload_margin"}
        weak = {cp.id: scores[cp.id] for cp in CHECKPOINTS
                if cp.id not in excluded and (not math.isfinite(scores[cp.id]) or scores[cp.id] < 0.99)}
        if weak:
            raise ValueError(f"{name}: reference below full credit: {weak}")


def _json_default(value):
    return value.tolist()


def validate_candidate(candidate: Path, measurements: dict) -> None:
    """Use a fresh interpreter so imported constants cannot retain old values."""
    with tempfile.TemporaryDirectory(prefix="task08-calibration-") as work:
        record = Path(work) / "measurements.json"
        record.write_text(json.dumps(measurements, default=_json_default, allow_nan=False))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(TASK), str(REPO))))
        subprocess.run(
            [sys.executable, "-m", "dev.calibrate", "--validate-candidate", str(candidate),
             "--measurements", str(record)], env=env, cwd=REPO, check=True)


def publish(values: dict, measurements: dict, destination: Path | None = None) -> None:
    """Validate first; any failure leaves the installed and output files intact."""
    validate_values(values)
    parent = destination.parent if destination is not None else None
    fd, path = tempfile.mkstemp(suffix=".py", prefix=".task08-thresholds-", dir=parent)
    candidate = Path(path)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(HEADER + "".join(f"{key} = {values[key]!r}\n" for key in THRESHOLD_NAMES))
            stream.flush()
            os.fsync(stream.fileno())
        validate_candidate(candidate, measurements)
        if destination is not None:
            candidate.chmod(destination.stat().st_mode & 0o777 if destination.exists() else 0o644)
            os.replace(candidate, destination)
    finally:
        candidate.unlink(missing_ok=True)


def _validate_worker(candidate: Path, record: Path) -> None:
    # Only this dev subprocess substitutes thresholds; the shipped harness has
    # no configuration override through environment variables or submissions.
    tree = ast.parse(candidate.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = ast.literal_eval(node.value)
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str)):
            raise ValueError("unexpected candidate content")
    validate_values(values)
    module = types.ModuleType("harness.thresholds")
    vars(module).update(values)
    sys.modules[module.__name__] = module
    validate_measurements(json.loads(record.read_text()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--check", action="store_true", help="check installed thresholds (default)")
    output.add_argument("--output", type=Path, help="write a validated candidate to this file")
    output.add_argument("--write", action="store_true", help="replace installed thresholds after validation")
    parser.add_argument("--validate-candidate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--measurements", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if bool(args.validate_candidate) != bool(args.measurements):
        parser.error("candidate validation requires a measurements file")
    try:
        if args.validate_candidate:
            _validate_worker(args.validate_candidate, args.measurements)
            return 0
        from harness import config

        destination = THRESHOLDS_PATH if args.write else args.output
        if destination is not None:
            destination = destination.resolve()
            if not args.write and destination in (THRESHOLDS_PATH.resolve(), (TASK / "harness/config.py").resolve()):
                raise ValueError("use --write to replace installed thresholds; config.py is not generated")
        measurements = measure_golden()
        values = (derive(measurements) if args.output or args.write
                  else {key: getattr(config, key) for key in THRESHOLD_NAMES})
        publish(values, measurements, destination)
        print(json.dumps(values, indent=2))
        print(f"validated {destination or 'installed thresholds'}; shelf control requires the Harbor Oracle")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"calibration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
