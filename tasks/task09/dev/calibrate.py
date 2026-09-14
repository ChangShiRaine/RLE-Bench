"""Measure and derive the task09 scoring bars for one follower.

    RLEBENCH_GELLO_VARIANT=<variant> python -m dev.calibrate [--output FILE]

Prints the entry for `codesign_variant_config.py`; pinning is an explicit
step so calibrating one follower cannot overwrite the others' bars. The
saved measurements live in dev/data/calibration_<variant>.json.
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import mujoco

from harness import spec as gspec
from harness.codesign import CodesignEnvelope, codesign_battery, local_provider
from harness.codesign_oracle import CODESIGN_MODELS, reference_trim_module
from harness.codesign_variant_config import HIDDEN_SEEDS


def measure() -> dict:
    mujoco.set_mju_user_warning(lambda message: None)
    env = CodesignEnvelope()
    module = reference_trim_module()
    model_dir = os.path.join(CODESIGN_MODELS, gspec.VARIANT_NAME)
    oracle_xml = os.path.join(model_dir, "lead_oracle.xml")
    sref_xml = os.path.join(model_dir, "lead_sref.xml")

    reference = codesign_battery(
        oracle_xml,
        HIDDEN_SEEDS,
        env,
        provider=local_provider(oracle_xml, module, HIDDEN_SEEDS, env, adapted=True),
    )
    sref_nominal = codesign_battery(
        sref_xml,
        HIDDEN_SEEDS,
        env,
        provider=local_provider(sref_xml, module, HIDDEN_SEEDS, env, adapted=False),
    )
    sref_adapted = codesign_battery(
        sref_xml,
        HIDDEN_SEEDS,
        env,
        provider=local_provider(sref_xml, module, HIDDEN_SEEDS, env, adapted=True),
    )
    return {"ref": reference, "sref_nom": sref_nominal, "sref_ada": sref_adapted}


def derive(measurements: dict) -> dict:
    reference = measurements["ref"]
    sref_nominal = measurements["sref_nom"]
    sref_adapted = measurements["sref_ada"]
    sref_passive_droop = sref_nominal["passive_droop"]
    ratio_reference = sref_nominal["sw_droop"] / sref_passive_droop
    return {
        "H1_RESIDUAL_BAR": round(1.15 * reference["device_residual"], 4),
        "H2_DROOP_BAR": round(1.15 * reference["passive_droop"], 4),
        "H3_EFFORT_BAR": round(1.10 * reference["passive_effort"], 4),
        "S1_RATIO_BAR": round(1.10 * ratio_reference, 4),
        "S2_DROOP_BAR": round(1.15 * sref_adapted["sw_droop"], 4),
        "S2_EFFORT_BAR": round(1.10 * sref_adapted["sw_effort"], 4),
        "B1_DROOP_BAR": round(1.15 * reference["sw_droop"], 4),
        "B2_EFFORT_BAR": round(1.10 * reference["sw_effort"], 4),
        "B3_HEADROOM_BAR": round(0.85 * reference["headroom"], 4),
        "B4_POKE_BAR": round(1.25 * reference["poke_final"], 4),
        "SREF_PASSIVE_DROOP": round(sref_passive_droop, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="save measurements and derived bars")
    args = parser.parse_args()
    measurements = measure()
    metric_names = (
        "device_residual",
        "passive_droop",
        "passive_effort",
        "sw_droop",
        "sw_effort",
        "headroom",
        "poke_final",
    )
    for name, result in measurements.items():
        values = " ".join(f"{key}={result[key]:.4f}" for key in metric_names)
        print(f"{name:>9}: {values}")

    bars = derive(measurements)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(variant=gspec.VARIANT_NAME,
            measurements=measurements, bars=bars), indent=2) + "\n")
    print(
        f"\nvariant={gspec.VARIANT_NAME!r}; paste these bars into "
        "tasks/task09/harness/codesign_variant_config.py"
    )
    for key, value in bars.items():
        print(f"  {key!r}: {value},")


if __name__ == "__main__":
    main()
