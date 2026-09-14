"""Score every ablation tier through the task09 scorer.

The full oracle must exceed each ablation on the weighted performance
stages, with the failure signature naming the missing ingredient.

Usage: python tasks/task09/dev/ablations/run_ablations.py [--seeds 311]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

FAMILY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, FAMILY)

from harness import codesign_variant_config as ccfg    # noqa: E402
from harness import spec as gspec                        # noqa: E402
from harness.codesign_oracle import (CODESIGN_MODELS,    # noqa: E402
                                     build_reference_submission)
from harness.codesign_scorer import score_submission     # noqa: E402

ORACLE = os.path.join(CODESIGN_MODELS, gspec.VARIANT_NAME, "lead_oracle.xml")
BARE = os.path.join(CODESIGN_MODELS, gspec.VARIANT_NAME, "lead_bare.xml")
SREF = os.path.join(CODESIGN_MODELS, gspec.VARIANT_NAME, "lead_sref.xml")


def make_clamp_xml(outpath: str, scale: float = 8.0) -> str:
    tree = ET.parse(ORACLE)
    for j in tree.getroot().iter("joint"):
        if j.get("stiffness"):
            j.set("stiffness", str(float(j.get("stiffness")) * scale))
    tree.write(outpath)
    return outpath


def strip_adapt(trim_path: str) -> None:
    """Keep make_trim, remove the adaptation entry points."""
    src = open(trim_path).read()
    src = src.replace("def adapt(", "def _disabled_adapt(")
    src = src.replace("def make_trim_adapted(", "def _disabled_make_trim_adapted(")
    with open(trim_path, "w") as f:
        f.write(src)


def build_ablations(root: str) -> dict:
    subs = {}
    subs["full_reference"] = build_reference_submission(
        os.path.join(root, "full"), ORACLE)
    d = build_reference_submission(os.path.join(root, "passive_only"), ORACLE)
    os.remove(os.path.join(d, "trim.py"))
    subs["passive_only"] = d
    subs["active_only"] = build_reference_submission(
        os.path.join(root, "active_only"), BARE)
    d = build_reference_submission(os.path.join(root, "no_adapt"), ORACLE)
    strip_adapt(os.path.join(d, "trim.py"))
    subs["no_adapt"] = d
    subs["lazy_hardware"] = build_reference_submission(
        os.path.join(root, "lazy_hardware"), SREF)
    clamp = make_clamp_xml(os.path.join(root, "_clamp.xml"))
    subs["clamp_x8"] = build_reference_submission(
        os.path.join(root, "clamp"), clamp)
    return subs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override hidden seeds (faster single-seed run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.seeds:
        ccfg.HIDDEN_SEEDS = tuple(args.seeds)

    root = args.out or tempfile.mkdtemp(prefix="task09_ablations_")
    subs = build_ablations(root)
    results = {}
    for name, subdir in subs.items():
        r = score_submission(subdir)
        results[name] = r
        print(f"{name:<16} reward={r['reward']:.4f} merit={r['merit']:.4f} "
              f"gated={r['gated']} stages={r['stages']}")
    with open(os.path.join(root, "ablation_report.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk in ("reward", "merit", "gated", "stages",
                                 "checkpoints", "battery", "s_tier")}
                   for k, v in results.items()}, f, indent=2)
    print(f"\nreport: {os.path.join(root, 'ablation_report.json')}")


if __name__ == "__main__":
    main()
