"""Pin task11's force limits and reference anchors (never shipped).

Runs the privileged golden packer and the public baseline on the hidden
evaluation seeds, sets the crush and tote-strike limits 1.3x above the
largest sustained force either reference exerts (with floors), and records
both reference scores in tasks/task11/harness/config.py.

    make calibrate-task11      (PYTHONPATH=tasks/task11 python -m dev.calibrate)
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "harness" / "config.py"
MARGIN = 1.3
DAMAGE_FLOOR_N, BIN_HIT_FLOOR_N = 250.0, 150.0


def _episode(args):
    which, seed = args
    from harness import runtime, scene, scorer
    if which == "golden":
        from harness.golden import GoldenPacker
        log = runtime.run_episode(GoldenPacker(), seed, render=False)
    else:
        from harness.baseline import Packer
        cell = runtime.default_cell_spec()
        cell["model_file"] = str(Path(tempfile.mkdtemp()) / "cell.mjb")
        scene.save_cell_mjb(cell["model_file"])
        log = runtime.run_episode(Packer(), seed, render=True, cell_spec=cell)
    m = log.metrics()
    m.pop("penalty_events")
    m["episode_score"] = scorer.episode_score(m)
    return which, seed, m


def _pin(text: str, name: str, value) -> str:
    return re.sub(rf"^{name} = .*$", f"{name} = {value!r}", text,
                  count=1, flags=re.M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from harness import scorer
    _, seeds = scorer._load_seeds()
    jobs = [(w, s) for w in ("golden", "baseline") for s in seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(_episode, jobs))

    by = {"golden": [], "baseline": []}
    for which, seed, m in results:
        by[which].append(m)
        print(which, seed, json.dumps({k: m[k] for k in (
            "episode_score", "n_packed", "utilization", "throughput_ppm",
            "bin_full", "pick_place_success_rate", "floor_drops",
            "damage_events", "bin_hits", "peak_sustained_force",
            "peak_tool_bin_force")}, default=str))
    peak = max(m["peak_sustained_force"] for m in by["golden"] + by["baseline"])
    tool = max(m["peak_tool_bin_force"] for m in by["golden"] + by["baseline"])
    ref = {w: round(sum(m["episode_score"] for m in ms) / len(ms), 4)
           for w, ms in by.items()}
    pins = {"DAMAGE_FORCE_N": max(DAMAGE_FLOOR_N, round(MARGIN * peak, -1)),
            "BIN_HIT_FORCE_N": max(BIN_HIT_FLOOR_N, round(MARGIN * tool, -1)),
            "GOLDEN_REF_SCORE": ref["golden"],
            "BASELINE_REF_SCORE": ref["baseline"]}
    print(json.dumps(pins, indent=2))
    if not args.dry_run:
        text = CONFIG.read_text()
        for k, v in pins.items():
            text = _pin(text, k, v)
        CONFIG.write_text(text)


if __name__ == "__main__":
    main()
