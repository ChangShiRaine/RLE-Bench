"""Freeze RoboTwin episode lists: solvable seeds plus their generated instructions.

    python robotwin_precompute.py --tasks-file /path/task_names.json --config clean \
        --seed-base 300000 --episodes 20 --instructions seen --out episodes.json [--workers 4]

Walks seeds upward from --seed-base for every task, keeps the ones RoboTwin's
expert script solves (its evaluation protocol), and records one instruction of
the requested kind per seed together with the full seen/unseen pools, so that
rollouts later need neither the expert nor the generator. No policy is
involved: rollout_robotwin.py runs in --book-only mode, which stops after the
expert check and the instruction draw.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NANOVLA_ACTION_DIM", "14")  # RoboTwin joint targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--config", choices=["clean", "randomized"], default="clean")
    parser.add_argument("--instructions", choices=["seen", "unseen"], default="seen")
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks_file.read_text())
    work = Path(tempfile.mkdtemp(prefix="rt-precompute-"))
    rows_path = work / "rows.jsonl"
    command = [sys.executable, str(Path(__file__).with_name("rollout_robotwin.py")),
               "--book-only", "--out", str(rows_path), "--resume", str(rows_path) + ".inc",
               "--workers", str(args.workers), "--tasks", ",".join(tasks), "--config", args.config,
               "--instructions", args.instructions, "--seed-base", str(args.seed_base),
               "--episodes", str(args.episodes)]
    subprocess.run(command, check=True, env={**os.environ, "NANOVLA_ACTION_DIM": "14"})
    book: dict[str, list[dict]] = {task: [] for task in tasks}
    for line in rows_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("solvable") and "instruction" in row:
            book[row["task"]].append({"seed": row["seed"], "config": args.config,
                                      "instruction": row["instruction"],
                                      "seen": row.get("seen", []), "unseen": row.get("unseen", [])})
    for task in tasks:
        book[task] = sorted(book[task], key=lambda e: e["seed"])[: args.episodes]
        if len(book[task]) < args.episodes:
            print(f"warning: {task} has only {len(book[task])} solvable seeds", file=sys.stderr)
    args.out.write_text(json.dumps(book, indent=1) + "\n")
    print(f"wrote {args.out}: " + ", ".join(f"{t}={len(v)}" for t, v in book.items()))


if __name__ == "__main__":
    main()
