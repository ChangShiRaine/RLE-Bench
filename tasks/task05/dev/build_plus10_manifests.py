#!/usr/bin/env python3
"""Sample the 02 verifier manifest from the LIBERO-plus libero_10 variants.

    python3 tasks/task05/dev/build_plus10_manifests.py --plus /path/LIBERO-plus/libero/libero \
        --shards /path/shards_l10_128 --per-task 50

Every base task contributes `per-task` variants drawn uniformly from its bddl
files (paraphrased instructions, lighting, table textures and backgrounds,
added objects, layout levels), each paired with a rotating init-state index.
The manifest carries the fields the evaluator reads: suite, base, bddl, init,
init_idx, lang, perturb, task (the shards' task index).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

SUITE = "libero_10"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plus", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True,
                        help="libero_10 shards; their task order numbers the manifest's base tasks")
    parser.add_argument("--per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path,
                        default=Path("tasks/task05/subtasks/02-libero-robustness/tests/plus10_manifest.json"))
    args = parser.parse_args()
    plus = args.plus
    rng = random.Random(args.seed)
    task_strs = json.loads((args.shards / "task_strs.json").read_text())

    inits = sorted(glob.glob(f"{plus}/init_files/{SUITE}/*.pruned_init"))
    bases = sorted((os.path.basename(p)[: -len(".pruned_init")] for p in inits), key=len, reverse=True)
    by_base: dict[str, list[str]] = {b: [] for b in bases}
    for path in sorted(glob.glob(f"{plus}/bddl_files/{SUITE}/*.bddl")):
        stem = os.path.basename(path)[: -len(".bddl")]
        if " copy" in stem:
            continue
        base = next((b for b in bases if stem == b or stem.startswith(b + "_")), None)
        if base is not None:
            by_base[base].append(path)
    manifest = []
    for base in sorted(by_base):
        lang_base = re.search(r"\(:language\s+([^)]*)\)", (plus / "bddl_files" / SUITE / f"{base}.bddl").read_text())
        task = task_strs.index(lang_base.group(1).strip())
        for i, path in enumerate(rng.sample(by_base[base], args.per_task)):
            m = re.search(r"\(:language\s+([^)]*)\)", open(path).read())
            stem = os.path.basename(path)[: -len(".bddl")]
            manifest.append({
                "suite": SUITE, "base": base, "bddl": os.path.relpath(path, plus),
                "init": f"init_files/{SUITE}/{base}.pruned_init",
                "init_idx": i % 10, "lang": m.group(1).strip() if m else base.replace("_", " "),
                "perturb": stem[len(base):].strip("_") or "base", "task": task,
            })
    for entry in manifest:
        for key in ("bddl", "init"):
            if not (plus / entry[key]).is_file():
                raise SystemExit(f"missing {plus / entry[key]}")
    if {e["task"] for e in manifest} != set(range(len(task_strs))):
        raise SystemExit("manifest does not cover every task")
    args.out.write_text(json.dumps(manifest, indent=0) + "\n")
    kinds = Counter(re.sub(r"_?\d+$", "", e["perturb"]) for e in manifest)
    print(f"{len(manifest)} entries -> {args.out}; {dict(kinds.most_common())}")


if __name__ == "__main__":
    main()
