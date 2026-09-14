#!/usr/bin/env python3
"""Cut the libero_10 subset out of the LIBERO 4-suite shards.

    python3 tasks/task05/dev/prepare_libero10.py --shards /path/shards_128 --out /path/shards_l10_128

The 4-suite shards store libero_10 as tasks 0-9 in the leading frames, so the
subset is a prefix; episode ends stay valid. Normalisation statistics, the task
list, the vocabulary and the token table are rebuilt from the subset alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = 10
IMG = 128


def tokenize(text: str, vocab: dict[str, int], text_len: int) -> list[int]:
    words = "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()
    ids = [vocab.get(w, 1) for w in words][:text_len]
    return ids + [0] * (text_len - len(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    src, out = args.shards, args.out

    meta = json.loads((src / "meta.json").read_text())
    task_idx = np.fromfile(src / "task_idx.u8", np.uint8)
    keep = np.flatnonzero(task_idx < TASKS)
    n = int(keep.size)
    if keep[0] != 0 or keep[-1] != n - 1:
        raise SystemExit("libero_10 frames are not a prefix of the shards")
    ep_end = np.fromfile(src / "ep_end.i32", np.int32)[:n]
    if ep_end.max() != n:
        raise SystemExit("episode ends do not close on the subset boundary")

    out.mkdir(parents=True, exist_ok=True)
    for name, width in (("agent.u8", IMG * IMG * 3), ("wrist.u8", IMG * IMG * 3)):
        with (src / name).open("rb") as fin, (out / name).open("wb") as fout:
            remaining = n * width
            while remaining:
                block = fin.read(min(remaining, 64 << 20))
                fout.write(block)
                remaining -= len(block)
    state = np.fromfile(src / "state.f32", np.float32).reshape(-1, meta["state_dim"])[:n]
    actions = np.fromfile(src / "actions.f32", np.float32).reshape(-1, meta["act_dim"])[:n]
    state.tofile(out / "state.f32")
    actions.tofile(out / "actions.f32")
    ep_end.tofile(out / "ep_end.i32")
    task_idx[:n].tofile(out / "task_idx.u8")

    def stats(x: np.ndarray) -> dict[str, list[float]]:
        return {"q01": np.percentile(x, 1, axis=0).tolist(),
                "q99": np.percentile(x, 99, axis=0).tolist()}

    (out / "norm_stats.json").write_text(json.dumps(
        {"actions": stats(actions), "state": stats(state)}, indent=2) + "\n")
    task_strs = json.loads((src / "task_strs.json").read_text())[:TASKS]
    (out / "task_strs.json").write_text(json.dumps(task_strs, indent=2) + "\n")
    words = sorted({w for t in task_strs for w in
                    "".join(ch if ch.isalnum() else " " for ch in t.lower()).split()})
    vocab = {w: i + 2 for i, w in enumerate(words)}
    text_len = len(json.loads((src / "task_tokens.json").read_text())[0])
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    (out / "task_tokens.json").write_text(json.dumps(
        [tokenize(t, vocab, text_len) for t in task_strs]) + "\n")
    (out / "meta.json").write_text(json.dumps({
        **meta, "n_frames": n, "n_episodes": int(np.unique(ep_end).size),
        "n_tasks": TASKS, "suite": "libero_10"}, indent=2) + "\n")
    print(f"{n} frames, {np.unique(ep_end).size} episodes, {TASKS} tasks -> {out}")


if __name__ == "__main__":
    main()
