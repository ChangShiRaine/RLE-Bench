#!/usr/bin/env python3
"""Build the RoboTwin shards from self-collected RoboTwin 2.0 clean demonstrations.

    python3 tasks/task05/dev/prepare_robotwin.py --demos /path/RoboTwin/data --out /path/shards_rt15_128

Input: <demos>/<task>/clean50/{data/episode*.hdf5, instructions/episode*.json}.
Cameras: head -> agent.u8, front -> wrist.u8 (JPEG bytes decoded, channel order
restored, resized to 128x128). State = [left_endpose(7), left_gripper,
right_endpose(7), right_gripper] (16). Action = joint_action/vector (14: left arm
6, left gripper, right arm 6, right gripper). Every episode keeps its full list
of "seen" instruction paraphrases in instructions.json; the "unseen" lists stay
out of the shards.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

SIZE = 128
STATE_DIM, ACT_DIM = 16, 14
EXCLUDE = {"move_stapler_pad"}  # 9 demos and a SAPIEN hang at evaluation
CONFIG = "clean50"


def episodes(demos: Path) -> list[tuple[str, Path, Path]]:
    found = []
    for task_dir in sorted(demos.iterdir()):
        if task_dir.name in EXCLUDE or not (task_dir / CONFIG / "data").is_dir():
            continue
        for h5 in sorted((task_dir / CONFIG / "data").glob("episode*.hdf5")):
            ins = task_dir / CONFIG / "instructions" / (h5.stem + ".json")
            if ins.is_file():
                found.append((task_dir.name, h5, ins))
    return found


def length(job):
    with h5py.File(job[1], "r") as f:
        return int(f["joint_action/vector"].shape[0])


def decode(blob: bytes) -> np.ndarray:
    # RoboTwin encodes frames through cv2 (BGR), so the stored JPEG is R/B-swapped.
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def convert(job):
    entries, out, n_total = job
    agent = np.memmap(out / "agent.u8", np.uint8, "r+", shape=(n_total, SIZE, SIZE, 3))
    wrist = np.memmap(out / "wrist.u8", np.uint8, "r+", shape=(n_total, SIZE, SIZE, 3))
    state = np.memmap(out / "state.f32", np.float32, "r+", shape=(n_total, STATE_DIM))
    actions = np.memmap(out / "actions.f32", np.float32, "r+", shape=(n_total, ACT_DIM))
    ep_end = np.memmap(out / "ep_end.i32", np.int32, "r+", shape=(n_total,))
    task_idx = np.memmap(out / "task_idx.u8", np.uint8, "r+", shape=(n_total,))
    episode_id = np.memmap(out / "episode_id.i32", np.int32, "r+", shape=(n_total,))
    for ep, task, h5, off in entries:
        with h5py.File(h5, "r") as f:
            n = int(f["joint_action/vector"].shape[0])
            heads = f["observation/head_camera/rgb"][:]
            fronts = f["observation/front_camera/rgb"][:]
            for i in range(n):
                agent[off + i] = decode(heads[i])
                wrist[off + i] = decode(fronts[i])
            state[off:off + n] = np.concatenate([
                f["endpose/left_endpose"][:], f["endpose/left_gripper"][:][:, None],
                f["endpose/right_endpose"][:], f["endpose/right_gripper"][:][:, None]], axis=1)
            actions[off:off + n] = f["joint_action/vector"][:]
            ep_end[off:off + n] = off + n
            task_idx[off:off + n] = task
            episode_id[off:off + n] = ep
    for m in (agent, wrist, state, actions, ep_end, task_idx, episode_id):
        m.flush()
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    t0 = time.time()
    found = episodes(args.demos)
    tasks = sorted({task for task, _, _ in found})
    print(f"{len(found)} episodes across {len(tasks)} tasks")
    with ProcessPoolExecutor(args.workers) as pool:
        lengths = list(pool.map(length, found))
    args.out.mkdir(parents=True, exist_ok=True)
    entries, instructions, off = [], [], 0
    for ep, ((task, h5, ins), n) in enumerate(zip(found, lengths)):
        entries.append((ep, tasks.index(task), h5, off))
        instructions.append(json.loads(ins.read_text())["seen"])
        off += n
    n_total = off
    for name, shape, dtype in (("agent.u8", (n_total, SIZE, SIZE, 3), np.uint8),
                               ("wrist.u8", (n_total, SIZE, SIZE, 3), np.uint8),
                               ("state.f32", (n_total, STATE_DIM), np.float32),
                               ("actions.f32", (n_total, ACT_DIM), np.float32),
                               ("ep_end.i32", (n_total,), np.int32),
                               ("task_idx.u8", (n_total,), np.uint8),
                               ("episode_id.i32", (n_total,), np.int32)):
        np.memmap(args.out / name, dtype, "w+", shape=shape).flush()
    jobs = [(entries[w::args.workers], args.out, n_total) for w in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        done = sum(pool.map(convert, jobs))
    print(f"converted {done} episodes, {n_total} frames ({time.time() - t0:.0f}s)")

    state = np.fromfile(args.out / "state.f32", np.float32).reshape(n_total, STATE_DIM)
    actions = np.fromfile(args.out / "actions.f32", np.float32).reshape(n_total, ACT_DIM)

    def stats(x: np.ndarray) -> dict:
        return {"q01": np.percentile(x, 1, axis=0).tolist(), "q99": np.percentile(x, 99, axis=0).tolist(),
                "min": x.min(0).tolist(), "max": x.max(0).tolist()}

    (args.out / "norm_stats.json").write_text(json.dumps({"actions": stats(actions), "state": stats(state)}, indent=2) + "\n")
    (args.out / "task_strs.json").write_text(json.dumps([t.replace("_", " ") for t in tasks], indent=2) + "\n")
    (args.out / "task_names.json").write_text(json.dumps(tasks, indent=2) + "\n")
    (args.out / "instructions.json").write_text(json.dumps(instructions) + "\n")
    per_task = {t: sum(1 for task, _, _ in found if task == t) for t in tasks}
    (args.out / "meta.json").write_text(json.dumps({
        "n_frames": n_total, "n_episodes": len(found), "img_size": SIZE, "state_dim": STATE_DIM,
        "act_dim": ACT_DIM, "n_tasks": len(tasks), "suite": "robotwin", "config": CONFIG,
        "cameras": ["head_camera", "front_camera"], "episodes_per_task": per_task,
        "source": "RoboTwin 2.0 aloha-agilex clean demonstrations"}, indent=2) + "\n")
    print(json.dumps(per_task, indent=1))


if __name__ == "__main__":
    main()
