"""Roll out a remote policy on LIBERO or LIBERO-plus episodes.

Standard episodes (LIBERO_CONFIG_PATH -> the LIBERO fork):
    python rollout.py --remote-socket S --out rows.jsonl --suite libero_10 --tasks 0-9 \
        --init files --episodes 10            # published init states 0..N-1 per task
    python rollout.py ... --init seeded --seed-base 7000 --episodes 50
                                              # env.seed(base + task*1000 + i); env.reset()
Plus episodes (LIBERO_CONFIG_PATH -> the LIBERO-plus fork, PYTHONPATH first):
    python rollout.py --remote-socket S --out rows.jsonl --plus-manifest M [--per-task 25]

Protocol per episode: 10 settle steps with a no-op action, then the policy's
chunks are executed in full until success or the suite's step cap. Cameras are
rotated 180 degrees and resized to 128x128 like the shards; the state is the raw
eight-dimensional proprioceptive vector; actions are clipped to [-1, 1].
Crash-tolerant: incremental rows, repair passes for missing episodes, watchdog.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

from recording import assemble, designate, recorder

IMG = 128
NUM_WAIT = 10
STALL_S = 600  # no episode finished in this long means a genuinely hung worker
DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
EPISODES_PER_ENV = 15


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-socket", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--render", type=int, default=256)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--init", choices=["files", "seeded"], default="files")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--plus-manifest", default="")
    ap.add_argument("--per-task", type=int, default=0, help="plus entries per base task (0 = all)")
    ap.add_argument("--record", default="", help="media directory for scored-episode videos")
    return ap.parse_args()


def parse_tasks(text: str) -> list[int]:
    tasks: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            tasks.extend(range(int(lo), int(hi) + 1))
        else:
            tasks.append(int(part))
    return sorted(set(tasks))


def episode_key(row: dict) -> tuple:
    if row.get("kind") == "plus":
        return ("plus", row["bddl"], int(row["init_idx"]))
    return ("standard", row["suite"], int(row["task"]), int(row["episode"]))


def worker(wid, args, jobs, results):
    try:
        _worker(wid, args, jobs, results)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        results.put(None)


def _egl_device(wid: int) -> str:
    import torch
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    visible = visible or [str(i) for i in range(torch.cuda.device_count())]
    sim = visible[1:] or visible  # keep the first GPU for the policy when there are several
    if not sim:
        raise RuntimeError("rollouts need at least one visible GPU")
    return sim[wid % len(sim)]


def _worker(wid, args, jobs, results):
    import faulthandler
    faulthandler.enable()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = _egl_device(wid)

    def log(msg):
        print(f"[w{wid}] {msg}", file=sys.stderr, flush=True)

    import numpy as np
    import torch
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from remote_protocol import RemoteClient

    client = RemoteClient(args.remote_socket)

    def pre_img(x):
        x = x[::-1, ::-1]
        return np.asarray(Image.fromarray(x).resize((IMG, IMG), Image.BILINEAR))

    def view(env):
        return env.sim.render(camera_name="agentview", width=640, height=480)[::-1, ::-1]

    def run_episode(env, suite, language, obs, rec=None):
        done, steps = False, 0
        for _ in range(NUM_WAIT):
            obs, _, done, _ = env.step(DUMMY)
        limit = MAX_STEPS[suite]
        while steps < limit and not done:
            cams = np.stack([pre_img(obs["agentview_image"]),
                             pre_img(obs["robot0_eye_in_hand_image"])])
            state = np.concatenate([obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"]]).astype(np.float32)
            chunk = client.predict(language, cams, state)
            for action in np.clip(chunk, -1.0, 1.0):
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                if rec is not None:
                    rec.capture(lambda: view(env))
                if done or steps >= limit:
                    break
        return bool(done), int(steps)

    def make_env(bddl):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.render,
                                 camera_widths=args.render)
        env.seed(7)
        return env

    if args.plus_manifest:
        plus_root = os.environ["NANOVLA_LIBERO_PLUS_ROOT"]
        for entry in iter(jobs.get, None):
            suite = entry["suite"]
            bddl = os.path.join(plus_root, entry["bddl"])
            try:
                env = make_env(bddl)
            except Exception as exc:
                log(f"env FAILED {entry['bddl']}: {type(exc).__name__}")
                results.put({"kind": "plus", **entry, "success": False, "steps": -1,
                             "error": type(exc).__name__})
                continue
            try:
                stem = os.path.basename(entry["bddl"])[: -len(".bddl")]
                if "_add_" in stem or "_level" in stem:
                    # added-object and clutter-level variants carry one init state of their own
                    init = torch.load(os.path.join(plus_root, "init_files", "libero_newobj", suite,
                                                   stem + ".pruned_init"), weights_only=False)
                    init_states, init_idx = np.asarray(init).reshape(1, -1), 0
                else:
                    init = torch.load(os.path.join(plus_root, entry["init"]), weights_only=False)
                    init_states, init_idx = np.asarray(init), int(entry["init_idx"])
                env.reset()
                obs = env.set_init_state(init_states[init_idx])
                rec = recorder(args, ("plus", entry["bddl"], int(entry["init_idx"])), every=2)
                try:
                    success, steps = run_episode(env, suite, entry["lang"], obs, rec)
                finally:
                    if rec is not None:
                        rec.close()
                results.put({"kind": "plus", **entry, "success": success, "steps": steps})
                log(f"ep done {stem[:60]} success={success}")
            except Exception as exc:
                log(f"ep ERROR {entry['bddl'][:60]}: {type(exc).__name__}: {exc}")
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        client.close()
        return

    from libero.libero import benchmark, get_libero_path
    suites = {}
    # Blocks come off a shared queue, so no worker idles while another finishes.
    for suite_name, task_id, episodes in iter(jobs.get, None):
        if suite_name not in suites:
            suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
        task = suites[suite_name].get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = None
        if args.init == "files":
            init_states = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder,
                                                  task.init_states_file), weights_only=False)
        env = make_env(bddl)
        on_env = 0
        for episode in episodes:
            if on_env >= EPISODES_PER_ENV:  # long-lived EGL sessions end in SIGABRT
                env.close()
                env = make_env(bddl)
                on_env = 0
            on_env += 1
            try:
                if init_states is not None:
                    env.reset()
                    obs = env.set_init_state(init_states[episode])
                else:
                    env.seed(args.seed_base + task_id * 1000 + episode)
                    obs = env.reset()
                rec = recorder(args, ("standard", suite_name, task_id, int(episode)), every=2)
                try:
                    success, steps = run_episode(env, suite_name, task.language, obs, rec)
                finally:
                    if rec is not None:
                        rec.close()
            except Exception as exc:
                log(f"ep ERROR {suite_name}/{task_id}#{episode}: {type(exc).__name__}: {exc}")
                continue
            results.put({"kind": "standard", "suite": suite_name, "task": task_id,
                         "episode": int(episode), "success": success, "steps": steps,
                         "lang": task.language})
            log(f"ep done {suite_name}/{task_id}#{episode} success={success} steps={steps}")
        env.close()
    client.close()


def expected_jobs(args):
    """Return (expected keys, job builder) for the requested episode set."""
    if args.plus_manifest:
        manifest = json.load(open(args.plus_manifest))
        if args.per_task:
            per_base = collections.defaultdict(list)
            for entry in manifest:
                per_base[entry["base"]].append(entry)
            manifest = [e for entries in per_base.values() for e in entries[: args.per_task]]
        keys = {("plus", e["bddl"], int(e["init_idx"])): e for e in manifest}

        def build(missing):
            return [keys[k] for k in missing]
        return set(keys), build
    tasks = parse_tasks(args.tasks)
    keys = {("standard", args.suite, t, e) for t in tasks for e in range(args.episodes)}

    def build(missing):
        per_task = collections.defaultdict(list)
        for _, suite, task, episode in sorted(missing):
            per_task[(suite, task)].append(episode)
        chunk = max(5, -(-args.episodes // max(1, args.workers // max(1, len(tasks)))))
        return [(suite, task, eps[i:i + chunk]) for (suite, task), eps in per_task.items()
                for i in range(0, len(eps), chunk)]
    return keys, build


def film_groups(args) -> dict[str, list[tuple]]:
    """Each task's episodes in scoring order, named for the videos."""
    if args.plus_manifest:
        groups = collections.defaultdict(list)
        for entry in json.load(open(args.plus_manifest)):
            groups[entry["base"]].append(("plus", entry["bddl"], int(entry["init_idx"])))
        return groups
    tasks = parse_tasks(args.tasks)
    try:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[args.suite]()
        names = {t: suite.get_task(t).name for t in tasks}
    except Exception:
        names = {t: f"{args.suite}_task{t}" for t in tasks}
    return {names[t]: [("standard", args.suite, t, e) for e in range(args.episodes)] for t in tasks}


def main():
    args = parse_args()
    import queue as pyqueue
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    expected, build = expected_jobs(args)
    args.clips = designate(film_groups(args)) if args.record else {}
    rows: dict[tuple, dict] = {}
    if args.resume and os.path.exists(args.resume):
        for line in open(args.resume):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.setdefault(episode_key(row), row)
        print(f"resumed {len(rows)} episodes", flush=True)
    inc = open(args.out + ".inc", "a")
    t0 = time.time()

    for attempt in range(8):
        missing = expected - set(rows)
        if not missing:
            break
        if attempt:
            print(f"[{time.time() - t0:6.0f}s] repair pass {attempt}: {len(missing)} missing", flush=True)
        jobs = build(missing)
        n_workers = min(args.workers, len(jobs))
        pool = mp.Queue()
        for job in jobs:
            pool.put(job)
        for _ in range(n_workers):
            pool.put(None)
        results = mp.Queue()
        procs = [mp.Process(target=worker, args=(i, args, pool, results), daemon=False)
                 for i in range(n_workers)]
        for p in procs:
            p.start()
        finished, last = 0, time.time()
        while finished < len(procs):
            try:
                row = results.get(timeout=20)
            except pyqueue.Empty:
                alive = sum(p.is_alive() for p in procs)
                if alive == 0:
                    break
                if time.time() - last > STALL_S:
                    print(f"[watchdog] no results for {STALL_S}s; terminating stragglers", flush=True)
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    break
                continue
            last = time.time()
            if row is None:
                finished += 1
                continue
            key = episode_key(row)
            if key in expected and key not in rows:
                rows[key] = row
                inc.write(json.dumps(row) + "\n")
                inc.flush()
            if len(rows) % 25 == 0:
                sr = sum(r["success"] for r in rows.values()) / len(rows)
                print(f"[{time.time() - t0:6.0f}s] {len(rows)}/{len(expected)} | SR {sr:.3f}", flush=True)
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    with open(args.out, "w") as stream:
        for row in rows.values():
            stream.write(json.dumps(row) + "\n")
    if args.clips:
        assemble(args.record, args.clips, {json.dumps(list(k)): r["success"] for k, r in rows.items()})
    by_group = collections.defaultdict(list)
    for row in rows.values():
        group = row["base"] if row["kind"] == "plus" else f"task {row['task']}"
        by_group[group].append(row["success"])
    for group, values in sorted(by_group.items()):
        print(f"{group}: {sum(values) / len(values):.3f} (n={len(values)})")
    done = [r["success"] for r in rows.values()]
    print(f"OVERALL: {sum(done) / max(1, len(done)):.4f} (n={len(done)}/{len(expected)}) | "
          f"wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
