"""Roll out a remote policy on RoboTwin 2.0 episodes (SAPIEN, aloha-agilex).

Fixed episode list (verifier and development service):
    python rollout_robotwin.py --remote-socket S --out rows.jsonl --episodes-file episodes.json \
        --tasks adjust_bottle,lift_pot --per-task 20
    episodes.json: {"<task>": [{"seed": int, "config": "clean"|"randomized",
                                "instruction": str}, ...]}
Official-protocol sweep (robotwin_precompute.py uses it to build those lists):
    python rollout_robotwin.py --remote-socket S --out rows.jsonl --tasks adjust_bottle \
        --config clean --instructions seen --seed-base 300000 --episodes 20
    Seeds are walked upward from --seed-base; a seed counts only if RoboTwin's
    expert script solves it (RoboTwin's own protocol), and the instruction is
    drawn from the episode's generated "seen" or "unseen" descriptions.

Protocol per episode: the policy receives the instruction, the head and front
cameras resized to 128x128 (like the shards) and the raw 16-dim endpose
state, and returns a chunk of 14-dim joint targets executed in order until
success or the task's step limit. Crash-tolerant: incremental rows, repair
passes, stall watchdog.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

from recording import assemble, designate, recorder

ROBOTWIN = os.environ.get("NANOVLA_ROBOTWIN_ROOT", "/opt/robotwin")
# RoboTwin's own task configs: clean50 (no randomisation) and rand50 (background,
# clutter, light, table height). In eval mode RoboTwin also draws its background
# textures from the held-out "unseen" pool, so the randomised setting is unseen twice over.
CONFIG_FILES = {"clean": "clean50", "randomized": "rand50"}
IMG = 128
EPISODES_PER_ENV = 5  # SAPIEN render cache is cleared this often (RoboTwin's clear_cache_freq)
BLOCK = 5  # episodes per (task, worker) block; blocks are interleaved across tasks
STALL_S = 3600  # no episode finished in this long means a genuinely hung worker
STATE_KEYS = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-socket", default="")
    ap.add_argument("--book-only", action="store_true",
                    help="sweep mode: record solvable seeds and instructions without rolling a policy out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=0, help="ignored: SAPIEN uses CUDA_VISIBLE_DEVICES")
    ap.add_argument("--tasks", required=True, help="comma list of RoboTwin task names")
    ap.add_argument("--episodes-file", default="", help="fixed per-task episode lists")
    ap.add_argument("--per-task", type=int, default=0, help="episodes per task from the file (0 = all)")
    ap.add_argument("--config", choices=["clean", "randomized"], default="clean")
    ap.add_argument("--instructions", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=0, help="sweep mode: solvable seeds per task")
    ap.add_argument("--record", default="", help="media directory for scored-episode videos")
    args = ap.parse_args()
    if not args.book_only and not args.remote_socket:
        raise SystemExit("--remote-socket is required unless --book-only is set")
    return args


def episode_key(row: dict) -> tuple:
    return (row["task"], int(row["seed"]))


# ------------------------------------------------------------------ RoboTwin glue

def task_args(config: str, task: str) -> dict:
    """RoboTwin's setup_demo kwargs, assembled like script/eval_policy.py."""
    import yaml
    with open(os.path.join(ROBOTWIN, "task_config", f"{CONFIG_FILES[config]}.yml")) as f:
        args = yaml.safe_load(f)
    with open(os.path.join(ROBOTWIN, "task_config", "_embodiment_config.yml")) as f:
        embodiments = yaml.safe_load(f)
    with open(os.path.join(ROBOTWIN, "task_config", "_camera_config.yml")) as f:
        cameras = yaml.safe_load(f)
    args["task_name"] = task
    args["task_config"] = config
    args["ckpt_setting"] = "nanovla"
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = cameras[head]["h"], cameras[head]["w"]
    kinds = args["embodiment"]
    robot = embodiments[kinds[0]]["file_path"]
    with open(os.path.join(robot, "config.yml")) as f:
        embodiment = yaml.safe_load(f)
    args["left_robot_file"] = args["right_robot_file"] = robot
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = args["right_embodiment_config"] = embodiment
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    args["save_data"] = False
    args["collect_data"] = False
    args["policy_name"] = "nanovla"
    return args


def make_env(task: str):
    import importlib
    module = importlib.import_module(f"envs.{task}")
    return getattr(module, task)()


def expert_state(env, task: str) -> None:
    """Set what the scripted expert would have decided before acting.

    A few RoboTwin tasks record the arm the expert picked and read it back in
    check_success. Policy rollouts never run the expert, so the value is derived
    here from the same scene quantity, which setup_demo has already fixed for the
    seed. The generated instruction names that arm, so requiring it is the task.
    """
    if task == "open_laptop":
        from envs.utils import ArmTag, get_face_prod
        face_prod = get_face_prod(env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        env.arm_tag = ArmTag("left" if face_prod > 0 else "right")


def expert_solves(env, task: str, seed: int, args: dict, log=None, clear_cache: bool = False):
    """RoboTwin's feasibility check: the scripted expert must solve the seed. Returns episode info or None.

    An unstable scene or an unsolvable plan is an ordinary rejection; anything
    else is reported, so a broken environment cannot masquerade as a run of
    unsolvable seeds.
    """
    from envs.utils.create_actor import UnStableError
    try:
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        info = env.play_once()
        ok = bool(env.plan_success and env.check_success())
    except UnStableError:
        ok, info = False, None
    except Exception as exc:
        ok, info = False, None
        if log:
            log(f"expert ERROR {task}#{seed}: {type(exc).__name__}: {exc}")
    finally:
        try:
            env.close_env(clear_cache=clear_cache)
        except Exception:
            pass
    return info if ok else None


def describe(task: str, info: dict, seed: int, kind: str, count: int = 100) -> list[str]:
    """Instruction candidates for one episode (RoboTwin's generator, seeded)."""
    import random
    import numpy as np
    from generate_episode_instructions import generate_episode_descriptions
    random.seed(seed)
    np.random.seed(seed % (2**32))
    results = generate_episode_descriptions(task, [info["info"]], count)
    return list(results[0][kind]) if results else []


def worker(wid, args, jobs, results):
    try:
        _worker(wid, args, jobs, results)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        results.put(None)


def _worker(wid, args, jobs, results):
    os.chdir(ROBOTWIN)
    for path in (ROBOTWIN, os.path.join(ROBOTWIN, "description", "utils")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import numpy as np
    from PIL import Image
    from remote_protocol import RemoteClient

    def log(msg):
        print(f"[w{wid}] {msg}", file=sys.stderr, flush=True)

    client = None if args.book_only else RemoteClient(args.remote_socket)

    def frame(rgb):
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        if img.size != (IMG, IMG):
            img = img.resize((IMG, IMG), Image.BILINEAR)
        return np.asarray(img)

    def observe(env):
        obs = env.get_obs()
        cams = np.stack([frame(obs["observation"]["head_camera"]["rgb"]),
                         frame(obs["observation"]["front_camera"]["rgb"])])
        ep = obs["endpose"]
        state = np.concatenate([np.asarray(ep["left_endpose"], np.float32).ravel(),
                                np.atleast_1d(np.float32(ep["left_gripper"])),
                                np.asarray(ep["right_endpose"], np.float32).ravel(),
                                np.atleast_1d(np.float32(ep["right_gripper"]))]).astype(np.float32)
        return cams, state

    def view(env):
        env.cameras.update_picture()
        rgb = env.cameras.get_rgb()
        return np.concatenate([rgb["head_camera"]["rgb"], rgb["front_camera"]["rgb"]], axis=1)

    def run_episode(env, instruction, rec=None):
        env.set_instruction(instruction=instruction)
        while env.take_action_cnt < env.step_lim and not env.eval_success:
            cams, state = observe(env)
            chunk = client.predict(instruction, cams, state)
            for action in chunk:
                env.take_action(np.asarray(action, np.float64), action_type="qpos")
                if rec is not None:
                    rec.capture(lambda: view(env))
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    break
        return bool(env.eval_success), int(env.take_action_cnt)

    # Blocks come off a shared queue, so no worker idles while another finishes.
    for task, config, episodes in iter(jobs.get, None):
        targs = task_args(config, task)
        env = make_env(task)
        done_on_env = tried = 0
        for entry in episodes:
            seed = int(entry["seed"])
            try:
                instruction = entry.get("instruction")
                if instruction is None:  # sweep mode: expert check + generated instruction
                    tried += 1
                    info = expert_solves(env, task, seed, targs, log,
                                         clear_cache=(tried % EPISODES_PER_ENV == 0))
                    if info is None:
                        results.put({"kind": "robotwin", "task": task, "seed": seed, "config": config,
                                     "solvable": False})
                        continue
                    pool = describe(task, info, seed, args.instructions)
                    if not pool:
                        results.put({"kind": "robotwin", "task": task, "seed": seed, "config": config,
                                     "solvable": False})
                        continue
                    instruction = pool[seed % len(pool)]
                    extra = {"seen": describe(task, info, seed, "seen"),
                             "unseen": describe(task, info, seed, "unseen")}
                    if args.book_only:
                        results.put({"kind": "robotwin", "task": task, "seed": seed, "config": config,
                                     "instruction_type": args.instructions, "instruction": instruction,
                                     "solvable": True, **extra})
                        log(f"seed ok {task}#{seed}")
                        continue
                else:
                    extra = {}
                env.setup_demo(now_ep_num=done_on_env, seed=seed, is_test=True, **targs)
                expert_state(env, task)
                rec = recorder(args, (task, seed), every=3)
                try:
                    success, steps = run_episode(env, instruction, rec)
                finally:
                    if rec is not None:
                        rec.close()
                done_on_env += 1
                try:
                    env.close_env(clear_cache=(done_on_env % EPISODES_PER_ENV == 0))
                except Exception:
                    pass
                results.put({"kind": "robotwin", "task": task, "seed": seed, "config": config,
                             "instruction_type": args.instructions, "instruction": instruction,
                             "solvable": True, "success": success, "steps": steps, **extra})
                log(f"ep done {task}#{seed} success={success} steps={steps}")
            except Exception as exc:
                log(f"ep ERROR {task}#{seed}: {type(exc).__name__}: {exc}")
                try:
                    env.close_env()
                except Exception:
                    pass
    if client is not None:
        client.close()


# ---------------------------------------------------------------------- driver

def load_episode_lists(args) -> dict[str, list[dict]]:
    tasks = [t for t in args.tasks.split(",") if t]
    if args.episodes_file:
        book = json.load(open(args.episodes_file))
        episodes = book["episodes"] if "episodes" in book else book  # {"tasks":..,"episodes":..} or {task: [...]}
        lists = {}
        for task in tasks:
            entries = episodes[task]
            lists[task] = entries[: args.per_task] if args.per_task else entries
        return lists
    if not args.episodes:
        raise SystemExit("sweep mode needs --episodes")
    # sweep: walk seeds upward; the worker reports unsolvable seeds so the driver keeps walking
    return {task: [{"seed": args.seed_base + i, "config": args.config} for i in range(args.episodes * 4)]
            for task in tasks}


def main():
    args = parse_args()
    import queue as pyqueue
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    lists = load_episode_lists(args)
    sweep = not args.episodes_file
    args.clips = designate({task: [(task, int(e["seed"])) for e in entries] for task, entries in lists.items()}) \
        if args.record and not sweep else {}
    rows: dict[tuple, dict] = {}
    if args.resume and os.path.exists(args.resume):
        for line in open(args.resume):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.setdefault(episode_key(row), row)
        print(f"resumed {len(rows)} rows", flush=True)
    inc = open(args.out + ".inc", "a")
    t0 = time.time()

    def pending() -> list[tuple[str, str, list[dict]]]:
        jobs = []
        for task, entries in lists.items():
            if sweep:
                solved = sum(1 for e in entries if rows.get((task, e["seed"]), {}).get("solvable"))
                if solved >= args.episodes:
                    continue
                todo = [e for e in entries if (task, e["seed"]) not in rows]
                todo = todo[: max(1, (args.episodes - solved) * 2)]
            else:
                todo = [e for e in entries if (task, e["seed"]) not in rows]
            config = entries[0].get("config", args.config)
            for i in range(0, len(todo), BLOCK):
                jobs.append((task, config, todo[i:i + BLOCK]))
        # Round-robin the blocks across tasks: a rollout that runs out of wall clock
        # then leaves a balanced sample instead of finishing the first few tasks and
        # never starting the rest.
        by_task: dict[str, list] = {}
        for job in jobs:
            by_task.setdefault(job[0], []).append(job)
        ordered = []
        for round_ in range(max((len(v) for v in by_task.values()), default=0)):
            for task in lists:
                if task in by_task and round_ < len(by_task[task]):
                    ordered.append(by_task[task][round_])
        return ordered

    for attempt in range(6):
        jobs = pending()
        if not jobs:
            break
        if attempt:
            print(f"[{time.time() - t0:6.0f}s] repair pass {attempt}: {sum(len(j[2]) for j in jobs)} episodes", flush=True)
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
                if sum(p.is_alive() for p in procs) == 0:
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
            if key not in rows:
                rows[key] = row
                inc.write(json.dumps(row) + "\n")
                inc.flush()
            scored = [r for r in rows.values() if "success" in r]
            if scored and len(scored) % 10 == 0:
                sr = sum(r["success"] for r in scored) / len(scored)
                print(f"[{time.time() - t0:6.0f}s] {len(scored)} episodes | SR {sr:.3f}", flush=True)
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    with open(args.out, "w") as stream:
        for row in rows.values():
            stream.write(json.dumps(row) + "\n")
    if args.clips:
        assemble(args.record, args.clips,
                 {json.dumps(list(k)): r["success"] for k, r in rows.items() if "success" in r})
    by_task = collections.defaultdict(list)
    for row in rows.values():
        if "success" in row:
            by_task[row["task"]].append(row["success"])
    for task, values in sorted(by_task.items()):
        print(f"{task}: {sum(values) / len(values):.3f} (n={len(values)})")
    scored = [v for values in by_task.values() for v in values]
    print(f"OVERALL: {sum(scored) / max(1, len(scored)):.4f} (n={len(scored)}) | wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
