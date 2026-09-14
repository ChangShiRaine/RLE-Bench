"""Development evaluation service: the sidecar beside the agent container.

    python eval_service.py --subtask 02-libero-robustness --socket /run/nanovla-eval/service.sock

Requests arrive on a Unix socket as u32-length-prefixed JSON:
    {"policy": "/run/nanovla-eval/<name>.sock", "tasks": [0, ...], "episodes": N}
and are answered the same way with per-episode results. The service holds the
simulator; development episodes are seeded resets (episode i of a task is the
same placement on every request), disjoint from anything the verifier scores.
The policy socket must be a socket owned by the agent UID inside the shared
directory. Requests run one at a time and draw on a fixed episode budget. Every
request is logged.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

from common import PRIVATE_DIR, SIMULATOR, VERIFIER_CODE, _reset_dir, evaluator_env

_U32 = struct.Struct("!I")
MAX_REQUEST = 64 * 1024
AGENT_UID = 1000
SUITE = "libero_10"
DEV_SEED_BASE = 100_000  # LIBERO development episodes: seed = base + task*1000 + episode
ROBOTWIN_DEV_BOOK = VERIFIER_CODE / "robotwin_dev_episodes.json"  # RoboTwin: frozen solvable seeds


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("client closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Service:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.shared = Path(args.socket).parent
        self.budget = args.budget
        self.log_path = Path(args.log_dir) / f"requests-{os.uname().nodename}-{int(time.time())}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.serial = 0
        self.simulator = SIMULATOR
        if self.simulator == "robotwin":
            self.book = json.loads(ROBOTWIN_DEV_BOOK.read_text())
            self.task_names = self.book["tasks"]
            self.max_episodes = min((len(v) for v in self.book["episodes"].values()), default=0)
        else:
            self.task_names = [str(i) for i in range(10)]
            self.max_episodes = 50
        self.n_tasks = len(self.task_names)

    def parse(self, request: dict) -> dict:
        policy = request.get("policy")
        if not isinstance(policy, str) or Path(policy).parent != self.shared:
            raise ValueError(f"policy socket must live directly under {self.shared}")
        try:
            info = Path(policy).lstat()
        except FileNotFoundError:
            raise ValueError("policy socket does not exist") from None
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != AGENT_UID:
            raise ValueError("policy must be a socket owned by the agent user")
        tasks = request.get("tasks", list(range(self.n_tasks)))
        if (not isinstance(tasks, list) or not tasks
                or any(type(t) is not int or not 0 <= t < self.n_tasks for t in tasks)):
            raise ValueError(f"tasks must be a non-empty list of integers in 0..{self.n_tasks - 1}")
        episodes = request.get("episodes", 10)
        if type(episodes) is not int or not 1 <= episodes <= self.max_episodes:
            raise ValueError(f"episodes must be an integer in 1..{self.max_episodes}")
        return {"policy": policy, "tasks": sorted(set(tasks)), "episodes": episodes}

    def run(self, spec: dict) -> dict:
        self.serial += 1
        total = len(spec["tasks"]) * spec["episodes"]
        if total > self.budget:
            raise ValueError(f"episode budget exhausted: {self.budget} left, {total} requested")
        self.budget -= total
        work = PRIVATE_DIR / f"request-{self.serial}"
        _reset_dir(work, mode=0o700)
        out = work / "rows.jsonl"
        if self.simulator == "robotwin":
            names = [self.task_names[t] for t in spec["tasks"]]
            subset = {"tasks": names, "episodes": {n: self.book["episodes"][n][: spec["episodes"]] for n in names}}
            (work / "episodes.json").write_text(json.dumps(subset))
            command = [sys.executable, str(VERIFIER_CODE / "rollout_robotwin.py"), "--remote-socket", spec["policy"],
                       "--out", str(out), "--workers", str(self.args.workers), "--tasks", ",".join(names),
                       "--episodes-file", str(work / "episodes.json")]
            timeout = min(5400.0, 300.0 + 240.0 * total / max(1, self.args.workers))
        else:
            command = [sys.executable, str(VERIFIER_CODE / "rollout.py"), "--remote-socket", spec["policy"],
                       "--out", str(out), "--workers", str(self.args.workers), "--gpus", str(self.args.gpus),
                       "--suite", SUITE, "--tasks", ",".join(map(str, spec["tasks"])), "--init", "seeded",
                       "--seed-base", str(DEV_SEED_BASE), "--episodes", str(spec["episodes"])]
            timeout = min(3600.0, 300.0 + 90.0 * total / max(1, self.args.workers))
        start = time.monotonic()
        with (work / "rollout.log").open("wb") as log:
            try:
                subprocess.run(command, env=evaluator_env(), cwd=VERIFIER_CODE,
                               stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                pass
        wall = time.monotonic() - start
        rows = []
        for candidate in (out, Path(str(out) + ".inc")):
            if candidate.is_file():
                rows += [json.loads(line) for line in candidate.read_text().splitlines() if line.strip()]
        seen, episodes = set(), []
        for row in rows:
            if self.simulator == "robotwin":
                key = (row.get("task"), row.get("seed"))
                task, episode = self.task_names.index(row["task"]), int(row["seed"])
            else:
                key = (row.get("task"), row.get("episode"))
                task, episode = int(row["task"]), int(row["episode"])
            if key in seen or type(row.get("success")) is not bool:
                continue
            seen.add(key)
            episodes.append({"task": task, "episode": episode,
                             "success": row["success"], "steps": int(row.get("steps", -1))})
        per_task: dict[str, dict] = {}
        for episode in episodes:
            bucket = per_task.setdefault(str(episode["task"]), {"successes": 0, "episodes": 0})
            bucket["successes"] += int(episode["success"])
            bucket["episodes"] += 1
        for bucket in per_task.values():
            bucket["success_rate"] = bucket["successes"] / bucket["episodes"]
        rate = sum(e["success"] for e in episodes) / total  # missing episodes count as failures
        return {"ok": True, "episodes": total, "completed": len(episodes),
                "success_rate": rate, "per_task": per_task, "results": episodes,
                "wall_s": wall, "budget_remaining": self.budget}

    def log(self, record: dict) -> None:
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def serve(self) -> None:
        path = Path(self.args.socket)
        self.shared.mkdir(parents=True, exist_ok=True)
        self.shared.chmod(0o1777)
        path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        path.chmod(0o666)
        listener.listen(16)
        print(f"evaluation service ready: {path} (subtask {self.args.subtask}, {self.simulator}, budget {self.budget})", flush=True)
        while True:
            conn, _ = listener.accept()
            with conn:
                started = time.time()
                try:
                    conn.settimeout(30.0)
                    length = _U32.unpack(_recv_exact(conn, _U32.size))[0]
                    if not 1 <= length <= MAX_REQUEST:
                        raise ValueError("request too large")
                    request = json.loads(_recv_exact(conn, length).decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request must be a JSON object")
                    spec = self.parse(request)
                    conn.settimeout(None)
                    reply = self.run(spec)
                    self.log({"time": started, "request": spec, "success_rate": reply["success_rate"],
                              "episodes": reply["episodes"], "completed": reply["completed"],
                              "wall_s": reply["wall_s"], "budget_remaining": self.budget})
                except Exception as exc:
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "budget_remaining": self.budget}
                    self.log({"time": started, "error": reply["error"]})
                payload = json.dumps(reply).encode("utf-8")
                try:
                    conn.sendall(_U32.pack(len(payload)) + payload)
                except OSError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtask", required=True)
    parser.add_argument("--socket", default="/run/nanovla-eval/service.sock")
    parser.add_argument("--log-dir", default="/logs/eval-service")
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("eval_service must run as root inside its own container")
    _reset_dir(PRIVATE_DIR, mode=0o700)
    Service(args).serve()


if __name__ == "__main__":
    main()
