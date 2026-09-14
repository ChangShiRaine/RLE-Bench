"""Client for the evaluation service that runs beside the development container.

Start your policy server first (see remote_protocol.py), then:

    python nanovla_eval.py --policy /run/nanovla-eval/policy.sock --episodes 10
    python nanovla_eval.py --policy /run/nanovla-eval/policy.sock --tasks 0,3 --episodes 5

The service owns the simulator. Development episode i of a task is always the
same seeded placement, so two policies evaluated on the same tasks and
episode counts see identical initial states. It reports per-episode success
and steps; the policy socket must live under /run/nanovla-eval/. Every request
draws on the session's episode budget.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys

SERVICE = "/run/nanovla-eval/service.sock"
_U32 = struct.Struct("!I")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("evaluation service closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def evaluate(policy: str, *, tasks: list[int] | None = None, episodes: int = 10,
             service: str = SERVICE, timeout_s: float = 7200.0) -> dict:
    """Run one evaluation request and return the service's JSON reply."""
    request = {"policy": policy, "episodes": int(episodes)}
    if tasks is not None:
        request["tasks"] = [int(t) for t in tasks]
    payload = json.dumps(request).encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(service)
        sock.sendall(_U32.pack(len(payload)) + payload)
        length = _U32.unpack(_recv_exact(sock, _U32.size))[0]
        return json.loads(_recv_exact(sock, length).decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="your policy server's Unix socket")
    parser.add_argument("--tasks", default="", help="comma list of task indices (default: all)")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per task")
    parser.add_argument("--json", default="", help="write the full reply here")
    args = parser.parse_args()
    tasks = [int(t) for t in args.tasks.split(",") if t.strip()] or None
    reply = evaluate(args.policy, tasks=tasks, episodes=args.episodes)
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(reply, stream, indent=2)
    if not reply.get("ok"):
        print(f"evaluation failed: {reply.get('error')}", file=sys.stderr)
        raise SystemExit(1)
    for task, stats in sorted(reply["per_task"].items(), key=lambda item: int(item[0])):
        print(f"task {int(task):2d}: {stats['success_rate']:.2f}  (n={stats['episodes']})")
    print(f"OVERALL {reply['success_rate']:.4f} over {reply['episodes']} episodes | "
          f"wall {reply['wall_s']:.0f}s | budget left {reply['budget_remaining']}")


if __name__ == "__main__":
    main()
