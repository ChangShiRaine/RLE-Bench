"""The harness's control plane. ROOT ONLY -- never staged into the agent's payload.

The agents drive the run; the harness decides which phase the run is in and when one
trial ends and the next begins. Those are different planes, and this module is the
second one. It ships only in the root-only /opt/private tree (see
tasks/task02/build_assets.py), so no agent can read it, and the daemon additionally
refuses these operations to any caller whose uid is not 0 -- hiding the module alone
would be obscurity, since the wire would still accept a guessed op string.

WHY THE SEPARATION IS SHARPER HERE THAN IN TASK01. Task01 learned this the hard way:
`begin_evaluation` once lived in the agent-facing client, a real agent called it during
the unscored development phase, and spent its single terminal submission -- evaluated
without ever being told it was being evaluated. Task02 adds a second, worse case.
`next-trial` ends one graded trial and opens the next, and each trial is meant for a
DIFFERENT agent with no memory of the last. An agent that could call it would drive its
successor's trial carrying everything it knows, which does not merely inflate a score --
it dismantles what the benchmark measures.

Used from Harbor collect hooks, which run as root between steps:

    PYTHONSAFEPATH=1 PYTHONPATH=/opt/private python -m harness.control open-evaluation
    PYTHONSAFEPATH=1 PYTHONPATH=/opt/private python -m harness.control next-trial

PYTHONSAFEPATH=1 is load-bearing. The hooks run from the image WORKDIR, /workspace, and
`python -m` searches the CWD before PYTHONPATH -- so without it a package the agent wrote
at /workspace/harness answers this import instead of the root-only tree.
"""

from __future__ import annotations

import argparse
import os
import pwd
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from . import protocol as P

DEFAULT_SOCKET = os.environ.get(
    "RLEBENCH_TOOLSMITH_SOCKET", "/run/rlebench/toolsmith.sock"
)


def stop_agent_processes() -> None:
    """Stop detached agent commands too; the root simulator must survive."""
    uid = pwd.getpwnam("agent").pw_uid
    if os.geteuid() != 0 or uid == 0:
        raise RuntimeError("agent cleanup requires root and a non-root agent UID")

    def live_pids() -> list[int]:
        found = []
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                status = dict(
                    line.split(":", 1)
                    for line in (proc / "status").read_text().splitlines()
                )
                if (int(status["Uid"].split()[0]) == uid
                        and not status["State"].lstrip().startswith("Z")):
                    found.append(int(proc.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
        return found

    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 3
        while pids := live_pids():
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        else:
            return
    if live_pids():
        raise RuntimeError("agent processes survived cleanup")


def _handoff_timeout(signum, frame) -> None:
    raise TimeoutError("phase handoff exceeded 150 seconds")


class ControlClient:
    """Speaks the same wire protocol as the agent's client, over the same socket.

    The separation is not the transport -- it is who may call, enforced by the daemon
    from the kernel-supplied peer uid.
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self.socket_path = str(socket_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)
        self._reader = P.LineReader(self._sock)

    def _request(self, op: str, **fields: Any) -> dict:
        P.send(self._sock, {"op": op, **fields})
        msg = self._reader.read()
        if msg is None:
            raise RuntimeError("daemon closed the connection")
        if not msg.get("ok", False):
            raise RuntimeError(str(msg.get("error", "unknown error")))
        return {k: v for k, v in msg.items() if k != "ok"}

    def open_evaluation(self) -> dict:
        """Close development, open the scored phase, and prepare the first trial.

        The agent's counterpart is `end_development()`, which only says "I am ready" and
        touches nothing. Development is closed here whether or not the agent said so: a
        run whose agent never got round to it must still be evaluated, and the ledger
        already records which of the two happened.
        """
        return self._request("open_evaluation")

    def next_trial(self) -> dict:
        """Score the live trial and open the next. THE STEP BOUNDARY.

        Called between Harbor steps, so the agent that drove trial N has already exited
        when trial N+1 is built. Advancing past the last trial closes the evaluation and
        writes its aggregate.
        """
        return self._request("next_trial")

    def seal(self) -> dict:
        """Close the session, seal the ledger and export the artifacts."""
        return self._request("seal")

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "ControlClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="task02 harness control plane (root only)")
    ap.add_argument("command", choices=("open-evaluation", "next-trial", "seal"))
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    args = ap.parse_args(argv)

    previous_handler = signal.signal(signal.SIGALRM, _handoff_timeout)
    signal.alarm(150)  # Finish before Harbor's 180-second collect-hook timeout.
    try:
        if args.command != "seal":
            stop_agent_processes()
        with ControlClient(args.socket) as ctl:
            if args.command == "open-evaluation":
                out = ctl.open_evaluation()
                print(f"[control] evaluation opened: "
                      f"{out.get('total_trials')} trial(s)")
            elif args.command == "next-trial":
                out = ctl.next_trial()
                if out.get("evaluation_done"):
                    print(f"[control] evaluation finished: "
                          f"{out.get('trials')} trial(s) graded")
                else:
                    print(f"[control] trial {out.get('trials_done')} of "
                          f"{out.get('total_trials')} scored; next trial open")
            else:
                out = ctl.seal()
                print(f"[control] sealed: {out.get('summary')}")
    except Exception as exc:  # noqa: BLE001
        # Harbor treats collect hooks as best-effort; report failures explicitly.
        print(f"[control] {args.command} FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
