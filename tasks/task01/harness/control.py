"""The harness's control plane. ROOT ONLY -- never staged into the agent's payload.

The agent drives the run; the harness decides which phase the run is in. Those are
different planes, and this module is the second one. It ships only in the root-only
/opt/private tree (see tasks/task01/build_assets.py), so the agent cannot read it, and
the daemon additionally refuses these operations to any caller whose uid is not 0 --
hiding the module alone would be obscurity, since the wire would still accept a guessed
op string.

Entering the scored phase cannot be the agent's call: an agent that could reach it would
spend its single terminal submission without ever being told it was being evaluated, which
is the only thing distinguishing the two phases.

Used from Harbor collect hooks, which run as root between steps:

    PYTHONPATH=/opt/private python -m harness.control open-evaluation
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Any

from . import protocol as P

DEFAULT_SOCKET = os.environ.get(
    "RLEBENCH_SPEEDRUN_SOCKET", "/run/rlebench/speedrun.sock"
)


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
        """Advance to the scored phase and prepare the simulator for the first trial.

        The agent's counterpart is `end_development()`, which only says "I am ready" and
        touches nothing.
        """
        return self._request("open_evaluation")

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
    ap = argparse.ArgumentParser(description="task01 harness control plane (root only)")
    ap.add_argument("command", choices=("open-evaluation", "seal"))
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    args = ap.parse_args(argv)

    try:
        with ControlClient(args.socket) as ctl:
            if args.command == "open-evaluation":
                out = ctl.open_evaluation()
                print(f"[control] evaluation opened: "
                      f"{out.get('total_trials')} trial(s)")
            else:
                out = ctl.seal()
                print(f"[control] sealed: {out.get('summary')}")
    except Exception as exc:  # noqa: BLE001
        # Collect hooks are best-effort in Harbor ("failures never abort the trial"), so
        # a failure here is silent unless it is said out loud.
        print(f"[control] {args.command} FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
