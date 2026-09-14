"""Stage `harness` into the task02 agent-image build context.

Docker cannot COPY from outside a build context, and the metering code lives in the repo
at tasks/task02/harness/. The staged copies are build inputs, not sources -- edit
tasks/task02/harness/ and re-run.

Boundary (CLAUDE.md invariant #2): the agent may read the client and the wire protocol;
the daemon, ledger writer, session logic, stage functions and scorer decide and record the
score, so the agent's uid must not be able to read them.

TASK02 ADDS ONE ITEM TO THAT LIST, AND IT IS THE MOST IMPORTANT ONE. `config.py` names the
held-out task the run is graded on, and `stages.py` names it again as a dict key. An agent
that read either would spend development practising the task it is about to face, and the
benchmark would measure memorisation rather than transfer.

The daemon runs INSIDE the agent's container (uid-separated), so both trees ship in the
one image every group runs, and the split is by file ownership rather than by image:

    payload_agent/   -> /opt/rlebench          world-readable (client, protocol)
    payload_private/ -> /opt/private/harness  root-only, mode 700 (everything)

Nothing is staged into the base image: it stays simulator-only, so anything deriving from
it cannot inherit the scoring side by accident.

The VERIFIER shares this environment and runs as root, importing the scorer from
/opt/private (tasks/task02/tests/test.sh). There is no third tree: a separate verifier
could only read the /logs/artifacts copy, which Harbor makes agent-writable.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tasks" / "task02" / "harness"

# What the agent legitimately needs in order to talk to the daemon and write a
# controller. Note what is NOT here: config.py, which the agent-facing modules therefore
# must not import at module scope.
AGENT_MODULES = ("__init__.py", "client.py", "protocol.py", "controller.py",
                 "perception.py")

# The trusted side gets everything, INCLUDING the scorer: the verifier shares this
# environment and runs as root, importing from /opt/private (see tests/test.sh). That is
# what lets it read the authoritative ledger at its root-only path instead of the
# agent-writable copy under /logs/artifacts.
DAEMON_MODULES = (
    "__init__.py", "client.py", "protocol.py", "controller.py", "perception.py",
    "config.py", "stages.py", "env.py", "ledger.py", "session.py", "evaluation.py",
    "service.py", "daemon_main.py", "control.py", "transcript.py", "debug.py",
    "scoring.py", "verify_main.py", "eval_video.py",
)

# Never readable by the agent's uid: these decide, record or REVEAL the score.
#
# `config.py` and `stages.py` head the list on purpose -- between them they name the
# held-out task, which is the one secret task02 has.
FORBIDDEN_IN_AGENT = (
    "config.py", "stages.py", "service.py", "ledger.py", "session.py", "env.py",
    "daemon_main.py", "evaluation.py", "scoring.py", "verify_main.py", "control.py",
    "debug.py", "transcript.py",
)


# rlebench.core beside the private harness: eval_video.py records with it.
CORE_FILES = ("__init__.py", "media.py")


def stage_core(dest_root: Path) -> None:
    rl = dest_root / "rlebench"
    if rl.exists():
        shutil.rmtree(rl)
    (rl / "core").mkdir(parents=True)
    shutil.copy2(REPO / "rlebench" / "__init__.py", rl / "__init__.py")
    for name in CORE_FILES:
        shutil.copy2(REPO / "rlebench" / "core" / name, rl / "core" / name)


# Harness modules every family on the RoboCasa layer shares, stored once under
# sim/robocasa and synced into this package (gitignored here) before anything is
# staged from it, so the three copies cannot drift.
SHARED_MODULES = {"eval_video.py": REPO / "sim" / "robocasa" / "eval_video.py"}


def sync_shared() -> None:
    for name, src in SHARED_MODULES.items():
        shutil.copy2(src, SRC / name)


def stage(dest_root: Path, modules: tuple[str, ...]) -> None:
    sync_shared()
    pkg = dest_root / "harness"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    for name in modules:
        shutil.copy2(SRC / name, pkg / name)


def check(agent_ctx: Path, eval_tasks: tuple[str, ...] = ()) -> list[str]:
    """Every way the evaluation split could reach the agent's tree.

    Takes the tree and the names to look for: the shared image carries every group's.
    """
    pkg = agent_ctx / "harness"
    leaks = [m for m in FORBIDDEN_IN_AGENT if (pkg / m).exists()]
    if not pkg.exists():
        return leaks
    # A module list is only half the boundary. An agent-visible module that IMPORTS a
    # forbidden one at module scope would pull it into the agent's process the moment
    # the client is imported -- and it would still be there to read, whatever the
    # directory permissions say about the file on disk.
    forbidden_names = {m[:-3] for m in FORBIDDEN_IN_AGENT}
    for path in sorted(pkg.glob("*.py")):
        text = path.read_text()
        for name in forbidden_names:
            if f"from .{name} import" in text or f"from . import {name}" in text:
                leaks.append(f"{path.name} imports {name}")
    # Belt and braces on the one thing that matters most: no evaluation task name may
    # appear anywhere in the agent's tree, however it got there.
    for path in sorted(pkg.glob("*.py")):
        text = path.read_text()
        for task in eval_tasks:
            if task in text:
                leaks.append(f"{path.name} names the evaluation task {task}")
    return leaks


def main() -> int:
    """Check the boundary of the staged image context. Staging itself belongs to
    `build_groups.py --emit`, which is the only thing that knows the splits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the boundary (kept for the Makefile)")
    ap.add_argument("--sync-shared", action="store_true",
                    help="copy the sim/robocasa shared modules into harness/")
    args = ap.parse_args()
    if args.sync_shared:
        sync_shared()
        return 0

    sys.path.insert(0, str(Path(__file__).parent))
    from build_groups import IMAGE_DIR, all_eval_tasks

    problems = check(IMAGE_DIR / "payload_agent", all_eval_tasks())
    if problems:
        print("BOUNDARY VIOLATION:", file=sys.stderr)
        for p in problems:
            print(f"  image/: {p}", file=sys.stderr)
        return 1
    print(f"boundary ok in image/ ({len(all_eval_tasks())} held-out names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
