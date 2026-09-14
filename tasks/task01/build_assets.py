"""Stage `harness` into a task01 image build context.

Docker cannot COPY from outside a build context, and the metering code lives in the
repo at tasks/task01/harness/. Edit there, then re-run `build_levels.py --emit-all`.

Boundary (CLAUDE.md invariant #2): the agent may read the client and the wire protocol;
the daemon, ledger writer and session logic decide and record the score, so the agent's
uid must not be able to read them. The daemon runs INSIDE the agent's container, so both
trees ship in the same image and the split is by file ownership:

    payload_agent/   -> /opt/rlebench          world-readable
    payload_private/ -> /opt/private/harness  root-only, mode 700

THE HARNESS LEVEL IS DECIDED HERE. A level is a statement about which modules the
agent's uid can open, so it is settled when the payload is staged, not at run time:

    L1  client, protocol, obs                            the control condition
    L2  the same, plus the harness library and its manual
    L3  the same as L2; the extra state reaches the agent over the METERED SOCKET,
        never as a module. `privileged.py` is forbidden in the agent tree at EVERY
        level, L3 included -- an importable copy would let an agent call it on an env
        of its own and step off-meter.

The VERIFIER shares this environment and runs as root, importing the scorer from
/opt/private (see _template/tests/test.sh). It reads the authoritative root-only ledger
in place; the copy Harbor makes agent-writable under /logs/artifacts is never scored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tasks" / "task01" / "harness"

# What the agent legitimately needs in order to talk to the daemon, at every level.
AGENT_MODULES = ("__init__.py", "client.py", "protocol.py", "obs.py")

# The harness library, staged into the agent tree at SKILLS_LEVELS and nowhere else.
# It carries its manual, which is the half of the harness that does the work.
SKILLS_PKG = "skills"

# The trusted side gets everything, including the scorer.
DAEMON_MODULES = (
    "__init__.py", "client.py", "protocol.py", "env.py", "ledger.py",
    "session.py", "service.py", "daemon_main.py", "obs.py",
    "config.py", "evaluation.py", "transcript.py", "control.py",
    "scoring.py", "verify_main.py", "debug.py", "privileged.py",
    "perception_service.py", "eval_video.py",
)

# Never readable by the agent's uid: these decide or record the score, or -- in
# privileged.py's case -- would hand out an off-meter route to the simulator's state.
# `transcript.py` is here because it carries `diagnose`, which reads object and fixture
# poses straight off the env: exactly what L1 strips out. `perception_service.py` holds no
# secret -- the models are a pure function of the image sent to them -- but the agent gets
# a SERVICE with a contract rather than a tree it could monkeypatch, so the image keeps one
# story about what its uid can open.
FORBIDDEN_IN_AGENT = (
    "service.py", "ledger.py", "session.py", "env.py", "daemon_main.py",
    "config.py", "evaluation.py", "scoring.py", "verify_main.py", "control.py",
    "debug.py", "privileged.py", "transcript.py", "perception_service.py",
)


def _levels():
    """The level constants, from the one place they are defined.

    Imported lazily and by path: this script runs under the Makefile's python, which is
    not the simulator venv, and `harness.config` imports nothing heavy.
    """
    sys.path.insert(0, str(REPO))
    from harness import config as C

    return C


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


def stage(dest_root: Path, modules: tuple[str, ...], level: str | None = None) -> None:
    """Write one payload tree. `level` staged into the AGENT tree adds the skills."""
    sync_shared()
    pkg = dest_root / "harness"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    for name in modules:
        shutil.copy2(SRC / name, pkg / name)
    if level is not None and level in _levels().SKILLS_LEVELS:
        shutil.copytree(SRC / SKILLS_PKG, pkg / SKILLS_PKG,
                        ignore=shutil.ignore_patterns("__pycache__"))


def check(agent_root: Path, level: str) -> list[str]:
    """Everything wrong with an agent payload, or [] if it is sound.

    Both directions matter. A forbidden module present is a metering hole; the skill
    library MISSING at L2 is a silently degraded harness, which would be scored as L2
    and is not.
    """
    pkg = agent_root / "harness"
    problems = [f"forbidden module {m}" for m in FORBIDDEN_IN_AGENT
                if (pkg / m).exists()]
    # A module list is only half the boundary. An agent-visible module that IMPORTS a
    # forbidden one at module scope would pull it into the agent's process the moment the
    # client is imported -- and it would still be there to read, whatever the directory
    # permissions say about the file on disk. The skill library is scanned too: skills
    # run in the AGENT's process, so an import there lands in the same place.
    forbidden_names = {m[:-3] for m in FORBIDDEN_IN_AGENT}
    for path in sorted(pkg.rglob("*.py")):
        text = path.read_text()
        # Both dot depths: the skills package sits one level down, so its route to a
        # sibling of the client is `from ..config import`, not `from .config import`.
        for name in sorted(forbidden_names):
            forms = (f"from .{name} import", f"from ..{name} import",
                     f"from . import {name}", f"from .. import {name}")
            if any(form in text for form in forms):
                problems.append(f"{path.relative_to(pkg)} imports {name}")
    has_skills = (pkg / SKILLS_PKG).is_dir()
    if level in _levels().SKILLS_LEVELS:
        if not has_skills:
            problems.append("no skill library, but L2 is the level that ships one")
        elif not (pkg / SKILLS_PKG / "MANUAL.md").is_file():
            problems.append("skill library ships without MANUAL.md")
    elif has_skills:
        problems.append(f"skill library present at {level}; only L2 ships one")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify every emitted image context without writing")
    ap.add_argument("--sync-shared", action="store_true",
                    help="copy the sim/robocasa shared modules into harness/")
    args = ap.parse_args()
    if args.sync_shared:
        sync_shared()
        return 0
    if not args.check:
        ap.error("staging happens through build_levels.py --emit-all; "
                 "this script only verifies")

    sys.path.insert(0, str(Path(__file__).parent))
    import build_levels

    bad = False
    for level in _levels().LEVELS:
        ctx = build_levels.image_dir(level)
        if not ctx.is_dir():
            print(f"{level}: not emitted; run build_levels.py --emit-all",
                  file=sys.stderr)
            bad = True
            continue
        for problem in check(ctx / "payload_agent", level):
            print(f"BOUNDARY VIOLATION in {level}: {problem}", file=sys.stderr)
            bad = True
    if bad:
        return 1
    print(f"boundary ok for {', '.join(_levels().LEVELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
