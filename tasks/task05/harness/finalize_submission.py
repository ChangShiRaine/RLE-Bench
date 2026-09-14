#!/usr/bin/env python3
"""Trusted collect hook: freeze one valid nanoVLA Python submission."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import time
from pathlib import Path

ARTIFACT_ROOT = Path("/logs/artifacts")
SOURCE_DIR = ARTIFACT_ROOT / "submission"
SOURCE = SOURCE_DIR / "solution.py"
HANDOFF_ROOT = Path("/logs/nanovla-handoff")
HANDOFF_DIR = HANDOFF_ROOT / "submission"
HANDOFF = HANDOFF_DIR / "solution.py"
MAX_BYTES = 1024 * 1024
AGENT_UID = 1000


def process_cgroup(pid: int) -> str | None:
    try:
        return (Path("/proc") / str(pid) / "cgroup").read_text()
    except (FileNotFoundError, PermissionError):
        return None


def process_start_time(pid: int) -> int:
    try:
        stat_line = (Path("/proc") / str(pid) / "stat").read_text()
        return int(stat_line[stat_line.rfind(")") + 2 :].split()[19])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return 2**63 - 1


def protected_ancestry() -> set[int]:
    """Return this trusted exec process and its parent chain."""
    protected = {1}
    pid = os.getpid()
    while pid > 0 and pid not in protected:
        protected.add(pid)
        try:
            fields = (Path("/proc") / str(pid) / "status").read_text().splitlines()
            ppid_line = next(line for line in fields if line.startswith("PPid:"))
            pid = int(ppid_line.split()[1])
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            break
    return protected


def kill_agent_processes() -> None:
    """Stop experiments without allowing the collect exec to kill itself."""
    protected = protected_ancestry()
    own_cgroup = process_cgroup(os.getpid())
    for _ in range(20):
        remaining = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in protected:
                continue
            if process_cgroup(pid) != own_cgroup:
                continue
            try:
                fields = (entry / "status").read_text().splitlines()
                uid_line = next(line for line in fields if line.startswith("Uid:"))
                state_line = next(line for line in fields if line.startswith("State:"))
                # a zombie is already dead: SIGKILL cannot act on it and it cannot
                # run or mutate the handoff; it only waits for PID 1 to reap it
                if int(uid_line.split()[1]) == AGENT_UID and not state_line.split()[1].startswith("Z"):
                    remaining.append(pid)
            except (FileNotFoundError, PermissionError, StopIteration, ValueError, IndexError):
                continue
        if remaining:
            protected.add(min(remaining, key=process_start_time))
            remaining = [pid for pid in remaining if pid not in protected]
        if not remaining:
            return
        details = []
        for pid in remaining:
            try:
                raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
                command = raw.replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                command = "<gone>"
            details.append(f"{pid}:{command[:160]}")
        print("UID1000 processes: " + "; ".join(details), flush=True)
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.25)  # 20 rounds = 5 s, well inside the 30 s collect budget
    raise SystemExit("agent processes survived SIGKILL; refusing mutable handoff")


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit(
            f"nanovla-finalize must run as root; effective UID is {os.geteuid()}"
        )
    print(f"finalizer root PID {os.getpid()}; freezing agent state", flush=True)
    kill_agent_processes()
    if HANDOFF_ROOT.exists():
        shutil.rmtree(HANDOFF_ROOT)
    HANDOFF_DIR.mkdir(parents=True, mode=0o700)
    HANDOFF_ROOT.chmod(0o700)

    if not ARTIFACT_ROOT.is_dir():
        raise SystemExit("missing /logs/artifacts")
    root_entries = sorted(entry.name for entry in ARTIFACT_ROOT.iterdir())
    if root_entries != ["submission"]:
        raise SystemExit(
            f"/logs/artifacts must contain only submission/; found {root_entries}"
        )
    if not SOURCE_DIR.is_dir() or SOURCE_DIR.is_symlink():
        raise SystemExit("missing /logs/artifacts/submission")
    # bytecode cache left by the agent compiling/importing its own file; carries nothing
    shutil.rmtree(SOURCE_DIR / "__pycache__", ignore_errors=True)
    entries = sorted(entry.name for entry in SOURCE_DIR.iterdir())
    if entries != ["solution.py"]:
        raise SystemExit(f"submission must contain only solution.py; found {entries}")
    info = SOURCE.lstat()
    if not stat.S_ISREG(info.st_mode) or SOURCE.is_symlink():
        raise SystemExit("solution.py must be a regular file, not a symlink")
    if not 1 <= info.st_size <= MAX_BYTES:
        raise SystemExit(f"solution.py must be 1..{MAX_BYTES} bytes")
    payload = SOURCE.read_bytes()
    try:
        text = payload.decode("utf-8")
        compile(text, "solution.py", "exec")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SystemExit(f"solution.py is not valid UTF-8 Python: {exc}") from exc

    HANDOFF.write_bytes(payload)
    HANDOFF.chmod(0o400)
    print(f"finalized {len(payload)} bytes for separate verification")


if __name__ == "__main__":
    main()
