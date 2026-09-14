#!/usr/bin/env python3
"""Atomically stage the one-file nanoVLA submission."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MAX_BYTES = 1024 * 1024
DESTINATION = Path("/logs/artifacts/submission/solution.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    source = args.candidate
    if source.is_symlink() or not source.is_file():
        raise SystemExit("candidate must be a regular file, not a symlink")
    payload = source.read_bytes()
    if not payload or len(payload) > MAX_BYTES:
        raise SystemExit(f"candidate must be 1..{MAX_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
        compile(text, "solution.py", "exec")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SystemExit(f"candidate is not valid UTF-8 Python: {exc}") from exc

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_name(f".solution.py.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.chmod(0o644)
        os.replace(temporary, DESTINATION)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"staged {len(payload)} bytes at {DESTINATION}")


if __name__ == "__main__":
    main()
