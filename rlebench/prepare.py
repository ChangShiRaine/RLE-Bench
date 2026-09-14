"""Per-family preparation: assets, generated trees, docker images.

The manifest (tasks/<family>/manifest.toml) names what to run; the heavy
lifting stays where it lives today -- build_assets.py, the matrix emitters,
make, sim/robocasa/robocasa.sh -- so prepare is a sequencer, not a second
implementation.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from .repo import REPO, Family

_PLACEHOLDERS = {
    "{py}": str(REPO / ".venv" / "bin" / "python"),
    "{python3}": sys.executable,
}


def _fill(cmd: list[str]) -> list[str]:
    return [_PLACEHOLDERS.get(tok, tok) for tok in cmd]


def image_exists(tag: str) -> bool:
    if not shutil.which("docker"):
        return False
    return (
        subprocess.run(
            ["docker", "image", "inspect", tag], capture_output=True
        ).returncode
        == 0
    )


def status(family: Family) -> dict[str, bool]:
    ok = {}
    for rel in family.verify_paths:
        ok[rel] = (REPO / rel).exists()
    for tag in family.verify_images:
        ok[f"image:{tag}"] = image_exists(tag)
    return ok


def prepare(family: Family, *, dry_run: bool = False) -> int:
    for note in family.prepare_env_notes:
        print(f"note: {note}")
    for cmd in family.prepare_commands:
        argv = _fill(cmd)
        print(f"$ {' '.join(argv)}")
        if dry_run:
            continue
        code = subprocess.run(argv, cwd=REPO).returncode
        if code != 0:
            print(f"error: {' '.join(argv)} exited {code}", file=sys.stderr)
            return code
    missing = [k for k, v in status(family).items() if not v]
    if missing and not dry_run:
        print(f"warning: still missing after prepare: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0
