"""Enforce Pyflakes on maintained sources and the assembled task08 controller."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

from pyflakes import api
from pyflakes.reporter import Reporter

from .repo import REPO

CONTROLLER = "tasks/task08/solution/controller.py"


def source_paths(root: Path) -> list[Path]:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "--", "rlebench", "sim", "tasks", "tests"], cwd=root,
    ).decode().split("\0")
    return [root / path for path in sorted(set(paths))
            if path.endswith(".py") and path != CONTROLLER
            and (root / path).is_file()]


def controller_source(root: Path) -> str:
    spec = importlib.util.spec_from_file_location(
        "task08_lint_stager", root / "tasks/task08/build_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solution_controller_source()


def check(root: Path = REPO) -> int:
    paths = source_paths(root)
    warnings = sum(api.checkPath(str(path)) for path in paths)
    source = controller_source(root)
    reporter = Reporter(sys.stdout, sys.stderr)
    warnings += api.check(source, f"{CONTROLLER} (assembled)", reporter)
    if not warnings:
        print(f"Pyflakes passed: {len(paths)} source files and the task08 controller.")
    return int(bool(warnings))


if __name__ == "__main__":
    raise SystemExit(check())
