"""Emit task04 as five independent Harbor tasks from one shared template.

    python3 tasks/task04/build_tasks.py --emit-all
    python3 tasks/task04/build_tasks.py --emit 02-fight
    python3 tasks/task04/build_tasks.py --list
    python3 tasks/task04/build_tasks.py --check

The emitted ``NN-slug/`` directories are gitignored build output.  Each task uses
one shared pair of prebuilt images; only the scored motion path and prose vary.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "_template"
MANIFEST = HERE / "motions.tsv"
TOKENS = ("@@SLUG@@", "@@MOTION@@", "@@TITLE@@")

sys.path.insert(0, str(HERE.parents[1]))
from rlebench import taskgen  # noqa: E402


@dataclass(frozen=True)
class MotionTask:
    slug: str
    motion: str
    frames: str
    title: str


def tasks() -> list[MotionTask]:
    result: list[MotionTask] = []
    for lineno, raw in enumerate(MANIFEST.read_text().splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 4:
            raise SystemExit(f"{MANIFEST}:{lineno}: expected four tab-separated fields")
        result.append(MotionTask(*fields))
    if len(result) != 5 or len({task.slug for task in result}) != len(result):
        raise SystemExit("motions.tsv must define five tasks with unique slugs")
    return result


def render(text: str, task: MotionTask) -> str:
    return taskgen.render(text, {
        "@@SLUG@@": task.slug,
        "@@MOTION@@": task.motion,
        "@@TITLE@@": task.title,
    })


def emit(task: MotionTask) -> Path:
    dest = HERE / task.slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    for name in ("environment", "tests", "solution"):
        shutil.copytree(TEMPLATE / name, dest / name,
                        ignore=shutil.ignore_patterns("Dockerfile"))
    (dest / "task.toml").write_text(
        render((TEMPLATE / "task.toml.in").read_text(), task))
    (dest / "instruction.md").write_text(
        render((TEMPLATE / "instruction.md.in").read_text(), task))
    return dest


def differences() -> list[str]:
    problems: list[str] = []
    expected = {task.slug for task in tasks()}
    actual = {path.name for path in HERE.glob("[0-9][0-9]-*") if path.is_dir()}
    if actual != expected:
        problems.append(f"generated task directories: expected {sorted(expected)}, got {sorted(actual)}")

    for task in tasks():
        dest = HERE / task.slug
        required = (
            "task.toml", "instruction.md", "environment/docker-compose.yaml",
            "tests/test.sh", "tests/score_task.py", "solution/solve.sh",
            "solution/payload/train.py",
        )
        for rel in required:
            if not (dest / rel).is_file():
                problems.append(f"{task.slug}: missing {rel}")
        if not dest.exists():
            continue
        expected_text = {
            "task.toml": render((TEMPLATE / "task.toml.in").read_text(), task),
            "instruction.md": render((TEMPLATE / "instruction.md.in").read_text(), task),
        }
        for rel, wanted in expected_text.items():
            path = dest / rel
            if path.is_file() and path.read_text() != wanted:
                problems.append(f"{task.slug}: stale {rel}")
        for rel in ("environment/docker-compose.yaml", "tests/test.sh",
                    "tests/score_task.py", "solution/solve.sh",
                    "solution/payload/train.py"):
            source = TEMPLATE / rel
            target = dest / rel
            if target.is_file() and target.read_bytes() != source.read_bytes():
                problems.append(f"{task.slug}: stale {rel}")
        for path in dest.rglob("Dockerfile"):
            problems.append(f"{task.slug}: generated task must use prebuilt images, found {path}")
        for path in (dest / "task.toml", dest / "instruction.md"):
            if path.is_file() and any(token in path.read_text() for token in TOKENS):
                problems.append(f"{task.slug}: unrendered token in {path.name}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-all", action="store_true")
    group.add_argument("--emit", metavar="SLUG")
    group.add_argument("--list", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    matrix = tasks()

    if args.list:
        for task in matrix:
            print(f"{task.slug}	{task.motion}	{task.frames}")
        return
    if args.emit_all:
        for task in matrix:
            print(emit(task).relative_to(HERE.parent))
        return
    if args.emit:
        selected = next((task for task in matrix if task.slug == args.emit), None)
        if selected is None:
            parser.error(f"unknown slug {args.emit!r}; choose: {', '.join(t.slug for t in matrix)}")
        print(emit(selected).relative_to(HERE.parent))
        return

    problems = differences()
    if problems:
        raise SystemExit("task04 matrix is stale:\n  - " + "\n  - ".join(problems))
    print("task04 matrix ok")


if __name__ == "__main__":
    main()
