"""Emit three tabletop tasks, the pocket cube, and hidden-COM.

Run with --emit-all to rebuild generated task directories and image contexts.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "_template"
IMAGE_DIR = HERE / "image"

sys.path.insert(0, str(HERE))

from tabletop.budgets import INTERACTION_STEPS  # noqa: E402

import build_assets  # noqa: E402
import build_hidden_com  # noqa: E402
import build_pocket  # noqa: E402

BLURB = "RGB cameras and robot proprioception through a metered control client."

IMAGE = "rlebench-task03-agent:dev"

# (env class, slug goal phrase, task notes spliced into the instruction)
SUBTASKS: tuple[tuple[str, str, str], ...] = (
    ("TowerMaxHeight",
     "stack mixed pieces into the tallest stable tower",
     "Some pieces cannot be stacked on; choose which pieces to use and in what "
     "order. Leave the completed tower standing on the table without support "
     "from the robot. Retract the arm before calling `finish()`."),
    ("CantileverOverhang",
     "extend a stack of identical blocks past the table edge",
     "Build a stable stack extending beyond any table edge. Leave it supported "
     "by the table without support from the robot. Retract the arm before "
     "calling `finish()`."),
    ("BalanceCoins",
     "find the heavy cube with the balance and deliver it to the answer mat",
     "Exactly one of the visually identical cubes is heavier. The attempt "
     "succeeds when that cube -- and only it -- rests on the green answer mat. "
     "Use as few weighings as possible: among successful attempts, fewer "
     "weighings earn a higher score. Plan comparisons before moving cubes. "
     "Observations from observe(), "
     "reset(), step(), and move() include cube_positions (coin_0 through coin_8 "
     "centers) and pan_positions (left and right upper-surface centers). "
     "All coordinates are live world-frame xyz in meters. Cube IDs persist "
     "across movement and reset; they do not encode weight. The enlarged "
     "pans have diameter 0.225 m and centers 0.5 m apart when level."),

)


def subtask_slug(index: int, task: str) -> str:
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", task).lower()
    return f"{index:02d}-{kebab}"


def subtasks() -> list[tuple[int, str, str, str, str]]:
    return [(i, task, subtask_slug(i, task), goal, note)
            for i, (task, goal, note) in enumerate(SUBTASKS, 1)]


# -- emitting -----------------------------------------------------------------

def emit_image() -> Path:
    dest = IMAGE_DIR
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(TEMPLATE / "image", dest)
    build_assets.stage(dest / "payload_private", build_assets.DAEMON_MODULES,
                       tabletop=True)
    build_assets.stage_core(dest / "payload_private")
    build_assets.stage(dest / "payload_agent", build_assets.AGENT_MODULES)
    problems = build_assets.check(dest / "payload_agent")
    if problems:
        raise SystemExit(f"BOUNDARY VIOLATION in image/: {problems}")
    return dest


def render(text: str, task: str, slug: str, goal: str,
           note: str) -> str:
    for token, value in (
        ("@@LEVEL@@", "L1"),
        ("@@SLUG@@", slug),
        ("@@TASK@@", task),
        ("@@INTERACTION_STEPS@@", str(INTERACTION_STEPS[task])),
        ("@@GOAL@@", goal),
        ("@@TASK_NOTES@@", note),
        ("@@LEVEL_BLURB@@", BLURB),
    ):
        text = text.replace(token, value)
    return text


def emit_task(index: int, task: str, slug: str, goal: str,
              note: str) -> Path:
    dest = HERE / slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    shutil.copytree(TEMPLATE / "environment", dest / "environment")
    shutil.copytree(TEMPLATE / "tests", dest / "tests")

    args = (task, slug, goal, note)
    (dest / "task.toml").write_text(
        render((TEMPLATE / "task.toml.in").read_text(), *args))

    shutil.copytree(TEMPLATE / "solution", dest / "solution",
                    ignore=shutil.ignore_patterns("actions"))
    actions = TEMPLATE / "solution/actions" / f"{task}.npz"
    if actions.is_file():
        shutil.copy2(actions, dest / "solution/actions.npz")
    body = render((TEMPLATE / "instruction.md.in").read_text(), *args)
    (dest / "instruction.md").write_text(body)
    return dest


def emit_all() -> list[Path]:
    emit_image()
    return [emit_task(*entry) for entry in subtasks()] + [build_pocket.emit(), build_hidden_com.emit()]


# -- checking -----------------------------------------------------------------

def differences() -> list[str]:
    problems: list[str] = []
    expected = subtasks()
    ctx = IMAGE_DIR
    if not (ctx / "Dockerfile").is_file():
        problems.append("image/ has no build context; run --emit-all")
    else:
        problems += [f"image/: {p}"
                     for p in build_assets.check(ctx / "payload_agent")]

    present = sorted(p.name for p in HERE.glob("[0-9][0-9]-*")
                     if p.is_dir())
    if present != sorted([s for _, _, s, _, _ in expected] + [build_pocket.SLUG, build_hidden_com.SLUG]):
        problems.append(f"tasks/task03/ holds {present}; expected "
                        f"{sorted([s for _, _, s, _, _ in expected] + [build_pocket.SLUG, build_hidden_com.SLUG])}")
        return problems

    for _, task, slug, _, _ in expected:
        toml = (HERE / slug / "task.toml").read_text()
        for want in (f'"rlebench/task03-{slug}"',
                     f'"{IMAGE}"',
                     f'RLEBENCH_TASK = "{task}"'):
            if want not in toml:
                problems.append(f"{slug}/task.toml is missing {want}")
        if "@@" in toml:
            problems.append(f"{slug}/task.toml has unrendered "
                            f"placeholders")
        instr = (HERE / slug / "instruction.md")
        if not instr.is_file():
            problems.append(f"{slug} has no instruction")
        elif "@@" in instr.read_text():
            problems.append(f"{slug} instruction has unrendered "
                            f"placeholders")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit-all", action="store_true", help="emit everything")
    ap.add_argument("--check", action="store_true", help="verify without writing")
    ap.add_argument("--list", action="store_true", help="show the matrix")
    args = ap.parse_args()

    if args.check and args.emit_all:
        ap.error("--emit-all writes, --check verifies; run them as two commands")

    if args.list:
        print(f"{len(SUBTASKS)} shared subtasks, image {IMAGE}\n")
        for i, task, slug, goal, _note in subtasks():
            print(f"  {slug:28s} {task:20s} {goal}")
        print(f"  {build_pocket.SLUG:28s} physical 2x2 cube, image {build_pocket.IMAGE}")
        print(f"  {build_hidden_com.SLUG:28s} single-stage, image {build_hidden_com.IMAGE}")
        return 0

    if args.check:
        problems = differences()
        if problems:
            print("TASK03 MATRIX OUT OF SYNC:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"matrix ok: {len(SUBTASKS)} shared subtasks + pocket cube + hidden-COM")
        return 0

    if args.emit_all:
        emitted = emit_all()
        print(f"emitted image/ + {len(emitted)} tasks (tabletop + pocket cube + hidden-COM)")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
