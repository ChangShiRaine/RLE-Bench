"""Emit task05's four nanoVLA subtasks from one shared template.

    python3 tasks/task05/build_subtasks.py --emit-all
    python3 tasks/task05/build_subtasks.py --emit 01-libero-open-design
    python3 tasks/task05/build_subtasks.py --list
    python3 tasks/task05/build_subtasks.py --check

The emitted ``NN-slug/`` directories are gitignored build output. Each subtask
names the agent image and encoder-bundle image of its `hf_bundle` and the
verifier base of its `simulator` (LIBERO for 01/02, RoboTwin for 03/04); the
per-subtask verifier images are built from the emitted tests/ contexts.
`subtasks.toml` carries every scalar the template substitutes;
`subtasks/<slug>/` carries what genuinely differs per subtask (instruction,
scorer, reference recipe, compose overlays, verifier data) and is copied
verbatim.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "_template"
OVERLAYS = HERE / "subtasks"
MANIFEST = HERE / "subtasks.toml"

VERBATIM = ("tests/test.sh", "solution/solve.sh")
OVERLAY_REQUIRED = ("instruction.md", "environment/docker-compose.yaml",
                    "tests/docker-compose.yaml", "tests/score_task.py",
                    "solution/solution.py")

sys.path.insert(0, str(HERE.parents[1]))
from rlebench import taskgen  # noqa: E402


def subtasks() -> dict[str, dict]:
    with MANIFEST.open("rb") as fh:
        matrix = tomllib.load(fh)
    if len(matrix) != 4:
        raise SystemExit("subtasks.toml must define four subtasks")
    return matrix


def tokens(slug: str, values: dict) -> dict[str, str]:
    def array(items: list[str]) -> str:
        return ", ".join(f'"{item}"' for item in items)

    return {
        "@@SLUG@@": slug,
        "@@DESCRIPTION@@": values["description"],
        "@@KEYWORDS@@": array(values["keywords"]),
        "@@TAGS@@": array(values["tags"]),
        "@@DIFFICULTY_EXPLANATION@@": values["difficulty_explanation"],
        "@@CPUS@@": str(values["cpus"]),
        "@@MEMORY_MB@@": str(values["memory_mb"]),
        "@@STORAGE_MB@@": str(values["storage_mb"]),
        "@@GPUS@@": str(values["gpus"]),
        "@@VERIFIER_TIMEOUT_SEC@@": str(values["verifier_timeout_sec"]),
        "@@AGENT_TIMEOUT_SEC@@": str(values["agent_timeout_sec"]),
        "@@README_SUMMARY@@": values["readme_summary"].rstrip("\n"),
        "@@README_ASSETS@@": values["readme_assets"].rstrip("\n"),
        "@@AGENT_IMAGE@@": f"rlebench-task05-agent-{values['hf_bundle']}:dev",
        "@@HF_IMAGE@@": f"rlebench-task05-hf-{values['hf_bundle']}:dev",
        "@@VERIFIER_BASE_IMAGE@@": ("rlebench-task05-robotwin-verifier-base:dev"
                                    if values.get("simulator") == "robotwin"
                                    else "rlebench-task05-libero-verifier-base:dev"),
    }


def rendered(slug: str, values: dict) -> dict[str, str]:
    mapping = tokens(slug, values)
    return {
        "task.toml": taskgen.render((TEMPLATE / "task.toml.in").read_text(), mapping),
        "README.md": taskgen.render((TEMPLATE / "README.md.in").read_text(), mapping),
        "tests/Dockerfile": taskgen.render((TEMPLATE / "tests" / "Dockerfile.in").read_text(), mapping),
    }


def overlay_files(slug: str) -> list[str]:
    root = OVERLAYS / slug
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    missing = [rel for rel in OVERLAY_REQUIRED if rel not in files]
    if missing:
        raise SystemExit(f"{slug}: overlay is missing {missing}")
    return files


def emit(slug: str, values: dict) -> Path:
    dest = HERE / slug
    if dest.exists():
        shutil.rmtree(dest)
    overlay_files(slug)
    shutil.copytree(OVERLAYS / slug, dest)
    for rel in VERBATIM:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEMPLATE / rel, dest / rel)
    for rel, text in rendered(slug, values).items():
        (dest / rel).write_text(text)
    return dest


def differences() -> list[str]:
    problems: list[str] = []
    matrix = subtasks()
    actual = {path.name for path in HERE.glob("[0-9][0-9]-*") if path.is_dir()}
    if actual != set(matrix):
        problems.append(f"emitted subtask directories: expected {sorted(matrix)}, got {sorted(actual)}")

    for slug, values in matrix.items():
        dest = HERE / slug
        if not dest.is_dir():
            continue
        for rel, wanted in rendered(slug, values).items():
            path = dest / rel
            if not path.is_file():
                problems.append(f"{slug}: missing {rel}")
            elif path.read_text() != wanted:
                problems.append(f"{slug}: stale {rel}")
            elif taskgen.unrendered(path.read_text()):
                problems.append(f"{slug}: unrendered token in {rel}")
        for source_root, rels in ((TEMPLATE, VERBATIM), (OVERLAYS / slug, overlay_files(slug))):
            for rel in rels:
                target = dest / rel
                if not target.is_file():
                    problems.append(f"{slug}: missing {rel}")
                elif target.read_bytes() != (source_root / rel).read_bytes():
                    problems.append(f"{slug}: stale {rel}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-all", action="store_true")
    group.add_argument("--emit", metavar="SLUG")
    group.add_argument("--list", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    matrix = subtasks()

    if args.list:
        for slug, values in matrix.items():
            print(f"{slug}\tgpus={values['gpus']}\tverifier_timeout={values['verifier_timeout_sec']}")
        return
    if args.emit_all:
        for slug, values in matrix.items():
            print(emit(slug, values).relative_to(HERE.parent))
        return
    if args.emit:
        if args.emit not in matrix:
            parser.error(f"unknown subtask {args.emit!r}; choose: {', '.join(matrix)}")
        print(emit(args.emit, matrix[args.emit]).relative_to(HERE.parent))
        return

    problems = differences()
    if problems:
        raise SystemExit("task05 matrix is stale:\n  - " + "\n  - ".join(problems))
    print("task05 matrix ok")


if __name__ == "__main__":
    main()
