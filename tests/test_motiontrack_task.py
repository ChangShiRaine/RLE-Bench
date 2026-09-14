"""The isolation boundary of the task04 images (CLAUDE.md invariant #2).

There is no build step to hide behind: the agent image's COPY list *is* the
boundary, so these read it directly. Ground truth — the scorer, the calibrated
thresholds, the calibration itself and the evaluation seeds — must never reach
the agent image, and no agent-facing file may name any of it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

TASK = Path("tasks/task04")
TEMPLATE = TASK / "_template"
SRC = Path("tasks/task04/harness")

GROUND_TRUTH_MODULES = ("scorer.py", "config.py", "calibrate.py", "eval_seeds.json")
FORBIDDEN_TOKENS = (
    "eval_seeds", "score_submission", "oracle_ref_score", "dev.calibrate",
    "GATE_CAP",
)

AGENT_FILES = (TEMPLATE / "instruction.md.in", TEMPLATE / "task.toml.in",
               TEMPLATE / "environment/Dockerfile",
               TEMPLATE / "environment/dev_eval.py",
               TEMPLATE / "solution/payload/train.py")

pytestmark = pytest.mark.skipif(not TASK.exists(), reason="task04 not present")


def agent_dockerfile() -> str:
    return (TEMPLATE / "environment/Dockerfile").read_text()


def copied_modules() -> set[str]:
    """Harness modules the agent image copies, from the Dockerfile itself."""
    return set(re.findall(r"tasks/task04/harness/(\w+\.\w+)", agent_dockerfile()))


def agent_visible_files():
    return AGENT_FILES + tuple(SRC / name for name in sorted(copied_modules()))


def test_agent_image_copies_no_ground_truth():
    leaked = copied_modules() & set(GROUND_TRUTH_MODULES)
    assert not leaked, f"agent image would ship ground truth: {sorted(leaked)}"


def test_agent_image_names_its_modules_individually():
    """Copying the package wholesale would sweep the scorer in the moment anyone
    adds it, so the COPY list must enumerate files, never the directory."""
    assert "COPY tasks/task04/harness /" not in agent_dockerfile()
    assert len(copied_modules()) >= 8


def test_every_copied_module_exists():
    missing = [m for m in copied_modules() if not (SRC / m).exists()]
    assert not missing, f"Dockerfile copies files that do not exist: {missing}"


def test_the_agent_gets_what_it_needs():
    """The boundary must not be so tight that the task is unusable."""
    needed = {"spec.py", "robot.py", "motion.py", "env.py", "mdp.py",
              "evaluator.py", "export.py", "progress.py", "metrics.py",
              "rotations.py"}
    assert needed <= copied_modules()


def test_verifier_image_gets_the_whole_harness():
    verifier = (TEMPLATE / "tests/Dockerfile").read_text()
    assert "COPY tasks/task04/harness /tests/harness" in verifier


def test_no_agent_visible_file_names_ground_truth():
    problems = []
    for path in agent_visible_files():
        text = path.read_text()
        problems += [f"{path}: {t!r}" for t in FORBIDDEN_TOKENS if t in text]
    assert not problems, problems


def test_no_agent_facing_file_leaks_an_evaluation_seed():
    seeds = json.loads((SRC / "eval_seeds.json").read_text())
    values = [str(seeds["smoke"])] + [str(s) for s in seeds["eval"]]
    problems = []
    for path in agent_visible_files():
        text = path.read_text()
        problems += [f"{path}: seed {v}" for v in values if v in text]
    assert not problems, problems


def test_dockerfile_pins_have_no_defaults():
    """A bare `docker build` must fail rather than silently fork the repo's pins."""
    for name in ("environment/Dockerfile", "tests/Dockerfile"):
        for line in (TEMPLATE / name).read_text().splitlines():
            if line.startswith("ARG "):
                assert "=" not in line, f"{name}: {line} carries a default"


def test_every_dockerfile_arg_is_a_real_pin():
    pins = {line.split("=", 1)[0].strip()
            for line in Path("sim/motiontrack/pins.env").read_text().splitlines()
            if line.strip() and not line.startswith("#") and "=" in line}
    for name in ("environment/Dockerfile", "tests/Dockerfile"):
        args = {line.split()[1] for line in (TEMPLATE / name).read_text().splitlines()
                if line.startswith("ARG ")}
        assert args <= pins, f"{name}: {sorted(args - pins)} not in pins.env"


EXPECTED_MATRIX = [
    ("01-dance", "dance1_subject2", "122:722"),
    ("02-fight", "fight1_subject2", "650:1250"),
    ("03-fall-and-get-up", "fallAndGetUp2_subject2", "1:601"),
    ("04-run", "run1_subject2", "3266:3866"),
    ("05-sprint", "sprint1_subject2", "1525:2125"),
]


def test_task04_is_a_five_task_matrix():
    rows = []
    for raw in (TASK / "motions.tsv").read_text().splitlines():
        if raw and not raw.startswith("#"):
            slug, motion, frames, _title = raw.split("\t")
            rows.append((slug, motion, frames))
    assert rows == EXPECTED_MATRIX
    assert (TASK / "build_tasks.py").is_file()


def test_templates_select_motion_at_runtime():
    task_toml = (TEMPLATE / "task.toml.in").read_text()
    instruction = (TEMPLATE / "instruction.md.in").read_text()
    assert 'MOTIONTRACK_MOTION = "/workspace/assets/motions/@@MOTION@@.npz"' in task_toml
    assert 'MOTIONTRACK_MOTION = "/tests/assets/motions/@@MOTION@@.npz"' in task_toml
    assert "@@MOTION@@" in instruction


def test_shared_images_copy_all_converted_motions():
    assert "COPY third_party/motiontrack/motions" in agent_dockerfile()
    verifier = (TEMPLATE / "tests/Dockerfile").read_text()
    assert "COPY third_party/motiontrack/motions" in verifier
