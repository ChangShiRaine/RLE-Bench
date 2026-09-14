from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCORE_TASK = REPO / "tasks" / "task08" / "tests" / "score_task.py"


def _load_score_task():
    spec = importlib.util.spec_from_file_location(
        "task08_score_task_for_test", SCORE_TASK
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_staging_ignores_dangling_submitted_assets_symlink(tmp_path):
    score_task = _load_score_task()

    artifacts = tmp_path / "artifacts"
    staging = tmp_path / "staging"
    canonical_assets = tmp_path / "canonical-assets"
    artifacts.mkdir()
    canonical_assets.mkdir()

    (artifacts / "robot.xml").write_text("<mujoco/>\n")
    (artifacts / "assets").symlink_to("/workspace/assets")
    (canonical_assets / "canonical.mesh").write_text("trusted\n")

    score_task.ARTIFACTS = str(artifacts)
    score_task.STAGING = str(staging)
    score_task.CANONICAL_ASSETS = str(canonical_assets)
    score_task._stage_submission()

    assert (staging / "robot.xml").read_text() == "<mujoco/>\n"
    assert not (staging / "assets").is_symlink()
    assert (staging / "assets" / "canonical.mesh").read_text() == "trusted\n"


def test_staging_replaces_submitted_assets_directory(tmp_path):
    score_task = _load_score_task()

    artifacts = tmp_path / "artifacts"
    submitted_assets = artifacts / "assets"
    staging = tmp_path / "staging"
    canonical_assets = tmp_path / "canonical-assets"
    submitted_assets.mkdir(parents=True)
    canonical_assets.mkdir()

    (submitted_assets / "untrusted.mesh").write_text("untrusted\n")
    (canonical_assets / "canonical.mesh").write_text("trusted\n")

    score_task.ARTIFACTS = str(artifacts)
    score_task.STAGING = str(staging)
    score_task.CANONICAL_ASSETS = str(canonical_assets)
    score_task._stage_submission()

    assert not (staging / "assets" / "untrusted.mesh").exists()
    assert (staging / "assets" / "canonical.mesh").read_text() == "trusted\n"


@pytest.mark.parametrize("name", ["robot.xml", "controller.py"])
def test_staging_rejects_symlink_entry_files(tmp_path, name):
    score_task = _load_score_task()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("private")
    (artifacts / name).symlink_to(secret)
    score_task.ARTIFACTS = str(artifacts)
    score_task.STAGING = str(tmp_path / "staging")
    with pytest.raises(OSError):
        score_task._stage_submission()


def test_all_arm_variants_score_all_twelve_public_targets():
    from harness.base_design.pickscene import scored_target_schedule
    from harness.sim.arm_variants import trusted_arm_specs

    panda_ref = str(REPO / "assets" / "robots"
                    / "franka_emika_panda" / "panda_nohand.xml")
    specs = trusted_arm_specs(panda_ref)
    assert set(specs) == {"panda", "ur5e", "xarm7"}

    for arm in specs.values():
        schedule = scored_target_schedule(0.40, arm)
        assert len(schedule) == 12
    assert [(index, name) for index, name, _ in schedule] == [
        (0, "low_front_left"), (1, "low_front_right"),
        (2, "low_back_left"), (3, "low_back_right"),
        (4, "middle_front_left"), (5, "middle_front_right"),
        (6, "middle_back_left"), (7, "middle_back_right"),
        (8, "high_front_left"), (9, "high_front_right"),
        (10, "high_back_left"), (11, "high_back_right"),
    ]
