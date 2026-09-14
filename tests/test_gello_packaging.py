"""Packaging, follower kinematics and per-variant reference scores for task09."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import mujoco
import numpy as np
import pytest

from harness.variants import VARIANTS


REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tasks" / "task09"
ASSETS = REPO / "tasks" / "task09" / "harness" / "assets" / "gello_codesign"
FOLLOWERS = {
    "ur5e": REPO / "assets" / "robots" / "universal_robots_ur5e" / "ur5e.xml",
    "xarm7": REPO / "assets" / "robots" / "ufactory_xarm7" / "xarm7_nohand.xml",
}


def test_time_budget():
    config = tomllib.loads((TASK / "task.toml").read_text())
    assert config["agent"]["timeout_sec"] == 10800.0
    assert config["verifier"]["timeout_sec"] == 3600.0


@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_single_task_contract_binds_variant_artifacts(variant):
    instruction = (TASK / "instruction.md").read_text()
    verifier = (TASK / "tests" / "score_task.py").read_text()
    solution = (TASK / "solution" / "solve.sh").read_text()
    artifact = f"/logs/artifacts/{variant}"
    assert f"/workspace/{variant}" in instruction
    assert artifact in instruction
    assert f'"{variant}"' in verifier
    assert "/logs/artifacts" in verifier
    assert "/solution/payload/." in solution
    workspace = TASK / "environment" / "assets" / variant
    lead = workspace / "lead.xml"
    assert lead.is_file()
    assert np.all(mujoco.MjModel.from_xml_path(str(lead)).jnt_stiffness == 0.0)
    assert not (workspace / "lead_sref.xml").exists()
    assert "lead_sref.xml" not in instruction
    assert (TASK / "tests" / "Dockerfile").is_file()
    assert (TASK / "tests" / "models" / "gello_codesign" / variant / "lead_sref.xml").is_file()
    payload = TASK / "solution" / "payload" / variant
    assert (payload / "lead.xml").is_file()
    assert (payload / "trim.py").is_file()


@pytest.mark.parametrize("name", ("ur5e", "xarm7"))
def test_variant_registry_matches_canonical_follower(name):
    variant = VARIANTS[name]
    model = mujoco.MjModel.from_xml_path(str(FOLLOWERS[name]))
    assert model.njnt == variant.n_joints
    for index, expected in enumerate(variant.follower_chain):
        body_id = model.jnt_bodyid[index]
        np.testing.assert_allclose(model.body_pos[body_id], expected["pos"], atol=1e-9)
        # Unit quaternions q and -q encode the same frame.
        actual_quat = np.asarray(model.body_quat[body_id])
        expected_quat = np.asarray(expected["quat"], dtype=float)
        expected_quat /= np.linalg.norm(expected_quat)
        assert abs(float(actual_quat @ expected_quat)) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(model.jnt_axis[index], expected["axis"], atol=1e-9)
        np.testing.assert_allclose(model.jnt_range[index], expected["range"], atol=1e-5)


@pytest.mark.heavy
@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_variant_assets_and_reference_score(variant):
    code = r'''
import json
import tempfile
from harness import spec
from harness.codesign import device_buildable
from harness.codesign_oracle import build_reference_submission
from harness.codesign_scorer import score_submission
from harness.scenarios import compose_lead
from harness.validity import check_validity

root = f"tasks/task09/harness/assets/gello_codesign/{spec.VARIANT_NAME}"
checks = {}
for kind in ("lead_bare.xml", "lead_sref.xml", "lead_oracle.xml"):
    path = f"{root}/{kind}"
    validity = check_validity(
        path, compose=lambda p, seed=0: compose_lead(
            p, seed=seed, perturb=False))
    checks[kind] = dict(
        valid=validity.get("ok", False) if kind != "lead_bare.xml" else
              all(validity[k] for k in ("loads", "inertia_ok", "mass_ok", "link_mass_ok"))
              and validity["density"]["ok"] and validity["structure"]["ok"],
        motion_failures=validity["motion_clearance"]["n_failed"],
        buildable=device_buildable(path)["ok"],
        dof=compose_lead(path, perturb=False)[0].nv,
        fk=validity["structure"]["max_fk_err"],
    )
with tempfile.TemporaryDirectory() as directory:
    build_reference_submission(directory)
    report = score_submission(directory, variant=spec.VARIANT_NAME)
print(json.dumps(dict(checks=checks, reward=report["reward"], stages=report["stages"], gated=report["gated"])))
'''
    env = os.environ.copy()
    env["RLEBENCH_GELLO_VARIANT"] = variant
    # the child needs the family harness importable, same as conftest gives us
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "tasks" / "task09"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    output = subprocess.check_output(
        [sys.executable, "-c", code], cwd=REPO, env=env, text=True)
    result = json.loads(output)
    expected_dof = VARIANTS[variant].n_joints
    for kind, check in result["checks"].items():
        assert check["motion_failures"] == 0
        assert check["valid"]
        assert check["buildable"]
        assert check["dof"] == expected_dof
        assert check["fk"] < 0.002
    assert not result["gated"]
    manifest = json.loads((TASK / "dev/data/oracle/manifest.json").read_text())
    assert result["reward"] == pytest.approx(
        manifest["variants"][variant]["reward"], abs=1e-4)


@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_oracle_packaging_preserves_pinned_submission(variant, tmp_path):
    """The helper and staged solution must carry each variant's exact design."""
    from harness.codesign_oracle import build_reference_submission
    pinned = TASK / "dev/data/oracle" / variant
    built = Path(build_reference_submission(str(tmp_path / variant), variant=variant))
    for name in ("lead.xml", "trim.py"):
        assert (built / name).read_bytes() == (pinned / name).read_bytes()
        assert (TASK / "solution/payload" / variant / name).read_bytes() == (pinned / name).read_bytes()
    assert (ASSETS / variant / "lead_oracle.xml").read_bytes() == (pinned / "lead.xml").read_bytes()
    assert (ASSETS / variant / "trim_oracle.py").read_bytes() == (pinned / "trim.py").read_bytes()
