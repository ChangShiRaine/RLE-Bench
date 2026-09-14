"""Regression guard: the task09 reference submission reproduces its
pinned score through the real verifier pipeline, passes basic validity,
is deterministic, and the scoring still discriminates (a submission
that abandons the passive design loses hardware credit)."""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from harness.codesign_oracle import build_reference_submission
from harness.codesign_scorer import score_submission
from harness.codesign_checkpoints import VALIDITY_PENALTIES

MANIFEST = Path(__file__).resolve().parents[1] / "tasks/task09/dev/data/oracle/manifest.json"
REFERENCE_REWARD = json.loads(MANIFEST.read_text())["variants"]["franka"]["reward"]


@pytest.fixture(scope="module")
def oracle_report(tmp_path_factory):
    sub = build_reference_submission(
        str(tmp_path_factory.mktemp("codesign_oracle") / "sub"))
    return score_submission(sub)


def test_reference_submission_contains_only_required_files(tmp_path):
    sub = build_reference_submission(str(tmp_path / "two_file_sub"))
    assert set(os.listdir(sub)) == {"lead.xml", "trim.py", "meshes"}


@pytest.mark.heavy
def test_reference_submission_matches_pinned_score(oracle_report):
    assert not oracle_report["gated"], oracle_report["gate_failed"]
    assert oracle_report["reward"] == pytest.approx(REFERENCE_REWARD, abs=1e-4)
    assert not oracle_report["errors"]


@pytest.mark.heavy
def test_reference_passes_validity_and_motion_clearance(oracle_report):
    assert set(oracle_report["deductions"]) == set(VALIDITY_PENALTIES)
    assert all(value == 0.0 for value in oracle_report["deductions"].values())
    assert oracle_report["checkpoints"]["C1.motion_clearance"] == 1.0
    assert oracle_report["motion_clearance"]["n_failed"] == 0
    for cid, score in oracle_report["checkpoints"].items():
        if cid.startswith("C1."):
            assert score == 1.0, cid
    assert oracle_report["stages"]["validity"] == pytest.approx(.10)
    assert oracle_report["validity_deduction"] == 0.0


@pytest.mark.heavy
def test_scoring_the_reference_submission_is_deterministic(tmp_path, oracle_report):
    """Same submission scored twice => identical reward and checkpoints."""
    sub = build_reference_submission(str(tmp_path / "sub"))
    repeated = score_submission(sub)
    assert oracle_report["reward"] == repeated["reward"]
    assert oracle_report["checkpoints"] == repeated["checkpoints"]


@pytest.mark.heavy
def test_stripping_the_passive_design_loses_hardware_credit(tmp_path,
                                                            oracle_report):
    """Discrimination guard: an arm with the springs removed still has to be
    a legal model, but must score materially worse on the hardware
    checkpoints than the gravity-compensated reference."""
    sub = build_reference_submission(str(tmp_path / "sub"))
    lead = os.path.join(sub, "lead.xml")
    tree = ET.parse(lead)
    for jnt in tree.getroot().iter("joint"):
        jnt.attrib.pop("stiffness", None)
        jnt.attrib.pop("springref", None)
    tree.write(lead)

    stripped = score_submission(sub)
    hw = [cid for cid in oracle_report["checkpoints"] if cid.startswith("H")]
    assert hw, "no H* hardware checkpoints found"
    ref_hw = sum(oracle_report["checkpoints"][c] for c in hw)
    bad_hw = sum(stripped["checkpoints"][c] for c in hw)
    assert bad_hw < ref_hw, (bad_hw, ref_hw)
    assert stripped["reward"] < oracle_report["reward"]


def test_render_submission_produces_views(tmp_path):
    """Post-eval renders (informational): the reference device yields both
    views. Skipped where no headless GL backend is available."""
    from harness.render import render_submission
    from rlebench.core.media import Media
    sub = build_reference_submission(str(tmp_path / "sub"))
    media = Media(tmp_path / "media")
    try:
        render_submission(media, os.path.join(sub, "lead.xml"), "franka")
    except Exception as e:
        pytest.skip(f"headless GL unavailable: {e}")
    index = media.close()
    assert index["files"] == ["franka_home.png", "franka_hold.png"]
    assert not index["skipped"]
    for name in index["files"]:
        assert os.path.getsize(tmp_path / "media" / name) > 10_000
