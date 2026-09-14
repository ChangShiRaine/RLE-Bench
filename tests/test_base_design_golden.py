"""Reference base-design and dynamics regressions.

One full scoring battery covers the reference chassis without a shelf policy.
Submitted shelf control is covered by the controller suite and the container
Oracle release gate; no shelf controller runs in this suite.
"""
import os
import xml.etree.ElementTree as ET

import pytest

# A module-scoped fixture shares one full scoring battery across assertions.
# The PR lane skips `heavy`; heavy-nightly.yml runs the heavy tests.
pytestmark = pytest.mark.heavy

HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..")
REFERENCE = os.path.join(REPO, "tasks", "task08", "reference", "robot.xml")
ARM_REF = os.path.join(REPO, "assets", "robots",
                       "franka_emika_panda", "panda_nohand.xml")

def _staged(dest_dir):
    """Copy the reference design with meshdir resolved for its new location
    (tasks/task08/build_assets.py does the same for the shipped payload)."""
    tree = ET.parse(REFERENCE)
    tree.getroot().find("compiler").set("meshdir", os.path.abspath(
        os.path.join(REPO, "assets", "robots",
                     "franka_emika_panda", "assets")))
    out = os.path.join(dest_dir, "robot.xml")
    tree.write(out)
    return out


@pytest.fixture(scope="module")
def reference_report(tmp_path_factory):
    from harness.base_design.scorer import score_submission

    sub = str(tmp_path_factory.mktemp("reference_sub"))
    _staged(sub)
    return score_submission(sub, arm_reference_xml=ARM_REF)


OPEN_ENDED = "S3.design_efficiency"
SHELF_CHECKPOINTS = {"S7.pick_compat", "S7.payload_margin"}


def test_reference_base_design_scores_at_its_bar(reference_report):
    r = reference_report
    assert not r["gated"], f"reference tripped the gate: {r['gate_failed']}"
    # The shelf pair contributes zero without a controller (0.15 total weight).
    assert r["reward"] >= 0.64, f"reward {r['reward']}: {r['checkpoints']}"
    assert all(r["checkpoints"][cid] == 0.0 for cid in SHELF_CHECKPOINTS)


def test_reference_design_aces_every_bounded_checkpoint(reference_report):
    cp = reference_report["checkpoints"]
    weak = {cid: s for cid, s in cp.items()
            if cid != OPEN_ENDED and cid not in SHELF_CHECKPOINTS
            and s < 0.99}
    assert not weak, f"below full credit: {weak} (report: {cp})"


def test_reference_design_earns_real_design_efficiency_credit(reference_report):
    """Open-ended, so not full credit — but a collapse to ~0 would mean the
    reference is at the budget limits, i.e. the envelope has drifted."""
    score = reference_report["checkpoints"][OPEN_ENDED]
    assert 0.05 <= score < 1.0, score


def test_reference_design_never_tips(reference_report):
    for sid, per_seed in reference_report["scenario_pass"].items():
        assert all(per_seed.values()), f"scenario {sid} failed: {per_seed}"
