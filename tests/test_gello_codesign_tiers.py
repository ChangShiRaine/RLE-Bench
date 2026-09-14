"""Task09 performance contract: the co-designed reference exceeds each
ablation on the weighted performance stages, and each failure signature
names the missing ingredient. Runs the complete hidden battery; the tiers
are built by tasks/task09/dev/ablations/run_ablations.py.
"""
import os
import sys

import pytest

from harness import codesign_variant_config as ccfg
from harness.codesign_scorer import score_submission

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tasks", "task09", "dev", "ablations"))
from run_ablations import build_ablations  # noqa: E402

pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("ablations"))
    subs = build_ablations(root)
    return {name: score_submission(d) for name, d in subs.items()}


def performance_score(report):
    # Interference credit differs with added structures, independently of
    # the hardware/software performance discrimination tested here.
    return sum(report["stages"][s] for s in ("hardware", "software", "codesign"))


def test_passive_only_misses_software(reports):
    r = reports["passive_only"]
    assert r["stages"]["software"] == 0.0
    assert r["stages"]["hardware"] == reports["full_reference"]["stages"]["hardware"]
    assert performance_score(r) <= performance_score(reports["full_reference"]) - 0.10


def test_active_only_misses_hardware(reports):
    r = reports["active_only"]
    assert r["stages"]["hardware"] <= 0.02          # crumbs only
    assert r["stages"]["software"] == reports["full_reference"]["stages"]["software"]
    assert r["stages"]["codesign"] <= 0.02
    assert performance_score(r) <= performance_score(reports["full_reference"]) - 0.19


def test_no_adapt_misses_identification(reports):
    r = reports["no_adapt"]
    assert r["checkpoints"]["S2.adapted_hold"] == 0.0
    assert r["checkpoints"]["S2.adapted_backdrive"] == 0.0
    assert r["stages"]["software"] == 0.0
    assert "S1.nominal_feedforward" not in r["checkpoints"]
    assert performance_score(r) <= performance_score(reports["full_reference"]) - 0.07


def test_lazy_hardware_caught_by_h_and_robustness(reports):
    r = reports["lazy_hardware"]
    full = reports["full_reference"]
    assert r["stages"]["hardware"] <= full["stages"]["hardware"] - 0.02
    # Lower inertia can improve poke recovery even while passive balance
    # and available servo headroom are worse.
    assert r["checkpoints"]["B3.headroom"] < full["checkpoints"]["B3.headroom"]
    assert performance_score(r) <= performance_score(full) - 0.03


def test_clamp_scores_low_through_the_continuous_terms(reports):
    """Over-stiff springs can settle at the wrong pose, but are neither
    balanced nor backdrivable and lose the continuous physics terms."""
    r = reports["clamp_x8"]
    assert not r["gated"]
    assert r["stages"]["hardware"] <= 0.02
    assert r["stages"]["codesign"] <= 0.02
    assert performance_score(r) <= performance_score(reports["full_reference"]) - 0.20


def test_unrunnable_device_loses_unmeasurable_credit_without_a_gate(tmp_path):
    """An invalid interface loses its own-arm metrics, with no extra cap."""
    import xml.etree.ElementTree as ET

    from harness import spec as gspec
    from harness.codesign_oracle import (CODESIGN_MODELS,
                                                 build_reference_submission)
    broken = tmp_path / "broken.xml"
    tree = ET.parse(os.path.join(CODESIGN_MODELS, gspec.VARIANT_NAME,
                                 "lead_oracle.xml"))
    for joint in tree.getroot().iter("joint"):
        if joint.get("name") == "lead_joint1":
            joint.set("name", "not_the_contract")
    tree.write(broken)
    subdir = build_reference_submission(str(tmp_path / "sub"), str(broken))
    saved = ccfg.HIDDEN_SEEDS
    ccfg.HIDDEN_SEEDS = (311,)
    try:
        r = score_submission(subdir)
    finally:
        ccfg.HIDDEN_SEEDS = saved
    assert not r["gated"]
    assert r["gate_failed"] == []
    assert r["checkpoints"]["C1.structure"] == 0.0
    assert r["deductions"]["C1.structure"] > 0.0
    assert r["stages"]["hardware"] == r["stages"]["codesign"] == 0.0
    assert r["validity_deduction"] <= .10
    assert r["reward"] == pytest.approx(max(0.0, r["raw_total"]))


def test_performance_tier_ordering(reports):
    order = ["full_reference", "lazy_hardware", "no_adapt",
             "passive_only", "active_only", "clamp_x8"]
    rewards = [performance_score(reports[n]) for n in order]
    assert rewards == sorted(rewards, reverse=True)
    assert rewards[0] - rewards[1] >= 0.03
    assert all(performance_score(reports[n]) < performance_score(reports["full_reference"])
               for n in order[1:])


def test_beating_the_reference_keeps_earning(reports):
    """The continuous contract: a hypothetical better design scores higher —
    verified at the checkpoint level via the credit curve directly."""
    from harness.codesign_checkpoints import _cont_down
    full = reports["full_reference"]
    h1 = full["checkpoints"]["H1.device_residual"]
    twice = _cont_down(full["battery"]["device_residual"] / 2.0,
                       ccfg.H1_RESIDUAL_BAR, headroom=1.0)
    # Combined hold is already saturated; residual torque still has headroom.
    assert twice > h1 * 1.5
