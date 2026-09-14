"""Regression tests for task09's agent/verifier boundary."""

import inspect

import numpy as np

from harness import (balance_codesign_public, scenarios,
                             scenarios_public)


PRIVATE_SCENARIO_NAMES = {
    "GelloEnvelope",
    "compose_instance",
    "fixed_hold_configs",
    "hold_configs",
    "sample_workspace",
    "teleop_path",
}


def test_public_scenarios_are_nominal_only():
    assert list(inspect.signature(scenarios_public.compose_lead).parameters) == [
        "lead_xml_path"
    ]
    for name in PRIVATE_SCENARIO_NAMES:
        assert not hasattr(scenarios_public, name)


def test_public_workspace_grid_matches_verifier():
    np.testing.assert_allclose(
        scenarios_public.workspace_grid(), scenarios.workspace_grid())


def test_public_balance_module_omits_simulation_scenarios():
    assert hasattr(balance_codesign_public, "balance_quality")
    assert hasattr(balance_codesign_public, "operator_effort")
    assert not hasattr(balance_codesign_public, "hold_sim")
    assert not hasattr(balance_codesign_public, "sag")
