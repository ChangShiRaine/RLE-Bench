"""Degenerate submissions must not score.

Each probe is a policy an agent could plausibly reach for that produces a number
without solving the task. They are scored through the real scorer on the real
hidden seeds — if one of these ever climbs, the reward function has a hole.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from harness import export, robot, scorer, spec  # noqa: E402

ROBOT_DIR = "third_party/motiontrack/unitree_g1"
NPZ = "third_party/motiontrack/motions/dance1_subject2.npz"

pytestmark = pytest.mark.skipif(
    not os.path.exists(NPZ), reason="run `make sim-motiontrack` first")


def linear_policy(weight: np.ndarray, bias: np.ndarray, path: str) -> str:
    layer = torch.nn.Linear(spec.OBS_DIM, spec.N_ACTIONS)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    export.export(layer, path)
    return path


def score(path: str) -> dict:
    return scorer.score_submission(path, ROBOT_DIR, NPZ)


@pytest.fixture(scope="module")
def parrot(tmp_path_factory):
    """Command the reference joint angles directly.

    The reference pose is literally the first 29 numbers of the observation, so
    one linear layer reproduces it exactly: action = (ref - default) / scale.
    This is the most obvious thing to try, and it is a trap — at BeyondMimic's
    compliant gains an open-loop joint tracker has no balance feedback and folds
    in about a second. The reward must not pay for the attempt.
    """
    weight = np.zeros((spec.N_ACTIONS, spec.OBS_DIM))
    command = spec.obs_slice("command")
    for i in range(spec.N_ACTIONS):
        weight[i, command.start + i] = 1.0 / robot.ACTION_SCALE[i]
    bias = -robot.DEFAULT_JOINT_POS / robot.ACTION_SCALE
    path = str(tmp_path_factory.mktemp("parrot"))
    return score(linear_policy(weight, bias, path))


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """Hold the default pose forever: zero actions, and perfectly smooth."""
    path = str(tmp_path_factory.mktemp("frozen"))
    return score(linear_policy(np.zeros((spec.N_ACTIONS, spec.OBS_DIM)),
                               np.zeros(spec.N_ACTIONS), path))


@pytest.fixture(scope="module")
def saturated(tmp_path_factory):
    """Slam every joint to its limit — maximum motion, zero tracking."""
    path = str(tmp_path_factory.mktemp("saturated"))
    return score(linear_policy(np.zeros((spec.N_ACTIONS, spec.OBS_DIM)),
                               np.full(spec.N_ACTIONS, 50.0), path))


def test_parroting_the_reference_does_not_score(parrot):
    assert parrot["reward"] < 0.15
    assert parrot["survived_s"] < 5.0, "an open-loop joint tracker cannot balance"


def test_a_frozen_policy_scores_nothing(frozen):
    """Smoothness must never be payable as credit: a frozen policy is perfectly
    smooth and completely useless."""
    assert frozen["reward"] == 0.0
    assert frozen["action_jerk"] == 0.0


def test_saturating_the_actuators_scores_nothing(saturated):
    assert saturated["reward"] < 0.05


def test_falling_early_beats_nothing(frozen, parrot):
    """Both fail, but the one that survives longer and tracks better must rank
    higher — the ordering is what makes the metric a gradient rather than a cliff."""
    assert parrot["tracking_score"] >= frozen["tracking_score"]


def test_the_gate_rejects_a_nondeterministic_policy(tmp_path):
    """A graph with sampling in it cannot be scored: the same seed must reproduce
    the same episode, or the battery measures noise."""
    class Noisy(torch.nn.Module):
        def forward(self, obs):
            return obs[:, :spec.N_ACTIONS] + torch.randn(1, spec.N_ACTIONS)

    export.export(Noisy(), str(tmp_path))
    report = score(str(tmp_path))
    assert not report["gate"]["deterministic"]
    assert report["reward"] <= 0.1, "a failed gate must cap the reward"


def test_an_oversized_policy_is_rejected_outright(tmp_path):
    big = torch.nn.Sequential(
        torch.nn.Linear(spec.OBS_DIM, 4000), torch.nn.ELU(),
        torch.nn.Linear(4000, 3000), torch.nn.ELU(),
        torch.nn.Linear(3000, spec.N_ACTIONS))
    export.export(big, str(tmp_path))
    report = score(str(tmp_path))
    assert report["reward"] == 0.0
    assert any("budget" in e for e in report["errors"])
