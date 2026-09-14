"""Pure configuration checks for task04's per-motion reward thresholds."""
from __future__ import annotations

import pytest

from harness import config

EXPECTED_MOTIONS = {
    "dance1_subject2",
    "fight1_subject2",
    "fallAndGetUp2_subject2",
    "run1_subject2",
    "sprint1_subject2",
}


def test_every_task_has_its_own_threshold_record():
    assert set(config.MOTION_THRESHOLDS) == EXPECTED_MOTIONS
    values = list(config.MOTION_THRESHOLDS.values())
    assert len({id(value) for value in values}) == 5


def test_uncalibrated_tasks_start_from_the_dance_placeholder():
    dance = config.MOTION_THRESHOLDS["dance1_subject2"]
    assert dance.calibrated
    for name in EXPECTED_MOTIONS - {"dance1_subject2"}:
        candidate = config.MOTION_THRESHOLDS[name]
        assert not candidate.calibrated
        assert candidate.jerk_cap == dance.jerk_cap
        assert candidate.oracle_ref_score == dance.oracle_ref_score
        assert candidate.null_ref_score == dance.null_ref_score


@pytest.mark.parametrize("name", sorted(EXPECTED_MOTIONS))
def test_thresholds_resolve_from_verifier_npz_path(name):
    path = f"/tests/assets/motions/{name}.npz"
    assert config.motion_name(path) == name
    assert config.thresholds_for_motion(path) is config.MOTION_THRESHOLDS[name]


def test_an_unknown_motion_fails_closed():
    with pytest.raises(KeyError, match="no task04 thresholds"):
        config.thresholds_for_motion("/tests/assets/motions/unknown.npz")
