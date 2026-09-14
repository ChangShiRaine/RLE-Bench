"""Equal performance rewards and penalty-only basic validity."""
import pytest

from harness.codesign_checkpoints import (
    CHECKPOINTS_CD, VALIDITY_PENALTIES, aggregate_codesign,
)


def perfect_scores():
    return {cp.id: 1.0 for cp in CHECKPOINTS_CD}


def test_ten_equal_rewards_and_no_nominal_credit():
    rewards = [cp for cp in CHECKPOINTS_CD if cp.weight]
    assert len(rewards) == 10
    assert all(cp.weight == .10 for cp in rewards)
    assert sum(VALIDITY_PENALTIES.values()) == pytest.approx(.10)
    scores = perfect_scores()
    assert 'S1.nominal_feedforward' not in scores
    assert aggregate_codesign(scores)['reward'] == 1.0
    scores['S1.nominal_feedforward'] = 0.0
    assert aggregate_codesign(scores)['reward'] == 1.0


@pytest.mark.parametrize('checkpoint', [cp.id for cp in CHECKPOINTS_CD if cp.weight])
def test_each_reward_is_worth_ten_percent(checkpoint):
    scores = perfect_scores()
    scores[checkpoint] = 0.0
    report = aggregate_codesign(scores)
    assert report['reward'] == .9
    assert report['validity_deduction'] == 0.0


def test_partial_validity_deducts_instead_of_awarding_credit():
    scores = perfect_scores()
    scores['C1.printability'] = .25
    scores['C1.buildable'] = .8
    expected = .75 * VALIDITY_PENALTIES['C1.printability'] + .2 * VALIDITY_PENALTIES['C1.buildable']
    report = aggregate_codesign(scores)
    assert report['reward'] == round(1 - expected, 4)
    assert report['validity_deduction'] == pytest.approx(expected, abs=1e-6)
    assert report['stages']['validity'] == round(.1 - expected, 4)
    assert sum(report['stages'].values()) == pytest.approx(report['raw_total'])


def test_missing_submission_has_no_positive_reward():
    report = aggregate_codesign({})
    assert report['reward'] == 0.0
    assert report['raw_total'] == -.10
    assert report['validity_deduction'] == .10


def test_all_basic_failures_deduct_at_most_ten_percent_without_a_gate():
    scores = perfect_scores()
    for cid in VALIDITY_PENALTIES:
        scores[cid] = 0.0
    report = aggregate_codesign(scores)
    assert report['reward'] == .9
    assert report['validity_deduction'] == .10
    assert not report['gated']
    assert report['gate_failed'] == []
