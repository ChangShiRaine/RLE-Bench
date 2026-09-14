"""Tests for the scorer: ledger in, reward.json out.

Two things are load-bearing and neither is obvious from reading the formula.

CUMULATIVE. Every one of the eleven Harbor steps runs this, and
`multi_step_reward_strategy = "final"` keeps whichever ran last. So a partial ledger must
produce a correct partial reward, not a refusal -- a run that solved six trials and then
had its chain aborted must score six trials' worth, not zero.

DENOMINATOR FROM THE START RECORD. Unreached trials count as zeros because the plan is
recorded before any agent ran. A truncated ledger that could shrink its own denominator
would score a run that gave up after one good trial the same as one that did them all.
"""

from __future__ import annotations

import json

import pytest

from harness import ledger as L
from harness import scoring


def build(tmp_path, *, planned=4, trials=(), steps=0, seal=True, result=True,
          budget=1000):
    """Write a ledger by hand, so a partial or malformed one can be scored on purpose."""
    led = L.Ledger(tmp_path / "cost.jsonl")
    led.append(L.KIND_START, train_tasks=["OpenDrawer"], split="pretrain",
               eval_split="target", interaction_budget=budget, planned_trials=planned)
    if steps:
        led.append(L.KIND_INTERACT, steps=steps, task="OpenDrawer")
    led.append(L.KIND_DEV_END, steps_used=steps, episodes=1, tasks_practised=1,
               steps_by_task={"OpenDrawer": steps})
    for i, (score, success) in enumerate(trials):
        led.append(L.KIND_TRIAL, index=i, task=f"Composite{i}", success=success,
                   score=score, steps=10, ended_by="agent_reset", attempted=True,
                   stages_at_reset=[], stages_best=[], error=None)
    if result:
        led.append(L.KIND_RESULT,
                   score=(sum(s for s, _ in trials) / planned) if planned else 0.0,
                   success_rate=0.0, trials=len(trials), planned_trials=planned,
                   successes=0, trials_attempted=len(trials), eval_steps=0, per_task={})
    if seal:
        count = sum(1 for _ in L.read_records(led.path)) + 1
        led.append(L.KIND_SEAL, record_count=count, transcript_sha256=None,
                   steps_used=steps, end_reason="harness_seal")
    return led.path


# -- the reward --------------------------------------------------------------

def test_reward_is_the_mean_stage_score_over_planned_trials(tmp_path):
    path = build(tmp_path, planned=4, trials=[(1.0, True), (0.5, False),
                                              (0.0, False), (0.5, False)])
    assert scoring.score_path(path)["reward"] == pytest.approx(0.5)


def test_a_run_that_solved_everything_scores_one(tmp_path):
    path = build(tmp_path, planned=2, trials=[(1.0, True), (1.0, True)])
    assert scoring.score_path(path)["reward"] == 1.0


def test_a_run_that_achieved_nothing_scores_zero(tmp_path):
    path = build(tmp_path, planned=2, trials=[(0.0, False), (0.0, False)])
    assert scoring.score_path(path)["reward"] == 0.0


def test_success_rate_is_reported_separately_from_the_reward(tmp_path):
    """One is the graded quantity; the other is what compares with any other RoboCasa
    result. On composite tasks they are expected to differ a lot."""
    path = build(tmp_path, planned=2, trials=[(0.5, False), (1.0, True)])
    result = scoring.score_path(path)
    assert result["reward"] == 0.75 and result["success_rate"] == 0.5


# -- partial ledgers ---------------------------------------------------------

def test_a_partial_ledger_scores_the_trials_it_has(tmp_path):
    """The reason every step scores. A chain aborted at step 7 must still be worth what
    it earned."""
    path = build(tmp_path, planned=10, trials=[(1.0, True)] * 6, seal=False,
                 result=False)
    result = scoring.score_path(path)
    assert result["ledger_ok"] and result["reward"] == pytest.approx(0.6)


def test_unreached_trials_count_as_zeros(tmp_path):
    """A truncated ledger must not shrink its own denominator: giving up after one good
    trial would otherwise score the same as doing all of them."""
    path = build(tmp_path, planned=10, trials=[(1.0, True)], seal=False, result=False)
    assert scoring.score_path(path)["reward"] == pytest.approx(0.1)


def test_a_partial_ledger_says_why_it_is_partial(tmp_path):
    path = build(tmp_path, planned=10, trials=[(1.0, True)], seal=False, result=False)
    assert "1 of 10 trials" in scoring.score_path(path)["ledger_reason"]


def test_an_unsealed_ledger_is_still_scored(tmp_path):
    """Task01 refuses one, correctly, because its seal happens on its only scored step.
    Here the seal happens once at the end and every earlier verifier would refuse."""
    path = build(tmp_path, planned=2, trials=[(1.0, True), (1.0, True)], seal=False)
    assert scoring.score_path(path)["ledger_ok"]
    assert scoring.score_path(path)["sealed"] is False


# -- gate failures -----------------------------------------------------------

def test_a_missing_ledger_scores_zero(tmp_path):
    result = scoring.score_path(tmp_path / "nothing.jsonl")
    assert result["reward"] == 0.0 and not result["ledger_ok"]
    assert "no ledger" in result["ledger_reason"]


def test_a_tampered_ledger_scores_zero(tmp_path):
    """Integrity, not authentication: the root-only path is what makes forgery
    impossible. This catches corruption and partial writes."""
    path = build(tmp_path, planned=2, trials=[(0.0, False), (0.0, False)])
    lines = path.read_text().splitlines()
    trial = next(i for i, line in enumerate(lines)
                 if json.loads(line)["kind"] == L.KIND_TRIAL)
    rec = json.loads(lines[trial])
    rec["payload"]["score"] = 1.0          # a real change, so the digest must disagree
    lines[trial] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    result = scoring.score_path(path)
    assert result["reward"] == 0.0 and "failed verification" in result["ledger_reason"]


def test_a_truncated_tail_is_caught_by_the_seal(tmp_path):
    """Chaining alone cannot detect dropping the last k records -- a valid prefix is a
    valid chain. The seal's declared record count is what catches it."""
    path = build(tmp_path, planned=2, trials=[(1.0, True), (1.0, True)])
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-2] + [lines[-1]]) + "\n")
    assert scoring.score_path(path)["reward"] == 0.0


def test_a_ledger_with_no_planned_trials_is_unscoreable(tmp_path):
    """No denominator means nothing to score -- as distinct from something that scored
    zero, which is what reporting 0.0 with ledger_ok would claim."""
    path = build(tmp_path, planned=0, trials=[])
    result = scoring.score_path(path)
    assert not result["ledger_ok"] and "no planned trials" in result["ledger_reason"]


def test_an_unreadable_ledger_does_not_crash_the_verifier(tmp_path):
    path = tmp_path / "garbage.jsonl"
    path.write_text("{not json\n")
    assert scoring.score_path(path)["reward"] == 0.0


# -- reward.json shape -------------------------------------------------------

def test_reward_json_is_flat_and_numeric(tmp_path):
    """Harbor parses this into dict[str, float | int]. Anything else is a step
    exception, which in an eleven-step task throws away every trial still to come."""
    payload = scoring.reward_json(scoring.score_path(
        build(tmp_path, planned=2, trials=[(1.0, True), (0.0, False)])))
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in payload.values())
    assert payload["reward"] == 0.5


def test_reward_json_omits_absent_counts_rather_than_sending_null(tmp_path):
    """A run with no ledger has no step count, and reporting 0 would read as 'ran and
    spent nothing'."""
    payload = scoring.reward_json(scoring.score_path(tmp_path / "nothing.jsonl"))
    assert "interaction_steps" not in payload
    assert payload["reward"] == 0.0


def test_reward_json_survives_a_gate_failure(tmp_path):
    payload = scoring.reward_json(scoring.score_path(tmp_path / "nothing.jsonl"))
    assert json.dumps(payload)


# -- diagnosis ---------------------------------------------------------------

def test_development_facts_are_reported_but_not_scored(tmp_path):
    """The interaction budget is a hard cap and nothing else in task02. Spending it all
    or none of it must not move the reward."""
    lean = scoring.score_path(build(tmp_path / "a", planned=2,
                                    trials=[(1.0, True), (1.0, True)], steps=1))
    heavy = scoring.score_path(build(tmp_path / "b", planned=2,
                                     trials=[(1.0, True), (1.0, True)], steps=999))
    assert lean["reward"] == heavy["reward"] == 1.0
    assert lean["interaction_steps"] == 1 and heavy["interaction_steps"] == 999


def test_diagnosis_carries_the_per_trial_breakdown(tmp_path):
    """A trial that scored 0.5 by placing the object but never turning the appliance on
    is a different result from one that did half of two independent things."""
    path = build(tmp_path, planned=2, trials=[(0.5, False), (1.0, True)])
    diagnosis = scoring.diagnosis_json(scoring.score_path(path))
    assert [t["score"] for t in diagnosis["per_trial"]] == [0.5, 1.0]
    assert [t["task"] for t in diagnosis["per_trial"]] == ["Composite0", "Composite1"]


def test_diagnosis_explains_a_zero(tmp_path):
    diagnosis = scoring.diagnosis_json(scoring.score_path(tmp_path / "nothing.jsonl"))
    assert diagnosis["ledger_reason"] and not diagnosis["ledger_ok"]
