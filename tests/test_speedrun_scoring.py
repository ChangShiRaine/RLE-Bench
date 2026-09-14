"""Tests for the task01 reward.

The reward reads the sealed ledger and nothing else:

    reward = W_OUTCOME * sr  +  W_EFFICIENCY * sr * (1 - dev_steps / budget)

Three properties carry it, and each is a way a run could otherwise look better than it
was:

  * a missing, broken or unsealed ledger must score ZERO, not "zero cost, therefore
    maximally efficient";
  * efficiency must be SCALED BY the success rate. An independent efficiency term pays
    the cheapest possible run -- never interact, never succeed -- for doing nothing.
    A flat "gate" component for a sealed ledger would be just as wrong: the harness
    seals every run from a root hook, so it would pay even a run whose agent never
    issued a single graded action;
  * efficiency is a FRACTION of the run's own budget, so overriding the budget rescales
    the curve instead of silently redefining what a score means.
"""

from __future__ import annotations

import json

import pytest

from harness import config as C
from harness import ledger as L
from harness import scoring as S


def build_ledger(tmp_path, *, steps=5_000, success_rate=1.0, sealed=True, submit=True,
                 name="cost", budget=None, trials=10, trials_attempted=None,
                 end_reason="agent_ended"):
    led = L.Ledger(tmp_path / f"{name}.jsonl")
    start = {"task": "CloseDrawer", "split": "pretrain"}
    if budget is not None:
        # What the real daemon records; the scorer grades efficiency as a fraction of it.
        start["interaction_budget"] = budget
    led.append(L.KIND_START, **start)
    if steps:
        led.append(L.KIND_INTERACT, steps=steps)
    n = 2 if steps else 1
    if submit:
        led.append(L.KIND_SUBMIT, submission_index=0, steps_cumulative=steps,
                   success_rate=success_rate, episodes=1, trials=trials,
                   trials_attempted=(trials if trials_attempted is None
                                     else trials_attempted))
        n += 1
    if sealed:
        led.append(L.KIND_SEAL, record_count=n + 1, end_reason=end_reason)
    return led.path


def reward_for(tmp_path, name, **kw):
    return S.score_path(build_ledger(tmp_path, name=name, **kw))


# -- an unusable ledger is not a cheap run -----------------------------------

def test_missing_ledger_scores_zero_not_free(tmp_path):
    """The dangerous failure: no ledger could be read as 'spent nothing'."""
    r = S.score_path(tmp_path / "absent.jsonl")
    assert r["reward"] == 0.0
    assert r["ledger_ok"] is False
    assert "no ledger" in r["ledger_reason"]


def test_unsealed_ledger_is_unscoreable(tmp_path):
    """A valid prefix is internally consistent, so only the seal proves completion."""
    r = reward_for(tmp_path, "unsealed", sealed=False)
    assert r["ledger_ok"] is False
    assert "not sealed" in r["ledger_reason"]
    assert r["reward"] == 0.0


def test_unsealed_without_a_submission_reads_as_in_progress(tmp_path):
    """The develop-step verifier reads the ledger mid-run by design; that must be
    tellable apart from a finished run whose seal never landed."""
    r = reward_for(tmp_path, "midrun", sealed=False, submit=False)
    assert r["ledger_ok"] is False and r["reward"] == 0.0
    assert r["in_progress"] is True
    assert S.diagnosis_json(r)["in_progress"] is True
    lost = reward_for(tmp_path, "lost", sealed=False, submit=True)
    assert lost.get("in_progress") is not True


def test_tampered_ledger_is_unscoreable(tmp_path):
    path = build_ledger(tmp_path, steps=100_000, name="tampered")
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["steps"] = 1          # pretend it was 100000x cheaper
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    r = S.score_path(path)
    assert r["ledger_ok"] is False
    assert "verification" in r["ledger_reason"]
    assert r["reward"] == 0.0


# -- no participation credit -------------------------------------------------

def test_a_sealed_ledger_alone_is_worth_nothing(tmp_path):
    """THE REGRESSION THIS FORMULA EXISTS FOR.

    The harness seals every run from a root collect hook, so "the ledger is sealed" is
    true of any run that reached the end of the last step -- including one whose agent
    died before issuing a single graded action. A real run scored 0.1 on exactly this.
    Completing the protocol must be worth zero.
    """
    r = reward_for(tmp_path, "idle", steps=0, submit=False)
    assert r["ledger_ok"] is True        # the ledger is readable and complete...
    assert r["reward"] == 0.0            # ...and that buys nothing at all


def test_driving_no_trials_scores_zero_and_says_so(tmp_path):
    """Distinguishable from losing every trial, without changing the reward."""
    r = reward_for(tmp_path, "nodrive", steps=45, success_rate=0.0,
                   trials=10, trials_attempted=0, end_reason="agent_exited")
    assert r["reward"] == 0.0
    d = S.diagnosis_json(r)
    assert d["trials_attempted"] == 0 and d["trials_total"] == 10
    assert d["end_reason"] == "agent_exited"


def test_efficiency_requires_success(tmp_path):
    """Spending nothing and solving nothing must not collect efficiency credit."""
    frugal_failure = reward_for(tmp_path, "frugal", steps=0, success_rate=0.0)
    assert frugal_failure["efficiency_score"] == 0.0
    assert frugal_failure["reward"] == 0.0


# -- the formula -------------------------------------------------------------

@pytest.mark.parametrize("sr,steps,expected", [
    (0.0, 0,       0.00),
    (0.0, 50_000,  0.00),
    (0.5, 0,       0.50),
    (0.5, 20_000,  0.48),
    (1.0, 20_000,  0.96),
    (1.0, 0,       1.00),
])
def test_reward_table(tmp_path, sr, steps, expected):
    r = reward_for(tmp_path, f"t{sr}_{steps}", steps=steps, success_rate=sr,
                   budget=100_000)
    assert r["reward"] == pytest.approx(expected)


def test_outcome_is_proportional_to_success_rate(tmp_path):
    half = reward_for(tmp_path, "half", success_rate=0.5)
    assert half["outcome_score"] == pytest.approx(C.W_OUTCOME * 0.5)


def test_a_cheaper_run_beats_an_equally_successful_dear_one(tmp_path):
    cheap = reward_for(tmp_path, "cheap", steps=10_000, budget=100_000)
    dear = reward_for(tmp_path, "dear", steps=90_000, budget=100_000)
    assert cheap["reward"] > dear["reward"]
    assert cheap["efficiency_score"] > dear["efficiency_score"]


def test_solving_more_beats_spending_less(tmp_path):
    """Outcome dominates: 0.8 of the weight is there, efficiency scales only 0.2."""
    thorough = reward_for(tmp_path, "thorough", steps=90_000, success_rate=1.0,
                          budget=100_000)
    frugal = reward_for(tmp_path, "frugalish", steps=100, success_rate=0.5,
                        budget=100_000)
    assert thorough["reward"] > frugal["reward"]


def test_efficiency_is_a_fraction_of_the_run_s_own_budget(tmp_path):
    """A run given a tenth of the budget and spending a tenth as much has been exactly
    as efficient, and must score the same."""
    big = reward_for(tmp_path, "big", steps=50_000, budget=100_000)
    small = reward_for(tmp_path, "small", steps=5_000, budget=10_000)
    assert big["efficiency_score"] == pytest.approx(small["efficiency_score"])
    assert small["interaction_budget"] == 10_000

    spent_all = reward_for(tmp_path, "all", steps=10_000, budget=10_000)
    assert spent_all["efficiency_score"] == 0.0


def test_overspending_the_budget_cannot_go_negative(tmp_path):
    """The daemon refuses steps past the budget, but the scorer must not depend on it."""
    r = reward_for(tmp_path, "over", steps=10**7, budget=100_000)
    assert r["efficiency_score"] == 0.0
    assert r["reward"] == pytest.approx(C.W_OUTCOME)


def test_a_missing_budget_record_falls_back_to_the_configured_default(tmp_path):
    """Ledgers written before the budget was recorded must still score."""
    r = reward_for(tmp_path, "nobudget", steps=10, budget=None)
    assert r["interaction_budget"] == C.INTERACTION_STEPS


def test_reward_stays_within_bounds(tmp_path):
    for steps in (0, 1, 10_000, 199_999, 10**7):
        for sr in (0.0, 0.37, 1.0):
            r = reward_for(tmp_path, f"b{steps}_{sr}", steps=steps, success_rate=sr)
            assert 0.0 <= r["reward"] <= 1.0


def test_a_perfect_free_run_scores_one(tmp_path):
    """The weights sum to 1.0, so solving everything with nothing spent is the top."""
    r = reward_for(tmp_path, "perfect", steps=0, success_rate=1.0)
    assert r["reward"] == 1.0


# -- weights are configurable, and only from the verifier's channel -----------

def test_weights_come_from_the_verifier_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RLEBENCH_W_OUTCOME", "0.5")
    monkeypatch.setenv("RLEBENCH_W_EFFICIENCY", "0.5")
    r = reward_for(tmp_path, "weighted", steps=0, success_rate=1.0)
    assert r["outcome_score"] == pytest.approx(0.5)
    assert r["efficiency_score"] == pytest.approx(0.5)


def test_default_weights_apply_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("RLEBENCH_W_OUTCOME", raising=False)
    monkeypatch.delenv("RLEBENCH_W_EFFICIENCY", raising=False)
    assert S.reward_weights() == (C.W_OUTCOME, C.W_EFFICIENCY)


def test_a_malformed_weight_is_refused_rather_than_ignored(monkeypatch):
    """Scoring a run under rules nobody chose is worse than failing loudly."""
    monkeypatch.setenv("RLEBENCH_W_OUTCOME", "most-of-it")
    with pytest.raises(ValueError):
        S.reward_weights()


# -- output shape ------------------------------------------------------------

def test_reward_json_is_flat_and_numeric(tmp_path):
    """Harbor parses reward.json into dict[str, float | int]. A nested object, a
    string or a null fails validation -- and Harbor records that as a step exception
    and aborts the remaining steps, so a malformed reward file LOSES the run rather
    than scoring it badly. This is the contract test for that.
    """
    payload = S.reward_json(reward_for(tmp_path, "shape"))
    for key, value in payload.items():
        assert isinstance(value, (int, float)), f"{key}={value!r} is not numeric"
        assert not isinstance(value, bool), f"{key} is a bool; Harbor wants 0/1"
    assert set(payload) >= {"reward", "success_rate", "component_outcome",
                            "component_efficiency", "interaction_steps",
                            "interaction_budget", "trials_total",
                            "trials_attempted"}
    json.dumps(payload)          # must be serialisable as-is


def test_reward_json_no_longer_carries_a_gate(tmp_path):
    payload = S.reward_json(reward_for(tmp_path, "nogate"))
    assert "component_gate" not in payload
    assert "gate_passed" not in payload
    assert "cleared" not in payload


def test_reward_json_omits_absent_counts_rather_than_sending_null(tmp_path):
    """None fails the schema, and 0 would read as 'ran and spent nothing'."""
    payload = S.reward_json(S.score_path(tmp_path / "absent.jsonl"))
    assert "interaction_steps" not in payload
    assert payload["reward"] == 0.0


def test_diagnosis_carries_what_the_reward_file_cannot(tmp_path):
    """The reason a run scored zero is the useful part, and it is not a number."""
    d = S.diagnosis_json(S.score_path(tmp_path / "absent.jsonl"))
    assert d["ledger_ok"] is False
    assert "no ledger" in d["ledger_reason"]


def test_absolute_metrics_are_reported_uncapped(tmp_path):
    """Kept so runs stay comparable if the weights change later."""
    payload = S.reward_json(reward_for(tmp_path, "abs", steps=123_456,
                                       budget=1_000_000))
    assert payload["interaction_steps"] == 123_456


# -- verifier entry point ----------------------------------------------------

def test_verifier_scores_the_ledger_it_is_pointed_at(tmp_path, monkeypatch):
    """The verifier reads the root-only ledger in place. Pointing it elsewhere is a
    test affordance; the trust is the hash chain either way -- see the tamper tests in
    test_speedrun_antigaming."""
    from harness import verify_main as V

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "cost.jsonl").write_text(
        build_ledger(tmp_path, steps=77, name="only").read_text())

    monkeypatch.setattr(V, "LEDGER", artifacts / "cost.jsonl")
    monkeypatch.setattr(V, "ARTIFACTS", artifacts)
    monkeypatch.setattr(V, "REWARD_PATH", tmp_path / "reward.json")
    V.main()

    payload = json.loads((tmp_path / "reward.json").read_text())
    diagnosis = json.loads((tmp_path / "diagnosis.json").read_text())
    assert payload["interaction_steps"] == 77
    assert diagnosis["ledger_source"] == str(artifacts / "cost.jsonl")


def test_verifier_writes_a_reward_file_even_with_no_ledger_anywhere(tmp_path, monkeypatch):
    from harness import verify_main as V

    monkeypatch.setattr(V, "LEDGER", tmp_path / "nope.jsonl")
    monkeypatch.setattr(V, "ARTIFACTS", tmp_path / "nothing")
    monkeypatch.setattr(V, "REWARD_PATH", tmp_path / "reward.json")
    assert V.main() == 0                       # never a non-zero exit for a bad run

    payload = json.loads((tmp_path / "reward.json").read_text())
    assert payload["reward"] == 0.0
    # Every value Harbor will parse must still be numeric on the failure path.
    assert all(isinstance(v, (int, float)) for v in payload.values())
