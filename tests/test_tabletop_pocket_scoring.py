"""Pocket-cube quality scoring with privileged counters and step reporting."""
import json
import importlib.util
import sys
from pathlib import Path

import pytest
from harness import ledger as L


@pytest.fixture
def experiment_scoring(monkeypatch):
    name = "harness._pocket_base_scoring"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "tasks/task01/harness/scoring.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    monkeypatch.setitem(sys.modules, name, original)
    path = Path(__file__).resolve().parents[1] / "tasks/task03/tabletop/pocket/experiment_scoring.py"
    spec = importlib.util.spec_from_file_location("experiment_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("RLEBENCH_W_OUTCOME", "0.80")
    monkeypatch.setenv("RLEBENCH_W_EFFICIENCY", "0.20")
    return module


@pytest.mark.parametrize("steps,quality,expected", [(0,1,1), (12500,1,1),
                                                    (50000,1,1), (12500,0,0)])
def test_sealed_trial_steps(tmp_path, experiment_scoring, steps, quality, expected):
    path = tmp_path / "cost.jsonl"
    ledger = L.Ledger(path)
    ledger.append(L.KIND_START, task="RubikCube", interaction_budget=50000)
    ledger.append(L.KIND_SUBMIT, success_rate=quality, trials=1,
                  trials_attempted=1, eval_steps=steps)
    ledger.append(L.KIND_SEAL, record_count=3)
    counters = tmp_path / "recovery.json"
    counters.write_text(json.dumps(dict(optimal_qtm=7, actual_qtm=7,
                                        unclassified_transitions=0, recoveries=0)))
    result = experiment_scoring.score_path(path, counters)
    assert result["ledger_ok"]
    assert result["reward"] == expected
    assert experiment_scoring.reward_json(result)["interaction_steps"] == steps


def test_missing_ledger_is_not_zero_step_success(tmp_path, experiment_scoring):
    result = experiment_scoring.score_path(tmp_path / "missing")
    assert not result["ledger_ok"]
    assert result["reward"] == 0
    assert "interaction_steps" not in experiment_scoring.reward_json(result)


@pytest.mark.parametrize("actual,expected", [(7, 1), (14, .5), (21, .25), (0, 1)])
def test_qtm_decay(experiment_scoring, actual, expected):
    assert experiment_scoring.task_quality(True, 7, actual) == expected
    assert experiment_scoring.task_quality(False, 7, actual) == 0


def test_missing_counters_fail_closed(tmp_path, experiment_scoring):
    path = tmp_path / "ledger"
    ledger = L.Ledger(path)
    ledger.append(L.KIND_START, task="RubikCube", interaction_budget=50000)
    ledger.append(L.KIND_SUBMIT, success_rate=1, trials=1, trials_attempted=1, eval_steps=0)
    ledger.append(L.KIND_SEAL, record_count=3)
    result = experiment_scoring.score_path(path, tmp_path / "missing")
    assert result["reward"] == 0 and not result["qtm_ok"]


@pytest.mark.parametrize("counts", [
    dict(optimal_qtm=7, actual_qtm=-1, unclassified_transitions=0, recoveries=0),
    dict(optimal_qtm=7, actual_qtm=True, unclassified_transitions=0, recoveries=0),
    {},
])
def test_invalid_counters_cannot_earn_reward(tmp_path, experiment_scoring, counts):
    path = tmp_path / "ledger"
    ledger = L.Ledger(path)
    ledger.append(L.KIND_START, task="RubikCube", interaction_budget=50000)
    ledger.append(L.KIND_SUBMIT, success_rate=1, trials=1, trials_attempted=1, eval_steps=0)
    ledger.append(L.KIND_SEAL, record_count=3)
    counters = tmp_path / "counters.json"
    counters.write_text(json.dumps(counts))
    result = experiment_scoring.score_path(path, counters)
    assert result["ledger_ok"] and result["success_rate"] == 1
    assert not result["qtm_ok"]
    assert result["task_score"] == result["outcome_score"] == result["efficiency_score"] == result["reward"] == 0


@pytest.mark.parametrize("unclassified", [0, 1])
@pytest.mark.parametrize("damage", ["unsealed", "tampered"])
def test_invalid_ledger_cannot_use_valid_counters(tmp_path, experiment_scoring, damage, unclassified):
    path = tmp_path / "ledger"
    ledger = L.Ledger(path)
    ledger.append(L.KIND_START, task="RubikCube", interaction_budget=50000)
    ledger.append(L.KIND_SUBMIT, success_rate=1, trials=1, trials_attempted=1, eval_steps=0)
    if damage == "tampered":
        ledger.append(L.KIND_SEAL, record_count=3)
        # Re-serialize a modified payload without changing its hash.
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["payload"]["success_rate"] = 0
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    counters = tmp_path / "counters.json"
    counters.write_text(json.dumps(dict(optimal_qtm=7, actual_qtm=7,
                                        unclassified_transitions=unclassified, recoveries=0)))
    result = experiment_scoring.score_path(path, counters)
    assert not result["ledger_ok"] and result["reward"] == 0


@pytest.mark.parametrize("success,actual,steps,expected", [
    (1, 0, 0, .5),
    (1, 7, 50000, .5),
    (1, 14, 12500, .5),
    (1, 21, 0, .25),
    (0, 0, 0, 0),
])
def test_incomplete_qtm_caps_total_without_awarding_a_floor(
        tmp_path, experiment_scoring, success, actual, steps, expected):
    path = tmp_path / "ledger"
    ledger = L.Ledger(path)
    ledger.append(L.KIND_START, task="RubikCube", interaction_budget=50000)
    ledger.append(L.KIND_SUBMIT, success_rate=success, trials=1,
                  trials_attempted=1, eval_steps=steps)
    ledger.append(L.KIND_SEAL, record_count=3)
    counters = tmp_path / "counters.json"
    counters.write_text(json.dumps(dict(optimal_qtm=7, actual_qtm=actual,
                                        unclassified_transitions=1, recoveries=0)))
    result = experiment_scoring.score_path(path, counters)
    assert result["ledger_ok"] and not result["qtm_ok"]
    assert experiment_scoring.reward_json(result)["reward"] == expected
    assert experiment_scoring.diagnosis_json(result)["qtm_ok"] is False
    assert result["reward"] <= max(0, round(result["outcome_score"] + result["efficiency_score"], 6))
