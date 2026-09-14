"""Score the privileged tabletop ledger using task quality."""
from . import _tabletop_base_scoring as shared
from .tabletop.reward import apply


def score_ledger(records, verify_error=None):
    result = shared.score_ledger(records, verify_error)
    return apply(result, result["success_rate"] if result["ledger_ok"] else 0.0)


def score_path(path):
    result = shared.score_path(path)
    return apply(result, result["success_rate"] if result["ledger_ok"] else 0.0)


def reward_json(result):
    return dict(shared.reward_json(result), task_score=result["task_score"])


def diagnosis_json(result):
    return shared.diagnosis_json(result)
