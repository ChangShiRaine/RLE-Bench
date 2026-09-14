"""Score the pocket cube from the sealed ledger and private turn counters."""
import json
from pathlib import Path

from harness import _pocket_base_scoring as shared
from harness import ledger as L
from harness.tabletop.reward import apply

INCOMPLETE_QTM_GATE_CAP = 0.5
COUNTERS = Path("/var/lib/rlebench/recovery.json")
COUNTER_KEYS = ("optimal_qtm", "actual_qtm", "unclassified_transitions", "recoveries")


def task_quality(success, optimal, actual):
    if not success:
        return 0.0
    return 2.0 ** (-max(0.0, actual / max(1, optimal) - (optimal > 0)))


def score_path(path, counters_path=COUNTERS):
    try:
        records = L.load_verified(path)
    except Exception as exc:
        result = shared.score_ledger(None, verify_error=f"unreadable or invalid ledger: {exc}")
    else:
        result = shared.score_ledger(records)
        if result["ledger_ok"]:
            # This single attempt meters physical actions as evaluation steps.
            result["interaction_steps"] += sum(
                int(record.payload.get("eval_steps", 0))
                for record in records if record.kind == L.KIND_SUBMIT)
            result["efficiency_fraction"] = shared.efficiency_fraction(
                result["interaction_steps"], result["interaction_budget"])

    counters_valid = False
    result["qtm_ok"] = False
    try:
        counts = json.loads(Path(counters_path).read_text())
        if any(type(counts[k]) is not int or counts[k] < 0 for k in COUNTER_KEYS):
            raise ValueError("Invalid QTM counters")
        counters_valid = True
        result.update({k: counts[k] for k in COUNTER_KEYS})
        result["qtm_ok"] = counts["unclassified_transitions"] == 0
    except (OSError, ValueError, KeyError, TypeError):
        pass

    quality = (task_quality(result["success_rate"], counts["optimal_qtm"], counts["actual_qtm"])
               if result["ledger_ok"] and counters_valid else 0.0)
    apply(result, quality)
    if counters_valid and not result["qtm_ok"]:
        result["reward"] = min(result["reward"], INCOMPLETE_QTM_GATE_CAP)
    return result


def reward_json(result):
    reward = shared.reward_json(result)
    reward.update({k: result[k] for k in (*COUNTER_KEYS, "task_score") if k in result})
    return reward


def diagnosis_json(result):
    diagnosis = shared.diagnosis_json(result)
    diagnosis["qtm_ok"] = result.get("qtm_ok", False)
    return diagnosis
