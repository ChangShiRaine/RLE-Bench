"""Turn a sealed cost ledger into `reward.json`.

Scoring reads the **ledger only**. That is not a shortcut: what task01 evaluates is the
agent's workflow with the LLM in the loop, which is not replayable, so there is nothing
for the verifier to re-execute. The ledger is harness instrumentation -- root-owned,
hash-chained, written solely by the metering daemon -- so reading it satisfies
invariant #1, whereas reading the transcript or diagnosis would not: those live in
/logs/artifacts, which Harbor makes agent-writable.

    reward = W_OUTCOME * sr  +  W_EFFICIENCY * sr * (1 - dev_steps / budget)

with three properties that matter:

  * an unverifiable, unsealed or missing ledger scores ZERO. There is no participation
    credit for reaching the end of the run: the harness seals every run from a root
    collect hook, so a flat gate component for a sealed ledger would pay even a run whose
    agent never issued a single graded action.
    Completing the protocol is not an achievement, it is the precondition for measuring
    one;
  * efficiency is SCALED BY the success rate, never added independently. An independent
    term pays the cheapest possible run -- never interact, never succeed -- for having
    done nothing;
  * efficiency is measured as a FRACTION of the run's own interaction budget, which the
    ledger records. The budget is deliberately overridable per run, and an absolute step
    target would silently redefine the grading curve every time it changed.

`dev_steps` is development interaction only. Evaluation stepping never touches the
interaction budget (session.Session._charge_steps is not called on the evaluation path),
so the ledger's interact records already carry exactly the right quantity. That is the
right currency for a speed-run: it asks how much experience the agent needed before it
was ready, and it must not make an agent pay for driving its graded trials thoroughly.
"""

from __future__ import annotations

from typing import Any

from . import config as C
from . import ledger as L


def reward_weights() -> tuple[float, float]:
    """(outcome, efficiency), overridable from `[verifier.env]` in task.toml.

    Read here rather than at import so a test can monkeypatch the environment, and read
    from the VERIFIER's channel rather than the container's so the agent cannot discover
    how it is being weighted.
    """
    return (
        C.env_float("RLEBENCH_W_OUTCOME", C.W_OUTCOME),
        C.env_float("RLEBENCH_W_EFFICIENCY", C.W_EFFICIENCY),
    )


def efficiency_fraction(steps: int, budget: int | None) -> float:
    """The unspent fraction of the interaction budget, clamped to [0, 1].

    Linear rather than saturating: with the term already gated behind the success rate
    there is nothing left for a saturation point to protect against, and a straight line
    is the version an agent can reason about while deciding whether one more experiment
    is worth it.
    """
    budget = C.INTERACTION_STEPS if budget is None else int(budget)
    if budget <= 0:
        return 0.0
    spent = min(1.0, max(0, int(steps)) / budget)
    return 1.0 - spent


def score_ledger(records: list[L.Record] | None, verify_error: str | None = None
                 ) -> dict[str, Any]:
    """Compute the reward breakdown from an already-read ledger.

    `records=None` means the ledger was absent or unreadable; `verify_error` carries a
    chain-verification failure. Both are gate failures.
    """
    out: dict[str, Any] = {
        "ledger_ok": False,
        "ledger_reason": None,
        "success_rate": 0.0,
        "interaction_steps": None,
        "efficiency_score": 0.0,
        "outcome_score": 0.0,
        "reward": 0.0,
    }

    if records is None:
        out["ledger_reason"] = verify_error or "ledger missing or unreadable"
        return out
    if verify_error:
        out["ledger_reason"] = verify_error
        return out

    summary = L.summarize(records)
    out["interaction_steps"] = summary["interaction_steps_total"]
    out["success_rate"] = float(summary["best_success_rate"])
    out["submissions_used"] = summary["submissions_used"]
    out["task"] = summary["task"]
    out["trials_total"] = summary.get("trials_total")
    out["trials_attempted"] = summary.get("trials_attempted")
    out["end_reason"] = summary.get("end_reason")
    # Reported regardless of outcome: it is what the efficiency axis is measured
    # against, so a reader cannot interpret interaction_steps without it. The fallback
    # covers ledgers written before the budget was recorded.
    out["interaction_budget"] = (summary.get("interaction_budget")
                                 or C.INTERACTION_STEPS)

    if not summary["sealed"]:
        # A prefix of a valid chain is internally consistent, so "unsealed" is the only
        # signal that the run was cut short before its results were written. There is
        # nothing to score, as distinct from something that scored badly. With no
        # submission driven yet the run is simply still going -- the develop-step
        # verifier reads the ledger mid-run by design.
        if summary["submissions_used"] == 0:
            out["ledger_reason"] = ("ledger is not sealed and no submission driven "
                                    "yet (run in progress)")
            out["in_progress"] = True
        else:
            out["ledger_reason"] = "ledger is not sealed (run did not complete)"
        return out

    out["ledger_ok"] = True

    w_outcome, w_efficiency = reward_weights()
    sr = out["success_rate"]
    unspent = efficiency_fraction(out["interaction_steps"],
                                  out["interaction_budget"])

    out["outcome_score"] = w_outcome * sr
    # Scaled by the success rate -- see the module docstring. A run that solved nothing
    # earns nothing here however little it spent.
    out["efficiency_score"] = w_efficiency * sr * unspent
    out["efficiency_fraction"] = unspent

    out["reward"] = round(out["outcome_score"] + out["efficiency_score"], 6)
    return out


def score_path(ledger_path) -> dict[str, Any]:
    """Read, verify and score a ledger file. Never raises: a broken ledger is a
    gate failure with a reason, not a crash in the verifier."""
    from pathlib import Path

    path = Path(ledger_path)
    if not path.exists():
        return score_ledger(None, verify_error=f"no ledger at {path}")
    try:
        records = list(L.read_records(path))
    except Exception as exc:  # noqa: BLE001
        return score_ledger(None, verify_error=f"unreadable ledger: {exc}")
    try:
        L.verify(records)
    except L.LedgerError as exc:
        return score_ledger(records, verify_error=f"ledger failed verification: {exc}")
    return score_ledger(records)


def reward_json(result: dict[str, Any]) -> dict[str, float | int]:
    """Shape the breakdown into what Harbor reads.

    Harbor parses reward.json into `VerifierResult.rewards: dict[str, float | int]`, so
    EVERY value must be a bare number. Nested objects, strings and nulls all fail
    validation -- and a validation failure is recorded as a step exception, which aborts
    the remaining steps, so a badly shaped reward file loses the run rather than
    scoring it badly. Hence: flat, numeric, no None.

    `reward` is the weighted total in [0, 1] and the key Harbor treats as canonical
    (`min_reward` gates on it). Components are flattened with a `component_` prefix
    rather than nested. Absolute counts stay uncapped so runs remain comparable if the
    weights change.

    `trials_attempted` is reported alongside `trials_total` and deliberately does NOT
    enter the reward: a run whose agent never reached a trial scores zero on merit, and
    this is what lets a reader tell that apart from a run that reached every trial and
    failed. One is an infrastructure failure worth discarding; the other is a result --
    including a trial the agent reset straight past without driving it, which is a
    choice it made and not a trial the harness lost.

    Everything non-numeric -- ledger_reason, task name, per-trial detail -- is
    diagnosis, not reward, and is written separately by the verifier.
    """
    payload: dict[str, float | int] = {
        "reward": float(result["reward"]),
        "success_rate": float(result["success_rate"]),
        "component_outcome": float(result["outcome_score"]),
        "component_efficiency": float(result["efficiency_score"]),
    }
    # Absent counts are omitted rather than sent as null: a run with no ledger has no
    # step count, and reporting 0 would read as "ran and spent nothing".
    for key in ("interaction_steps", "interaction_budget",
                "trials_total", "trials_attempted"):
        value = result.get(key)
        if value is not None:
            payload[key] = int(value)
    return payload


def diagnosis_json(result: dict[str, Any]) -> dict[str, Any]:
    """The non-numeric half of the breakdown, for humans.

    Kept out of reward.json because Harbor's schema rejects it, but it is the part that
    explains *why* a run scored what it did.
    """
    return {
        "task": result.get("task"),
        "ledger_ok": bool(result["ledger_ok"]),
        "ledger_reason": result.get("ledger_reason"),
        "in_progress": bool(result.get("in_progress", False)),
        "success_rate": float(result["success_rate"]),
        "interaction_steps": result.get("interaction_steps"),
        "interaction_budget": result.get("interaction_budget"),
        "efficiency_fraction": result.get("efficiency_fraction"),
        # How the run ended and how much of the evaluation the agent actually drove.
        # Neither affects the reward; both decide whether a zero is a result or a
        # broken run.
        "end_reason": result.get("end_reason"),
        "trials_total": result.get("trials_total"),
        "trials_attempted": result.get("trials_attempted"),
    }
