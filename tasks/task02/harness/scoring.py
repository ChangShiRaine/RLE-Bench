"""Turn a cost ledger into `reward.json`.

    reward = mean stage score over all PLANNED evaluation trials

Scoring reads the **ledger only**. That is not a shortcut: what task02 evaluates is a
handoff between agents with an LLM in the loop on both sides, which is not replayable, so
there is nothing for the verifier to re-execute. The ledger is harness instrumentation --
root-owned, hash-chained, written solely by the metering daemon -- so reading it satisfies
invariant #1, whereas reading the manual or the controllers would not: those live in
/logs/artifacts, which Harbor makes agent-writable.

CUMULATIVE. Every step's verifier scores the ledger as it then stands, and
`multi_step_reward_strategy = "final"` takes the last one that ran. Any step can abort the
chain (`MultiStepTrial._should_stop_after_step`), so a scorer that only produced a number
at the end would produce none at all for a run that crashed part-way. Unreached trials
count as zeros: the denominator comes from the START record, written before any agent ran,
so a truncated run cannot shrink what it is measured against.

THE SEAL IS NOT A PRECONDITION FOR SCORING, only for completeness. It happens once, on the
last step, so every earlier verifier would otherwise refuse a sound ledger. The gate is the
CHAIN, verifiable at every point.

An unverifiable or missing ledger scores zero, and there is no participation credit: the
harness seals every run from a root hook, so a seal says nothing about the agent.

`dev_steps` is reported and weighted ZERO -- task02 asks whether the harness transfers, not
how cheaply it was built. Still worth recording: an agent that spent nothing and one that
spent everything are different runs even when they score the same.
"""

from __future__ import annotations

from typing import Any

from . import ledger as L


def score_ledger(records: list[L.Record] | None, verify_error: str | None = None
                 ) -> dict[str, Any]:
    """Compute the reward breakdown from an already-read ledger.

    `records=None` means the ledger was absent or unreadable; `verify_error` carries a
    chain-verification failure. Both are gate failures.
    """
    out: dict[str, Any] = {
        "ledger_ok": False,
        "ledger_reason": None,
        "reward": 0.0,
        "success_rate": 0.0,
        "trials_recorded": 0,
        "planned_trials": None,
        "interaction_steps": None,
        "sealed": False,
    }

    if records is None:
        out["ledger_reason"] = verify_error or "ledger missing or unreadable"
        return out
    if verify_error:
        out["ledger_reason"] = verify_error
        return out

    summary = L.summarize(records)
    if not summary["planned_trials"]:
        # No START record, or one that planned nothing. There is no denominator, so
        # there is nothing to score -- as distinct from something that scored zero.
        out["ledger_reason"] = "ledger records no planned trials"
        return out

    out["ledger_ok"] = True
    # THE REWARD. Already normalised against the planned trial count by `summarize`, so
    # this is a bare read rather than an arithmetic step that could disagree with the
    # ledger's own view of the same run.
    out["reward"] = round(float(summary["mean_trial_score"]), 6)
    out["success_rate"] = float(summary["success_rate"])
    out["successes"] = int(summary["successes"])
    out["trials_recorded"] = int(summary["trials_recorded"])
    out["planned_trials"] = int(summary["planned_trials"])
    out["trials_attempted"] = int(summary["trials_attempted"])
    out["interaction_steps"] = int(summary["interaction_steps_total"])
    out["interaction_budget"] = summary["interaction_budget"]
    out["development_ended"] = bool(summary["development_ended"])
    out["tasks_practised"] = summary["tasks_practised"]
    # WEIGHTED ZERO, like every development figure. Reported because `tasks_practised`
    # alone cannot tell a phase that finished nothing from one that finished most of its
    # curriculum. None for a ledger predating the per-episode records.
    out["train_tasks_solved"] = summary["train_tasks_solved"]
    out["train_tasks_solved_names"] = summary["train_tasks_solved_names"]
    out["dev_episodes"] = summary["dev_episodes"]
    out["dev_successes"] = summary["dev_successes"]
    out["dev_by_task"] = summary["dev_by_task"]
    out["sealed"] = bool(summary["sealed"])
    out["evaluation_closed"] = bool(summary["evaluation_closed"])
    out["end_reason"] = summary["end_reason"]
    out["per_trial"] = summary["per_trial"]
    # Not a gate -- see the module docstring -- but the difference between a run that
    # finished and one that was cut off, which is the first thing to look at when a
    # score is lower than expected.
    if summary["trials_recorded"] < summary["planned_trials"]:
        out["ledger_reason"] = (
            f"{summary['trials_recorded']} of {summary['planned_trials']} trials "
            "were graded; the rest are scored as zeros"
        )
    return out


def score_path(ledger_path) -> dict[str, Any]:
    """Read, verify and score a ledger file. Never raises: a broken ledger is a gate
    failure with a reason, not a crash in the verifier."""
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
    every remaining step, throwing away every trial after it. Hence: flat, numeric, no None.

    `reward` is the mean trial score in [0, 1] and the key Harbor treats as canonical
    (`min_reward` gates on it).

    `success_rate` is reported alongside and is deliberately NOT the reward: it is the
    fraction of trials that satisfied the environment's own predicate outright, which on
    composite tasks is expected to be far below the stage score. Both belong in the file
    -- one is the graded quantity, the other is the one that is comparable with any other
    RoboCasa result.

    `trials_attempted` never enters the reward either: a run whose agents never reached a
    trial scores zero on merit, and this is what lets a reader tell that apart from a run
    that reached every trial and failed. One is an infrastructure failure worth
    discarding; the other is a result -- including a trial the agent reset straight past
    without driving it, which is a choice it made and not a trial the harness lost.
    """
    payload: dict[str, float | int] = {
        "reward": float(result["reward"]),
        "success_rate": float(result["success_rate"]),
    }
    # Absent counts are omitted rather than sent as null: a run with no ledger has no
    # step count, and reporting 0 would read as "ran and spent nothing".
    # The development figures ride along for the same reason `tasks_practised` does: they
    # are numeric, weighted zero, and change how a low reward reads.
    for key in ("successes", "trials_recorded", "planned_trials", "trials_attempted",
                "interaction_steps", "interaction_budget", "tasks_practised",
                "train_tasks_solved", "dev_episodes", "dev_successes"):
        value = result.get(key)
        if value is not None:
            payload[key] = int(value)
    return payload


def diagnosis_json(result: dict[str, Any]) -> dict[str, Any]:
    """The non-numeric half of the breakdown, for humans.

    Kept out of reward.json because Harbor's schema rejects it, but it is the part that
    explains *why* a run scored what it did -- and for task02 the per-trial stage vectors
    are most of that explanation. A trial that scored 0.5 because it placed the object
    and never turned the appliance on is a different result from one that scored 0.5 by
    doing half of two independent things, and only this file says which.
    """
    return {
        "ledger_ok": bool(result["ledger_ok"]),
        "ledger_reason": result.get("ledger_reason"),
        "reward": float(result["reward"]),
        "success_rate": float(result["success_rate"]),
        "successes": result.get("successes"),
        "trials_recorded": result.get("trials_recorded"),
        "planned_trials": result.get("planned_trials"),
        "trials_attempted": result.get("trials_attempted"),
        # Development facts. None of them is scored; all of them change how a zero reads.
        "interaction_steps": result.get("interaction_steps"),
        "interaction_budget": result.get("interaction_budget"),
        "development_ended": result.get("development_ended"),
        "tasks_practised": result.get("tasks_practised"),
        # What development FINISHED, not just what it touched. `dev_by_task` breaks it
        # down per training task, and is where to look when a harness scored zero.
        "train_tasks_solved": result.get("train_tasks_solved"),
        "train_tasks_solved_names": result.get("train_tasks_solved_names"),
        "dev_episodes": result.get("dev_episodes"),
        "dev_successes": result.get("dev_successes"),
        "dev_by_task": result.get("dev_by_task"),
        # How the run ended, and whether it got to the end at all.
        "sealed": result.get("sealed"),
        "evaluation_closed": result.get("evaluation_closed"),
        "end_reason": result.get("end_reason"),
        "per_trial": result.get("per_trial"),
    }
