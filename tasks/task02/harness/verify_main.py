"""Verifier entry point. Runs as ROOT in the agent's own container, once per step.

The verifier SHARES the agent's environment, and that is a deliberate security choice
rather than a convenience. Protection is file ownership -- the same mechanism that
already hides the simulator and the metering daemon:

    /opt/private        root:root 0700   this module, scoring.py, config.py, ledger.py
    /var/lib/rlebench  root:root 0700   the authoritative ledger

Sharing is what lets the scorer read the ledger **in place**, at its root-only path. The
alternative -- a separate verifier image -- can only reach the copy the daemon exports to
/logs/artifacts, and Harbor deliberately makes that directory agent-writable (it chmods
the mount chain to 0777 so tasks can publish from it). A reward computed from a file the
agent can delete and rewrite is not a reward: a task01 experiment doing exactly that,
with a from-scratch chain claiming a one-step flawless run, verified perfectly and scored
1.0.

So the ledger under /logs/artifacts is exported for HUMANS and is never scored. There is
deliberately no fallback to it: if the root-only ledger is missing, the run is broken and
scores zero, rather than quietly falling back to the one file an agent could have written.

RUN ONCE PER STEP. Every step writes a cumulative reward and Harbor's
`multi_step_reward_strategy = "final"` keeps the last one, so an aborted chain still
yields the correct partial score. That makes this module's cheapness a requirement: it
reads a file and does arithmetic, and must not need the simulator.

What sharing costs: verification needs this environment, so an archived run cannot be
re-scored on a plain CPU box. That was judged the right trade -- being unable to re-score
an old run offline is an inconvenience; scoring a forged one is a broken benchmark.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rlebench.core.media import Media

from . import scoring

# The authoritative ledger: root-owned, mode 600, never in the agent's reach.
LEDGER = Path(os.environ.get("RLEBENCH_LEDGER", "/var/lib/rlebench/cost.jsonl"))
# Human-facing export. Read only to report what a reader will find there; NEVER scored.
ARTIFACTS = Path(os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts/toolsmith"))
REWARD_PATH = Path(os.environ.get("RLEBENCH_REWARD", "/logs/verifier/reward.json"))
# Evaluation-trial videos the daemon recorded into the root-only tree (eval_video.py).
EVAL_MEDIA = Path(os.environ.get("RLEBENCH_EVAL_MEDIA_DIR", "/opt/private/media"))

# Where the development agent was told to leave its harness. Recorded as DIAGNOSIS ONLY
# -- see below.
HARNESS_DIR = Path(os.environ.get("RLEBENCH_HARNESS_DIR", "/workspace/agent_harness"))


def harness_facts() -> dict:
    """What the development agent left behind, for a human reader.

    NEVER SCORED, and the reason is worth stating: gating on file presence would pay an
    agent for touching a file, and a run that somehow solved trials without writing a
    manual is a legitimate -- if surprising -- result. What these facts are for is
    reading a zero. An evaluation that scored nothing with no MANUAL.md is a development
    phase that failed; the same zero with a full manual is a harness that did not
    transfer, and those are different findings about the same number. The two layers are
    reported separately for the same reason: motion with nothing to aim it and perception
    nothing consumes are different ways to hand over half a harness.
    """
    def _modules(name: str) -> list[str]:
        directory = HARNESS_DIR / name
        try:
            return sorted(p.name for p in directory.glob("*.py")) \
                if directory.is_dir() else []
        except OSError:
            return []

    manual = HARNESS_DIR / "MANUAL.md"
    try:
        manual_bytes = manual.stat().st_size if manual.is_file() else 0
    except OSError:
        manual_bytes = 0
    return {
        "harness_dir": str(HARNESS_DIR),
        "manual_present": manual_bytes > 0,
        "manual_bytes": manual_bytes,
        "controller_modules": _modules("controllers"),
        "perception_modules": _modules("perception"),
    }


def main(argv: list[str] | None = None) -> int:
    result = scoring.score_path(LEDGER)

    # reward.json is Harbor's, and its schema is dict[str, float | int] -- flat and
    # numeric. Anything else fails validation, which Harbor records as a step exception
    # and which aborts the remaining steps -- throwing away every trial still to come,
    # not just this one. Everything descriptive goes to diagnosis.json next to it.
    payload = scoring.reward_json(result)

    diagnosis = scoring.diagnosis_json(result)
    diagnosis["ledger_source"] = str(LEDGER)
    diagnosis["transcript_present"] = (ARTIFACTS / "transcript.jsonl").exists()
    diagnosis.update(harness_facts())

    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (REWARD_PATH.parent / "diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    try:
        exported = Media.export(EVAL_MEDIA, REWARD_PATH.parent / "media")
        print(f"[verifier] media: {len(exported)} evaluation video(s) exported")
    except Exception as exc:  # noqa: BLE001
        print(f"[verifier] media export skipped: {exc}", file=sys.stderr)

    # Both go to stdout, which Harbor captures as test-stdout.txt -- the reward file
    # alone does not explain a zero.
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    if not diagnosis["ledger_ok"]:
        print(f"[verifier] LEDGER UNUSABLE: {diagnosis['ledger_reason']}",
              file=sys.stderr)
    elif not diagnosis.get("trials_attempted"):
        # Scores zero on merit, but for a very different reason than losing every trial,
        # and the two are worth telling apart in a log someone reads later.
        print("[verifier] no evaluation trial was ever reached "
              f"(end_reason={diagnosis.get('end_reason')})", file=sys.stderr)
    elif not diagnosis.get("manual_present"):
        print("[verifier] no MANUAL.md was left in the harness directory; the "
              "evaluation agents had controllers but no instructions for them",
              file=sys.stderr)
    # Exit 0 regardless: the reward file is the result, and a non-zero exit reads as a
    # broken verifier rather than a run that scored badly.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
