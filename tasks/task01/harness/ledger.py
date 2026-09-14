"""Append-only, hash-chained ledger of interaction and submission cost.

This file IS the score. Task01 grades how cheaply the agent reached the success
threshold, along two currencies (env.step calls, and evaluation attempts), so the
ledger must be trustworthy in the same way `harness.base_design` is for other tasks:
per CLAUDE.md invariant #1, nothing here may come from an artifact the agent wrote.

THE SCORED LEDGER IS NEVER AGENT-WRITABLE. It is owned by root, mode 600, in a
root-only directory, and the agent runs as an unprivileged user, so it cannot append,
edit or truncate it. The verifier shares this container and reads that original in
place. That single fact is the security story -- there is no key to manage and nothing
cryptographic to get right.

It is worth stating why, because the alternative was tried and measured. With a
SEPARATE verifier image the only ledger reachable is the copy the daemon exports to
/logs/artifacts -- and Harbor deliberately makes that directory agent-writable (it
chmods the mount chain to 0777 so tasks can publish). A run's whole reward then rests on
a file the agent can delete and rewrite. It was rewritten, as an experiment: a
from-scratch chain claiming a one-step flawless run verified perfectly and scored 1.0.
Reading the root-only original removes that file from the trust path entirely.

The chain that remains is an INTEGRITY check, not an authentication one: each record
commits to the previous record's digest, so accidental corruption or a partial write is
detected rather than silently scored. It is deliberately not asked to do more, because
the exported copy is for humans and is never what decides a reward.

Chaining cannot detect truncation of the TAIL (dropping the last k records leaves a
valid prefix). Countered by recording the expected record count in the final SEAL
record; the verifier treats a missing seal as a failed run rather than a zero-cost one.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64

# Record kinds.
KIND_START = "start"          # session opened: task, split, budgets
KIND_INTERACT = "interact"    # metered env.step calls
KIND_SUBMIT = "submit"        # an evaluation attempt and its outcome
KIND_DEV_END = "dev_end"      # agent declared development finished (no sim effect)
KIND_SEAL = "seal"            # end of session: totals, for tail-truncation checks


def _canonical(payload: dict[str, Any]) -> bytes:
    """Stable serialization, so a digest depends on content and not dict order."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _digest(prev_hash: str, index: int, kind: str, payload: dict[str, Any]) -> str:
    """Integrity digest for one record, chained to the previous one."""
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(str(index).encode())
    h.update(kind.encode())
    h.update(_canonical(payload))
    return h.hexdigest()


@dataclass(frozen=True)
class Record:
    index: int
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "index": self.index,
                "kind": self.kind,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
                "hash": self.hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Record":
        return Record(
            index=int(d["index"]),
            kind=str(d["kind"]),
            payload=dict(d["payload"]),
            prev_hash=str(d["prev_hash"]),
            hash=str(d["hash"]),
        )


class LedgerError(RuntimeError):
    pass


class Ledger:
    """Writer. Only the metering daemon (running as root) constructs this."""

    def __init__(self, path: str | os.PathLike[str], mode: int = 0o600):
        self.path = Path(path)
        self._mode = mode
        self._index = 0
        self._last_hash = GENESIS
        self._sealed = False
        if self.path.exists():
            records = list(read_records(self.path))
            if records:
                verify(records)  # never append onto a chain we cannot vouch for
                self._index = records[-1].index + 1
                self._last_hash = records[-1].hash
                self._sealed = any(r.kind == KIND_SEAL for r in records)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Create with restrictive permissions from the outset -- never a window
            # where the ledger exists world-readable.
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, mode)
            os.close(fd)
        try:
            os.chmod(self.path, mode)
        except PermissionError:
            pass

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def append(self, kind: str, **payload: Any) -> Record:
        if self._sealed:
            raise LedgerError("ledger is sealed; no further records accepted")
        rec = Record(
            index=self._index,
            kind=kind,
            payload=payload,
            prev_hash=self._last_hash,
            hash=_digest(self._last_hash, self._index, kind, payload),
        )
        # O_APPEND write + fsync: a crash mid-session must not lose accrued cost,
        # since a lost record reads as a cheaper run than actually happened.
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, (rec.to_json() + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        self._index += 1
        self._last_hash = rec.hash
        if kind == KIND_SEAL:
            self._sealed = True
        return rec


def read_records(path: str | os.PathLike[str]) -> Iterator[Record]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Record.from_dict(json.loads(line))


def verify(records: list[Record]) -> None:
    """Raise LedgerError unless the chain is intact and complete.

    Checks link integrity, contiguous indices, and -- if a seal is present -- that
    the record count matches, which is what catches tail truncation.
    """
    prev = GENESIS
    for i, rec in enumerate(records):
        if rec.index != i:
            raise LedgerError(f"index gap at position {i}: record says {rec.index}")
        if rec.prev_hash != prev:
            raise LedgerError(f"chain broken at record {i}: prev_hash mismatch")
        expected = _digest(rec.prev_hash, rec.index, rec.kind, rec.payload)
        if rec.hash != expected:
            raise LedgerError(f"record {i} was modified: digest mismatch")
        prev = rec.hash

    seals = [r for r in records if r.kind == KIND_SEAL]
    if seals:
        if seals[-1].index != len(records) - 1:
            raise LedgerError("records appear after the seal")
        declared = seals[-1].payload.get("record_count")
        if declared is not None and int(declared) != len(records):
            raise LedgerError(
                f"seal declares {declared} records but {len(records)} present "
                "(tail truncated?)"
            )


def load_verified(path: str | os.PathLike[str]) -> list[Record]:
    records = list(read_records(path))
    verify(records)
    return records


def summarize(records: list[Record]) -> dict[str, Any]:
    """Reduce a verified ledger to the quantities the scorer needs.

    `interaction_steps_total` is DEVELOPMENT interaction: evaluation stepping is never
    charged against the budget, so it never appears as an interact record.
    """
    start = next((r for r in records if r.kind == KIND_START), None)
    submits = [r for r in records if r.kind == KIND_SUBMIT]
    interacts = [r for r in records if r.kind == KIND_INTERACT]
    seal = next((r for r in records if r.kind == KIND_SEAL), None)
    dev_end = next((r for r in records if r.kind == KIND_DEV_END), None)

    total_steps = sum(int(r.payload.get("steps", 0)) for r in interacts)
    budget = start.payload.get("interaction_budget") if start else None

    def _from_submits(key: str) -> int | None:
        values = [r.payload[key] for r in submits if r.payload.get(key) is not None]
        return int(max(values)) if values else None

    return {
        "task": (start.payload.get("task") if start else None),
        "split": (start.payload.get("split") if start else None),
        # The budget this run actually ran under. Recorded by the root daemon at
        # session start, so the scorer can express efficiency as a fraction of it
        # rather than against a constant that a per-run override would invalidate.
        "interaction_budget": (int(budget) if budget is not None else None),
        "sealed": seal is not None,
        "development_ended": dev_end is not None,
        "interaction_steps_total": total_steps,
        "submissions_used": len(submits),
        "best_success_rate": max(
            [float(r.payload.get("success_rate", 0.0)) for r in submits],
            default=0.0,
        ),
        # Diagnosis, not reward: how much of the evaluation the agent actually drove,
        # and how the run ended. What separates "failed the task" from "never
        # attempted it".
        "trials_total": _from_submits("trials"),
        "trials_attempted": _from_submits("trials_attempted"),
        "end_reason": (seal.payload.get("end_reason") if seal else None),
    }
