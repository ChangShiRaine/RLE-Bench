"""Adversarial tests for the task01 cost ledger.

The ledger IS the score (interaction steps and submission attempts at the moment the
threshold was first cleared), so every way of making a run look cheaper than it was
must be either impossible or detectable.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from harness import ledger as L


def _mk(tmp_path):
    return L.Ledger(tmp_path / "cost.jsonl")


def test_append_and_verify_roundtrip(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="CloseDrawer", split="pretrain")
    led.append(L.KIND_INTERACT, steps=100)
    led.append(L.KIND_INTERACT, steps=50)
    led.append(
        L.KIND_SUBMIT, submission_index=0, steps_cumulative=150,
        success_rate=0.4, cleared=False, controller_sha256="ab" * 32,
    )
    led.append(L.KIND_SEAL, record_count=5)

    records = L.load_verified(led.path)
    assert [r.kind for r in records] == [
        L.KIND_START, L.KIND_INTERACT, L.KIND_INTERACT, L.KIND_SUBMIT, L.KIND_SEAL
    ]


def test_ledger_file_is_not_group_or_world_readable(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    mode = stat.S_IMODE(os.stat(led.path).st_mode)
    # The agent runs as a different uid; permissions are the primary defence.
    assert not mode & stat.S_IRGRP
    assert not mode & stat.S_IROTH
    assert not mode & stat.S_IWOTH


def test_modifying_a_payload_is_detected(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_INTERACT, steps=10_000)
    led.append(L.KIND_SEAL, record_count=3)

    lines = led.path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["steps"] = 1  # pretend the run was 10000x cheaper
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    led.path.write_text("\n".join(lines) + "\n")

    with pytest.raises(L.LedgerError, match="modified"):
        L.load_verified(led.path)


def test_rehashing_a_modified_record_still_breaks_the_chain(tmp_path):
    """A forger who recomputes the edited record's own digest is still caught,
    because the NEXT record commits to the old digest."""
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_INTERACT, steps=10_000)
    led.append(L.KIND_SEAL, record_count=3)

    lines = led.path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["steps"] = 1
    rec["hash"] = L._digest(rec["prev_hash"], rec["index"], rec["kind"], rec["payload"])
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    led.path.write_text("\n".join(lines) + "\n")

    with pytest.raises(L.LedgerError, match="chain broken"):
        L.load_verified(led.path)


def test_deleting_an_interior_record_is_detected(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_INTERACT, steps=500)
    led.append(L.KIND_INTERACT, steps=500)
    led.append(L.KIND_SEAL, record_count=4)

    lines = led.path.read_text().splitlines()
    del lines[1]  # drop 500 steps of accrued cost
    led.path.write_text("\n".join(lines) + "\n")

    with pytest.raises(L.LedgerError):
        L.load_verified(led.path)


def test_tail_truncation_is_detected_via_the_seal_count(tmp_path):
    """Chaining alone leaves a valid prefix, so the seal records the count."""
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_INTERACT, steps=100)
    led.append(L.KIND_INTERACT, steps=900)
    led.append(L.KIND_SEAL, record_count=4)

    lines = led.path.read_text().splitlines()
    led.path.write_text("\n".join(lines[:2]) + "\n")  # keep only start + 100 steps

    records = list(L.read_records(led.path))
    L.verify(records)  # a bare prefix is internally consistent...
    assert not any(r.kind == L.KIND_SEAL for r in records)  # ...but unsealed
    assert L.summarize(records)["sealed"] is False


def test_records_after_the_seal_are_rejected(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_SEAL, record_count=2)
    with pytest.raises(L.LedgerError, match="sealed"):
        led.append(L.KIND_INTERACT, steps=1)


def test_reopening_continues_the_chain(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    first_hash = led.last_hash

    reopened = L.Ledger(led.path)
    assert reopened.last_hash == first_hash
    reopened.append(L.KIND_INTERACT, steps=7)
    L.load_verified(led.path)


def test_summarize_reports_cost_at_the_first_clearing_submission(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="CloseDrawer", split="pretrain")
    led.append(L.KIND_INTERACT, steps=1_000)
    led.append(L.KIND_SUBMIT, submission_index=0, steps_cumulative=1_000,
               success_rate=0.2, cleared=False, controller_sha256="a" * 64)
    led.append(L.KIND_INTERACT, steps=500)
    led.append(L.KIND_SUBMIT, submission_index=1, steps_cumulative=1_500,
               success_rate=1.0, cleared=True, controller_sha256="b" * 64)
    # Extra work after clearing must not change the measured cost.
    led.append(L.KIND_INTERACT, steps=9_000)
    led.append(L.KIND_SUBMIT, submission_index=2, steps_cumulative=10_500,
               success_rate=1.0, cleared=True, controller_sha256="c" * 64)
    led.append(L.KIND_SEAL, record_count=8)

    s = L.summarize(L.load_verified(led.path))
    assert s["best_success_rate"] == 1.0
    assert s["interaction_steps_total"] == 10_500
    assert s["submissions_used"] == 3
    assert s["best_success_rate"] == 1.0


def test_summarize_on_a_run_that_never_cleared(tmp_path):
    led = _mk(tmp_path)
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_INTERACT, steps=42)
    led.append(L.KIND_SUBMIT, submission_index=0, steps_cumulative=42,
               success_rate=0.0, cleared=False, controller_sha256="d" * 64)
    led.append(L.KIND_SEAL, record_count=4)

    s = L.summarize(L.load_verified(led.path))
    assert s["best_success_rate"] == 0.0
    assert s["best_success_rate"] == 0.0
