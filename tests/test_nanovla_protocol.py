"""Pure host-side tests for the verifier's freezing, scoring and socket framing."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tasks" / "task05" / "harness"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


finalizer = load_module("nanovla_finalizer", HARNESS / "finalize_submission.py")
common = load_module("nanovla_verifier_common", HARNESS / "verifier" / "common.py")
protocol = load_module(
    "nanovla_remote_protocol", HARNESS / "runtime" / "remote_protocol.py"
)


def test_finalizer_protects_its_own_process_ancestry():
    protected = finalizer.protected_ancestry()
    assert 1 in protected
    assert os.getpid() in protected
    assert finalizer.process_cgroup(os.getpid()) is not None
    assert finalizer.process_start_time(os.getpid()) > 0


def test_missing_rows_are_failures_in_fixed_denominator(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    rows = [
        {"kind": "standard", "suite": "libero_10", "task": 0, "episode": 0, "success": True},
        {"kind": "standard", "suite": "libero_10", "task": 0, "episode": 1, "success": False},
        # A string must not be coerced to True.
        {"kind": "standard", "suite": "libero_10", "task": 0, "episode": 2, "success": "yes"},
        # An unexpected key cannot enlarge or improve the denominator.
        {"kind": "standard", "suite": "libero_10", "task": 9, "episode": 49, "success": True},
        # A plus row never counts towards a standard set.
        {"kind": "plus", "bddl": "x.bddl", "init_idx": 0, "success": True},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected = common.standard_expected("libero_10", [0], 4)
    score, completed = common.score_rows(path, expected)
    assert score == pytest.approx(0.25)
    assert completed == 2
    plus_expected = common.plus_expected([{"bddl": "x.bddl", "init_idx": 0}, {"bddl": "y.bddl", "init_idx": 3}])
    assert common.score_rows(path, plus_expected) == (0.5, 1)


def freeze_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(common, "OUTPUT_DIR", output)
    monkeypatch.setattr(common, "FROZEN_DIR", tmp_path / "frozen")
    monkeypatch.setattr(common.os, "geteuid", lambda: 1)  # no chown on the host
    return output


def test_output_freeze_is_an_immutable_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = freeze_fixture(tmp_path, monkeypatch)
    (output / "ckpt.pt").write_bytes(b"checkpoint-A")
    (output / "meta.json").write_text('{"chunk": 16}')
    frozen = common.freeze_output()
    assert frozen == tmp_path / "frozen"
    assert (frozen / "ckpt.pt").read_bytes() == b"checkpoint-A"
    assert (frozen / "ckpt.pt").stat().st_mode & 0o777 == 0o444
    assert frozen.stat().st_mode & 0o777 == 0o555
    assert not output.exists()


@pytest.mark.parametrize("kind", ["extra", "symlink", "fifo", "oversize", "bad-meta"])
def test_output_freeze_rejects_invalid_layouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    output = freeze_fixture(tmp_path, monkeypatch)
    source = output / "ckpt.pt"
    if kind == "extra":
        source.write_bytes(b"payload")
        (output / "notes.txt").write_text("x")
    elif kind == "symlink":
        target = output / "target"
        target.write_bytes(b"payload")
        source.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(source)
    elif kind == "oversize":
        with source.open("wb") as stream:
            stream.truncate(common.MAX_CHECKPOINT_BYTES + 1)
    else:
        source.write_bytes(b"payload")
        (output / "meta.json").write_text("[1, 2]")
    with pytest.raises(common.ScoreFailure):
        common.freeze_output()


def run_protocol_server(path: Path, count: int, action: np.ndarray, observed: dict) -> threading.Thread:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def serve():
        try:
            conn, _ = listener.accept()
            with conn:
                language, cameras, state = protocol._read_request(conn)
                observed.update(language=language, cameras=cameras, state=state)
                conn.sendall(protocol._U32.pack(count) + action.astype("<f4").tobytes())
        finally:
            listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    return thread


def test_remote_protocol_round_trip_carries_variable_chunks(tmp_path: Path):
    socket_path = tmp_path / "policy.sock"
    expected = np.arange(21, dtype=np.float32).reshape(3, 7)
    observed = {}
    thread = run_protocol_server(socket_path, 21, expected, observed)
    client = protocol.RemoteClient(str(socket_path))
    cameras = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)
    actual = client.predict("pick the bowl", cameras, state)
    client.close()
    thread.join(timeout=2)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(observed["cameras"], cameras)
    np.testing.assert_array_equal(observed["state"], state)
    assert observed["language"] == "pick the bowl"


@pytest.mark.parametrize("count", [6, 7 * 65])
def test_remote_protocol_rejects_malformed_chunks(tmp_path: Path, count: int):
    socket_path = tmp_path / "policy.sock"
    thread = run_protocol_server(socket_path, count, np.zeros(count, dtype=np.float32), {})
    client = protocol.RemoteClient(str(socket_path))
    with pytest.raises(ValueError, match="invalid response length"):
        client.predict("task", np.zeros((2, 8, 8, 3), np.uint8), np.zeros(8, np.float32))
    client.close()
    thread.join(timeout=2)


def test_serve_helper_batches_and_validates_replies(tmp_path: Path):
    socket_path = tmp_path / "policy.sock"
    seen = []

    def infer(requests):
        seen.append(len(requests))
        return [np.full((2, 7), 0.5, dtype=np.float32) for _ in requests]

    thread = threading.Thread(target=protocol.serve, args=(str(socket_path), infer), daemon=True)
    thread.start()
    client = protocol.RemoteClient(str(socket_path), connect_timeout_s=10.0)
    action = client.predict("task", np.zeros((2, 8, 8, 3), np.uint8), np.zeros(8, np.float32))
    client.close()
    assert action.shape == (2, 7) and float(action[0, 0]) == 0.5
    assert seen == [1]
