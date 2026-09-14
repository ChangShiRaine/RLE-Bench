"""Agent-visible RPC client for private design-set diagnostics.

The implementation and exact labels stay in the root-owned service. This
module only sends a request and decodes its aggregate response.
"""
from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = "/run/rlebench/task06-training.sock"


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("diagnostic service closed an incomplete response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def evaluate() -> dict:
    """Evaluate the candidate currently staged in ``/logs/artifacts``.

    Returns aggregate translation and wrapped-rotation error summaries for
    fixed single-frame and push-episode diagnostics derived from the public
    design seeds. Six calls are allowed per task container. Values are rounded;
    push-frame confidence intervals resample whole episodes rather than treating
    correlated frames as independent. Error summaries cover finite predictions;
    frame counts and valid_fraction report failures.
    """
    path = os.environ.get("RLEBENCH_TRAINING_SOCKET", DEFAULT_SOCKET)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(3600.0)
        sock.connect(path)
        sock.sendall(b'{"op":"evaluate"}\n')
        stream = sock.makefile("rb")
        line = stream.readline(4097)
        if not line or len(line) > 4096:
            raise RuntimeError("invalid response from diagnostic service")
        header = json.loads(line)
        if not header.get("ok"):
            raise RuntimeError(header.get("error", "diagnostic service error"))
        payload = _read_exact(stream, int(header["size"]))
    return json.loads(payload)
