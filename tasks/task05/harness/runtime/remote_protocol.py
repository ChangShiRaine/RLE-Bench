"""Unix-socket protocol between the evaluator and a submitted policy server.

The evaluator owns simulation, episode identities and success decisions. The
policy server sees one observation per request and answers with one action
chunk. Only fixed-size numeric arrays cross the wire; nothing is unpickled.

Request (evaluator -> policy):
    u32 header length, JSON header
        {"language": str, "camera_shape": [C, H, W, 3], "state_floats": D}
    C*H*W*3 uint8 camera bytes, then D little-endian float32 state values.
    C = 2 cameras at 128x128 with the shard preprocessing applied; D raw
    proprioceptive values, as stored in the shards.
Reply (policy -> evaluator):
    u32 count, then count little-endian float32 values forming a
    (count/ACTION_DIM, ACTION_DIM) chunk in the environment's action space,
    1..MAX_CHUNK rows.
    count 0 rejects the request. Every returned action is executed in order.

Serve with `serve(socket_path, infer)`: `infer` receives a list of `Request`
objects (batched while the queue is busy) and returns an array of shape
(batch, K, ACTION_DIM) or a list of (K, ACTION_DIM) arrays.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

ACTION_DIM = int(os.environ.get("NANOVLA_ACTION_DIM", "7"))  # 7 for LIBERO, 14 for RoboTwin
MAX_CHUNK = 64
_U32 = struct.Struct("!I")
_MAX_HEADER = 16 * 1024
_MAX_ACTION_FLOATS = MAX_CHUNK * ACTION_DIM


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("peer closed the inference socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class RemoteClient:
    """Evaluator-side client; one connection, one request in flight."""

    def __init__(self, path: str, connect_timeout_s: float = 90.0,
                 request_timeout_s: float = 180.0):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(request_timeout_s)
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self._sock.connect(path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"inference server did not accept {path}")
                time.sleep(0.05)

    def predict(self, language: str, cameras: np.ndarray,
                state: np.ndarray) -> np.ndarray:
        cameras = np.ascontiguousarray(cameras, dtype=np.uint8)
        state = np.ascontiguousarray(state, dtype="<f4").reshape(-1)
        if cameras.ndim != 4 or cameras.shape[-1] != 3:
            raise ValueError(f"invalid camera tensor shape {cameras.shape}")
        header = json.dumps(
            {"language": str(language), "camera_shape": list(cameras.shape),
             "state_floats": int(state.size)},
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        if len(header) > _MAX_HEADER:
            raise ValueError("inference request header is too large")
        self._sock.sendall(_U32.pack(len(header)) + header +
                           cameras.tobytes() + state.tobytes())
        count = _U32.unpack(_recv_exact(self._sock, _U32.size))[0]
        if count == 0:
            raise RuntimeError("policy server rejected the request")
        if count % ACTION_DIM or not 1 <= count // ACTION_DIM <= MAX_CHUNK:
            raise ValueError(f"invalid response length {count}")
        action = np.frombuffer(_recv_exact(self._sock, count * 4),
                               dtype="<f4").copy().reshape(-1, ACTION_DIM)
        if not np.isfinite(action).all():
            raise ValueError("policy server returned non-finite actions")
        return action

    def close(self) -> None:
        self._sock.close()


@dataclass
class Request:
    language: str
    cameras: np.ndarray  # (C, H, W, 3) uint8
    state: np.ndarray    # (D,) float32
    reply: queue.Queue


def _read_request(conn: socket.socket) -> tuple[str, np.ndarray, np.ndarray]:
    header_len = _U32.unpack(_recv_exact(conn, _U32.size))[0]
    if not 1 <= header_len <= _MAX_HEADER:
        raise ValueError(f"invalid request header length {header_len}")
    header = json.loads(_recv_exact(conn, header_len).decode("utf-8"))
    shape = tuple(int(x) for x in header["camera_shape"])
    state_floats = int(header["state_floats"])
    if len(shape) != 4 or shape[-1] != 3 or any(x <= 0 for x in shape):
        raise ValueError(f"invalid camera shape {shape}")
    camera_bytes = int(np.prod(shape, dtype=np.int64))
    if camera_bytes > 8 * 1024 * 1024 or not 1 <= state_floats <= 1024:
        raise ValueError("request exceeds protocol limits")
    cameras = np.frombuffer(_recv_exact(conn, camera_bytes),
                            dtype=np.uint8).copy().reshape(shape)
    state = np.frombuffer(_recv_exact(conn, state_floats * 4),
                          dtype="<f4").copy()
    return str(header["language"]), cameras, state


def _connection_loop(conn: socket.socket, requests: queue.Queue) -> None:
    with conn:
        while True:
            try:
                language, cameras, state = _read_request(conn)
            except EOFError:
                return
            except Exception:
                try:
                    conn.sendall(_U32.pack(0))
                finally:
                    return
            reply: queue.Queue = queue.Queue(maxsize=1)
            requests.put(Request(language, cameras, state, reply))
            result = reply.get()
            if isinstance(result, BaseException):
                conn.sendall(_U32.pack(0))
                continue
            action = np.ascontiguousarray(result, dtype="<f4").reshape(-1)
            if (not action.size or action.size > _MAX_ACTION_FLOATS or
                    action.size % ACTION_DIM or not np.isfinite(action).all()):
                conn.sendall(_U32.pack(0))
                continue
            conn.sendall(_U32.pack(action.size) + action.tobytes())


def _accept_loop(listener: socket.socket, requests: queue.Queue) -> None:
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=_connection_loop, args=(conn, requests),
                         daemon=True).start()


def serve(socket_path: str, infer, *, batch_size: int = 32,
          batch_wait_ms: float = 3.0) -> None:
    """Serve `infer` on `socket_path` until the process is terminated."""
    requests: queue.Queue = queue.Queue()
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    os.chmod(socket_path, 0o666)
    listener.listen(128)
    threading.Thread(target=_accept_loop, args=(listener, requests), daemon=True).start()
    print(f"policy server ready: {socket_path}", flush=True)

    while True:
        first = requests.get()
        batch = [first]
        deadline = time.monotonic() + batch_wait_ms / 1000.0
        while len(batch) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(requests.get(timeout=remaining))
            except queue.Empty:
                break
        try:
            output = infer(batch)
            if len(output) != len(batch):
                raise ValueError("inference batch size mismatch")
            for request, action in zip(batch, output):
                request.reply.put(np.asarray(action, dtype=np.float32))
        except BaseException as exc:
            for request in batch:
                request.reply.put(exc)
