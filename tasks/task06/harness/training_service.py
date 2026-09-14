"""Root-only rendering service for the Task 06 family design data.

This process is the sole owner of the block meshes in the agent container.
Requests contain only public generation parameters. Responses contain RGB-D
observations and pose labels, never the selected shape name.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
import socket

import numpy as np

from . import design_evaluator, episodes, spec

SOCKET_MODE = 0o666
MAX_REQUEST_BYTES = 4096
MAX_EVALUATOR_CALLS = 6
EVALUATOR_DECIMALS = 4
_evaluator_calls = 0
LABEL_XY_STD_M = 0.005
LABEL_THETA_STD_RAD = np.deg2rad(3.0)
STREAM_SINGLE_LABEL_NOISE = 12051
STREAM_PUSH_LABEL_NOISE = 12052


def _validate_seed(value) -> int:
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    seed = int(value)
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    return seed


def _noisy_gts(frames, seed: int, operation: str) -> np.ndarray:
    stream = (STREAM_SINGLE_LABEL_NOISE if operation == "single_frame"
              else STREAM_PUSH_LABEL_NOISE)
    rng = np.random.default_rng([seed, stream])
    truth = np.asarray([frame.gt for frame in frames], dtype=np.float64)
    noisy = truth.copy()
    noisy[:, :2] += rng.normal(0.0, LABEL_XY_STD_M, size=(len(frames), 2))
    noisy[:, 2] += rng.normal(0.0, LABEL_THETA_STD_RAD, size=len(frames))
    noisy[:, 2] = np.pi - np.mod(np.pi - noisy[:, 2], 2 * np.pi)
    return noisy


def _encode(frames, modalities: str | None = None, *,
            label_seed: int | None = None,
            operation: str = "single_frame") -> bytes:
    if not frames or any(frame.obs is None for frame in frames):
        raise RuntimeError("renderer returned no observations")
    modalities = modalities or os.environ.get(
        "RLEBENCH_MODALITIES", "rgb,depth")
    requested = {item.strip() for item in modalities.split(",") if item.strip()}
    if not requested or not requested <= {"rgb", "depth"}:
        raise RuntimeError("invalid RLEBENCH_MODALITIES")
    first = frames[0].obs
    arrays = {
        "K": np.asarray(first["K"]),
        "T_cam_table": np.asarray(first["T_cam_table"]),
        "t": np.asarray([frame.t for frame in frames], dtype=np.float64),
        "gt": (_noisy_gts(frames, label_seed, operation)
               if label_seed is not None
               else np.asarray([frame.gt for frame in frames],
                               dtype=np.float64)),
        "occluded": np.asarray(
            [frame.occluded for frame in frames], dtype=np.bool_),
    }
    if "rgb" in requested:
        arrays["rgb"] = np.stack([frame.obs["rgb"] for frame in frames])
    if "depth" in requested:
        arrays["depth"] = np.stack([frame.obs["depth"] for frame in frames])
    buf = BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _rounded_report(value):
    if isinstance(value, float):
        return round(value, EVALUATOR_DECIMALS)
    if isinstance(value, dict):
        return {key: _rounded_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_report(item) for item in value]
    return value


def dispatch(message: dict) -> bytes:
    global _evaluator_calls
    op = message.get("op")
    if op == "evaluate":
        if _evaluator_calls >= MAX_EVALUATOR_CALLS:
            raise RuntimeError("diagnostic query budget exhausted")
        _evaluator_calls += 1
        variant = os.environ.get("RLEBENCH_VARIANT", "d")
        report = _rounded_report(
            design_evaluator.evaluate_submission(variant))
        report["queries_remaining"] = MAX_EVALUATOR_CALLS - _evaluator_calls
        return json.dumps(report, allow_nan=False).encode()
    seed = _validate_seed(message.get("seed"))
    if op == "single_frame":
        frame = episodes.single_frame(
            seed, render=True, shapes=spec.BLOCK_SHAPES)
        return _encode([frame], label_seed=seed, operation=op)
    if op == "push_episode":
        n_strokes = int(message.get("n_strokes", episodes.MAX_STROKES))
        max_frames = int(message.get("max_frames", 100))
        if not 1 <= n_strokes <= episodes.MAX_STROKES:
            raise ValueError(f"n_strokes must be in [1, {episodes.MAX_STROKES}]")
        if not 1 <= max_frames <= 100:
            raise ValueError("max_frames must be in [1, 100]")
        episode = episodes.push_episode(
            seed,
            n_strokes=n_strokes,
            render=True,
            max_frames=max_frames,
            shapes=spec.BLOCK_SHAPES,
        )
        return _encode(episode.frames, label_seed=seed, operation=op)
    raise ValueError("unknown operation")


def _serve_connection(conn: socket.socket) -> None:
    stream = conn.makefile("rb")
    line = stream.readline(MAX_REQUEST_BYTES + 1)
    if not line or len(line) > MAX_REQUEST_BYTES:
        raise ValueError("invalid request")
    message = json.loads(line)
    payload = dispatch(message)
    header = json.dumps({"ok": True, "size": len(payload)}).encode() + b"\n"
    conn.sendall(header)
    conn.sendall(payload)


def serve(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        os.chmod(path, SOCKET_MODE)
        server.listen(8)
        print(f"[task06-training] ready on {path}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    _serve_connection(conn)
                except Exception as exc:
                    error = type(exc).__name__
                    reply = json.dumps(
                        {"ok": False, "error": f"request failed: {error}"}
                    ).encode() + b"\n"
                    try:
                        conn.sendall(reply)
                    except OSError:
                        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--socket",
        default=os.environ.get(
            "RLEBENCH_TRAINING_SOCKET",
            "/run/rlebench/task06-training.sock",
        ),
    )
    args = parser.parse_args(argv)
    serve(args.socket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
