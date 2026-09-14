"""Agent-visible client for the Task 06 family private rendering service.

The service owns MuJoCo and the meshes. This module reconstructs the public
Frame/Episode API from arrays returned over a Unix socket; no shape identity
crosses that boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
import os
import socket

import numpy as np

DEFAULT_SOCKET = "/run/rlebench/task06-training.sock"
MAX_STROKES = 5


@dataclass
class Frame:
    obs: dict | None
    gt: tuple
    occluded: bool
    t: float


@dataclass
class Episode:
    seed: int
    frames: list = field(default_factory=list)

    @property
    def gts(self):
        return [f.gt for f in self.frames]

    @property
    def occluded_flags(self):
        return [f.occluded for f in self.frames]


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("training service closed an incomplete response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _request(message: dict) -> dict:
    path = os.environ.get("RLEBENCH_TRAINING_SOCKET", DEFAULT_SOCKET)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(1800.0)
        sock.connect(path)
        sock.sendall(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        stream = sock.makefile("rb")
        line = stream.readline(4097)
        if not line or len(line) > 4096:
            raise RuntimeError("invalid response from training service")
        header = json.loads(line)
        if not header.get("ok"):
            raise RuntimeError(header.get("error", "training service error"))
        payload = _read_exact(stream, int(header["size"]))
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _frames(arrays: dict, render: bool) -> list[Frame]:
    out = []
    for i in range(len(arrays["t"])):
        obs = None
        if render:
            obs = {
                "K": arrays["K"].copy(),
                "T_cam_table": arrays["T_cam_table"].copy(),
                "t": float(arrays["t"][i]),
            }
            for channel in ("rgb", "depth"):
                if channel in arrays:
                    obs[channel] = arrays[channel][i]
        out.append(Frame(
            obs=obs,
            gt=tuple(float(v) for v in arrays["gt"][i]),
            occluded=bool(arrays["occluded"][i]),
            t=float(arrays["t"][i]),
        ))
    return out


def single_frame(seed: int, render: bool = True) -> Frame:
    """Return one labeled design observation; the active shape is not named."""
    arrays = _request({"op": "single_frame", "seed": int(seed)})
    return _frames(arrays, render)[0]


def push_episode(seed: int, n_strokes: int = MAX_STROKES, render: bool = True,
                 max_frames: int = 100) -> Episode:
    """Return one labeled push episode; all frames use one unnamed shape."""
    arrays = _request({
        "op": "push_episode",
        "seed": int(seed),
        "n_strokes": int(n_strokes),
        "max_frames": int(max_frames),
    })
    return Episode(seed=int(seed), frames=_frames(arrays, render))
