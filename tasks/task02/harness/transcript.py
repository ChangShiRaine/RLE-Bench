"""Human-readable record of an evaluation: what happened, and what the env thought.

Two audiences, and they need different things:

  * the SCORER reads the ledger -- root-owned, hash-chained, minimal;
  * a HUMAN debugging a run needs the story: which controllers ran, when control came
    back and why, what the environment looked like when a trial ended, and pictures.

This module writes the second. It is explicitly **not** evidence. Artifacts land in
/logs/artifacts, which Harbor makes agent-writable, so anything here could in principle
be edited after the fact; scoring therefore never reads it. To keep it trustworthy
anyway, the transcript's digest is recorded in the ledger's seal, so tampering with the
human record is detectable even though it is not scored.

Records are newline-delimited JSON:

    {"kind": "trial_start", "trial": 0, "scene": 1, "instruction": "...", ...}
    {"kind": "segment",     "trial": 0, "index": 0, "steps": 12, "reason": "...", ...}
    {"kind": "trial_end",   "trial": 0, "ended_by": "agent_reset", "diagnosis": {...}}
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Observation keys worth keeping verbatim: small, and they are what a human actually
# looks at first. Images are handled separately (saved as PNG, not inlined).
_ROBOT_KEYS = (
    "robot0_eef_pos", "robot0_eef_quat", "robot0_base_pos", "robot0_base_quat",
    "robot0_gripper_qpos", "robot0_joint_pos",
)


def _jsonable(value: Any) -> Any:
    """Convert to something json.dumps accepts, recursing into containers.

    Containers must be recursed rather than str()'d: numpy arrays have .tolist() and so
    survived, but plain lists and dicts fell through to str() and were silently
    stringified, which turns a readable record into quoted noise.
    """
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def diagnose(env, obs: dict | None) -> dict:
    """What the ENVIRONMENT says about the current state, for a human to read.

    Deliberately generic: it collects what every env exposes and degrades gracefully
    rather than special-casing any task. Per-task ground truth lives in stages.py, which
    is scored; this is not. Every lookup is guarded -- a diagnosis that raises would take
    down a trial it was only meant to describe.
    """
    d: dict[str, Any] = {}
    try:
        d["success"] = bool(env._check_success())
    except Exception as exc:  # noqa: BLE001
        d["success"] = None
        d["success_error"] = f"{type(exc).__name__}: {exc}"

    try:
        meta = env.get_ep_meta() or {}
        d["instruction"] = meta.get("lang")
        d["layout_id"] = meta.get("layout_id")
        d["style_id"] = meta.get("style_id")
    except Exception:  # noqa: BLE001
        pass

    if obs:
        d["robot"] = {k: _jsonable(obs[k]) for k in _ROBOT_KEYS if k in obs}
        # Object poses: everything pose-shaped that is not the robot or an image.
        d["objects"] = {
            k: _jsonable(v) for k, v in sorted(obs.items())
            if (k.endswith("_pos") or k.endswith("_quat"))
            and not k.startswith("robot0_") and not k.endswith("_image")
        }

    # Articulated fixtures expose a normalized open/closed state, which is usually the
    # single most informative number for drawer/door/cabinet tasks.
    for attr in ("drawer", "door", "cab", "fixture", "microwave"):
        fixture = getattr(env, attr, None)
        getter = getattr(fixture, "get_door_state", None) if fixture else None
        if callable(getter):
            try:
                d["fixture_state"] = {attr: {k: float(v)
                                             for k, v in getter(env=env).items()}}
            except Exception:  # noqa: BLE001
                pass
            break
    return d


def save_frames(obs: dict | None, dest: Path, prefix: str) -> list[str]:
    """Write the camera views as PNGs. Best-effort: a missing encoder must not break
    a run, and the frames are the first thing a human wants when a trial fails."""
    if not obs:
        return []
    written = []
    try:
        from PIL import Image
    except ImportError:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    for key in sorted(k for k in obs if k.endswith("_image")):
        try:
            import numpy as np

            arr = np.asarray(obs[key])
            if arr.ndim != 3:
                continue
            # NO FLIP. Frames arrive upright: env.make_env sets robosuite's image
            # convention at the source, so the array saved here is byte-identical to
            # the one the agent was given. Flipping again would invert it, and would
            # also mean this record no longer shows what the agent actually saw.
            path = dest / f"{prefix}_{key}.png"
            Image.fromarray(arr.astype("uint8")).save(path)
            written.append(path.name)
        except Exception:  # noqa: BLE001
            continue
    return written


class Transcript:
    """Append-only evaluation log plus per-trial diagnosis and frames."""

    def __init__(self, path: str | os.PathLike[str], frames_dir: str | os.PathLike[str]
                 | None = None, save_images: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir = Path(frames_dir) if frames_dir else self.path.parent / "frames"
        self.save_images = save_images

    def _write(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def trial_start(self, trial: int, task: str, seed: int,
                    env=None, obs: dict | None = None) -> None:
        # Keyed by TASK rather than by (scene, layout, style): task02 runs one composite
        # task per trial and does not pin scenes, so the task name is what identifies a
        # trial and the layout it drew is a detail `diagnose` already records.
        rec = {"kind": "trial_start", "trial": trial, "task": task, "seed": seed}
        if env is not None:
            rec["diagnosis"] = diagnose(env, obs)
        if self.save_images:
            rec["frames"] = save_frames(obs, self.frames_dir, f"trial{trial:03d}_start")
        self._write(rec)

    def step(self, trial: int, index: int, steps: int, ended, success: bool) -> None:
        """One `step` call: how much of it was applied, and whether it ended the trial."""
        self._write({
            "kind": "step", "trial": trial, "index": index,
            "steps": steps, "ended": ended, "success": bool(success),
        })

    def trial_end(self, trial: int, ended_by: str, success: bool, steps: int,
                  env=None, obs: dict | None = None) -> None:
        rec = {"kind": "trial_end", "trial": trial, "ended_by": ended_by,
               "success": bool(success), "steps": steps}
        if env is not None:
            rec["diagnosis"] = diagnose(env, obs)
        if self.save_images:
            rec["frames"] = save_frames(obs, self.frames_dir, f"trial{trial:03d}_end")
        self._write(rec)

    def summary(self, result: dict) -> None:
        self._write({"kind": "summary", **{k: _jsonable(v) for k, v in result.items()}})

    def digest(self) -> str | None:
        """SHA-256 of the transcript, recorded in the ledger seal so that tampering
        with the human-facing record is detectable even though it is not scored."""
        if not self.path.exists():
            return None
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
