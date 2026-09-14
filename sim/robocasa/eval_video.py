"""Evaluation-trial videos: what the agent's cameras saw during the scored phase.

Recorded by the ROOT daemon through rlebench.core.media into a tree the agent
cannot traverse (/opt/private, 0700), and copied next to reward.json by
verify_main once the phase is over. Evidence for a human, never an input to the
score.

The frames are the observations the harness already holds -- the three agent
cameras tiled side by side, exactly what the agent saw -- so recording renders
nothing extra and discloses nothing the agent did not already have. It still
stays root-only until the verifier exports it: the agent is running while the
trials are recorded and must not be able to read or replace the files.

Development episodes are not recorded: unbounded in number, and not the scored
run. RLEBENCH_MEDIA=0 turns recording off. Every hook is safe to call
unconditionally and nothing here can raise into the session.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rlebench.core.media import VIDEO_FPS, Media, Recorder, enabled as media_enabled

from .debug import tile_frame

DEFAULT_ROOT = "/opt/private/media"
EVERY_N_STEPS = 2        # 20 Hz control -> 10 fps video, the debug view's cadence


class EvalVideoRecorder:
    """Recorder-shaped, like debug.DebugRecorder; the session calls both."""

    def __init__(self, root: str | os.PathLike[str] = DEFAULT_ROOT, *,
                 enabled: bool = True, every_n_steps: int = EVERY_N_STEPS,
                 fps: int = VIDEO_FPS):
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.every_n_steps = max(1, int(every_n_steps))
        self.fps = int(fps)
        self.media = Media(self.root, on=self.enabled)
        self._recording = False
        self._trial = 0
        self._rec: Recorder | None = None

    # -- lifecycle -----------------------------------------------------------
    def episode_started(self, phase: str, index: int, obs: dict | None = None) -> None:
        if not self.enabled:
            return
        self._finish()
        self._recording = phase == "evaluation"
        self._trial = int(index)
        self.frame(obs)

    def frame(self, obs: dict | None) -> None:
        if not (self.enabled and self._recording):
            return
        try:
            if self._rec is None:
                writer = self.media.video(f"trial{self._trial:03d}.mp4", fps=self.fps)
                self._rec = Recorder(writer, every=self.every_n_steps) if writer else None
            if self._rec is not None and self._rec.wants_frame():
                img = tile_frame(obs)
                if img is not None:
                    self._rec.add(img)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval-video] {exc}", flush=True)
            self._finish()
            self._recording = False

    def event(self, op: str, **fields: Any) -> None:
        if op == "trial_end":
            self._finish()

    def diagnosis(self, *args: Any, **kwargs: Any) -> None:
        """Ground truth stays out of this tree by construction."""

    def status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        if self.enabled:
            self._finish()

    def _finish(self) -> None:
        rec, self._rec = self._rec, None
        if rec is None:
            return
        try:
            self.media.finish(rec.stream)
            self.media.close()          # index.json stays current for the verifier
        except Exception as exc:  # noqa: BLE001
            print(f"[eval-video] {exc}", flush=True)


class Recorders:
    """Fan-out to several recorder-shaped objects; one failing never reaches
    another, or the session."""

    def __init__(self, *recorders):
        self.recorders = [r for r in recorders if r is not None]
        self.enabled = any(getattr(r, "enabled", False) for r in self.recorders)

    def __getattr__(self, name: str):
        def call(*args, **kwargs):
            for r in self.recorders:
                fn = getattr(r, name, None)
                if fn is None:
                    continue
                try:
                    fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    print(f"[recorder] {type(r).__name__}.{name}: {exc}", flush=True)
        return call


def from_env() -> EvalVideoRecorder:
    """RLEBENCH_MEDIA=0 turns it off; RLEBENCH_EVAL_MEDIA_DIR relocates it."""
    return EvalVideoRecorder(os.environ.get("RLEBENCH_EVAL_MEDIA_DIR", DEFAULT_ROOT),
                             enabled=media_enabled())
