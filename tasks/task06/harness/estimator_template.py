"""Starter interface for code-based Task 06 submissions.

Copy this file to ``/logs/artifacts/estimator.py`` and implement
``Estimator.update``. This source file is a template; it is not loaded
automatically from the agent workspace.
"""
from __future__ import annotations

from typing import Any


class Estimator:
    """Stateful planar-pose estimator called once per observation frame."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear temporal state before a push episode or independent frame."""

    def update(
        self,
        *,
        rgb: Any,
        K: Any,
        T_cam_table: Any,
        t: float,
        depth: Any | None = None,
    ) -> tuple[float, float, float, int]:
        """Return ``(x, y, theta, shape_id)``; IDs are 0=T, 1=C, 2=F.

        ``x`` and ``y`` are meters. ``theta`` is radians about +z, wrapped to
        ``(-pi, pi]``. The RGB-only task omits ``depth``. Implementations may
        update internal tracking state before returning the pose and class.
        """
        raise NotImplementedError


def make_estimator() -> Estimator:
    """Factory used to create one estimator instance."""
    return Estimator()
