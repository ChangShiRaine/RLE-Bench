"""Static Stability Factor (SSF) = track_width / (2 * CoM_height).

Cheap design-space scalar (automotive rollover metric). Diagnostic only,
not a pass gate. Higher = more rollover-resistant.
"""
from __future__ import annotations


def static_stability_factor(track_width: float, com_height: float) -> float:
    """track_width / (2 * com_height); requires a positive CoM height."""
    if com_height <= 0.0:
        raise ValueError(f"com_height must be > 0, got {com_height}")
    if track_width < 0.0:
        raise ValueError(f"track_width must be >= 0, got {track_width}")
    return track_width / (2.0 * com_height)
