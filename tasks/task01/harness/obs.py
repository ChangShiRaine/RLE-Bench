"""What an observation should contain.

`ObsSpec` is the one thing the agent passes in. How you organise the code around
`sim.step` and `sim.observe` is entirely yours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObsSpec:
    """How big the pictures are, which of them you want, and whether range comes too.

    The same object for looking and for acting, so what you look at and what you act on
    cannot disagree:

        sim.observe(ObsSpec(width=384))                  # a free look, your size
        sim.step(action, obs_spec=ObsSpec(width=384))    # the same, on what comes back
        sim.step(action, obs_spec=ObsSpec(cameras=()))   # no pictures at all

    Per call; nothing is remembered between calls, and a batch renders ONCE, for the
    observation it hands back. `None` fields mean "the task default", so `ObsSpec()`
    changes nothing. Sizes above the maximum are clamped, not refused, and unknown
    camera names are ignored.

    Both directions cost wall clock, never interaction budget: a bigger `width` is
    render, transfer and decode time on every step; `cameras=()` removes images
    entirely. `depth=True` adds a `<camera>_depth` map beside each `<camera>_image`, in
    METRES, same cameras and same size -- off by default, since it roughly doubles what
    an observation costs.
    """

    width: int | None = None
    height: int | None = None            # None mirrors width, so squares are one number
    cameras: tuple[str, ...] | None = None    # None = all of them; () = none at all
    # Not `None`-defaulted like the others: there is no "task default" for depth to
    # differ from. Off unless asked for, in both phases.
    depth: bool = False

    def as_wire(self) -> dict:
        """The JSON-safe form; empty when nothing is constrained."""
        out: dict[str, Any] = {}
        if self.width is not None:
            out["width"] = int(self.width)
        if self.height is not None:
            out["height"] = int(self.height)
        if self.cameras is not None:
            out["cameras"] = list(self.cameras)
        # Omitted when false, so an older peer's messages decode unchanged.
        if self.depth:
            out["depth"] = True
        return out

    @classmethod
    def from_wire(cls, raw: Any) -> "ObsSpec | None":
        if not isinstance(raw, dict):
            return None
        cameras = raw.get("cameras")
        return cls(
            width=raw.get("width"),
            height=raw.get("height"),
            cameras=None if cameras is None else tuple(cameras),
            depth=bool(raw.get("depth", False)),
        )
