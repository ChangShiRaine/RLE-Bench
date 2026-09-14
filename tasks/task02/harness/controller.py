"""What an observation contains.

A CONTROLLER holds `sim`, acts, and decides when to stop:

    def reach(sim, target, tol=0.01, max_steps=200):
        for _ in range(max_steps):
            obs = sim.step(delta_towards(target, obs))["obs"]
            if close_enough(obs, target, tol):
                return {"ok": True}
        return {"ok": False, "reason": "stalled"}

It is an ordinary function -- there is no interface to implement and nothing to submit.
Looking (`sim.observe()`) is free and works inside the loop, so a controller can verify
its own progress, and controllers compose into controllers.

`ObsSpec` is the one shared vocabulary: it says what a render contains, in both
`sim.observe()` and `sim.step()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObsSpec:
    """How big the pictures are, and which ones.

        sim.observe(ObsSpec(width=512, depth=True))   # free, any time
        sim.step(action, ObsSpec(width=256))          # render this step

    `None` means "the task default", so `ObsSpec()` changes nothing. Oversized values are
    clamped, not refused; unknown camera names are ignored.

    Size costs wall clock, never interaction budget: a bigger `width` is render, transfer
    and decode time, and `cameras=()` makes a step much cheaper. `depth=True` adds a
    `<camera>_depth` map in METRES (z-depth along the camera's optical axis, not distance
    along the pixel's ray) beside each `<camera>_image`, for the same cameras at
    the same size, and roughly doubles what an observation costs.
    """

    width: int | None = None
    height: int | None = None            # None mirrors width, so squares are one number
    cameras: tuple[str, ...] | None = None    # None = all of them; () = none at all
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


# The cheapest observation there is: proprioception only, no render. What `sim.step()`
# returns unless asked for more.
PROPRIO = ObsSpec(cameras=())


class Ended:
    """Why the episode or trial ended. `None` while it is still alive.

    All of these are the harness's, not a controller's: a controller stops by returning,
    and cannot claim success by stopping.
    """

    ENV_SUCCESS = "env_success"            # the task's own success predicate fired
    ENV_DONE = "env_done"                  # env returned done
    HORIZON = "horizon"                    # episode step limit
    BUDGET_EXHAUSTED = "budget_exhausted"  # interaction budget ran out
