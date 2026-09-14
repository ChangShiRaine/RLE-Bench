"""Agent-side client for the metering daemon.

This is the ONLY way to reach the simulator: the environment lives in the daemon, so
there is nothing here that can step it without being counted.

    from harness.client import SpeedrunClient, ObsSpec

    with SpeedrunClient() as sim:
        info = sim.task_info()          # action layout, dim, budgets, clock
        obs = sim.reset()               # opens an episode; costs one step
        look = sim.observe(width=256)   # free: see without acting

        for _ in range(20):
            res = sim.step(my_action(obs))
            obs = res["obs"]
            if res["episode_over"]:     # success, horizon, trial cap, budget
                sim.reset()             # dev: next episode | eval: score, next
                break

`step` and `observe` are the whole acting surface, and they work the same way in both
phases. There is no class to implement and nothing to hand over.

The session lives in the daemon, not in your process, and survives disconnecting -- so
one script can run, and a different script can carry on from where it stopped.

Entering the scored phase is not here, by design: say `end_development()` when you are
ready, and the harness opens the evaluation between phases.
"""

from __future__ import annotations

import os
import socket
from typing import Any

from . import protocol as P
from .obs import ObsSpec

DEFAULT_SOCKET = os.environ.get(
    "RLEBENCH_SPEEDRUN_SOCKET", "/run/rlebench/speedrun.sock"
)

class RemoteError(RuntimeError):
    """The daemon refused a request. `kind` distinguishes budget exhaustion from a
    malformed request, so a caller can tell 'out of budget' from 'my bug'."""

    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


class BudgetExhausted(RemoteError):
    pass


class SpeedrunClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self.socket_path = str(socket_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)
        self._reader = P.LineReader(self._sock)

    # -- plumbing ------------------------------------------------------------
    def _request(self, op: str, **fields: Any) -> dict:
        """Send one request, read one reply. Request/reply throughout -- the daemon
        never calls back into this process, which is what lets you drive the simulator
        from an ordinary loop in your own code."""
        try:
            P.send(self._sock, {"op": op, **fields})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the daemon may have refused and hung up first; its reply is still queued
        try:
            msg = self._reader.read()
        except OSError:
            msg = None
        if msg is None:
            raise RemoteError("daemon closed the connection")
        if not msg.get("ok", False):
            kind = msg.get("kind")
            err = str(msg.get("error", "unknown error"))
            if kind == "budget_exhausted":
                raise BudgetExhausted(err, kind)
            raise RemoteError(err, kind)
        return msg

    # -- API -----------------------------------------------------------------
    def task_info(self) -> dict:
        return self._request("task_info")

    def status(self) -> dict:
        """Budgets spent and remaining, and the time left in this phase. Free.

        The step figures are phase-local: the interaction budget in development, the
        current trial's count and remaining ceiling in evaluation.

        `seconds_remaining` is this phase's wall clock. It is the real number for this
        run rather than any default, so read it rather than assuming one.
        """
        return self._request("status")

    def observe(self, spec: ObsSpec | None = None, *, width: int | None = None,
                height: int | None = None, cameras=None, depth: bool = False) -> dict:
        """Look at the current state without acting. FREE, in BOTH phases.

        Returns {"obs", "instruction", "resolution", "max_resolution", "live"}. Nothing
        is stepped, charged or advanced, so a graded trial can be inspected as much as
        you like before you decide what to run.

        `live` is False once the episode or trial is over. An observation taken then is
        thinner than the one you asked for -- there is nothing to render against -- so
        check it rather than discovering the gap as a `KeyError` further down.

            look = sim.observe(ObsSpec(width=384))
            look = sim.observe(width=384)          # the same thing, spelled shorter
            look = sim.observe(width=512, depth=True)      # + <camera>_depth, in metres

        Takes the same ObsSpec `step` does. A request above `max_resolution` is clamped
        rather than refused; bigger frames cost render and transfer time, which is wall
        clock, and wall clock is a budget.
        """
        if spec is None:
            spec = ObsSpec(width=width, height=height,
                           cameras=None if cameras is None else tuple(cameras),
                           depth=bool(depth))
        r = self._request("observe", **spec.as_wire())
        return {k: v for k, v in r.items() if k != "ok"}

    def reset(self, seed: int | None = None) -> dict:
        """Start a new episode -- or, during evaluation, GIVE UP the current trial.

        Outside evaluation this resets the workspace and arm and returns an
        observation, and COSTS ONE INTERACTION STEP -- a reset rebuilds or
        re-randomises a scene, and a free one would make abandoning an episode cheaper
        than finishing it.

        During evaluation reset is a special move meaning "abandon this trial": the
        trial is scored immediately where it stands, and the reply describes the next
        trial, or the finished evaluation.
        """
        r = self._request("reset", **({} if seed is None else {"seed": seed}))
        if r.get("gave_up_trial"):
            return {k: v for k, v in r.items() if k not in ("ok", "gave_up_trial")}
        return r["obs"]

    def end_development(self) -> dict:
        """Declare development finished, when you judge yourself ready.

        It closes the development phase and records the interaction spent up to that
        moment. It starts no evaluation and does not touch the simulator; the harness
        opens the scored phase afterwards. Development interaction is refused after
        this, so call it when you mean it. Calling it twice is harmless.
        """
        r = self._request("end_development")
        return {k: v for k, v in r.items() if k != "ok"}

    def phase(self) -> str:
        """'development', 'evaluation', or 'finished' once the run is over.

        Worth checking: during evaluation, reset() ends and scores the current
        trajectory instead of merely restarting it."""
        return str(self.status()["phase"])

    def trial_info(self) -> dict | None:
        """The trial currently in progress, or None outside evaluation.

        The harness opens the evaluation before the agent starts, so use this to see
        the trial you have been dropped into (observation, instruction, progress).
        """
        return self._request("trial_info")["trial"]

    def step(self, actions, obs_spec: ObsSpec | None = None) -> dict:
        """Act in the current episode or trial. CHARGED, one unit per action.

        `actions` is one action or a sequence of them, applied in order:

            sim.step(action)                  # one, closed loop
            sim.step([hold] * 15)             # fifteen, one round trip

        Returns {"obs", "steps", "success", "episode_over", "ended", "steps_remaining"}.
        `steps` is how many were APPLIED, fewer than asked for when the episode ended
        part-way; `ended` says why, and is None while it is alive. A sequence stops
        there rather than stepping a finished episode.
        """
        r = self._request(
            "step", actions=[_as_list(a) for a in _as_batch(actions)],
            **({} if obs_spec is None else {"obs_spec": obs_spec.as_wire()}))
        return {k: v for k, v in r.items() if k != "ok"}

    def disconnect(self) -> None:
        """Drop this connection, leaving the run open. State lives in the daemon, so
        opening and closing clients freely is fine -- ONE AT A TIME: the daemon
        refuses a second concurrent client, so close this one before the next."""
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "SpeedrunClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Disconnect only: leaving a `with` block must not end the run.
        self.disconnect()


def _as_list(action) -> list:
    tolist = getattr(action, "tolist", None)
    return tolist() if callable(tolist) else list(action)


def _as_batch(actions) -> list:
    """A batch is a sequence of actions; a single action is a sequence of numbers."""
    ndim = getattr(actions, "ndim", None)
    if ndim is not None:
        return list(actions) if int(ndim) > 1 else [actions]
    if len(actions) and hasattr(actions[0], "__len__"):
        return list(actions)
    return [actions]
