"""Agent-side client for the metering daemon.

The ONLY way to reach the simulator: the environment lives in the daemon, so nothing here
can step it without being counted.

Two calls do the work, and they behave the same in both phases:

    sim.observe(spec)    look. FREE, unlimited, changes nothing.
    sim.step(actions)    act. Charged. One action or a list of them.

Development -- practise across the training tasks:

    from harness.client import ToolsmithClient

    with ToolsmithClient() as sim:
        tasks = sim.list_tasks()           # the training split, in full
        sim.reset(task=tasks[0])           # pick what to practise on
        res = my_controller(sim, target)   # your loop, holding sim
        if res["episode_over"]:
            sim.reset()                    # same task again, or name another
        sim.end_development()              # when your harness is ready

Evaluation -- one graded trial, driven by an agent who did not develop this:

    with ToolsmithClient() as sim:
        print(sim.trial_info()["instruction"])   # the goal, from the simulator
        my_controller(sim, target)               # the same loop, unchanged

Entering the scored phase is deliberately not here: call `end_development()` when ready
and the harness opens the evaluation. Nor is advancing between trials -- the harness moves
between them.
"""

from __future__ import annotations

import os
import socket
from typing import Any

from . import protocol as P
from .controller import ObsSpec

DEFAULT_SOCKET = os.environ.get(
    "RLEBENCH_TOOLSMITH_SOCKET", "/run/rlebench/toolsmith.sock"
)


class RemoteError(RuntimeError):
    """The daemon refused a request. `kind` distinguishes budget exhaustion from a
    malformed request, so a caller can tell 'out of budget' from 'my bug'."""

    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


class BudgetExhausted(RemoteError):
    pass


class ToolsmithClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self.socket_path = str(socket_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)
        self._reader = P.LineReader(self._sock)

    # -- plumbing ------------------------------------------------------------
    def _request(self, op: str, **fields: Any) -> dict:
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
    def list_tasks(self) -> list[str]:
        """The training tasks you may practise on. Free.

        Each is a multi-step procedure, not a single motion. Dividing a finite budget
        across them is part of the problem -- you are not expected to master all of them.
        """
        return list(self._request("list_tasks")["tasks"])

    def task_info(self) -> dict:
        """Action layout, dimension, budgets, the current instruction, the clock. Free."""
        return self._request("task_info")

    def status(self) -> dict:
        """Budgets spent and remaining, and the time left in this phase. Free.

        `seconds_remaining` is this phase's wall clock -- the real number for this run
        rather than any default, so read it rather than assuming one.
        """
        return self._request("status")

    def observe(self, spec: ObsSpec | None = None, *, width: int | None = None,
                height: int | None = None, cameras=None, depth: bool = False) -> dict:
        """Look at the current state without acting. FREE, in BOTH phases.

        DEFAULT SIZE IS THE TASK DEFAULT, which is what the step pipeline already
        rendered, so the default look costs a camera selection rather than a re-render.
        `resolution` in the reply says what you got and `max_resolution` is the ceiling.
        Ask for more when you need the detail.

        Returns {"obs", "instruction", "resolution", "max_resolution", "live"}. Nothing
        is stepped, charged or advanced, so a graded trial can be inspected as much as
        you like before deciding what to run.

        `live` is False once the episode or trial is over. An observation taken then is
        thinner than the one you asked for -- there is nothing to render against -- so
        check it rather than discovering the gap as a `KeyError` further down.

        Takes the same `ObsSpec` as `step()`, and works inside a controller's own loop --
        that is how a controller verifies its progress without paying for it:

            look = sim.observe(ObsSpec(width=384))
            look = sim.observe(width=384)          # the same, spelled shorter
            img = look["obs"]["robot0_agentview_left_image"]

        `depth=True` adds `<camera>_depth`, an HxW float32 map in METRES, for the same
        cameras at the same size. It is Z-DEPTH -- distance along the camera's optical
        axis, not along the pixel's own ray:

            look = sim.observe(width=512, depth=True)
            d = look["obs"]["robot0_eye_in_hand_depth"]   # metres along the optical axis
            print(d[v, u])                                # z-depth at pixel (u, v)

        Requests above `max_resolution` are clamped, not refused. Bigger frames cost only
        render and transfer time -- wall clock, which is also a budget.
        """
        if spec is None:
            spec = ObsSpec(width=width, height=height,
                           cameras=None if cameras is None else tuple(cameras),
                           depth=bool(depth))
        r = self._request("observe", **spec.as_wire())
        return {k: v for k, v in r.items() if k != "ok"}

    def reset(self, task: str | None = None, seed: int | None = None) -> dict:
        """Start a new episode -- or, during evaluation, GIVE UP the graded trial.

        DEVELOPMENT: `task` must be one of `list_tasks()`. The first reset has to name
        one; after that, omitting it stays on the current task. COSTS ONE INTERACTION
        STEP, charged to the task being reset to.

        EVALUATION: reset means "I am done with this attempt". The trial is scored
        immediately in whatever state it is in. The reply tells you it is over; it does
        not tell you how it scored.
        """
        fields: dict[str, Any] = {}
        if task is not None:
            fields["task"] = str(task)
        if seed is not None:
            fields["seed"] = int(seed)
        r = self._request("reset", **fields)
        if r.get("gave_up_trial"):
            return {k: v for k, v in r.items() if k not in ("ok", "gave_up_trial")}
        return r["obs"]

    def end_development(self) -> dict:
        """Declare development finished. Your signal, and the only one needed.

        It starts no evaluation and touches no simulator: it closes the development phase
        and records the interaction spent at the moment you judged yourself ready. The
        harness opens the scored phase afterwards, for a different agent.

        Development interaction is refused after this, so call it when you mean it.
        Calling it twice is harmless; never calling it ends development with the clock.
        """
        r = self._request("end_development")
        return {k: v for k, v in r.items() if k != "ok"}

    def phase(self) -> str:
        """'development' or 'evaluation'. Worth checking: during evaluation, reset() ends
        and scores the current trajectory instead of restarting it."""
        return str(self.status()["phase"])

    def trial_info(self) -> dict | None:
        """The graded trial in progress, or None outside evaluation.

        The harness opens the trial before its agent starts, so this is how you see what
        you were dropped into: the instruction, the observation, and where it sits.
        """
        return self._request("trial_info")["trial"]

    def step(self, actions, obs_spec: ObsSpec | None = None) -> dict:
        """Act in the current episode or trial. CHARGED, one unit per action.

        `actions` is one action or a sequence of them. A sequence is applied in order and
        costs exactly what it applies -- use it for open-loop stretches (holding the
        gripper shut, letting the arm settle), where it saves a round trip per step:

            sim.step(action)                  # one, closed loop
            sim.step([hold] * 15)             # fifteen, one round trip

        Returns {"obs", "steps", "episode_over", "success", "ended", "steps_remaining"}.
        `steps` is how many were APPLIED, which is fewer than asked for when the episode
        ended part-way; `ended` says why, and is None while it is alive. Applying a
        sequence stops there rather than stepping a finished episode.

        `obs` is proprioception only unless `obs_spec` asks for more, because most steps
        do not need pixels and rendering is what makes a step expensive in wall clock.
        Looking is free and separate: call `observe()` whenever you want to see.
        """
        batch = actions if _is_batch(actions) else [actions]
        r = self._request(
            "step", actions=[_as_list(a) for a in batch],
            **({} if obs_spec is None else {"obs_spec": obs_spec.as_wire()}),
        )
        # `ok` is wire framing, stripped here as it is from every other reply -- it is
        # always True by the time `_request` returns, since a false one raised.
        return {k: v for k, v in r.items() if k != "ok"}

    def disconnect(self) -> None:
        """Drop this connection, leaving the run open.

        Connecting is cheap and the session lives in the daemon, so opening and closing
        clients freely is fine -- state persists across connections and across steps.
        ONE CONNECTION AT A TIME, though: the daemon refuses a second concurrent
        client, so close this one before opening the next.
        """
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "ToolsmithClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Disconnect only: leaving a `with` block must not end the run.
        self.disconnect()


def _as_list(action) -> list:
    tolist = getattr(action, "tolist", None)
    return tolist() if callable(tolist) else list(action)


def _is_batch(actions) -> bool:
    """A batch is a sequence of actions; a single action is a sequence of numbers."""
    ndim = getattr(actions, "ndim", None)
    if ndim is not None:
        return int(ndim) > 1
    return len(actions) > 0 and hasattr(actions[0], "__len__")
