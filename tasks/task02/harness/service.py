"""Metering daemon. Runs as root; the agent talks to it over a Unix socket.

The daemon owns every environment. No agent code runs on the trusted side, and there is no
env in the agent's process to step off-meter: the agent's controller loops in its own
process and reaches the simulator only through the ops below, which are what count.

ONE DAEMON, TWO AGENTS. This shapes everything below: the daemon serves the development
agent and then the evaluation agent, a fresh session with no memory of the first, in the
same container. So the daemon is the only thing in the run that remembers anything, and
what it tells each caller is the task's whole information boundary:

  * the TRAINING split is published (`list_tasks`) -- the agent is meant to practise on it;
  * the EVALUATION split is never named, in any reply, ever;
  * no score is reported in either phase. Telling one trial's agent how it did is a
    channel to the next one, through whatever it writes to disk.

Message flow (newline-delimited JSON, see protocol.py):

    agent -> {"op": "reset", "task": "OpenDrawer"}
    agent <- {"ok": true, "obs": {...}, "instruction": "..."}

    agent -> {"op": "step", "actions": [[...12 floats...], ...]}
    agent <- {"ok": true, "obs": {...}, "steps": 12, "ended": null,
              "episode_over": false, "steps_remaining": 41234}

Two planes over one socket: agent ops are dispatched by `_dispatch`, harness ops by
`_dispatch_control`, which refuses any caller whose kernel-supplied peer uid is not the
control uid. See control.py.

Errors are returned as {"ok": false, "error": ...} and never take the daemon down: a
crash would strand the ledger unsealed and lose every trial still to come.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
import traceback
from typing import Any, Callable

from . import config as C
from . import ledger as L
from . import protocol as P
from .controller import ObsSpec
from .session import (AgentFacingError, Budgets, BudgetExhausted, ConnectionBusy,
                      MeteredSession)

# Operations the HARNESS may invoke and the agent may not. Absent from the agent-facing
# client AND refused here by the caller's uid, because hiding a module is obscurity: the
# wire would still accept a guessed op.
#
# `next_trial` matters most. It is the step boundary, and an agent able to call it could
# score its own trial early and drive its successor's with a context that successor is
# defined not to have -- the task's premise disassembled, not a better score.
CONTROL_OPS = frozenset({"open_evaluation", "next_trial", "seal"})

# The agent connects as a different uid, so the socket has to be reachable by it. 0o666
# is deliberate and safe: reaching the socket only lets you spend YOUR OWN budget, and
# the control ops above are gated on uid, not on reachability. Authority lives in the
# ledger, which stays root-owned mode 600.
SOCKET_MODE = 0o666

DEFAULT_SOCKET = "/run/rlebench/toolsmith.sock"


class Service:
    def __init__(
        self,
        train_tasks: tuple[str, ...],
        budgets: Budgets,
        ledger_path: str,
        env_factory: Callable[..., Any],
        eval_plan: tuple[tuple[str, int], ...],
        max_episode_steps: int | None = None,
        max_steps_per_trial: int | None = None,
        env_cache_size: int = C.ENV_CACHE_SIZE,
        transcript=None,
        recorder=None,
        artifacts_dir: str | None = None,
        control_uid: int = 0,
    ):
        # Every MeteredSession knob is forwarded. Keeping this signature in step with
        # MeteredSession matters: the daemon builds the Service, so a parameter added to
        # the session but not here fails only at container start, where the unit tests
        # (which build MeteredSession directly) cannot see it.
        self._session = MeteredSession(
            train_tasks=train_tasks,
            budgets=budgets,
            ledger=L.Ledger(ledger_path),
            env_factory=env_factory,
            eval_plan=eval_plan,
            max_episode_steps=max_episode_steps,
            max_steps_per_trial=max_steps_per_trial,
            env_cache_size=env_cache_size,
            transcript=transcript,
            recorder=recorder,
        )
        self._transcript = transcript
        self._recorder = self._session.recorder
        self._artifacts_dir = artifacts_dir
        # Which uid may invoke the control plane. Root in production -- the daemon runs
        # as root and Harbor's collect hooks declare user = "root". Injectable only so
        # the unit tests, which are not root, can exercise the harness side at all.
        self._control_uid = int(control_uid)
        self._action_dim: int | None = None
        # Wall-clock left in the phase, so the agent can budget against TIME as well as
        # steps -- a run can exhaust its clock at 5% of its step budget and be cut off
        # mid-experiment.
        #
        # HERE and not in MeteredSession on purpose: CLAUDE.md invariant #3 keeps wall
        # clock out of the scored path, so this never reaches the ledger, the session
        # state or the reward. A fact reported to the agent, not a fact about it.
        self._phase_start = time.monotonic()
        self._phase_seconds = _phase_seconds_from_env("RLEBENCH_DEVELOP_SECONDS")
        self._trial_seconds = _phase_seconds_from_env("RLEBENCH_TRIAL_SECONDS")
        self._srv: socket.socket | None = None
        self._socket_path: str | None = None
        self._stopping = False
        # Connections are served concurrently so a control hook is never blocked behind an
        # idle agent connection -- a lingering agent socket would otherwise make
        # `next_trial` hang until its timeout and fail silently, losing every trial after
        # it. The lock keeps the session single-threaded, so a control op waits for an
        # in-flight agent op rather than interleaving with it.
        self._lock = threading.RLock()
        # AGENT connections are limited to ONE AT A TIME, refused at accept. The lock
        # above only serializes single requests: an agent running two clients in
        # parallel would still interleave whole reset()/task_info() exchanges and each
        # would be handed the other's episode. Control connections are exempt -- they
        # are what the concurrency exists for. Guarded by its own lock because _lock is
        # held for the length of a sim-stepping request, and the accept loop must not
        # wait on that.
        self._agent_conn: socket.socket | None = None
        self._agent_conn_lock = threading.Lock()

    # -- helpers -------------------------------------------------------------
    # Observations are narrowed to images + proprioception on the way out (see
    # config.agent_visible_obs). The daemon keeps the full dict for success checking, for
    # the stage functions and for the human transcript; only the agent's view is
    # filtered, and only here, so there is no path to the object poses that does not
    # cross this boundary.

    def _peer_uid(self, sock) -> int | None:
        """The connecting process's uid, supplied by the kernel and unforgeable."""
        try:
            creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                    struct.calcsize("3i"))
            return struct.unpack("3i", creds)[1]
        except (OSError, AttributeError):  # pragma: no cover - non-Linux
            return None

    def _dispatch_control(self, op, msg, sock) -> dict:
        """The harness's plane. Reached over the same socket, refused for anyone but
        root, and never named in the agent's client.

        The error for a refused control op is deliberately the SAME "unknown op" a
        nonsense op gets. An agent probing the wire learns that the op does not exist for
        it, not that it exists and is guarded.
        """
        uid = self._peer_uid(sock)
        if uid != self._control_uid:
            raise P.ProtocolError(f"unknown op {op!r}")

        if op in ("open_evaluation", "next_trial"):
            # Retire the old client's socket before exposing the next trial.
            with self._agent_conn_lock:
                conn = self._agent_conn
                if conn is not None:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    conn.close()
                    self._agent_conn = None

        if op == "open_evaluation":
            # Advance to the scored phase and prepare the first trial. The clock restarts
            # here: this is where the first evaluation step begins.
            self._start_phase_clock(self._trial_seconds)
            return {"ok": True, **self._session.open_evaluation()}

        if op == "next_trial":
            # THE STEP BOUNDARY. Scores the live trial and opens the next, between Harbor
            # steps, so the agent that drove trial N is gone before trial N+1 exists.
            # Each trial gets its own clock because each gets its own agent.
            self._start_phase_clock(self._trial_seconds)
            return {"ok": True, **self._session.advance_trial()}

        if op == "seal":
            summary = self._session.close()
            self.export_artifacts()
            return {"ok": True, "summary": summary}

        raise P.ProtocolError(f"unknown control op {op!r}")

    def _log_internal_error(self, op, exc) -> None:
        """Put the full detail where an operator can read it and an agent cannot.

        NOT stdout: the daemon's stdout is the container log, which ends up in Harbor's
        captured output, which someone may later hand to an agent -- the same reason the
        startup banner does not print the evaluation tasks. This goes beside the ledger
        in /var/lib/rlebench (root:root 0700) instead. Best-effort: an error while
        recording an error must not become the error.
        """
        detail = (f"op={op!r} {type(exc).__name__}: {exc}\n"
                  f"{traceback.format_exc()}\n")
        try:
            path = self._session.ledger.path.parent / "internal_errors.log"
            with open(path, "a") as f:
                f.write(detail)
        except Exception:  # noqa: BLE001
            pass
        # One redacted line on stdout, so a run that is failing is visible live without
        # the log carrying anything private.
        print(f"[service] internal error on op={op!r}: {type(exc).__name__} "
              f"(detail in /var/lib/rlebench/internal_errors.log)", flush=True)

    def _start_phase_clock(self, seconds: float | None) -> None:
        self._phase_start = time.monotonic()
        self._phase_seconds = seconds

    def _clock(self) -> dict:
        """How much wall clock is left in this phase, if anyone configured it.

        Omitted entirely when unset rather than reported as a guess: an agent told it has
        `null` seconds left is worse off than one told nothing.
        """
        if not self._phase_seconds:
            return {}
        elapsed = time.monotonic() - self._phase_start
        return {
            "phase_seconds": round(self._phase_seconds, 1),
            "seconds_remaining": max(0.0, round(self._phase_seconds - elapsed, 1)),
        }

    def warm_up(self) -> dict:
        """One free reset at startup, learning the action contract.

        Required before `serve`: `env.action_spec` is only valid after a reset, so without
        this the agent's first `task_info()` would report action_dim=None. It is the ONLY
        place the dimension is learned -- a second lazy path would be dead code in
        production and would quietly hide a daemon that never warmed up.

        The contract is uniform across all 317 RoboCasa tasks (audited in task01), so
        warming on one training task answers for the evaluation's composites too.
        """
        info = self._session.probe()
        self._action_dim = int(info["action_dim"])
        return info

    def _require_action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("service was not warmed up; call warm_up() before serve()")
        return self._action_dim

    def _instruction(self) -> str | None:
        """The current episode's instruction, straight from the simulator.

        Read live rather than cached. It changes on every reset, and during evaluation it
        is the ONLY thing that tells an agent what it has been asked to do -- a stale
        instruction there would be worse than none.
        """
        from .session import episode_instruction

        env = self._session._current_env()
        return episode_instruction(env) if env is not None else None

    def handle_connection(self, sock) -> None:
        reader = P.LineReader(sock)
        while True:
            try:
                msg = reader.read()
            except P.ProtocolError as exc:
                P.send(sock, {"ok": False, "error": f"protocol: {exc}"})
                continue
            if msg is None:
                return
            op = msg.get("op")
            try:
                with self._lock:
                    if sock.fileno() < 0:
                        return  # A phase boundary retired this client's queued request.
                    if op in CONTROL_OPS:
                        reply = self._dispatch_control(op, msg, sock)
                    else:
                        reply = self._dispatch(op, msg)
                    # `step` is excluded: one event per action would bury the record it
                    # is meant to make readable. Episodes are recorded by the session.
                    if op != "step":
                        self._recorder.event(op or "?", ok=True)
                    self._recorder.status(
                        self._session.status(),
                        recompute_reward=op in ("reset", "seal", "next_trial",
                                                "end_development"))
            except BudgetExhausted as exc:
                reply = {"ok": False, "error": str(exc), "kind": "budget_exhausted"}
            except (P.ProtocolError, ValueError) as exc:
                reply = {"ok": False, "error": str(exc), "kind": "bad_request"}
            except AgentFacingError as exc:
                # Refusals whose text is written for the agent -- "this trial has already
                # been scored", "development is over". Returned verbatim because the
                # class says they are safe to; see session.AgentFacingError. Everything
                # NOT under that class falls through to the redacted handler below.
                reply = {"ok": False, "error": str(exc), "kind": "refused"}
            except ConnectionError:
                return  # agent went away; the session stays intact for the next step
            except Exception as exc:  # noqa: BLE001
                # Never die on an unexpected error: an unsealed ledger reads as an
                # incomplete run, so a crash would destroy the graded step still to come,
                # not just this request.
                #
                # THE DETAIL GOES TO THE OPERATOR, NOT TO THE AGENT, and here that is the
                # difference between keeping the split and losing it. An internal error is
                # a fault in the trusted side, so its traceback carries source lines and
                # paths from /opt/private -- including stages.py -- and its MESSAGE is
                # whatever the simulator chose to say. A scene that fails to build during
                # a graded trial raises from `env_factory(task=...)`, and RoboCasa names
                # the environment it could not construct: the held-out task, handed
                # straight back over the wire. The agent gets the exception TYPE, which is
                # enough to tell "the harness broke" from "my request was wrong".
                #
                # Reading the reply is not the only route to it -- the socket is 0666 and
                # protocol.py is agent-readable, so an agent can speak the wire itself.
                # Which is why the fix is to not send it, rather than to not surface it
                # in the client.
                self._log_internal_error(op, exc)
                reply = {
                    "ok": False,
                    "error": f"internal harness error ({type(exc).__name__}); "
                             "it has been logged for the operator",
                    "kind": "internal",
                }
            P.send(sock, reply)
            if op == "seal":
                return

    def _dispatch(self, op, msg) -> dict:
        if op == "status":
            return {"ok": True, **self._session.status(), **self._clock()}

        if op == "list_tasks":
            # The TRAINING split, in full. Published on purpose: deciding where to spend
            # a finite budget across these is part of the problem. Answered during
            # evaluation too, with `phase` alongside -- that agent cannot practise on
            # them, but it inherited a manual that names them, and refusing would leave
            # it unable to check what a controller was built for.
            return {
                "ok": True,
                "tasks": self._session.list_tasks(),
                "phase": self._session.phase,
            }

        if op == "task_info":
            # Everything the agent legitimately needs to write a controller. The
            # instruction comes from the simulator (it is episode-specific), and the
            # action layout is published so nobody has to reverse-engineer it.
            return {
                "ok": True,
                "phase": self._session.phase,
                "current_task": (self._session.state.current_task
                                 if not self._session.evaluating else None),
                "action_dim": self._action_dim,
                "action_layout": {
                    "arm_osc_pose": [0, 6],
                    "gripper": [6, 7],
                    "base": [7, 10],
                    "torso": [10, 11],
                    "base_mode": 11,
                },
                "instruction": self._instruction(),
                "action_range": [-1.0, 1.0],
                "action_reference_frame": "robot base (NOT world)",
                "interaction_budget": self._session.budgets.interaction_steps,
                "max_steps_per_trial": self._session.max_steps_per_trial,
                # The development episode ceiling. Published alongside the trial's so the
                # agent can see they match, rather than inferring its own by hitting it.
                "max_episode_steps": self._session.max_episode_steps,
                "obs_resolution": C.OBS_RESOLUTION,
                "obs_max_resolution": C.OBS_MAX_RESOLUTION,
                # The phase clock is reported here as well as in `status` so an agent
                # finds it whichever one it asks. It is enforced by Harbor, not by the
                # daemon -- this is a report, never a guarantee.
                **self._clock(),
            }

        if op == "reset":
            seed = msg.get("seed")
            was_evaluating = self._session.evaluating
            out = self._session.reset(
                task=msg.get("task"),
                seed=int(seed) if seed is not None else None,
            )
            if was_evaluating:
                # During evaluation reset is the give-up move: the reply says the trial
                # is over and how much of the evaluation remains. It carries no
                # observation, because there is nothing left to act in, and no score.
                return {"ok": True, "gave_up_trial": True, **out}
            return {
                "ok": True,
                "obs": C.agent_visible_obs(out),
                "task": self._session.state.current_task,
                "action_dim": self._action_dim,
                "instruction": self._instruction(),
            }

        if op == "observe":
            # FREE and available in BOTH phases, unlike `step`. It advances nothing, so
            # there is no reason to meter it and every reason to allow it in the scored
            # phase: an evaluation agent wakes with no context, and looking is how it
            # finds out what it is facing before committing its one shot to an action.
            out = self._session.observe(ObsSpec.from_wire(msg))
            return {
                "ok": True,
                "obs": C.agent_visible_obs(out["obs"]),
                "instruction": out["instruction"],
                "resolution": out["resolution"],
                "max_resolution": C.OBS_MAX_RESOLUTION,
                "live": out["live"],
            }

        if op == "step":
            actions = P.validate_actions(msg.get("actions"),
                                         self._require_action_dim())
            res = self._session.step(
                actions, spec=ObsSpec.from_wire(msg.get("obs_spec")))
            return {"ok": True, **res, "obs": C.agent_visible_obs(res["obs"])}

        if op == "end_development":
            # The agent's own "I am ready" signal. Ends development and nothing else --
            # no evaluation opened, no environment built, no scene touched.
            return {"ok": True, **self._session.end_development()}

        if op == "trial_info":
            return {"ok": True, "trial": self._session.trial_info()}

        raise P.ProtocolError(f"unknown op {op!r}")

    # -- server --------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop `serve` from outside it, by closing the listener so accept() fails.

        The daemon proper exits on SIGTERM; this exists so an in-process server (the
        tests) can be stopped without waiting on a thread that never returns.
        """
        self._stopping = True
        srv, self._srv = self._srv, None
        if srv is None:
            return
        path = self._socket_path
        try:
            srv.close()
        except OSError:
            pass
        # Closing a listener from another thread does not reliably wake a blocked
        # accept(), so poke it with one connection. Without this the server thread sits
        # in accept() until the process exits.
        if path:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as poke:
                    poke.settimeout(0.5)
                    poke.connect(path)
            except OSError:
                pass

    def serve(self, socket_path: str = DEFAULT_SOCKET, *, once: bool = True) -> None:
        path = str(socket_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv = srv
        self._socket_path = path
        self._stopping = False
        try:
            srv.bind(path)
            os.chmod(path, SOCKET_MODE)
            srv.listen(8)
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    if self._stopping:
                        return
                    raise
                if once:
                    try:
                        self.handle_connection(conn)
                    finally:
                        conn.close()
                    return
                is_agent = self._peer_uid(conn) != self._control_uid
                if is_agent:
                    with self._agent_conn_lock:
                        if self._agent_conn is not None:
                            busy = ConnectionBusy(
                                "the simulator serves one connection at a time and "
                                "another client is already connected -- close it and "
                                "reconnect; state persists across sequential "
                                "connections")
                            # No op has been read yet, so the refusal is written
                            # directly rather than raised through handle_connection's
                            # ladder; kind "refused" is what AgentFacingError maps to
                            # there, so the client surfaces the message verbatim.
                            try:
                                P.send(conn, {"ok": False, "error": str(busy),
                                              "kind": "refused"})
                            except OSError:
                                pass
                            try:
                                conn.close()
                            except OSError:
                                pass
                            continue
                        self._agent_conn = conn
                threading.Thread(target=self._serve_connection, args=(conn, is_agent),
                                 daemon=True).start()
        finally:
            srv.close()
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _serve_connection(self, conn, holds_agent_slot: bool = False) -> None:
        try:
            self.handle_connection(conn)
        except Exception:  # noqa: BLE001
            pass          # one bad connection must never take the daemon down
        finally:
            try:
                conn.close()
            except OSError:
                pass
            if holds_agent_slot:
                with self._agent_conn_lock:
                    if self._agent_conn is conn:
                        self._agent_conn = None

    def seal_if_open(self) -> dict:
        """Seal the ledger on shutdown paths nobody closed cleanly."""
        summary = self._session.close()
        self.export_artifacts()
        return summary

    def export_artifacts(self) -> None:
        """Copy the ledger and transcript where Harbor will collect them.

        FOR HUMANS ONLY. The scorer reads the root-only original at
        /var/lib/rlebench/cost.jsonl, never this copy -- the destination is a Harbor
        publish mount that the agent can write, so nothing here is evidence.
        """
        if not self._artifacts_dir:
            return
        import shutil
        from pathlib import Path

        dest = Path(self._artifacts_dir)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._session.ledger.path, dest / "cost.jsonl")
            if self._transcript is not None and self._transcript.path.exists():
                shutil.copy2(self._transcript.path, dest / "transcript.jsonl")
                frames = self._transcript.frames_dir
                if frames.exists():
                    shutil.copytree(frames, dest / "frames", dirs_exist_ok=True)
            self._hand_over_export(dest)
        except Exception as exc:  # noqa: BLE001
            # Never fail the run over artifact export: it is a convenience copy, and the
            # ledger the SCORER reads is the root-only one in /var/lib.
            print(f"[service] artifact export failed: {exc}", flush=True)

    def _hand_over_export(self, dest) -> None:
        """Give the exported copy to whoever owns the publish mount. See debug.match_owner
        for why: Harbor cannot archive a root-owned export."""
        from .debug import match_owner

        match_owner([dest, *dest.rglob("*")], dest.parent)


def _phase_seconds_from_env(name: str) -> float | None:
    """A phase's wall-clock budget, scaled the same way Harbor scales its timeout.

    Harbor's `--agent-timeout-multiplier` scales every step's timeout_sec while keeping
    their ratio, so the agent-facing number has to be scaled by the same factor or it
    would drift the moment anyone changed the run length. One shell variable drives both
    : set RLEBENCH_TIMEOUT_MULT in the shell that launches harbor, with the same
    value you pass to --agent-timeout-multiplier.

    Returns None when unconfigured, and the clock is then simply not reported.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        base = float(raw)
        mult = float(os.environ.get("RLEBENCH_TIMEOUT_MULT", "1") or 1)
    except ValueError:
        return None
    return base * mult if base > 0 and mult > 0 else None
