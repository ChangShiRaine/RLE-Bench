"""End-to-end tests for task02's metering daemon, over a real Unix socket.

Runs the daemon in a thread against a fake env, so the whole client/daemon contract is
covered in the fast dev suite. What matters here is that going through the socket does
not weaken any guarantee the in-process session already has -- and, specific to task02,
that the socket is the information boundary between eleven different agents:

  * the TRAINING split is reachable; the EVALUATION split is not, by any op;
  * no reply ever carries a score, so one trial's agent cannot pass its grade to the next
    through whatever it writes to disk;
  * the control plane -- open_evaluation, next_trial, seal -- is refused to the agent's
    uid, and refused with the same error a nonsense op gets.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from harness import ledger as L
from harness import protocol as P
from harness import stages as S
from harness.client import RemoteError, ToolsmithClient
from harness.controller import Ended
from harness.service import Service
from harness.session import Budgets

TRAIN = ("OpenDrawer", "CloseDrawer", "TurnOnStove")
PLAN = (("SecretCompositeA", 11), ("SecretCompositeB", 22))

A = [0.0] * 12


class FakeEnv:
    CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")

    def __init__(self, task: str, split: str = "pretrain", dim: int = 12):
        import numpy as np

        self.task = task
        self.split = split
        self.dim = dim
        self.steps = 0
        self.stage_flags = [False, False]
        self.rng = np.random.default_rng(0)
        self.draws: list = []

    @property
    def action_spec(self):
        return [-1.0] * self.dim, [1.0] * self.dim

    def _images(self):
        import numpy as np

        return {f"{cam}_image": np.zeros((8, 8, 3), dtype="uint8")
                for cam in self.CAMERAS}

    def _obs(self):
        # Carries the ground-truth object keys the real env does, so the filter is
        # tested against something rather than passing trivially.
        return {"robot0_proprio-state": [float(self.steps), 1.0],
                "robot0_eef_pos": [0.0, 0.0, 0.0],
                "step": self.steps,
                "fruit_pos": [1.0, 2.0, 3.0],
                "object-state": [0.0] * 42,
                **self._images()}

    def reset(self, seed=None):
        self.steps = 0
        self.stage_flags = [False, False]
        self.draws.append(int(self.rng.integers(0, 2**31 - 1)))
        return self._obs()

    def step(self, action):
        assert len(action) == self.dim, "daemon must validate before the sim sees it"
        self.steps += 1
        return self._obs(), 0.0, False, {}

    def _check_success(self):
        return all(self.stage_flags)

    def get_ep_meta(self):
        return {"lang": "put the thing in the other thing"}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_stages(monkeypatch):
    from harness import evaluation as E

    def read(task, env):
        if env is None or not hasattr(env, "stage_flags"):
            return None
        return [(f"s{i}", bool(v)) for i, v in enumerate(env.stage_flags)]

    monkeypatch.setattr(S, "read_stages", read)
    monkeypatch.setattr(E.S, "read_stages", read)


class Harness:
    """Daemon on a background thread plus a connected client."""

    def __init__(self, tmp_path, *, steps=200, max_episode_steps=10, control_uid=None,
                 connect=True, sock_name="toolsmith.sock"):
        self.envs: list[FakeEnv] = []
        self.client = None

        def factory(task, split="pretrain"):
            env = FakeEnv(task=task, split=split)
            self.envs.append(env)
            return env

        self.ledger_path = tmp_path / "cost.jsonl"
        self.service = Service(
            train_tasks=TRAIN,
            budgets=Budgets(interaction_steps=steps),
            ledger_path=str(self.ledger_path),
            env_factory=factory,
            eval_plan=PLAN,
            max_episode_steps=max_episode_steps,
            max_steps_per_trial=max_episode_steps,
            # not root under pytest, so the harness plane has to accept this uid
            control_uid=os.getuid() if control_uid is None else control_uid,
        )
        # As production does (daemon_main calls this before serve): one free reset so the
        # action contract is known before the agent's first task_info().
        self.service.warm_up()
        self.sock_path = str(tmp_path / sock_name)
        self._ready = threading.Event()

        def run():
            self._ready.set()
            self.service.serve(self.sock_path, once=False)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self._ready.wait(timeout=5)
        for _ in range(500):  # wait for bind
            try:
                if connect:
                    self.client = ToolsmithClient(self.sock_path)
                elif not os.path.exists(self.sock_path):
                    raise FileNotFoundError(self.sock_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.01)
        else:
            raise RuntimeError("daemon never bound its socket")

    def raw(self, **msg) -> dict:
        """Send an arbitrary op, bypassing the client. This is how an agent that had
        read the wire protocol would probe for hidden operations.

        Retries while the previous raw socket's agent slot is still being released --
        the daemon may close a busy connection before the probe can send or read."""
        import socket as _socket

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as sock:
                sock.settimeout(max(0.001, deadline - time.monotonic()))
                sock.connect(self.sock_path)
                try:
                    P.send(sock, msg)
                    reply = P.LineReader(sock).read()
                except (BrokenPipeError, ConnectionResetError):
                    reply = None
            if (reply is not None
                    and "one connection at a time" not in str(reply.get("error", ""))):
                return reply
            time.sleep(0.01)
        raise AssertionError("daemon did not accept the raw probe within 5 seconds")

    def stop(self):
        try:
            if self.client is not None:
                self.client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.service.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self.thread.join(timeout=5)


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.stop()


# -- the training split is reachable -----------------------------------------

def test_list_tasks_returns_the_training_split(h):
    assert h.client.list_tasks() == list(TRAIN)


def test_reset_selects_a_training_task(h):
    h.client.reset(task="TurnOnStove")
    assert h.client.task_info()["current_task"] == "TurnOnStove"


def test_resetting_outside_the_split_is_refused_over_the_wire(h):
    with pytest.raises(RemoteError, match="not in the training split"):
        h.client.reset(task="SecretCompositeA")


# -- the evaluation split is not ---------------------------------------------

def test_no_reply_names_an_evaluation_task(h):
    """The one thing task02 must keep private. An agent that learned the split could
    have developed against it, and the whole result would mean nothing."""
    h.service._session.open_evaluation()
    replies = [
        h.client.status(), h.client.task_info(), h.client.trial_info(),
        h.client.observe(), {"tasks": h.client.list_tasks()},
    ]
    for reply in replies:
        assert "SecretComposite" not in repr(reply)


def test_the_ledger_does_name_them(h):
    """The private side must record what the public side hides -- otherwise nobody can
    tell afterwards which tasks a run was actually graded on."""
    h.service._session.open_evaluation()
    h.service._session.advance_trial()
    records = [r for r in L.load_verified(h.ledger_path) if r.kind == L.KIND_TRIAL]
    assert records[0].payload["task"] == "SecretCompositeA"


# -- no score ever crosses the wire ------------------------------------------

def test_no_reply_carries_a_score(h):
    """Each trial is driven by a different agent. Telling one how it did is a channel to
    the next, through whatever it writes to disk."""
    h.service._session.open_evaluation()
    h.client.step([A])
    for reply in (h.client.status(), h.client.trial_info(), h.client.reset()):
        assert not ({"score", "success_rate", "successes", "reward"} & set(reply or {}))


# -- the control plane is not the agent's ------------------------------------

@pytest.mark.parametrize("op", ["open_evaluation", "next_trial", "seal"])
def test_control_ops_are_refused_to_a_non_control_uid(tmp_path, op):
    """Refused by UID, not by being unlisted: hiding the module alone is obscurity,
    because the wire would still accept a guessed op string."""
    # connect=False: this uid is an agent peer here, and a held client connection
    # would have the raw probe refused as a second concurrent agent connection.
    harness = Harness(tmp_path, control_uid=os.getuid() + 1, connect=False)
    try:
        reply = harness.raw(op=op)
        assert not reply.get("ok")
        assert "unknown op" in reply.get("error", "")
    finally:
        harness.stop()


def test_a_refused_control_op_is_indistinguishable_from_a_nonsense_one(tmp_path):
    """An agent probing the wire must learn that the op does not exist for it, not that
    it exists and is guarded."""
    harness = Harness(tmp_path, control_uid=os.getuid() + 1, connect=False)
    try:
        guarded = harness.raw(op="next_trial")
        nonsense = harness.raw(op="definitely_not_an_op")
        assert guarded.get("error") == nonsense.get("error").replace(
            "definitely_not_an_op", "next_trial")
    finally:
        harness.stop()


def test_next_trial_is_absent_from_the_agent_client():
    """Defence in depth: the uid gate is the enforcement, but an op the agent can read
    about is an op it will try."""
    assert not hasattr(ToolsmithClient, "next_trial")
    assert not hasattr(ToolsmithClient, "open_evaluation")


# -- observation filtering ----------------------------------------------------

def test_ground_truth_object_state_never_crosses_the_wire(h):
    """A composite success predicate is a conjunction over named objects, so an
    unfiltered observation would hand over the exact quantities being scored."""
    h.client.reset(task="OpenDrawer")
    obs = h.client.observe()["obs"]
    assert "fruit_pos" not in obs and "object-state" not in obs
    assert "robot0_proprio-state" in obs


def test_the_trial_descriptor_is_filtered_too(h):
    """Filtered where descriptors are BUILT, not at each call site: task01 shipped both
    descriptor-returning ops leaking object poses because the filter was per-reply."""
    h.service._session.open_evaluation()
    assert "fruit_pos" not in repr(h.client.trial_info())


def test_a_stepping_controller_is_shown_the_filtered_observation(h):
    h.client.reset(task="OpenDrawer")
    seen = h.client.step([A])["obs"]
    assert "fruit_pos" not in seen and "robot0_proprio-state" in seen


# -- budgets ------------------------------------------------------------------

def test_the_interaction_budget_is_enforced_over_the_wire(tmp_path):
    """Refused, not silently truncated: the call applies what it can, then names the cap."""
    harness = Harness(tmp_path, steps=3)
    try:
        harness.client.reset(task="OpenDrawer")           # costs 1
        harness.client.step([A] * 1)
        assert harness.client.status()["steps_remaining"] == 1

        # Spends the last step, then hits the cap inside the same call.
        res = harness.client.step([A] * 5)
        assert res["steps"] == 1
        assert res["ended"] == Ended.BUDGET_EXHAUSTED
        assert res["episode_over"]
    finally:
        harness.stop()


def test_evaluation_steps_are_not_charged_to_the_interaction_budget(h):
    """Mixing the two would make an agent that drove its graded trial thoroughly look
    like one that had spent its practice.

    Read from the session, not from `status()`: during evaluation `status()` deliberately
    reports the TRIAL's budget, so the development figure does not show up there.
    """
    h.service._session.open_evaluation()
    h.client.step([A] * 3)
    assert h.service._session.state.steps_used == 0     # development budget untouched
    assert h.client.status()["steps_used"] == 3         # the trial's own count


def test_observe_is_free_in_both_phases(h):
    """Free even though reset is not: looking must never be a reason to spend."""
    h.client.reset(task="OpenDrawer")
    spent = h.client.status()["steps_used"]
    for _ in range(5):
        h.client.observe()
    assert h.client.status()["steps_used"] == spent
    h.service._session.open_evaluation()
    for _ in range(5):
        h.client.observe(width=32)
    assert h.client.status()["steps_used"] == 0


# -- action validation --------------------------------------------------------

def test_a_wrong_length_action_is_refused_before_the_simulator_sees_it(h):
    """Refused on the wire, so nothing reaches the environment and nothing is charged."""
    h.client.reset(task="OpenDrawer")
    spent = h.client.status()["steps_used"]
    with pytest.raises(RemoteError):
        h.client.step([0.0] * 3)
    assert h.client.status()["steps_used"] == spent


def test_a_non_numeric_action_is_refused(h):
    h.client.reset(task="OpenDrawer")
    spent = h.client.status()["steps_used"]
    with pytest.raises(RemoteError):
        h.client.step(["not a number"] * 12)
    assert h.client.status()["steps_used"] == spent


def test_one_bad_action_rejects_the_whole_batch(h):
    """All or nothing: a caller never has to work out how much of a partially applied
    sequence reached the simulator."""
    h.client.reset(task="OpenDrawer")
    spent = h.client.status()["steps_used"]
    with pytest.raises(RemoteError):
        h.client.step([A, A, [0.0] * 3])
    assert h.client.status()["steps_used"] == spent


# -- the phase protocol -------------------------------------------------------

def test_the_agent_cannot_open_the_evaluation_by_calling_end_development(h):
    """task01's actual failure: an agent read `begin_evaluation` in its own client and
    spent its single terminal submission during the unscored phase."""
    h.client.reset(task="OpenDrawer")
    h.client.end_development()
    assert h.client.phase() == "development"


def test_the_segment_ops_are_gone_from_the_wire(h):
    """A stale harness must fail loudly rather than be quietly redirected."""
    for op in ("run_segment", "rollout"):
        with pytest.raises(RemoteError, match="unknown op"):
            h.client._request(op)


def test_status_reports_the_trial_budget_during_evaluation_over_the_wire(h):
    """The development figure must not survive into evaluation, where it neither moves
    nor describes anything the agent can spend."""
    h.client.reset(task="OpenDrawer")                    # costs 1
    h.client.step([A] * 2)
    assert h.client.status()["steps_used"] == 3          # development budget

    h.service._session.open_evaluation()
    assert h.client.status()["steps_used"] == 0          # the trial's, not the phase's
    h.client.step([A] * 1)
    assert h.client.status()["steps_used"] == 1


def test_step_after_the_trial_ended_is_refused_not_advanced(h):
    """It would consume the trial that belongs to the next agent."""
    h.service._session.open_evaluation()
    h.client.reset()
    with pytest.raises(RemoteError, match="already been scored"):
        h.client.step([A])


def test_the_trial_survives_the_agent_disconnecting(h, tmp_path):
    """The whole task depends on this: one agent's session ends at every step boundary
    and the next connects to the same daemon."""
    h.service._session.open_evaluation()
    h.client.step([A])
    h.client.disconnect()
    second = ToolsmithClient(h.sock_path)
    try:
        assert second.trial_info()["index"] == 0
    finally:
        second.disconnect()


def test_the_daemon_survives_an_internal_error(h):
    """An unsealed ledger says the run did not finish. With every evaluation step still
    to come, a daemon crash loses every one of them, not just this request."""
    assert not h.raw(op="step", actions="not a list").get("ok")
    assert h.client.status()["phase"] == "development"


# -- the wall clock -----------------------------------------------------------

def test_the_clock_is_omitted_when_unconfigured(h):
    """An agent told it has `null` seconds left is worse off than one told nothing."""
    assert "seconds_remaining" not in h.client.status()


def test_each_trial_gets_its_own_clock(tmp_path, monkeypatch):
    """Each trial is a separate Harbor step with a separate agent, so the clock it is
    told about must restart with it."""
    monkeypatch.setenv("RLEBENCH_TRIAL_SECONDS", "600")
    harness = Harness(tmp_path)
    try:
        harness.service._dispatch_control("open_evaluation", {}, _FakeSock())
        first = harness.client.status()["seconds_remaining"]
        time.sleep(0.05)
        harness.service._dispatch_control("next_trial", {}, _FakeSock())
        assert harness.client.status()["seconds_remaining"] >= first - 0.01
    finally:
        harness.stop()


class _FakeSock:
    """Stands in for a control-plane connection whose peer uid is ours."""

    def getsockopt(self, *_args):
        import struct

        return struct.pack("3i", 0, os.getuid(), 0)


# -- internal errors must not become a channel -------------------------------

def test_an_internal_error_is_redacted_before_it_reaches_the_agent(h, monkeypatch):
    """Neither `str(exc)` nor the traceback may reach the agent.

    Both are made of private material. The traceback carries source lines and paths from
    /opt/private -- stages.py included -- and the MESSAGE is whatever the simulator chose
    to say: a scene that fails to build during a graded trial raises from the env factory,
    and RoboCasa names the environment it could not construct, which is the held-out task.

    Redaction is done in the daemon rather than in the client on purpose. The socket is
    0666 and protocol.py is agent-readable, so an agent can speak the wire itself and read
    the raw reply; not surfacing it client-side would protect nobody.
    """
    secret = "DumpLeftovers"

    def explode(*a, **kw):
        raise RuntimeError(f"could not construct environment {secret}")

    monkeypatch.setattr(h.service._session, "observe", explode)

    with pytest.raises(RemoteError) as caught:
        h.client.observe()

    message = str(caught.value)
    assert secret not in message
    assert "could not construct" not in message
    assert "Traceback" not in message
    # The TYPE survives, which is all an agent can act on: "the harness broke" as
    # distinct from "my request was malformed".
    assert "RuntimeError" in message
    assert caught.value.kind == "internal"


def test_the_raw_wire_reply_carries_no_traceback_field(h, monkeypatch):
    """The client raises on `ok: false`, so the check above cannot see the whole reply.
    An agent writing its own client would. Assert on the reply itself."""
    def explode(*a, **kw):
        raise RuntimeError("SecretCompositeA could not be built")

    monkeypatch.setattr(h.service._session, "observe", explode)

    P.send(h.client._sock, {"op": "observe"})
    reply = h.client._reader.read()

    assert reply["ok"] is False
    assert "traceback" not in reply
    assert "SecretCompositeA" not in str(reply)


def test_refusals_written_for_the_agent_still_reach_it_verbatim(h):
    """The other half of the boundary. Redacting everything would be safe and useless:
    an agent must still be told that its trial is over or its development closed, and
    those messages are written for it. `session.AgentFacingError` is what marks a message
    as safe to send, and this is what stops the redaction swallowing them."""
    h.client.reset(task=TRAIN[0])
    h.client.end_development()

    with pytest.raises(RemoteError) as caught:
        h.client.reset(task=TRAIN[0])
    assert "internal harness error" not in str(caught.value)
    assert caught.value.kind == "refused"


def test_a_second_concurrent_agent_connection_is_refused(tmp_path):
    """Two agent clients interleaving on the one session hand each other's episodes
    back -- reset(task=X) then task_info() is not atomic across connections, and a
    real agent that ran two probes in parallel practised against the wrong
    instructions. The daemon refuses the second connection at accept."""
    h = Harness(tmp_path, control_uid=os.getuid() + 1)  # this uid is an agent peer
    try:
        second = ToolsmithClient(h.sock_path)
        with pytest.raises(RemoteError) as caught:
            second.status()
        assert caught.value.kind == "refused"
        assert "one connection at a time" in str(caught.value)
        second.disconnect()
    finally:
        h.stop()


def test_the_connection_slot_frees_on_disconnect(tmp_path):
    """Sequential reconnection is the documented workflow and must keep working:
    the slot is released when the holder's socket closes."""
    h = Harness(tmp_path, control_uid=os.getuid() + 1)
    try:
        assert h.client.status()["phase"] == "development"
        h.client.disconnect()
        for _ in range(200):  # EOF processing frees the slot near-instantly
            try:
                fresh = ToolsmithClient(h.sock_path)
                assert fresh.status()["phase"] == "development"
                break
            except RemoteError:
                time.sleep(0.01)
        else:
            raise AssertionError("slot never freed after disconnect")
        fresh.disconnect()
    finally:
        h.stop()


def test_control_uid_connections_are_never_refused(h):
    """The concurrency exists FOR the control plane: a collect hook must reach the
    daemon past an idle agent socket. The default harness grants control to this
    uid, so a second concurrent connection must be served, not refused."""
    assert h.raw(op="status").get("ok") is True   # concurrent with h.client


@pytest.mark.parametrize("last_trial_succeeds", [False, True])
def test_abandoned_clients_do_not_prevent_remaining_trials(
        tmp_path, monkeypatch, last_trial_succeeds):
    """Timed-out clients never disconnect; every subsequent trial still runs."""
    h = Harness(tmp_path, control_uid=os.getuid() + 1)
    peer_uid = h.service._peer_uid
    monkeypatch.setattr(h.service, "_peer_uid", lambda sock:
                        h.service._control_uid if isinstance(sock, _FakeSock)
                        else peer_uid(sock))
    clients = [h.client]
    try:
        assert h.client.status()["phase"] == "development"
        with h.service._lock:
            h.service._dispatch_control("open_evaluation", {}, _FakeSock())
        for index in range(len(PLAN)):
            client = ToolsmithClient(h.sock_path)
            clients.append(client)
            assert client.status()["trials_done"] == index
            if last_trial_succeeds and index == len(PLAN) - 1:
                h.envs[-1].stage_flags = [True, True]
            client.step([A])
            # The controller is abandoned mid-trial with its socket still open.
            with h.service._lock:
                out = h.service._dispatch_control("next_trial", {}, _FakeSock())
            assert h.service._agent_conn is None
        assert out["evaluation_done"]
        assert out["trials"] == len(PLAN)
        assert out["trials_attempted"] == len(PLAN)
        assert out["steps_used"] == len(PLAN)
        summary = L.summarize(list(L.read_records(h.ledger_path)))
        assert summary["per_trial"][0]["score"] == 0.0
        assert summary["per_trial"][-1]["score"] == float(last_trial_succeeds)
        assert summary["successes"] == int(last_trial_succeeds)
        assert summary["mean_trial_score"] == float(last_trial_succeeds) / len(PLAN)
    finally:
        for client in clients:
            client.disconnect()
        h.stop()
