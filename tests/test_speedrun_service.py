"""End-to-end tests for the metering daemon over a real Unix socket.

Runs the daemon in a thread against a fake env, so the whole client/daemon contract
is covered in the fast dev suite. What matters here is that going through the socket
does not weaken any guarantee the in-process session already has: budgets are still
enforced, the two currencies stay separate, feedback stays aggregate-only, and a
malformed action is refused rather than reaching the simulator.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from harness import ledger as L
from harness import protocol as P
from harness.client import RemoteError, SpeedrunClient
from harness.session import Termination
from harness.service import Service
from harness.session import Budgets


class FakeEnv:
    def __init__(self, success_after: int = 3, split: str = "pretrain", dim: int = 12):
        self.success_after = success_after
        self.split = split
        self.dim = dim
        self.steps = 0
        self._success = False
        # Mirrors the real path: no seed on reset(); the harness re-seeds env.rng.
        import numpy as np

        self.rng = np.random.default_rng(0)
        self.draws: list = []

    # action_spec is only valid after reset in the real thing; mirror that loosely.
    @property
    def action_spec(self):
        return [-1.0] * self.dim, [1.0] * self.dim

    # The three cameras the real env carries. Small, but present: without images an
    # observation-shaping test passes trivially and proves nothing.
    CAMERAS = ("robot0_agentview_left", "robot0_agentview_right",
               "robot0_eye_in_hand")

    def _images(self):
        import numpy as np

        return {f"{cam}_image": np.zeros((8, 8, 3), dtype="uint8")
                for cam in self.CAMERAS}

    def reset(self, seed=None):
        self.steps = 0
        self._success = False
        self.draws.append(int(self.rng.integers(0, 2**31 - 1)))
        return {"robot0_proprio-state": [0.0, 1.0], "robot0_eef_pos": [0.0, 0.0, 0.0],
                "step": 0, "drawer_obj_pos": [1.0, 2.0, 3.0],
                "object-state": [0.0] * 42, **self._images()}

    def step(self, action):
        assert len(action) == self.dim, "daemon must validate before the sim sees it"
        self.steps += 1
        if self.success_after >= 0 and self.steps >= self.success_after:
            self._success = True
        return ({"robot0_proprio-state": [float(self.steps), 1.0],
                 "robot0_eef_pos": [0.0, 0.0, 0.0], "step": self.steps,
                 "drawer_obj_pos": [1.0, 2.0, 3.0], "object-state": [0.0] * 42,
                 **self._images()},
                0.0, False, {})

    def _check_success(self):
        return self._success

    def close(self):
        pass


class Harness:
    """Daemon on a background thread plus a connected client."""

    def __init__(self, tmp_path, *, steps=200, submissions=3,
                 success_after=3, seeds=(101, 202), max_episode_steps=10,
                 control_uid=None, connect=True, sock_name="speedrun.sock",
                 env_cls=None):
        self.envs = []
        self.client = None

        def factory(task, split="pretrain", scene=None):
            # `env_cls` lets a caller swap in a richer fake without a second harness --
            # test_speedrun_levels.py needs one that answers scene questions an
            # observation cannot (objects, fixtures, grasp).
            env = (env_cls or FakeEnv)(success_after=success_after, split=split)
            self.envs.append(env)
            return env

        self.ledger_path = tmp_path / "cost.jsonl"
        self.service = Service(
            task="FakeTask",
            budgets=Budgets(interaction_steps=steps, submissions=submissions),
            ledger_path=str(self.ledger_path),
            env_factory=factory,
            trial_seeds=list(seeds),
            max_episode_steps=max_episode_steps,
            # not root under pytest, so the harness plane has to accept this uid
            control_uid=os.getuid() if control_uid is None else control_uid,
        )
        # As production does (daemon_main calls this before serve): one free reset so
        # the action contract is known before the agent's first task_info().
        self.service.warm_up()
        self.sock_path = str(tmp_path / sock_name)
        self._ready = threading.Event()

        def run():
            self._ready.set()
            self.service.serve(self.sock_path, once=False)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self._ready.wait(timeout=5)
        import time
        for _ in range(500):  # wait for bind
            try:
                if connect:
                    self.client = SpeedrunClient(self.sock_path)
                elif not os.path.exists(self.sock_path):
                    raise FileNotFoundError(self.sock_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.01)
        else:
            raise RuntimeError("daemon never bound its socket")

    def stop(self):
        try:
            if self.client is not None:
                self.client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.service.seal_if_open()      # sealing is the harness's job now
        except Exception:  # noqa: BLE001
            pass
        self.service.shutdown()
        self.thread.join(timeout=5)

    def control(self):
        """A harness-plane connection to the same socket."""
        from harness.control import ControlClient

        return ControlClient(self.sock_path)

    def open_evaluation(self):
        with self.control() as ctl:
            return ctl.open_evaluation()

    def drive_evaluation(self, actions=None, guard=200):
        """The agent's loop, identical in both phases: step, and reset when the episode
        is over. Exactly what the instructions tell the agent to write."""
        actions = actions if actions is not None else [[0.0] * 12] * 5000
        out = None
        for _ in range(guard):
            if self.client.phase() != "evaluation":
                break
            res = self.client.step(actions)
            info = res.get("info") or {}
            if info.get("evaluation_done"):
                out = info
                break
            if res.get("episode_over"):
                r = self.client.reset()
                if isinstance(r, dict) and r.get("evaluation_done"):
                    out = r
                    break
        return out


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.stop()


def test_reset_and_act_over_the_socket(harness):
    """`step` is the whole acting surface, and it is a plain request/reply: the agent
    sends actions and reads what came back, with no callback into its process."""
    obs = harness.client.reset()
    assert "robot0_proprio-state" in obs
    res = harness.client.step([[0.0] * 12] * 1)
    assert res["steps"] == 1
    assert res["obs"]["robot0_proprio-state"][0] == 1
    assert res["success"] is False and res["episode_over"] is False
    assert harness.client.status()["steps_used"] == 2      # the reset, then the action


def test_there_is_exactly_one_acting_path(harness):
    """`step` and nothing else. The callback-driven `run_segment`/`rollout` ops are
    gone: requiring the agent to hand over a controller constrained how it could
    organise its own code for no benefit the harness needed."""
    assert not hasattr(harness.client, "run_segment")
    assert not hasattr(harness.client, "rollout")
    for gone in ("run_segment", "rollout"):
        with pytest.raises(RemoteError, match="unknown op"):
            harness.client._request(gone)
    assert not hasattr(harness.client, "_await_reply")   # no callback plumbing left
    # The wire takes `actions`, always plural. A singular key is refused rather than
    # guessed at: silently reading one action as a batch of twelve components would
    # spend twelve steps for one requested.
    with pytest.raises(RemoteError, match="actions must be a list"):
        harness.client._request("step", action=[0.0] * 12)
    # And an empty batch is refused rather than applied as nothing, so a caller never
    # has to tell "did nothing" from "did nothing yet".
    with pytest.raises(RemoteError, match="must not be empty"):
        harness.client._request("step", actions=[])
    assert harness.client.status()["steps_used"] == 0


def test_task_info_publishes_the_action_contract(harness):
    harness.client.reset()
    info = harness.client.task_info()
    assert info["action_dim"] == 12
    assert info["action_range"] == [-1.0, 1.0]
    # The base-frame requirement is the single largest gotcha; it must be stated.
    assert "base" in info["action_reference_frame"].lower()
    assert info["action_layout"]["base_mode"] == 11


def test_status_is_free(harness):
    harness.client.reset()
    spent = harness.client.status()["steps_used"]
    for _ in range(5):
        harness.client.status()
    assert harness.client.status()["steps_used"] == spent


def test_malformed_action_is_refused_and_never_reaches_the_env(harness):
    """Refused outright, not coerced. The FakeEnv asserts its own action length, so
    reaching it at all would fail there instead."""
    harness.client.reset()
    for bad in ([0.0] * 7, [float("nan")] * 12):    # wrong length, then not a number
        with pytest.raises(RemoteError):
            harness.client.step(bad)
    # A whole batch is validated before any of it is applied, so one bad action late in
    # a batch cannot leave the episode half-stepped.
    with pytest.raises(RemoteError):
        harness.client.step([[0.0] * 12] * 5 + [[0.0] * 7])
    # Refused actions must not be charged. (One step is already spent: `step` opened
    # the episode, which costs the same as the reset that would have.)
    assert harness.client.status()["steps_used"] == 1
    # And the daemon survives them.
    harness.client.step([[0.0] * 12] * 1)
    assert harness.client.status()["steps_used"] == 2


def test_repeated_episodes_charge_every_step(harness):
    """The agent's own loop over several episodes -- no harness call runs one for it."""
    spent = 0
    for _ in range(2):
        harness.client.reset()
        res = harness.client.step([[0.0] * 12] * 10)
        spent += res["steps"]
        assert res["success"] is True     # FakeEnv succeeds at step 3
    assert spent == 6
    assert harness.client.status()["steps_used"] == 8      # + one per reset


def test_evaluation_does_not_charge_the_interaction_budget(harness):
    """Two currencies: evaluation stepping is a submission cost, never learning."""
    harness.client.reset()
    harness.client.step([[0.0] * 12] * 1)
    before = harness.client.status()["steps_used"]
    harness.open_evaluation()
    harness.drive_evaluation()
    assert harness.client.status()["steps_used"] == before


def test_evaluation_returns_aggregates_only(harness):
    harness.open_evaluation()
    out = harness.drive_evaluation()
    assert out is not None, "the evaluation never finished"
    assert set(out) >= {"success_rate", "trials", "trials_attempted"}
    assert "per_episode" not in out and "per_scene" not in out
    blob = repr(out)
    for seed in ("101", "202"):     # the hidden battery must not leak
        assert seed not in blob


def test_submit_uses_the_eval_split_and_hidden_seeds(harness):
    """The battery must actually apply its hidden seeds.

    reset() takes no seed in robosuite, so the check is that env.rng was re-seeded to
    each battery seed: the recorded draws must equal what those seeds produce.
    """
    import numpy as np

    harness.client.reset()                      # dev env
    harness.open_evaluation()
    harness.drive_evaluation()
    assert harness.envs[0].split == "pretrain"
    assert harness.envs[1].split == "target"

    expected = [int(np.random.default_rng(s).integers(0, 2**31 - 1))
                for s in (101, 202)]
    assert harness.envs[1].draws == expected


def test_budget_exhaustion_is_reported_as_such(tmp_path):
    """Reported, never raised, and reported the same way every time.

    With one acting path there is one behaviour: a call with nothing left applies
    nothing and says why. It is also answered BEFORE "this episode has ended" -- running
    out of budget is what ended it, and telling the agent to reset would send it round a
    loop that cannot help it.
    """
    h = Harness(tmp_path, steps=2, success_after=-1)
    try:
        h.client.reset()                                  # costs one of the two
        assert h.client.step([[0.0] * 12] * 2)["steps"] == 1
        for _ in range(3):
            res = h.client.step([[0.0] * 12] * 1)
            assert res["steps"] == 0
            assert res["ended"] == Termination.BUDGET_EXHAUSTED
        # A batch that runs out part-way charges only what it applied.
        assert h.client.status()["steps_used"] == 2
    finally:
        h.stop()


def test_submission_budget_exhaustion(tmp_path):
    """One terminal submission: the harness cannot open a second evaluation either."""
    h = Harness(tmp_path, submissions=1, success_after=-1)
    try:
        h.open_evaluation()
        h.drive_evaluation()
        with pytest.raises(RuntimeError, match="submission budget exhausted"):
            h.open_evaluation()
    finally:
        h.stop()


def test_close_seals_a_verifiable_ledger(tmp_path):
    h = Harness(tmp_path, success_after=3, seeds=(7,))
    try:
        h.client.reset()
        # Four development steps in two episodes: the FakeEnv succeeds at step 3, which
        # ends the segment AND the episode, so the fourth needs a reset first.
        h.client.step([[0.0] * 12] * 4)
        h.client.reset()
        h.client.step([[0.0] * 12] * 1)
        h.open_evaluation()
        h.drive_evaluation()   # clears at 4 steps
        with h.control() as ctl:
            summary = ctl.seal()["summary"]          # sealing is the harness's plane
    finally:
        h.thread.join(timeout=5)

    assert summary["sealed"] is True
    assert summary["best_success_rate"] == 1.0
    assert summary["interaction_steps_total"] == 6      # 4 steps + two resets
    L.load_verified(h.ledger_path)     # chain intact end to end


def test_unknown_op_is_rejected_without_killing_the_daemon(harness):
    with pytest.raises(RemoteError, match="unknown op"):
        harness.client._request("definitely_not_an_op")
    harness.client.reset()             # daemon still alive
    # Only the reset: a refused op charges nothing.
    assert harness.client.status()["steps_used"] == 1


# -- acting over the socket ----------------------------------------------

def test_env_success_is_authoritative_over_the_wire(harness):
    """FakeEnv succeeds at step 3; a batch that asks for more must still be stopped by
    the harness, and the reply must say the environment decided it."""
    harness.client.reset()
    res = harness.client.step([[0.0] * 12] * 100)
    assert res["ended"] == Termination.ENV_SUCCESS
    assert res["success"] is True
    assert res["steps"] == 3                     # the rest were never applied
    assert harness.client.status()["steps_used"] == 4      # + the reset


def test_a_batch_costs_one_round_trip_and_charges_what_it_applies(harness):
    """The reason batching exists: an open-loop stretch should not pay a round trip per
    step. It still costs exactly the steps it applies."""
    harness.client.reset()
    res = harness.client.step([[0.0] * 12] * 2)
    assert res["steps"] == 2
    assert harness.client.status()["steps_used"] == 3      # + the reset


def test_work_chains_in_one_episode_over_the_socket(harness):
    """The agent works in pieces: send a few actions, look, send more -- one episode,
    no reset between, and the session lives in the daemon so nothing is carried in the
    agent's process."""
    harness.client.reset()
    a = harness.client.step([0.0] * 12)
    b = harness.client.step([0.0] * 12)
    assert a["steps"] == 1 and b["steps"] == 1
    assert a["episode_over"] is False
    assert harness.client.status()["steps_used"] == 3      # the reset, then two actions


def test_a_run_survives_disconnecting_between_pieces(tmp_path):
    """What makes 'write a script, look, write another' work: the session is the
    daemon's, so a fresh client picks up exactly where the last one stopped."""
    h = Harness(tmp_path, success_after=-1)
    try:
        h.client.reset()
        h.client.step([[0.0] * 12] * 4)
        h.client.disconnect()

        from harness.client import SpeedrunClient

        h.client = SpeedrunClient(h.sock_path)
        assert h.client.status()["steps_used"] == 5      # 4 steps + the reset
        assert h.client.step([0.0] * 12)["steps"] == 1
        assert h.client.status()["steps_used"] == 6
    finally:
        h.stop()


# -- interactive evaluation over the socket ----------------------------------

def test_interactive_evaluation_over_the_socket(tmp_path):
    """The protocol end to end: open a submission, drive inside trials, give up one,
    and receive the aggregate when the last trial ends."""
    h = Harness(tmp_path, submissions=2, success_after=-1, max_episode_steps=3)
    try:
        info = h.open_evaluation()          # the HARNESS opens the scored phase
        assert info["total_trials"] >= 1
        assert "obs" in info["trial"] and "instruction" in info["trial"]
        assert h.client.phase() == "evaluation"

        # Give up the first trial: reset is the special move during evaluation.
        out = h.client.reset()
        assert "trial" in out or out.get("evaluation_done")

        # Keep giving up until the evaluation closes, then check the aggregate.
        guard = 0
        while h.client.phase() == "evaluation" and guard < 50:
            out = h.client.reset()
            guard += 1
        assert out["evaluation_done"] is True
        assert out["success_rate"] == 0.0
        assert out["trials"] >= 1
    finally:
        h.stop()


def test_reset_outside_evaluation_returns_an_observation(harness):
    obs = harness.client.reset()
    assert "robot0_proprio-state" in obs   # an observation, not a trial descriptor
    assert harness.client.phase() == "development"


def test_service_forwards_every_session_option(tmp_path):
    """If Service's signature drifts from MeteredSession's, the daemon dies at container
    start with a TypeError on an option the session gained. Unit tests build
    MeteredSession directly, so only this check sees the gap."""
    import inspect

    from harness.session import MeteredSession

    service_params = set(inspect.signature(Service.__init__).parameters)
    session_params = set(inspect.signature(MeteredSession.__init__).parameters)
    # Service constructs the ledger from a path and fixes the splits, so those differ.
    session_only = session_params - service_params - {"ledger", "dev_split",
                                                      "eval_split"}
    assert not session_only, (
        f"MeteredSession options not accepted by Service: {sorted(session_only)}"
    )


# -- the context manager must not end the run --------------------------------

def _serving_daemon(tmp_path, **kw):
    """A daemon with NO client attached, so a test can open its own connections.

    Deliberately the same Harness rather than a parallel copy: a copy can drift and
    skip warming the service up, which is exactly the production step these tests
    exist to exercise.
    """
    kw.setdefault("submissions", 1)
    kw.setdefault("max_episode_steps", 5)
    return Harness(tmp_path, connect=False, seeds=(101,), sock_name="multi.sock", **kw)


def test_leaving_a_with_block_does_not_seal_the_run(tmp_path):
    """The instructions document `with SpeedrunClient() as sim:` as the way in, so
    if __exit__ sealed, the first block an agent left would end its run during
    development -- before evaluation, with no way back. It must only disconnect.
    """
    h = _serving_daemon(tmp_path)

    with SpeedrunClient(h.sock_path) as sim:
        sim.reset()
        sim.step([[0.0] * 12] * 1)
        spent = sim.status()["steps_used"]
    assert spent == 2                                  # the reset, then the action

    # A second connection must find the SAME session, still open and still metered.
    with SpeedrunClient(h.sock_path) as sim:
        assert sim.status()["steps_used"] == spent      # state persisted
        sim.step([[0.0] * 12] * 1)
        assert sim.status()["steps_used"] == spent + 1
        assert sim.phase() == "development"

    h.service.seal_if_open()


def test_seal_refreshes_the_status_snapshot(tmp_path):
    """The SIGTERM seal path bypasses the dispatch chokepoint, so without a refresh in
    seal_if_open the final status.json still says the run never completed."""
    from harness.debug import DebugRecorder

    rec = DebugRecorder(root=tmp_path / "dbg", enabled=True, every_n_steps=1,
                        ledger_path=tmp_path / "cost.jsonl")
    svc = Service(
        task="FakeTask", budgets=Budgets(interaction_steps=200, submissions=1),
        ledger_path=str(tmp_path / "cost.jsonl"),
        env_factory=lambda task, split="pretrain", scene=None: FakeEnv(
            success_after=3, split=split),
        trial_seeds=[101], max_episode_steps=10, control_uid=os.getuid(),
        recorder=rec)
    svc.warm_up()
    svc.seal_if_open()

    snap = json.loads((tmp_path / "dbg" / "status.json").read_text())
    assert snap["reward_now"]["ledger_ok"] is True
    assert "not sealed" not in (snap["reward_now"].get("ledger_reason") or "")


def test_only_the_harness_can_seal(tmp_path):
    """Sealing ends the run, so it belongs to the harness. The agent has no close()
    at all now -- and cannot reach the control op that replaced it."""
    from harness.control import ControlClient

    h = _serving_daemon(tmp_path)

    assert not hasattr(SpeedrunClient, "close")   # the agent cannot end its own run
    with SpeedrunClient(h.sock_path) as sim:
        sim.reset()
    # (refusal by uid is covered by test_agent_cannot_open_its_own_evaluation; this
    #  daemon deliberately grants control to the test's own uid so it can seal at all)
    with ControlClient(h.sock_path) as ctl:
        assert ctl.seal()["summary"]["sealed"] is True

    with SpeedrunClient(h.sock_path) as sim:
        with pytest.raises(RemoteError, match="closed"):
            sim.reset()


# -- entering the scored phase is the harness's decision ---------------------

def test_agent_cannot_open_its_own_evaluation(tmp_path):
    """Observed in a real run: an agent read client.py, found begin_evaluation and
    spent its single terminal submission during the UNSCORED development phase -- so
    it was evaluated without ever being told it was being evaluated, which is the only
    thing distinguishing the two phases. The wire must refuse until the harness arms it.
    """
    h = Harness(tmp_path, control_uid=0)     # only real root may control it
    try:
        assert not hasattr(SpeedrunClient, "begin_evaluation")
        # Even guessing the op name gets nowhere: the daemon checks the caller's uid,
        # which the kernel supplies.
        with pytest.raises(RemoteError):
            h.client._request("open_evaluation")
        assert h.client.phase() == "development"
    finally:
        h.stop()


def test_submit_is_not_on_the_agent_plane_at_all(tmp_path):
    """submit() was hand-off-and-wait: give the harness a controller and read the
    verdict. Evaluation is meant to be interactive, so it is gone from the agent's
    client AND from the wire -- not merely gated."""
    h = Harness(tmp_path, control_uid=0)
    try:
        assert not hasattr(SpeedrunClient, "submit")
        with pytest.raises(RemoteError, match="unknown op"):
            h.client._request("submit")
    finally:
        h.stop()


def test_the_harness_opens_the_evaluation_and_the_agent_then_drives_it(tmp_path):
    """The two planes meeting: the harness advances the phase, and from there the
    agent works exactly as it did in development."""
    h = Harness(tmp_path, success_after=2, max_episode_steps=50)
    try:
        assert h.client.phase() == "development"
        info = h.open_evaluation()                # harness plane
        assert info["total_trials"] >= 1
        assert h.client.phase() == "evaluation"

        res = h.client.step([[0.0] * 12] * 1000)   # agent plane, same call
        assert res["success"] is True
        assert res["episode_over"] is True        # the environment ended the trial
    finally:
        h.stop()


def test_end_development_is_available_to_the_agent_and_touches_no_simulator(tmp_path):
    """The agent's readiness signal: it must work while the evaluation is still shut."""
    h = Harness(tmp_path)
    try:
        h.client.reset()
        h.client.step([[0.0] * 12] * 1)
        out = h.client.end_development()
        assert out["development_ended"] is True
        assert out["steps_used"] == 2                  # the reset, then the action
        assert len(h.envs) == 1                   # no new env was constructed
        assert h.client.phase() == "awaiting_evaluation"  # closed, but NOT evaluating
        with pytest.raises(RemoteError, match="development is over"):
            h.client.step([[0.0] * 12] * 1)
    finally:
        h.stop()


# -- the agent sees images and proprioception, never object poses ------------

def test_object_state_never_crosses_the_wire(harness):
    """RoboCasa hands out ground-truth poses for every object and fixture. A real robot
    would need perception and localisation to get them, so giving them to the agent
    turns a perception problem into arithmetic -- the drawer can be closed by reading
    its world pose and subtracting, without ever looking at a pixel.

    Every path that returns an observation must be filtered, not just the obvious one.
    """
    privileged = {"drawer_obj_pos", "object-state"}
    kept = {"robot0_proprio-state", "robot0_eef_pos"}

    obs = harness.client.reset()
    assert kept <= set(obs)
    assert not (privileged & set(obs)), "reset() leaked object state"

    res = harness.client.step([[0.0] * 12] * 1)
    assert not (privileged & set(res["obs"])), "run_segment() leaked object state"

    seen = {}

    res = harness.client.step([[0.0] * 12] * 2)
    seen.update(res.get("obs") or {})
    assert seen, "the step reply carried no observation"
    assert not (privileged & set(seen)), "the step reply leaked object state"


PRIVILEGED = {"drawer_obj_pos", "object-state"}


def privileged_anywhere(payload) -> set:
    """Privileged keys ANYWHERE in a reply, at any nesting depth.

    Checking the top level is what let this through: the leak was an observation nested
    inside a trial descriptor, so every key-by-key assertion above passed while
    `trial_info` shipped the drawer's world pose one level down.
    """
    found, stack = set(), [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            found |= PRIVILEGED & set(node)
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return found


def test_no_agent_op_leaks_object_state_at_any_depth(tmp_path):
    """THE REGRESSION. `trial_info` and evaluation `reset` return a trial DESCRIPTOR,
    which contains an observation, and both were missed when the filter was applied
    per-op at the wire -- so a real agent was handed `drawer_obj_pos` and
    `drawer_obj_to_robot0_eef_pos` for every graded trial.

    Enumerating the whole agent surface, in both phases, is the point: a new op that
    forgets to filter fails here rather than in a scored run.
    """
    h = Harness(tmp_path, sock_name="leakall.sock")
    try:
        for label, reply in (
            ("task_info", h.client.task_info()),
            ("status", h.client.status()),
            ("reset", h.client.reset()),
            ("observe", h.client.observe()),
            ("observe(depth)", h.client.observe(width=16, depth=True)),
            ("step", h.client.step([0.0] * 12)),
            ("step(batch)", h.client.step([[0.0] * 12] * 2)),
            ("trial_info(dev)", h.client.trial_info()),
        ):
            assert not privileged_anywhere(reply), f"{label} leaked object state"

        h.open_evaluation()
        for label, reply in (
            ("trial_info(eval)", h.client.trial_info()),
            ("observe(eval)", h.client.observe()),
            ("step(eval)", h.client.step([[0.0] * 12] * 2)),
            ("reset(eval)", h.client.reset()),
        ):
            assert not privileged_anywhere(reply), f"{label} leaked object state"
    finally:
        h.stop()


def test_the_harness_itself_still_sees_everything(harness):
    """The filter is for the agent, not for the harness: success checking and the
    human-facing transcript need the full observation."""
    raw = harness.envs[0].reset() if harness.envs else None
    if raw is None:
        harness.client.reset()
        raw = harness.envs[0].reset()
    assert "drawer_obj_pos" in raw and "object-state" in raw


# -- the wall clock reported to the agent ------------------------------------

def test_task_info_also_carries_the_clock_and_the_budgets(tmp_path, monkeypatch):
    """The wall clock is configured in task.toml and reported at run time, never
    written into the instruction text -- so it has to be findable from whichever call
    the agent happens to make. `status` and `task_info` both carry it."""
    monkeypatch.setenv("RLEBENCH_DEVELOP_SECONDS", "600")
    h = Harness(tmp_path, sock_name="ticlock.sock")
    try:
        info = h.client.task_info()
        assert info["phase_seconds"] == 600.0
        assert 0 < info["seconds_remaining"] <= 600.0
        assert info["interaction_budget"] > 0
        # Both step ceilings are published, not left to be inferred by hitting them.
        assert info["max_episode_steps"] == info["max_steps_per_trial"] == 10
        assert info["obs_max_resolution"] >= info["obs_resolution"]
        # The success threshold is gone: the reward is continuous in the success rate,
        # and telling an agent it must reach some bar would be telling it something
        # untrue.
        assert "threshold" not in info
    finally:
        h.stop()


def test_observe_over_the_wire_is_free_and_leaks_no_object_state(tmp_path):
    """The new op has to respect the same wire filter as every other observation."""
    h = Harness(tmp_path, sock_name="obs.sock")
    try:
        h.client.reset()
        before = h.client.status()["steps_used"]
        look = h.client.observe()
        assert h.client.status()["steps_used"] == before
        assert set(look) >= {"obs", "instruction", "resolution", "max_resolution"}
        assert look["obs"], "observe returned nothing"
        assert all(k.startswith("robot") for k in look["obs"]), look["obs"].keys()
    finally:
        h.stop()


def test_an_obs_spec_crosses_the_wire(tmp_path):
    """The agent chooses what an observation contains, per call: the spec goes out with
    the actions and shapes what comes back."""
    from harness.obs import ObsSpec

    h = Harness(tmp_path, sock_name="specwire.sock")
    try:
        h.client.reset()
        wide = h.client.step([0.0] * 12)
        assert any(k.endswith("_image") for k in wide["obs"])

        narrow = h.client.step([0.0] * 12, obs_spec=ObsSpec(cameras=()))
        assert not any(k.endswith("_image") for k in narrow["obs"])
    finally:
        h.stop()


def test_a_narrowed_observation_still_carries_proprioception(tmp_path):
    """`cameras=()` drops pictures, not the robot's own state -- an agent servoing
    without looking still has to be able to."""
    from harness.obs import ObsSpec

    h = Harness(tmp_path, sock_name="blindwire.sock")
    try:
        h.client.reset()
        obs = h.client.step([0.0] * 12, obs_spec=ObsSpec(cameras=()))["obs"]
        assert "robot0_proprio-state" in obs
        assert not any(k.endswith("_image") for k in obs)
    finally:
        h.stop()


def test_status_reports_the_wall_clock_when_configured(tmp_path, monkeypatch):
    """The phase clock is configuration, not prose: the instructions name no numbers,
    so status() is where the agent finds the real one for this run."""
    monkeypatch.setenv("RLEBENCH_DEVELOP_SECONDS", "600")
    monkeypatch.setenv("RLEBENCH_EVALUATE_SECONDS", "120")
    h = Harness(tmp_path, sock_name="clock.sock")
    try:
        st = h.client.status()
        assert st["phase_seconds"] == 600.0
        assert 0 < st["seconds_remaining"] <= 600.0
        # The evaluate phase gets its own budget, and the clock restarts with it.
        with h.control() as ctl:
            ctl.open_evaluation()
        st = h.client.status()
        assert st["phase_seconds"] == 120.0
        assert 0 < st["seconds_remaining"] <= 120.0
    finally:
        h.stop()


def test_the_clock_is_scaled_by_the_same_multiplier_harbor_uses(tmp_path, monkeypatch):
    """`--agent-timeout-multiplier` scales what Harbor ENFORCES; this scales what the
    agent is TOLD. Reporting one while enforcing the other is worse than silence."""
    monkeypatch.setenv("RLEBENCH_DEVELOP_SECONDS", "600")
    monkeypatch.setenv("RLEBENCH_TIMEOUT_MULT", "0.5")
    h = Harness(tmp_path, sock_name="scaled.sock")
    try:
        assert h.client.status()["phase_seconds"] == 300.0
    finally:
        h.stop()


def test_no_clock_is_reported_when_none_is_configured(tmp_path, monkeypatch):
    """Told nothing beats told a guess: an agent handed a null budget is worse off."""
    monkeypatch.delenv("RLEBENCH_DEVELOP_SECONDS", raising=False)
    h = Harness(tmp_path, sock_name="noclock.sock")
    try:
        st = h.client.status()
        assert "seconds_remaining" not in st and "phase_seconds" not in st
    finally:
        h.stop()


# -- internal errors must not become a channel -------------------------------

def test_an_internal_error_is_redacted_before_it_reaches_the_agent(harness, monkeypatch):
    """Neither `str(exc)` nor the traceback may reach the agent; both are made
    of private material: the traceback carries source lines and paths from /opt/private
    (privileged.py and the scorer among them), and the message is whatever the simulator
    or a harness assertion chose to say.

    Redacted in the DAEMON, not in the client: the socket is 0666 and protocol.py is
    agent-readable, so an agent can speak the wire itself and read the raw reply.
    """
    def explode(*a, **kw):
        raise RuntimeError("scene 7 layout 3 seed 918273 failed to build")

    monkeypatch.setattr(harness.service._session, "observe", explode)

    P.send(harness.client._sock, {"op": "observe"})
    reply = harness.client._reader.read()

    assert reply["ok"] is False
    assert "traceback" not in reply
    assert "918273" not in str(reply) and "layout 3" not in str(reply)
    # The TYPE survives, which is all an agent can act on.
    assert "RuntimeError" in reply["error"]
    assert reply["kind"] == "internal"


def test_refusals_written_for_the_agent_still_reach_it_verbatim(harness):
    """Redacting everything would be safe and useless. `session.AgentFacingError` marks
    a message as written for the agent, and those must still arrive intact."""
    harness.client.reset()
    harness.client.end_development()

    with pytest.raises(RemoteError) as caught:
        harness.client.step([0.0] * 12)
    assert "internal harness error" not in str(caught.value)
    assert caught.value.kind in ("refused", "bad_request")


def test_a_second_concurrent_agent_connection_is_refused(tmp_path):
    """Two agent clients interleaving on the one session hand each other's episodes
    back, so the daemon refuses the second connection at accept. Control
    connections stay exempt -- they are what the concurrency exists for."""
    h = Harness(tmp_path, control_uid=0)     # this uid is an agent peer
    try:
        second = SpeedrunClient(h.sock_path)
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
    import time

    h = Harness(tmp_path, control_uid=0)
    try:
        assert h.client.status()["phase"] == "development"
        h.client.disconnect()
        for _ in range(200):  # EOF processing frees the slot near-instantly
            try:
                fresh = SpeedrunClient(h.sock_path)
                assert fresh.status()["phase"] == "development"
                break
            except RemoteError:
                time.sleep(0.01)
        else:
            raise AssertionError("slot never freed after disconnect")
        fresh.disconnect()
    finally:
        h.stop()


def test_control_uid_connections_are_never_refused(harness):
    """A collect hook must reach the daemon past an idle agent socket. The default
    harness grants control to this uid, so a second concurrent connection must be
    served, not refused."""
    import socket as _socket

    sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    sock.connect(harness.sock_path)           # concurrent with harness.client
    try:
        P.send(sock, {"op": "status"})
        assert (P.LineReader(sock).read() or {}).get("ok") is True
    finally:
        sock.close()
