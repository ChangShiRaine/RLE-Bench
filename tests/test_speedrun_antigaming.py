"""Anti-gaming suite: every way a run could collect reward it did not earn.

Task01 pays for two things — solving the task, and doing so having consumed little
simulator interaction. So the attacks worth defending are: inflate the outcome, or
understate the cost. Each test below names a concrete attack and asserts it fails.

Ledger forgery itself (payload edits, re-hashed records, interior deletion, tail
truncation, post-seal appends) is covered in test_speedrun_ledger.py; here we check that
those failures actually reach the reward.
"""

from __future__ import annotations

import json

import pytest

from harness import config as C
from harness import ledger as L
from harness import scoring as S
from harness.session import Termination
from harness.session import Budgets, MeteredSession
from test_speedrun_session import FakeEnv


def make_session(tmp_path, *, name, plan=None, steps=10_000, submissions=1,
                 success_after=-1, max_episode_steps=1_000):
    made = []

    def factory(task, split="pretrain", scene=None):
        env = FakeEnv(success_after=success_after, split=split)
        made.append(env)
        return env

    sess = MeteredSession(
        task="T",
        budgets=Budgets(interaction_steps=steps, submissions=submissions),
        ledger=L.Ledger(tmp_path / f"{name}.jsonl"),
        env_factory=factory,
        max_episode_steps=max_episode_steps,
        eval_plan_fn=lambda t: plan or [(1, 1, 1, 7)],
    )
    return sess, made



# More actions than any episode here can absorb, so only the
# harness can stop it.
MANY = [[0.0] * 12] * 5000


# -- understating cost -------------------------------------------------------

def test_attack_never_submit_to_keep_cost_low(tmp_path):
    """Spend nothing, submit nothing: the cheapest run possible, and it must be worth
    exactly zero. Efficiency is scaled by the success rate, so being cheap buys nothing
    on its own -- and completing the protocol is not itself worth any credit."""
    sess, _ = make_session(tmp_path, name="nosub")
    summary = sess.close()
    r = S.score_path(sess.ledger.path)
    assert summary["submissions_used"] == 0
    assert r["efficiency_score"] == 0.0
    assert r["outcome_score"] == 0.0
    assert r["reward"] == 0.0


def test_attack_learn_during_evaluation_where_steps_are_free(tmp_path):
    """Evaluation steps are charged to the submission, not to the interaction budget,
    so an agent could try to do its learning there and report a tiny interaction cost.

    It cannot pay: a trial burned on learning is a trial that failed, and the outcome
    term is 0.8 of the reward while the whole efficiency term is worth 0.2. Sacrificing
    a trial to save interaction always loses more than it gains.
    """
    plan = [(1, 1, 1, 11), (1, 1, 1, 12)]
    # Succeeds only from the second trial onwards -- i.e. "learned" during trial 1.
    sess, made = make_session(tmp_path, name="freelearn", plan=plan,
                              success_after=-1, max_episode_steps=5)

    sess.begin_evaluation()
    # Trial 1 is spent "learning" and fails...
    sess.step(MANY)
    # ...after which the controller works perfectly.
    for env in made:
        env.success_after = 1
    while sess.evaluating:
        res = sess.reset()
    assert res["success_rate"] < 1.0        # the sacrificed trial is simply lost


def test_attack_abandon_a_batch_to_avoid_paying_for_it(tmp_path):
    """Actions that reached the environment are charged, whatever the agent's own code
    does next. With the agent driving its own loop, crashing after acting is simply a
    crash -- the steps are already spent and already journalled."""
    sess, _ = make_session(tmp_path, name="raise")
    sess.reset()
    sess.step([[0.0] * 12] * 3)
    assert sess.state.steps_used == 4            # the reset + three real steps
    sess.close()
    interacts = [r for r in L.load_verified(sess.ledger.path)
                 if r.kind == L.KIND_INTERACT]
    assert sum(int(r.payload["steps"]) for r in interacts) == 4


def test_attack_forged_ledger_does_not_reach_the_reward(tmp_path):
    """Ledger tampering is detected in the ledger module; confirm it actually gates
    the reward rather than merely raising somewhere."""
    led = L.Ledger(tmp_path / "forge.jsonl")
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_INTERACT, steps=99_000)
    led.append(L.KIND_SUBMIT, submission_index=0, steps_cumulative=99_000,
               success_rate=1.0, cleared=True, episodes=1)
    led.append(L.KIND_SEAL, record_count=4)

    lines = led.path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["steps"] = 10               # claim a near-free run
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    led.path.write_text("\n".join(lines) + "\n")

    r = S.score_path(led.path)
    assert r["ledger_ok"] is False
    assert r["reward"] == 0.0                  # not "cheap and successful"


def test_attack_truncate_the_ledger_to_hide_spending(tmp_path):
    """Dropping the tail leaves a valid prefix, so the seal is what catches it."""
    led = L.Ledger(tmp_path / "trunc.jsonl")
    led.append(L.KIND_START, task="T", split="pretrain")
    led.append(L.KIND_INTERACT, steps=100)
    led.append(L.KIND_INTERACT, steps=90_000)
    led.append(L.KIND_SEAL, record_count=4)

    lines = led.path.read_text().splitlines()
    led.path.write_text("\n".join(lines[:2]) + "\n")   # keep only the cheap prefix

    r = S.score_path(led.path)
    assert r["ledger_ok"] is False
    assert "not sealed" in r["ledger_reason"]


def test_attack_delete_the_ledger_entirely(tmp_path):
    """Destroying the evidence must not read as 'spent nothing'."""
    r = S.score_path(tmp_path / "gone.jsonl")
    assert r["reward"] == 0.0 and r["ledger_ok"] is False


# -- inflating the outcome ---------------------------------------------------

def test_attack_declare_success(tmp_path):
    """There is no channel for it, and that is the design: `success` is read off the
    environment on every reply, and nothing the agent sends is consulted."""
    sess, _ = make_session(tmp_path, name="claim", success_after=-1)
    sess.begin_evaluation()
    assert sess.step([0.0] * 12)["success"] is False
    while sess.evaluating:
        out = sess.reset()
    assert out["success_rate"] == 0.0


def test_attack_reset_spam_to_reach_an_easier_trial(tmp_path):
    """Resetting abandons the current trial and scores it where it stands, so
    skipping through trials to find a favourable one costs every skipped trial."""
    plan = [(1, 1, 1, i) for i in range(4)]
    sess, _ = make_session(tmp_path, name="spam", plan=plan, success_after=-1)
    sess.begin_evaluation()
    for _ in range(4):
        out = sess.reset()
    assert out["evaluation_done"] is True
    assert out["trials"] == 4
    assert out["success_rate"] == 0.0        # all four were forfeited


def test_attack_second_submission_after_a_bad_first(tmp_path):
    """One terminal submission: a poor result cannot be retried."""
    from harness.session import BudgetExhausted

    sess, _ = make_session(tmp_path, name="retry", success_after=-1)
    sess.begin_evaluation()
    while sess.evaluating:
        sess.reset()
    with pytest.raises(BudgetExhausted):
        sess.begin_evaluation()


# -- exceeding limits --------------------------------------------------------

def test_attack_stall_forever_to_avoid_ending_a_trial(tmp_path):
    """The environment never signals done (ignore_done=True), so a controller that
    never terminates would run without end if the harness did not cap the trial."""
    sess, _ = make_session(tmp_path, name="stall", max_episode_steps=25)
    sess.begin_evaluation()
    res = sess.step(MANY)
    assert res["steps"] == 25
    assert res["info"]["trial_ended"] == "max_steps"
    # The cap is reported as `ended`, like a development horizon -- not left to be
    # deduced from episode_over alone.
    assert res["ended"] == "horizon"
    assert res["episode_over"] is True


def test_attack_send_no_actions_forever(tmp_path):
    """An agent that keeps asking without ever acting gets nothing for it: empty calls
    are free, change nothing, and score nothing."""
    sess, _ = make_session(tmp_path, name="idle")
    sess.begin_evaluation()
    for _ in range(50):
        assert sess.step([])["steps"] == 0
    assert sess.state.steps_used == 0
    assert sess.close()["best_success_rate"] == 0.0


def test_attack_exceed_the_interaction_budget(tmp_path):
    """Requests past the budget are refused, never silently truncated, so an agent
    cannot quietly overspend and report the capped number."""
    sess, _ = make_session(tmp_path, name="over", steps=10, max_episode_steps=1_000)
    sess.reset()
    # Truncated at the budget and SAID SO, rather than either overspending or pretending
    # the whole batch ran.
    res = sess.step(MANY)
    assert res["steps"] == 9                     # the reset already took one of the ten
    assert res["ended"] == Termination.BUDGET_EXHAUSTED
    assert sess.state.steps_used == 10
    # And nothing gets through afterwards.
    assert sess.step(MANY)["steps"] == 0
    assert sess.state.steps_used == 10


def test_attack_unbounded_submission_cost(tmp_path):
    """Coming back for another batch must not let one submission consume unbounded
    simulator time."""
    plan = [(1, 1, 1, i) for i in range(4)]

    sess, _ = make_session(tmp_path, name="cap", plan=plan, max_episode_steps=5)
    sess.begin_evaluation()
    for _ in range(200):                       # one action at a time, coming back each
        if not sess.evaluating:
            break
        if sess.step([0.0] * 12)["episode_over"]:
            sess.reset()
    records = L.load_verified(sess.ledger.path)
    submit = [r for r in records if r.kind == L.KIND_SUBMIT][-1]
    assert submit.payload["eval_steps"] <= 20


def test_attack_ask_for_a_camera_the_task_does_not_publish(tmp_path):
    """The observation contract is three specific viewpoints. `sim.render` will resolve
    ANY camera in the MuJoCo model, and a kitchen carries more than three, so a request
    naming one of them must not be honoured -- a scene overview would turn the
    perception problem into a different, easier one."""
    from harness import env as ENV

    assert ENV.allowed_cameras(("robot0_eye_in_hand", "birdview", "frontview")) == \
        ("robot0_eye_in_hand",)
    assert ENV.allowed_cameras(("birdview",)) == ()
    assert ENV.allowed_cameras(None) == ENV.DEFAULT_CAMERAS


def test_attack_a_malformed_obs_spec_cannot_take_the_daemon_down(tmp_path):
    """The spec is agent-authored input crossing INTO the trusted side. A daemon that
    died on it would strand the ledger unsealed, which reads as a failed run."""
    from harness.client import RemoteError
    from test_speedrun_service import Harness

    h = Harness(tmp_path, sock_name="badspec.sock")
    try:
        for bad in ({"width": "huge"}, {"width": {}}, {"cameras": "notalist"},
                    {"width": -5}, {"height": 10**9}):
            try:
                h.client._request("observe", **bad)
            except RemoteError:
                pass                      # refused is fine; crashing is not
        # Still alive and still metering. A refused spec charges nothing, and looking
        # never opens an episode -- so there is nothing to see until a reset.
        assert h.client.status()["steps_used"] == 0
        assert h.client.observe()["live"] is False
        h.client.reset()
        assert h.client.observe()["obs"]
    finally:
        h.stop()


def test_attack_smuggle_an_extra_camera_through_a_running_controller(tmp_path):
    """Same hole, reached the other way: via a controller's declared spec rather than
    an explicit observe()."""
    from harness.obs import ObsSpec

    rendered = []

    sess, _ = make_session(tmp_path, name="greedy")
    sess.reset()
    import harness.env as ENV

    real = ENV.render_frames
    ENV.render_frames = (lambda env, cams, w, h, depth=False:
                         rendered.append(tuple(cams)) or {})
    try:
        sess.step([0.0] * 12, obs_spec=ObsSpec(
            width=256, cameras=("robot0_eye_in_hand", "birdview")))
    finally:
        ENV.render_frames = real

    assert rendered == [("robot0_eye_in_hand",)], "an unpublished camera was rendered"


# -- configuration sanity ----------------------------------------------------

def test_budgets_are_the_agreed_values():
    """These are decisions; a change should be a deliberate edit, not drift."""
    assert C.INTERACTION_STEPS == 100_000    # the default; overridable per run
    assert C.MAX_STEPS_PER_TRIAL == 1_000    # RoboCasa's own default horizon
    assert C.SUBMISSIONS == 1
    # Efficiency is a fraction of the run's own budget, so an override rescales the
    # curve rather than redefining what a score means.
    assert C.W_OUTCOME == 0.80 and C.W_EFFICIENCY == 0.20


def _forge_chain(records):
    """Rebuild a ledger from scratch the way an AGENT could: public sha256 over the
    public fields, no key. This is exactly the attack the HMAC exists to stop."""
    import hashlib
    import json

    prev, out = "0" * 64, []
    for i, (kind, payload) in enumerate(records):
        h = hashlib.sha256()
        h.update(prev.encode())
        h.update(str(i).encode())
        h.update(kind.encode())
        h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        d = h.hexdigest()
        out.append(json.dumps({"index": i, "kind": kind, "payload": payload,
                               "prev_hash": prev, "hash": d},
                              sort_keys=True, separators=(",", ":")))
        prev = d
    return "\n".join(out) + "\n"


def test_the_scorer_never_reads_the_agent_writable_copy(tmp_path, monkeypatch):
    """The one that matters most.

    The ledger the daemon exports to /logs/artifacts sits in a directory Harbor
    deliberately makes AGENT-WRITABLE (it chmods the publish mount chain to 0777 so
    tasks can publish). So the agent CAN delete that file and write a replacement --
    and the chain is a plain sha256 over public fields, so the replacement verifies
    perfectly. Measured: a from-scratch ledger claiming a one-step flawless run scored
    1.0 through a verifier that read the copy.

    The defence is not cryptography, it is which file gets read. The verifier shares the
    agent's container and scores the root-only original; the export is for humans. This
    pins that: a perfect forgery sitting in the artifacts dir must earn nothing when the
    authoritative ledger is absent -- never a silent fallback onto it.
    """
    from harness import verify_main as V

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "cost.jsonl").write_text(_forge_chain([
        ("start", {"task": "CloseDrawer", "split": "pretrain",
                   "interaction_budget": 100_000, "submission_budget": 1,
                   "success_threshold": 1.0, "planned_trials": 1}),
        ("interact", {"steps": 1}),
        ("submit", {"submission_index": 0, "steps_cumulative": 1,
                    "success_rate": 1.0, "cleared": True, "episodes": 1}),
        ("seal", {"record_count": 4, "steps_used": 1, "submissions_used": 1,
                  "cleared": True, "best_success_rate": 1.0}),
    ]))

    monkeypatch.setattr(V, "LEDGER", tmp_path / "var" / "cost.jsonl")  # absent
    monkeypatch.setattr(V, "ARTIFACTS", artifacts)
    monkeypatch.setattr(V, "REWARD_PATH", tmp_path / "reward.json")
    V.main()

    payload = json.loads((tmp_path / "reward.json").read_text())
    assert payload["reward"] == 0.0
    diagnosis = json.loads((tmp_path / "diagnosis.json").read_text())
    assert diagnosis["ledger_ok"] is False
    assert diagnosis["ledger_source"] == str(V.LEDGER)   # never the artifacts copy


def test_the_scorer_ships_only_where_the_agent_cannot_read_it(tmp_path):
    """The scorer decides the reward, so it must not be readable by the thing it
    scores. It lives in the root-only tree; the boundary check enforces that."""
    import pathlib

    images = pathlib.Path(__file__).resolve().parents[1] / "tasks/task01/images"
    contexts = sorted(images.glob("L*"))
    if not contexts:
        pytest.skip("image contexts not generated")

    # EVERY level, not one of them. The levels differ in what the agent tree holds, so
    # a boundary checked on L1 alone says nothing about L2 or L3 -- and L3, which hands
    # out privileged state, is exactly the one worth doubting.
    for ctx in contexts:
        payload = ctx / "payload_agent" / "harness"
        private = ctx / "payload_private" / "harness"
        for module in ("scoring.py", "verify_main.py", "config.py", "ledger.py",
                       "privileged.py"):
            assert not (payload / module).exists(), f"{ctx.name}: {module}"
            assert (private / module).exists(), f"{ctx.name}: {module}"


def test_inflating_the_recorded_budget_breaks_the_chain(tmp_path):
    """Cost is graded as a fraction of the budget in the ledger's start record, so a
    larger recorded budget would make the same spend look more efficient. That record
    is written by the root daemon and hash-chained, so editing it fails the gate."""
    import json

    from harness import scoring as S

    led = L.Ledger(tmp_path / "budget.jsonl")
    led.append(L.KIND_START, task="CloseDrawer", split="pretrain",
               interaction_budget=10_000)
    led.append(L.KIND_INTERACT, steps=9_000)
    led.append(L.KIND_SUBMIT, submission_index=0, steps_cumulative=9_000,
               success_rate=1.0, cleared=True, episodes=1)
    led.append(L.KIND_SEAL, record_count=4)

    honest = S.score_path(led.path)
    assert honest["efficiency_score"] < C.W_EFFICIENCY      # 90% of budget spent

    lines = led.path.read_text().splitlines()
    first = json.loads(lines[0])
    first["payload"]["interaction_budget"] = 10_000_000   # "I had a huge budget"
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    led.path.write_text("\n".join(lines) + "\n")

    forged = S.score_path(led.path)
    assert forged["ledger_ok"] is False
    assert forged["reward"] == 0.0


def test_the_harness_must_cap_trials_because_the_env_never_stops():
    """robosuite computes done = (timestep >= horizon) and not ignore_done, and
    RoboCasa sets ignore_done=True -- so a zero/None cap would mean 'run forever'."""
    assert C.MAX_STEPS_PER_TRIAL and C.MAX_STEPS_PER_TRIAL > 0


# -- the agent image must not carry ground truth ------------------------------

def test_agent_payload_contains_no_scoring_ground_truth():
    """The boundary audit, made permanent.

    build_assets.py already refuses to ship the scoring MODULES to the agent. This
    catches the subtler regression: a threshold, weight, evaluation seed or success
    predicate copied into a module that *is* agent-visible. An agent that can read the
    reward function can optimise against it instead of the task.
    """
    import pathlib

    images = pathlib.Path(__file__).resolve().parents[1] / "tasks/task01/images"
    payloads = [c / "payload_agent" for c in sorted(images.glob("L*"))]
    if not payloads:                              # generated tree; skip if unbuilt
        pytest.skip("image contexts not generated")

    forbidden = ("W_OUTCOME", "W_EFFICIENCY", "INTERACTION_STEPS",
                 "reward_weights", "eval_trials", "TARGET_SCENE_IDS",
                 "_EVAL_DOMAIN", "_check_success", "score_ledger", "EXCLUDED_TASKS")
    leaks = [
        f"{path.relative_to(payload.parent.parent)}: {name}"
        for payload in payloads
        for path in payload.rglob("*.py")
        for name in forbidden
        if name in path.read_text()
    ]
    assert not leaks, "scoring ground truth reachable from the agent: " + "; ".join(leaks)
