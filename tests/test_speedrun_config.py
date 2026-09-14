"""Tests for task01's evaluation plan, budgets and thresholds.

Every number lives in `config.py`, which ships only in the root-only tree. The plan is
the single source of which trials are run; there is no separate "test battery",
because with the agent in the loop the verifier cannot re-run an evaluation.

What has to hold: the plan is reproducible for a build, specific to the task,
free of repeats, and not enumerable by the agent.
"""

from __future__ import annotations

import pytest

from harness import config as C
from harness import ledger as L
from harness.session import Budgets, MeteredSession
from test_speedrun_session import FakeEnv


# -- plan derivation ---------------------------------------------------------

def test_plan_is_deterministic():
    assert C.eval_trials("CloseDrawer") == C.eval_trials("CloseDrawer")


def test_plan_differs_between_tasks():
    """Otherwise every task would be graded on the same episodes."""
    assert C.eval_trials("CloseDrawer") != C.eval_trials("OpenDrawer")


def test_trial_seeds_are_unique_within_a_plan():
    """A repeated seed would silently make the evaluation smaller than it claims."""
    plan = C.eval_trials("CloseDrawer", plan=((1, 8), (2, 8)))
    seeds = [p[3] for p in plan]
    assert len(set(seeds)) == len(seeds) == 16


def test_a_salt_changes_the_plan():
    """The salt is what lets a deployment keep its trials private even if the code
    is public."""
    assert C.eval_trials("CloseDrawer") != C.eval_trials("CloseDrawer", salt="secret")


def test_adding_trials_does_not_renumber_the_existing_ones():
    """Growing a scene's trial count must not change the trials already in it, or
    results from different plan sizes would not be comparable."""
    small = C.eval_trials("CloseDrawer", plan=((1, 2),))
    large = C.eval_trials("CloseDrawer", plan=((1, 5),))
    assert large[:2] == small


def test_seeds_are_in_a_valid_range():
    for entry in C.eval_trials("CloseDrawer", plan=((1, 16),)):
        assert 0 <= entry[3] < 2**31 - 1


def test_scene_ids_map_to_target_split_pairs():
    """Scene ids are 1-based into the target split's own (layout, style) pairs, so
    pinning a scene stays inside RoboCasa's distribution."""
    plan = C.eval_trials("CloseDrawer", plan=((1, 1), (4, 1)))
    assert [(p[1], p[2]) for p in plan] == [C.TARGET_SCENE_IDS[0],
                                            C.TARGET_SCENE_IDS[3]]
    # A scene's trials come out together and in plan order, so a reader can map a
    # trial index back to the scene it ran in.
    wider = C.eval_trials("CloseDrawer", plan=((1, 2), (3, 1)))
    assert [p[0] for p in wider] == [1, 1, 3]


def test_plan_rejects_impossible_schedules():
    for bad in ((), ((1, 0),), ((0, 1),), ((99, 1),)):
        with pytest.raises(ValueError):
            C.eval_trials("CloseDrawer", plan=bad)


# -- excluded tasks ----------------------------------------------------------

def test_the_reset_satisfied_task_is_excluded():
    """PrepareBroilingStation's predicate is true at reset, so a do-nothing agent
    scores 100% -- it measures nothing and must not be scoreable."""
    assert not C.is_scoreable("PrepareBroilingStation")
    assert C.is_scoreable("CloseDrawer")


def run_evaluation(sess):
    """Drive a whole evaluation the way the agent does: step, reset, repeat. There is no
    harness call that runs one for you -- that is the point of the interface."""
    sess.begin_evaluation()
    out = None
    while sess.evaluating:
        res = sess.step([[0.0] * 12] * 5000)
        if sess.evaluating and (res["episode_over"] or res["steps"] == 0):
            out = sess.reset()
    return out if isinstance(out, dict) and "success_rate" in (out or {}) \
        else sess.state.history[-1]


# -- cost bounds -------------------------------------------------------------

def test_submission_step_ceiling_is_enforced(tmp_path):
    """An agent that never finishes must not spend unbounded steps in one submission by
    coming back for another batch."""

    def factory(task, split="pretrain", scene=None):
        return FakeEnv(success_after=-1, split=split)

    sess = MeteredSession(
        task="T",
        budgets=Budgets(interaction_steps=10, submissions=2),
        ledger=L.Ledger(tmp_path / "cap.jsonl"),
        env_factory=factory,
        trial_seeds=[1, 2, 3, 4],
        max_episode_steps=5,
    )
    out = run_evaluation(sess)
    assert out["success_rate"] == 0.0
    submit = [r for r in L.load_verified(sess.ledger.path) if r.kind == L.KIND_SUBMIT][-1]
    assert submit.payload["eval_steps"] <= 5 * 4
    # Submissions are a separate currency: the interaction budget is untouched.
    assert sess.state.steps_used == 0


def test_submission_cap_does_not_charge_the_interaction_budget(tmp_path):
    def factory(task, split="pretrain", scene=None):
        return FakeEnv(success_after=2, split=split)

    sess = MeteredSession(
        task="T",
        budgets=Budgets(interaction_steps=5, submissions=1),
        ledger=L.Ledger(tmp_path / "sep.jsonl"),
        env_factory=factory,
        trial_seeds=[1, 2, 3],
        max_episode_steps=50,
    )
    run_evaluation(sess)
    assert sess.state.steps_used == 0
    assert sess.steps_remaining == 5


# -- documented decisions ----------------------------------------------------

def test_recorded_working_assumptions():
    """Recorded so a change is a deliberate edit here, not drift elsewhere."""
    # One terminal submission: the score is final, there is no interact/evaluate loop.
    assert C.SUBMISSIONS == 1
    assert C.MAX_STEPS_PER_TRIAL > 0
    # The two reward weights are the whole formula and must sum to 1.0, or a perfect
    # free run would not score 1.0 and scores would stop being comparable across
    # deployments that tuned them.
    assert C.W_OUTCOME + C.W_EFFICIENCY == pytest.approx(1.0)
    assert C.W_OUTCOME > C.W_EFFICIENCY, "solving the task must dominate"


def test_the_default_plan_can_resolve_a_success_rate(tmp_path):
    """One trial makes the score binary, which cannot rank anything between total
    failure and total success. The default must be wide enough to have a middle."""
    assert C.plan_trial_count() >= 5
    scenes = {scene for scene, _ in C.EVAL_PLAN}
    assert len(scenes) > 1, "a single scene measures luck, not generalisation"


# -- the evaluation plan is configurable from task.toml ----------------------

def test_plan_spec_round_trips():
    assert C.parse_eval_plan("1x2,2x2,3x2") == ((1, 2), (2, 2), (3, 2))
    assert C.parse_eval_plan(" 4x1 ") == ((4, 1),)


@pytest.mark.parametrize("spec", [
    "",                # empty
    "1",               # no trial count
    "1x",              # no trial count
    "axb",             # not integers
    "1x0",             # a scene with no trials
    "0x2",             # scene ids are 1-based
    "99x2",            # beyond the target split's scenes
])
def test_a_malformed_plan_is_refused(spec):
    """The daemon parses this at startup so a bad plan fails where it is visible,
    rather than at the first graded trial."""
    with pytest.raises(ValueError):
        C.parse_eval_plan(spec)


def test_plan_trial_count_follows_the_plan():
    assert C.plan_trial_count(C.parse_eval_plan("1x3,2x4")) == 7


def test_env_overrides_are_read_and_validated(monkeypatch):
    monkeypatch.setenv("RLEBENCH_INTERACTION_STEPS", "250")
    assert C.env_int("RLEBENCH_INTERACTION_STEPS", C.INTERACTION_STEPS) == 250
    monkeypatch.setenv("RLEBENCH_INTERACTION_STEPS", "lots")
    with pytest.raises(ValueError):
        C.env_int("RLEBENCH_INTERACTION_STEPS", C.INTERACTION_STEPS)
    monkeypatch.delenv("RLEBENCH_INTERACTION_STEPS")
    assert C.env_int("RLEBENCH_INTERACTION_STEPS", 7) == 7


def test_a_session_needs_either_a_plan_or_trial_seeds(tmp_path):
    """The two are alternatives, and neither being present is a caller bug."""
    with pytest.raises(ValueError, match="eval_plan_fn"):
        MeteredSession(
            task="T",
            budgets=Budgets(interaction_steps=1, submissions=1),
            ledger=L.Ledger(tmp_path / "none.jsonl"),
            env_factory=lambda task, split="pretrain", scene=None: FakeEnv(),
        )


def _task_toml():
    """The TEMPLATE task.toml, not an emitted one.

    The emitted matrix is build output and is gitignored, so a checkout that has not run
    `make task01` does not have it -- and the dev suite must not depend on having built
    anything. The template is also where a regression would actually be introduced:
    `build_levels.py` only substitutes placeholders into it.

    It parses as TOML with the placeholders in place, since every one of them sits
    inside a quoted string.
    """
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[1]
    return tomllib.loads(
        (root / "tasks/task01/_template/task.toml.in").read_text())


def test_scoring_weights_travel_on_the_verifier_channel_only():
    """[environment.env] reaches the whole container, so the agent can read it out of
    its own /proc/self/environ. How the agent is being weighted must not be in there."""
    cfg = _task_toml()
    agent_visible = cfg["environment"]["env"]
    verifier_only = cfg["verifier"]["env"]

    for key in ("RLEBENCH_W_OUTCOME", "RLEBENCH_W_EFFICIENCY"):
        assert key in verifier_only, f"{key} must be configurable"
        assert key not in agent_visible, f"{key} is readable by the agent"

    assert (float(verifier_only["RLEBENCH_W_OUTCOME"])
            + float(verifier_only["RLEBENCH_W_EFFICIENCY"])) == pytest.approx(1.0)


def test_the_configured_knobs_parse():
    """A knob that reaches the container as an unparseable string is a run scored under
    rules nobody chose, so the values in task.toml have to be checked here."""
    env = _task_toml()["environment"]["env"]
    assert int(env["RLEBENCH_INTERACTION_STEPS"]) > 0
    assert int(env["RLEBENCH_MAX_STEPS_PER_TRIAL"]) > 0
    # The plan is deliberately NOT among them -- it travels by the root-only override
    # file, so the default in config.py is what a run without one gets.
    assert C.plan_trial_count(C.EVAL_PLAN) >= 5


def test_task_toml_declares_no_knob_that_nothing_reads():
    """A setting that looks live and is not is worse than no setting: tuning it changes
    nothing, silently, and a value that merely happens to match the constant in config.py
    reads as confirmation. Every RLEBENCH_* name task.toml sets must be read by something.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    declared = {k for k in _task_toml()["environment"]["env"] if k.startswith("RLEBENCH_")}
    # Every reader: the package, and the shell that builds the daemon's environment.
    readers = list((root / "tasks/task01/harness").rglob("*.py"))
    readers += list((root / "tasks/task01/_template").rglob("*.sh"))
    src = "\n".join(p.read_text() for p in readers)
    unread = sorted(k for k in declared if not re.search(rf"\b{k}\b", src))
    assert not unread, f"task.toml sets {unread}, which nothing reads"


def test_the_daemon_only_secrets_never_travel_by_environment():
    """The salt makes the trial seeds underivable and the plan says which episodes are
    graded. NEITHER may reach the agent, and an environment variable cannot carry either.

    Setting both in [environment.env] and `unset`-ing them in entrypoint.sh leaves the
    plan fully readable. The unset does nothing: `docker exec` builds each new process's environment from the container's
    stored config rather than from PID 1's live environ, and Harbor starts the agent with
    `docker exec`. So the assertion is on the CHANNEL: no such variable exists at all.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    entrypoint = (root / "tasks/task01/_template/image/entrypoint.sh").read_text()
    env = _task_toml()["environment"]["env"]

    for key in ("RLEBENCH_BATTERY_SALT", "RLEBENCH_EVAL_PLAN"):
        assert key not in env, (
            f"{key} is in [environment.env], which the agent reads out of its own "
            f"/proc/self/environ. It belongs in {C.EVAL_OVERRIDES_PATH}.")
    # An unset in the entrypoint does not protect it.
    assert "unset RLEBENCH" not in entrypoint

    # The daemon reads them from the root-only file instead, and from nowhere else.
    daemon = (root / "tasks/task01/harness/daemon_main.py").read_text()
    assert "eval_overrides" in daemon
    for key in ("RLEBENCH_BATTERY_SALT", "RLEBENCH_EVAL_PLAN"):
        assert key not in daemon, f"{key} is still read from the environment"


def test_the_override_file_is_read_only_from_the_root_only_path():
    """A plan file anywhere the agent can traverse is the same leak by another route."""
    assert C.EVAL_OVERRIDES_PATH.startswith("/opt/private/")
    # Absent is the normal case, and must mean "use the defaults", not "fail".
    assert C.eval_overrides("/nonexistent/eval_plan.txt") == {}


def test_the_reported_clock_matches_the_enforced_timeout():
    """task.toml states each phase's wall clock TWICE -- as the timeout Harbor enforces
    and as the seconds the daemon reports to the agent -- because a task cannot read its
    own step config back at run time. Nothing stops those drifting apart except this.

    Drift is not cosmetic: the agent would budget against one number and be killed by
    another, which is the failure this clock was added to prevent.
    """
    cfg = _task_toml()
    env = cfg["environment"]["env"]

    for step in cfg["steps"]:
        key = f"RLEBENCH_{step['name'].upper()}_SECONDS"
        assert key in env, f"{step['name']} has no reported clock ({key})"
        assert float(env[key]) == float(step["agent"]["timeout_sec"]), (
            f"{step['name']}: agent is told {env[key]}s but Harbor enforces "
            f"{step['agent']['timeout_sec']}s"
        )


# -- the agent-readable surface ----------------------------------------------

def _repo():
    import pathlib
    return pathlib.Path(__file__).resolve().parents[1]


def test_the_isolation_check_covers_every_forbidden_module():
    """check_isolation.sh proves the agent cannot import the scoring side, so its list
    has to BE build_assets.FORBIDDEN_IN_AGENT. When they drifted, `config` and
    `evaluation` were forbidden but unchecked -- the two modules holding the evaluation
    plan and the reward weights.
    """
    import sys

    sys.path.insert(0, str(_repo() / "tasks/task01"))
    import build_assets

    script = (_repo() / "tasks/task01/_template/image/check_isolation.sh").read_text()
    for module in build_assets.FORBIDDEN_IN_AGENT:
        name = module[:-3]
        if name in ("transcript",):
            continue    # not staged into the agent tree at any level, so nothing to import
        assert f"harness.{name}" in script, (
            f"{name} is forbidden in the agent payload but check_isolation.sh never "
            f"tries to import it")


def test_the_evaluate_step_scrubs_the_previous_verifier_output():
    """/logs/verifier is shared by both steps and Harbor empties it only immediately
    BEFORE each verifier runs -- which is after the next agent has come and gone. So the
    develop step's reward.json and diagnosis.json are readable for the whole evaluation
    phase unless setup.sh removes them, and setup.sh is the only hook that runs in that
    window. Task02 shares this layout and leaks its held-out task name this way.
    """
    setup = (_repo()
             / "tasks/task01/_template/steps/evaluate/workdir/setup.sh").read_text()
    assert "/logs/verifier/diagnosis.json" in setup
    assert "/logs/verifier/reward.json" in setup
    # Before anything else in the file that could fail and skip it.
    assert setup.index("rm -f /logs/verifier") < setup.index("python")


# --- container readiness -------------------------------------------------------

def _repo_file(rel):
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / rel).read_text()


def test_the_healthcheck_waits_for_every_service_the_level_ships():
    """Not just the metering daemon.

    `test -S speedrun.sock` passed as soon as the daemon bound, which at L2 and L3 is up to
    a minute before SAM3 and Contact-GraspNet finish loading -- so Harbor could `docker
    exec` the agent into a container where `segment_by_text` raises. The entrypoint writes
    one marker after everything it started is answering, and the healthcheck tests that,
    which also keeps it from having to know which level it is running on.
    """
    toml = _repo_file("tasks/task01/_template/task.toml.in")
    assert 'command = "test -f /run/rlebench/ready"' in toml
    assert "test -S /run/rlebench/speedrun.sock" not in toml


def test_the_marker_is_written_after_both_waits_and_cleared_first():
    entry = _repo_file("tasks/task01/_template/image/entrypoint.sh")
    write = entry.index(": > /run/rlebench/ready")
    assert entry.index("rm -f /run/rlebench/ready") < write, "a stale marker would pass"
    for wait in ("daemon ready on", "perception ready on"):
        assert entry.index(wait) < write, f"marker written before {wait!r}"
    assert write < entry.index('exec "$@"')


def test_the_perception_service_is_started_only_where_it_ships():
    """Presence of the vendored tree decides, so there is no second source of truth about
    the level -- the L2/L3 image build copies it and L1's does not."""
    entry = _repo_file("tasks/task01/_template/image/entrypoint.sh")
    assert entry.count("[ -d /opt/perception ]") == 2      # launch, then wait
    assert "perception_service" in entry


def test_the_isolation_check_agrees_with_the_levels_that_ship_the_library():
    """check_isolation.sh runs AS THE AGENT, which cannot read config.py, so its level
    list is hardcoded. If it drifts from config, every run at the affected level fails
    its own isolation check and reports a broken harness."""
    script = _repo_file("tasks/task01/_template/image/check_isolation.sh")
    wanted = "|".join(C.SKILLS_LEVELS)
    assert f"    {wanted}) wants_skills=1 ;;" in script, (
        f"check_isolation.sh must branch on {wanted}, matching config.SKILLS_LEVELS")
