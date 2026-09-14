"""The daemon's own startup path.

THE GAP THIS CLOSES. Every other test builds `MeteredSession` or `Service` directly, so
nothing executed `daemon_main.main()` -- the code the container actually runs. A removed
constant left a dangling `args.threshold` in the startup banner, one line after the
warm-up and one line before `serve()`. The full suite stayed green, and the failure
surfaced only as a real run dying with

    HealthcheckError: test -S /run/rlebench/speedrun.sock

after burning ~3 minutes building an environment it then threw away. The banner is
printed *after* the expensive warm-up, so anything wrong there costs a full env
construction before it shows.

`Service` is stubbed out here, so these run in milliseconds and need no simulator,
assets or GPU. What they cover is argument parsing, config resolution, the wiring handed
to `Service`, and every line of `main()` up to and including `serve()`.
"""

from __future__ import annotations

import pytest

from harness import config as C
from harness import daemon_main as D


class StubService:
    """Records what the daemon built, and never touches a simulator."""

    last: "StubService | None" = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.served = None
        StubService.last = self

    def warm_up(self):
        return {"action_dim": 12, "instruction": "Close the left drawer."}

    def serve(self, socket_path, once=False):
        self.served = socket_path

    def seal_if_open(self):
        pass


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "Service", StubService)
    # Nothing below should be reached, but make failure loud rather than silent if it is.
    monkeypatch.setattr("harness.env.make_env",
                        lambda **kw: pytest.fail("the daemon built a real env"))
    for var in ("RLEBENCH_EVAL_PLAN", "RLEBENCH_INTERACTION_STEPS",
                "RLEBENCH_SUBMISSIONS", "RLEBENCH_MAX_STEPS_PER_TRIAL",
                "RLEBENCH_MAX_EPISODE_STEPS", "RLEBENCH_BATTERY_SALT",
                "RLEBENCH_DEBUG"):
        monkeypatch.delenv(var, raising=False)
    # The plan and the salt come from a root-only FILE, never a variable, so point the
    # daemon at a writable one. Absent by default, which is the case most tests want.
    monkeypatch.setattr(C, "EVAL_OVERRIDES_PATH", str(tmp_path / "eval_plan.txt"))

    def run(*extra, task="CloseDrawer"):
        return D.main([
            "--task", task,
            "--socket", str(tmp_path / "s.sock"),
            "--ledger", str(tmp_path / "cost.jsonl"),
            "--transcript", str(tmp_path / "t.jsonl"),
            "--artifacts-dir", str(tmp_path / "art"),
            *extra,
        ])

    return run


@pytest.fixture
def overrides(monkeypatch, tmp_path):
    """Write the root-only plan/salt file the `daemon` fixture is pointed at."""
    path = tmp_path / "eval_plan.txt"
    monkeypatch.setattr(C, "EVAL_OVERRIDES_PATH", str(path))

    def write(text: str) -> None:
        path.write_text(text if text.endswith("\n") else text + "\n")

    return write


def test_the_daemon_starts_and_serves(daemon, tmp_path):
    """The regression: main() must reach serve() without raising.

    A dangling attribute in the startup banner is invisible to every other test and
    fatal in the container -- the socket is never bound, so the healthcheck fails 40
    times and the trial errors before the agent runs at all.
    """
    assert daemon() == 0
    assert StubService.last.served == str(tmp_path / "s.sock")


def test_the_banner_only_names_things_that_exist(daemon, capsys):
    """It is printed after the ~40 s warm-up, so a bad reference there is expensive to
    discover. Assert it renders and carries the run's actual configuration."""
    daemon()
    out = capsys.readouterr().out
    assert "[daemon] warm-up: action_dim=12" in out
    assert f"trials={C.plan_trial_count()}" in out
    assert "threshold" not in out, "the threshold concept was removed"


def test_the_run_s_own_plan_is_used_and_reported(daemon, overrides, capsys):
    """Not the default plan. The banner is how an operator confirms an override took,
    and reporting the default would quietly contradict what is actually being run."""
    overrides("plan=1x3,2x3")
    daemon()

    assert "trials=6" in capsys.readouterr().out
    plan_fn = StubService.last.kwargs["eval_plan_fn"]
    assert len(plan_fn("CloseDrawer")) == 6


def test_budgets_come_from_the_environment(daemon, monkeypatch):
    monkeypatch.setenv("RLEBENCH_INTERACTION_STEPS", "250")
    monkeypatch.setenv("RLEBENCH_MAX_STEPS_PER_TRIAL", "77")
    daemon()

    budgets = StubService.last.kwargs["budgets"]
    assert budgets.interaction_steps == 250
    assert StubService.last.kwargs["max_episode_steps"] == 77


def test_a_malformed_plan_fails_at_startup_not_at_the_first_trial(daemon, overrides):
    """Where the message is visible in the container log, rather than mid-evaluation."""
    overrides("plan=not-a-plan")
    with pytest.raises(ValueError):
        daemon()


def test_a_malformed_override_file_fails_at_startup_too(daemon, overrides):
    """A typo'd key means the run is about to be graded under a plan nobody chose. Only
    an operator can write this file, so silence would hide their mistake, not an agent's.
    """
    overrides("plann=1x2")
    with pytest.raises(ValueError, match="unknown key"):
        daemon()


def test_an_unscoreable_task_is_refused(daemon):
    """Its success predicate is already true at reset, so it measures nothing."""
    with pytest.raises(SystemExit):
        daemon(task=next(iter(C.EXCLUDED_TASKS)))


def test_the_salt_reaches_the_plan_but_the_default_plan_does_not_depend_on_it(
        daemon, overrides):
    """The salt is what makes trial seeds underivable. It reaches the daemon by the
    root-only file and by no other route -- an environment variable could not hold it,
    since `docker exec` hands the agent whatever the container config set."""
    daemon()
    unsalted = StubService.last.kwargs["eval_plan_fn"]("CloseDrawer")

    overrides("salt=a-secret")
    daemon()
    salted = StubService.last.kwargs["eval_plan_fn"]("CloseDrawer")

    assert [p[3] for p in salted] != [p[3] for p in unsalted]


def test_the_salt_is_never_taken_from_the_environment(daemon, monkeypatch):
    """Setting the environment variable must change nothing: if it moved the seeds, the secret would be one an agent can
    read out of its own /proc/self/environ."""
    daemon()
    unsalted = StubService.last.kwargs["eval_plan_fn"]("CloseDrawer")

    monkeypatch.setenv("RLEBENCH_BATTERY_SALT", "a-secret")
    daemon()
    assert [p[3] for p in StubService.last.kwargs["eval_plan_fn"]("CloseDrawer")] \
        == [p[3] for p in unsalted]
