"""Tests for task02's Harbor wiring: task.toml and the generated step directories.

Every failure this catches is SILENT. A misplaced `artifacts` key parses clean and
collects nothing; a missing `next-trial` hook parses clean and leaves a trial ungraded; a
seal hook on the wrong step parses clean and kills the daemon mid-run. None of them
raises, and all of them are only visible in a finished run's reward.

`artifacts` in particular was written three lines too low and nested under
`[steps.agent]`, where Harbor ignores it -- caught here, not by reading it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from harness import config as C

TASK_DIR = Path(__file__).resolve().parents[1] / "tasks" / "task02"

# Shared template wiring needs representative groups; the coverage audit below
# still checks every active group's split.
GROUP_DIRS = [TASK_DIR / name for name in ("01-washing-dishes", "06-setting-the-table")]
for group in GROUP_DIRS:
    if not (group / "task.toml").is_file():
        subprocess.run(
            [sys.executable, str(TASK_DIR / "build_groups.py"), "--emit", group.name],
            check=True, capture_output=True,
        )


@pytest.fixture(scope="module", params=GROUP_DIRS, ids=lambda p: p.name)
def task_dir(request) -> Path:
    return request.param


@pytest.fixture(scope="module")
def cfg(task_dir) -> dict:
    return tomllib.loads((task_dir / "task.toml").read_text())


@pytest.fixture(scope="module")
def steps(cfg) -> list[dict]:
    return cfg["steps"]


@pytest.fixture(scope="module")
def eval_tasks(task_dir) -> tuple[str, ...]:
    """The split the image ships for THIS group, not the canonical config."""
    sys.path.insert(0, str(TASK_DIR))
    from build_groups import staged_eval_tasks

    return staged_eval_tasks(task_dir.name)


def hooks(step: dict) -> list[str]:
    return [h["command"] for h in step.get("verifier", {}).get("collect", [])]


def handoff_dir(cfg: dict) -> str:
    """Where the development agent is told to leave its harness. One value, read from the
    config rather than restated, so a rename cannot leave half the assertions behind."""
    return cfg["environment"]["env"]["RLEBENCH_HARNESS_DIR"]


# -- step structure ----------------------------------------------------------

def test_one_develop_step_then_one_step_per_trial(steps):
    """One trial is one Harbor step is one fresh agent, so the step list IS the trial
    count. A mismatch is silent at run time: surplus steps advance past the end of the
    plan, missing ones leave trials unreached and scored 0."""
    expected = ["develop"] + [f"eval_{i:02d}" for i in range(1, C.TRIALS_PER_TASK + 1)]
    assert [s["name"] for s in steps] == expected


def test_every_eval_step_advances_a_trial(steps):
    for step in steps[1:]:
        assert any("next-trial" in h for h in hooks(step)), step["name"]


def test_the_workspace_is_collected_once(steps, cfg):
    assert steps[0].get("artifacts") == ["/workspace"]
    assert not cfg.get("artifacts")
    assert not any(s.get("artifacts") for s in steps[1:])


def test_the_reward_strategy_is_final(cfg):
    """Every step writes a CUMULATIVE reward, so 'final' is always the most complete
    number. 'mean' would average one cumulative snapshot per step of the same run, which
    is not a quantity."""
    assert cfg["multi_step_reward_strategy"] == "final"


# -- collect hooks -----------------------------------------------------------

def test_the_develop_step_opens_the_evaluation(steps):
    assert any("open-evaluation" in h for h in hooks(steps[0]))


def test_the_develop_step_does_not_advance_a_trial(steps):
    assert not any("next-trial" in h for h in hooks(steps[0]))


def test_only_the_last_step_seals(steps):
    """Sealing kills the daemon, which serves every step. On any earlier step it
    would destroy every trial still to come."""
    sealing = [s["name"] for s in steps if any("seal" in h for h in hooks(s))]
    assert sealing == [steps[-1]["name"]]


def test_the_last_step_scores_its_trial_before_sealing(steps):
    """Order matters: sealing first would close the session with the last trial ungraded."""
    commands = hooks(steps[-1])
    assert commands.index(next(c for c in commands if "next-trial" in c)) \
        < commands.index(next(c for c in commands if "seal" in c))


def test_every_control_hook_runs_as_root(steps):
    """The control plane is uid-gated in the daemon. A hook that ran as the agent user
    would be refused, silently, because collect-hook failures never abort a trial."""
    for step in steps:
        for hook in step.get("verifier", {}).get("collect", []):
            assert hook["user"] == "root", step["name"]


def test_control_hooks_import_from_the_root_only_tree(steps):
    """PYTHONPATH=/opt/private is the point: the control module must never be importable
    from anywhere the agent's uid can write.

    PYTHONSAFEPATH=1 is what makes PYTHONPATH decisive. Harbor execs collect hooks from
    the image WORKDIR, /workspace, and `python -m` searches the CWD first -- so on its
    own the PYTHONPATH above loses to any package the agent left there."""
    for step in steps:
        for hook in step.get("verifier", {}).get("collect", []):
            if "harness.control" in hook["command"]:
                assert "PYTHONPATH=/opt/private" in hook["command"]
                assert "PYTHONSAFEPATH=1" in hook["command"]


def test_the_handoff_directory_cannot_shadow_the_root_only_package(cfg):
    """PYTHONPATH=/opt/private is NOT sufficient on its own, and this is the assertion
    that says so.

    The image WORKDIR is /workspace and Harbor execs the collect hooks and the verifier
    from it. `python -m harness.control` prepends the CWD to sys.path AHEAD of PYTHONPATH,
    so a handoff directory named `harness` -- which the agent owns and is told to fill --
    resolves before /opt/private/harness the moment it contains an __init__.py. The
    control plane and the scorer both go with it, and Harbor treats a failed collect hook
    as best effort: the run finishes, recorded as a completed zero.
    """
    handoff = handoff_dir(cfg)
    assert PurePosixPath(handoff).name != "harness", handoff
    assert cfg["verifier"]["env"]["RLEBENCH_HARNESS_DIR"] == handoff


@pytest.mark.parametrize("shadow", ["harness", None], ids=["shadowed", "clean"])
def test_the_control_plane_survives_a_package_the_agent_wrote(cfg, tmp_path, shadow):
    """The same claim, executed rather than asserted.

    Lay out the shipped arrangement -- the agent's deliverable under the workspace, the
    trusted tree on PYTHONPATH, the hook's own working directory -- and run the control
    module the way task.toml runs it. `shadow` is the adversarial case: an agent that
    names a package `harness` regardless of where it was told to put its harness. It owns
    /workspace, so nothing stops it."""
    workspace = tmp_path / "workspace"
    (workspace / PurePosixPath(handoff_dir(cfg)).name).mkdir(parents=True)
    if shadow:
        (workspace / shadow).mkdir()
        (workspace / shadow / "__init__.py").write_text(
            "raise ImportError('the agent package answered')\n")

    private = tmp_path / "private" / "harness"
    private.mkdir(parents=True)
    (private / "__init__.py").write_text("")
    (private / "control.py").write_text("print('private control plane')\n")

    proc = subprocess.run(
        [sys.executable, "-m", "harness.control"],
        cwd=workspace,
        env={**os.environ, "PYTHONPATH": str(private.parent), "PYTHONSAFEPATH": "1"},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "private control plane" in proc.stdout


# -- artifacts ---------------------------------------------------------------

def test_the_workspace_is_collected_from_the_develop_step(steps):
    """THE bug this file exists for. Written after `[steps.agent]` it becomes
    `steps.agent.artifacts`, which Harbor ignores -- and the deliverable, the whole
    harness the evaluation agents inherit, is never collected."""
    assert steps[0].get("artifacts") == ["/workspace"]


# -- the split travels on the right channel ----------------------------------

def test_neither_split_is_in_the_container_environment(cfg):
    """Anything in [environment.env] is agent-readable: `docker exec` builds its
    environment from the container's CONFIG, not from PID 1's live environ, and Harbor
    starts every agent that way. Both splits are fixed constants in the root-only config
    module instead, so there is no run-time channel to leak."""
    env = cfg["environment"]["env"]
    for key in ("RLEBENCH_TRAIN_TASKS", "RLEBENCH_EVAL_TASKS", "RLEBENCH_EVAL_SALT"):
        assert key not in env


def test_the_evaluation_split_never_travels_by_environment_variable(cfg):
    """Putting the split in [environment.env] and unsetting it in entrypoint.sh does
    not work. `docker exec` builds its environment from
    the container's CONFIG, not from PID 1's live environ, and Harbor starts every agent
    with `docker exec`, so the unset removed the variable from nothing the agent could
    reach. Measured: `docker run -e SECRET=x` + an entrypoint that unsets it, then
    `docker exec -u agent ... printenv SECRET` prints `x`, for every user.

    Anything in [environment.env] is agent-readable, full stop. The split travels as a
    root-only bind-mounted file instead.
    """
    env = cfg["environment"]["env"]
    assert "RLEBENCH_EVAL_TASKS" not in env
    assert "RLEBENCH_EVAL_SALT" not in env




def test_overridable_knobs_are_templated(cfg):
    """A BARE value in [environment.env] does not merely fail to read the shell -- it
    pins the knob and silently discards whatever the operator set. The documented
    reduced-scale command ran the full split and said nothing."""
    env = cfg["environment"]["env"]
    for key in ("RLEBENCH_INTERACTION_STEPS", "RLEBENCH_DEVELOP_SECONDS",
                "RLEBENCH_TRIAL_SECONDS"):
        assert env[key].startswith("${"), f"{key} is not overridable from the shell"


def test_the_training_split_is_not_unset(cfg, task_dir):
    """Isolation that blocks the task is a bug: the agent must be able to practise."""
    entrypoint = (TASK_DIR / "image" / "entrypoint.sh").read_text()
    assert "unset RLEBENCH_TRAIN_TASKS" not in entrypoint


def test_no_evaluation_task_is_named_in_task_toml(task_dir, eval_tasks):
    """It is world-readable inside the container the agent runs in."""
    text = (task_dir / "task.toml").read_text()
    for task in eval_tasks:
        assert task not in text


# -- clocks stay in step with the timeouts Harbor enforces -------------------

def _default(value: str) -> str:
    """The fallback out of a `${VAR:-default}` template, or a bare value unchanged.

    These knobs are templated so the launching shell can override them; the DEFAULT is
    what the assertions below are about.
    """
    import re

    match = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}", value)
    return match.group(1) if match else value


def test_the_reported_clocks_mirror_the_enforced_timeouts(cfg, steps):
    """The daemon only REPORTS a clock; Harbor enforces it. Drift between them tells the
    agent something false about how long it has."""
    env = cfg["environment"]["env"]
    assert float(_default(env["RLEBENCH_DEVELOP_SECONDS"])) \
        == steps[0]["agent"]["timeout_sec"]
    for step in steps[1:]:
        assert float(_default(env["RLEBENCH_TRIAL_SECONDS"])) \
            == step["agent"]["timeout_sec"]


def test_the_interaction_budget_matches_the_configured_default(cfg):
    env = cfg["environment"]["env"]
    assert int(_default(env["RLEBENCH_INTERACTION_STEPS"])) == C.INTERACTION_STEPS
    assert int(_default(env["RLEBENCH_MAX_STEPS_PER_TRIAL"])) == C.MAX_STEPS_PER_TRIAL


# -- generated step directories ----------------------------------------------

def test_every_step_has_an_instruction(steps, task_dir):
    for step in steps:
        path = task_dir / "steps" / step["name"] / "instruction.md"
        assert path.is_file() and path.stat().st_size > 0, step["name"]


def test_the_develop_instruction_names_the_handoff_directory(cfg, task_dir):
    text = (task_dir / "steps" / "develop" / "instruction.md").read_text()
    assert handoff_dir(cfg) in text and "MANUAL.md" in text


def test_every_evaluation_instruction_points_at_the_same_directory(cfg, steps, task_dir):
    for step in steps[1:]:
        text = (task_dir / "steps" / step["name"] / "instruction.md").read_text()
        assert handoff_dir(cfg) in text and "MANUAL.md" in text, step["name"]


def test_no_instruction_names_an_evaluation_task(task_dir, eval_tasks):
    """They are the agent's brief. Naming a graded task there would hand over the split
    in the most direct way possible."""
    for path in (task_dir / "steps").rglob("instruction.md"):
        text = path.read_text()
        for task in eval_tasks:
            assert task not in text, f"{path} names {task}"


# -- the shared simulator layer -----------------------------------------------

SIM_DIR = TASK_DIR.parents[1] / "sim" / "robocasa"


def test_task02_builds_from_the_shared_simulator_image(task_dir):
    dockerfile = (TASK_DIR / "image" / "Dockerfile").read_text()
    assert "ARG SIM_IMAGE=" in dockerfile
    assert "FROM ${SIM_IMAGE}" in dockerfile


def test_the_simulator_pins_live_in_exactly_one_file():
    """pins.env is read by base/Dockerfile (via --build-arg), sim/robocasa/robocasa.sh and
    the Makefile. A default in the Dockerfile would let the image's pins fork from the
    venv's silently, which is the failure this arrangement rules out."""
    pins = dict(
        line.split("=", 1)
        for line in (SIM_DIR / "pins.env").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    for key in ("ROBOSUITE_SHA", "ROBOCASA_SHA", "ROBOCASA_ASSET_VERSION"):
        assert pins.get(key), key
        # Declared, never defaulted: `ARG X` and not `ARG X=...`.
        assert f"\nARG {key}\n" in (SIM_DIR / "base" / "Dockerfile").read_text(), key

    # The image build lives in the script, so that is where the pins must reach the
    # Dockerfile from. The Makefile still includes pins.env, for ROBOCASA_SIM_IMAGE.
    script = (TASK_DIR.parents[1] / "sim" / "robocasa" / "robocasa.sh").read_text()
    assert "pins.env" in script
    for key in ("ROBOSUITE_SHA", "ROBOCASA_SHA", "ROBOCASA_ASSET_VERSION"):
        assert f'--build-arg {key}="${key}"' in script, key
        # Reading it (`sha=$ROBOSUITE_SHA`) is the point; assigning a LITERAL is the
        # drift this rules out. So: no `KEY=` that is not immediately a `$expansion`.
        assert re.search(rf'{key}=(?!["\']?\$)', script) is None, key

    assert "include sim/robocasa/pins.env" in (TASK_DIR.parents[1] / "Makefile").read_text()


def test_the_vendored_tree_is_in_repo_and_never_committed():
    """third_party/ holds robosuite, robocasa and the ~15 GB dataset, so every default
    path derives from the repo's own location -- and none of it can be staged."""
    repo = TASK_DIR.parents[1]
    script = (repo / "sim" / "robocasa" / "robocasa.sh").read_text()
    assert 'VENDOR="${ROBOCASA_VENDOR:-$REPO/third_party}"' in script
    assert "third_party/" in (repo / ".gitignore").read_text()


def test_every_group_runs_the_shared_image_and_selects_itself(cfg, task_dir):
    """One image for every group: task.toml must name the image the Makefile builds and
    select its own entry in the root-only split table."""
    makefile = (TASK_DIR.parents[1] / "Makefile").read_text()
    assert cfg["environment"]["docker_image"] == "rlebench-task02-agent:dev"
    assert cfg["environment"]["env"]["RLEBENCH_GROUP"] == task_dir.name
    assert "-t rlebench-task02-agent:dev" in makefile
    assert "tasks/task02/image" in makefile


def test_every_step_has_a_reference_solution(steps, task_dir):
    """Harbor's oracle runs steps/<name>/solution/solve.sh. A missing one makes the
    end-to-end check silently skip that step."""
    for step in steps:
        path = task_dir / "steps" / step["name"] / "solution" / "solve.sh"
        assert path.is_file(), step["name"]


# -- no machine-specific paths -----------------------------------------------

# HOST path roots that belong to one developer's box. Container paths (/opt,
# /workspace, /logs, /var/lib, /run) and published conventions (/mnt/robocasa-assets)
# are fine -- they are the same everywhere.
_PERSONAL_ROOTS = ("/scratch/", "/nethome/", "/home/", "/Users/")


def _tracked_config_files():
    root = TASK_DIR.parents[1]
    yield root / "Makefile"
    for pattern in ("*.md", "*.toml", "*.sh", "*.py", "*.yaml", "*.env"):
        yield from TASK_DIR.rglob(pattern)
        yield from (root / "sim").rglob(pattern)


def test_no_machine_specific_paths_are_checked_in():
    """A path from one developer's box resolves to nothing on anyone else's, and the
    failure surfaces deep inside a container as a missing model.xml rather than as
    "you have not configured this". Paths are derived from the repo or required
    explicitly -- see the Makefile's `robocasa-require-*` guards."""
    offenders = []
    for path in _tracked_config_files():
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue          # prose and examples may cite a path
            for root in _PERSONAL_ROOTS:
                if root in line:
                    offenders.append(f"{path.relative_to(TASK_DIR.parents[1])}:{lineno}")
    assert not offenders, offenders


# -- the training set covers the evaluation task -----------------------------
# The benchmark's central claim: the graded task recombines what was practised. A primitive
# it needs that no training task provides makes that clause unsolvable by construction, so
# its zero would measure the split rather than the agent.
#
# These read tasks/task02/dev/data/task02_primitive_audit.json, which is checked in, so no simulator is
# needed. Regenerate it after a RoboCasa pin bump:
#     .venv-robocasa/bin/python tasks/task02/dev/decompose.py --audit \
#         --out tasks/task02/dev/data/task02_primitive_audit.json

def _build_groups():
    import sys

    sys.path.insert(0, str(TASK_DIR))
    import build_groups

    return build_groups


def test_every_group_has_a_well_formed_training_set():
    """`train` is taken as written -- any tasks, any number -- so this asserts only what
    would BREAK a build rather than what would merely make a different one.
    Group.__post_init__ enforces the same two at import; asserting them here names the
    failure instead of collapsing the whole module into a collection error."""
    B = _build_groups()
    for g in B.GROUPS:
        # `held_out` names the graded task and indexes the subtask counts the writer
        # emits, so a non-member crashes `--emit`.
        assert g.held_out in set(g.names), f"{g.name}: held-out task is not a member"
        assert g.held_out not in g.train, f"{g.name}: trains on its own graded task"
        # Not enforced at import, but a repeat is always a typo: the daemon would round
        # -robin a task twice and the split would misreport its own width.
        assert len(set(g.train)) == len(g.train), f"{g.name}: train repeats a task"
        # `composites`/`atomics` partition `train` -- they are derived, so this is really
        # a check that nothing has broken the two properties.
        assert set(g.train) == set(g.composites) | set(g.atomics), f"{g.name}: split"


def test_no_active_group_ships_an_uncovered_evaluation_task():
    """Every primitive the held-out task needs has a provider among the training tasks.

    This is the benchmark's central claim -- that the graded task recombines what was
    practised -- and it is the one that was asserted rather than checked.
    """
    B = _build_groups()
    audit = B.load_audit()
    broken = {g.name: B.disqualified(g, audit) for _, g in B.active_groups()
              if B.disqualified(g, audit)}
    assert not broken, broken


def test_inactive_groups_really_cannot_meet_the_bar():
    """The other direction: `active=False` is a finding, not a preference. A group that
    could ship but is switched off is silently costing the suite a group."""
    B = _build_groups()
    audit = B.load_audit()
    excusable = [g.name for g in B.GROUPS
                 if not g.active
                 and B.why_inactive(g, audit).startswith("marked inactive by hand")]
    assert not excusable, (
        f"{excusable} are inactive but their splits are sound; either reactivate them or "
        "record why they are held back")


def test_the_audit_covers_every_task_the_table_names():
    """A GROUPS edit that adds a task the audit has never seen would make its coverage
    unknowable, and `coverage()` treats a missing row as providing nothing -- which reads
    as a gap in whichever group happens to reference it."""
    B = _build_groups()
    audit = B.load_audit()
    missing = sorted({t for g in B.GROUPS for t in g.names} - set(audit))
    assert not missing, (
        f"{missing} are not in the primitive audit; regenerate it with decompose.py "
        "--audit --out tasks/task02/dev/data/task02_primitive_audit.json")


# -- the verifier's output must not reach the next trial's agent --------------

def test_every_eval_step_scrubs_the_previous_verifier_output(task_dir, eval_tasks):
    """THE LEAK THIS CLOSES. /logs/verifier is one directory shared by every step, and
    Harbor empties it only immediately BEFORE each verifier runs -- which is after the
    next agent has already come and gone. So `diagnosis.json` from trial N sits there,
    world-readable (Harbor chmods the directory 0777), for the whole of trial N+1's agent
    phase. It carries `per_trial[].task`: the held-out task name, which every other
    channel withholds on purpose -- the trial descriptor omits it, `task_info` returns
    None for it during evaluation, and the daemon banner refuses to print it.

    setup.sh is the only hook that runs in that window, and it runs before the agent, so
    an agent cannot skip it.
    """
    for step in sorted((task_dir / "steps").glob("eval_*")):
        setup = step / "workdir" / "setup.sh"
        assert setup.is_file(), f"{step.name} has no setup.sh to scrub with"
        text = setup.read_text()
        assert "/logs/verifier/diagnosis.json" in text, f"{step.name} leaves diagnosis.json"
        assert "/logs/verifier/reward.json" in text, f"{step.name} leaves reward.json"
        assert text.index("rm -f /logs/verifier") < text.index("python")


def test_the_held_out_task_is_in_no_agent_readable_file(task_dir, eval_tasks):
    """The one secret, checked across everything the image makes world-readable: the
    agent payload, the step instructions, task.toml (readable from /proc/self/environ)
    and the scripts chmod 755 into /opt."""
    image = TASK_DIR / "image"
    readable = [
        *(image / "payload_agent").rglob("*.py"),
        *(task_dir / "steps").rglob("instruction.md"),
        task_dir / "task.toml",
        task_dir / "tests" / "test.sh",
        image / "check_isolation.sh",
        image / "entrypoint.sh",
    ]
    for path in readable:
        if not path.is_file():
            continue
        text = path.read_text()
        for task in eval_tasks:
            assert task not in text, f"{path} names the held-out task {task}"
