"""Interactive evaluation: a submission evaluates the agent's WORKFLOW.

A submission is not "here is a frozen controller, go score it". The agent stays in the
loop: within each trial it runs controllers, and whenever one hands back control
it can inspect the history and choose the next move. What is being measured is the
whole orchestration, not a single controller.

Structure of one submission:

    scene 1 ── trial 1, trial 2, ... trial k
    scene 2 ── trial 1, trial 2, ... trial k
    ...                                        success rate = mean over ALL trials

Scenes are fixed (layout, style) pairs drawn from the target split's own scene set;
trials within a scene vary object instances and placements.

A trial ends in one of these ways (the TRIAL_END_* constants below):

  agent_reset             the agent chose to reset while the trial was still running.
                          During evaluation that is a SPECIAL MOVE meaning "give up
                          this trial": it is scored immediately, in whatever state it
                          is in. This is what makes giving up cost something.
  env_success / env_done  the environment stopped -- its success predicate fired, or
                          it returned done.
  max_steps               the trial hit its own step limit.
  submission_step_budget  the run was sealed before this trial was reached. Scored
                          as a failure.
  trial_error             the trial could not be set up or driven -- a scene that fails
                          to build, an env that raises. Scored as a failure, and the
                          run CONTINUES to the next trial.

Only `agent_reset` is the agent's choice. After a trial ends the agent calls `reset` to
begin the next one, so the same loop (segment, then reset on `episode_over`) is correct
in both phases.

FAILURE OF ONE TRIAL IS NOT FAILURE OF THE RUN. A trial that cannot be built or driven
is scored zero and the evaluation moves on, because the alternative -- one broken scene
wedging the trials behind it -- turns a partial result into no result at all.

Scoring is always the environment's own predicate at the moment the trial ends. A
controller or agent cannot assert success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# RoboCasa's own held-out split. Evaluation always runs here -- it is the test
# protocol, not a knob, which is why no caller may choose it.
EVAL_SPLIT = "target"

TRIAL_END_RESET = "agent_reset"
TRIAL_END_ENV_SUCCESS = "env_success"
TRIAL_END_ENV_DONE = "env_done"
TRIAL_END_MAX_STEPS = "max_steps"
TRIAL_END_SUBMISSION_BUDGET = "submission_step_budget"
TRIAL_END_ERROR = "trial_error"


@dataclass
class TrialRecord:
    scene_index: int
    trial_index: int
    success: bool
    steps: int
    ended_by: str
    # Did the agent get this trial at all? True once it drove a step, and also for a
    # trial it reset straight past -- it reached that one and chose to skip it. False
    # only for trials the run never got to, which is the case worth telling apart.
    attempted: bool = False
    error: str | None = None


@dataclass
class EvaluationRun:
    """One submission in progress. Owned by the daemon; the agent drives it."""

    task: str
    plan: tuple[tuple[int, int, int, int], ...]   # (scene_i, layout, style, seed)
    env_factory: Callable[..., Any]
    max_steps_per_trial: int | None = None

    position: int = 0
    segment_index: int = 0
    steps_used: int = 0
    trial_steps: int = 0
    results: list[TrialRecord] = field(default_factory=list)
    finished: bool = False
    # Is a trial actually in progress? `position` cannot answer this: it advances on
    # end_trial, so "trial N ended" and "trial N+1 started" look identical from it.
    # The distinction is what lets reset() know whether it has anything to forfeit.
    trial_live: bool = False
    _env: Any = None
    _env_scene: tuple[int, int] | None = None
    _obs: Any = None

    # -- trial lifecycle -----------------------------------------------------
    @property
    def total_trials(self) -> int:
        return len(self.plan)

    def current(self) -> tuple[int, int, int, int] | None:
        if self.position >= len(self.plan):
            return None
        return self.plan[self.position]

    def _scene_env(self, layout: int, style: int):
        """One env per scene, reused across that scene's trials.

        Building an env costs ~20 s, so rebuilding per trial would dominate the
        submission. Trials still differ because the trial seed is applied to the
        env's rng before each reset.
        """
        if self._env is not None and self._env_scene == (layout, style):
            return self._env
        self._close_env()
        self._env = self.env_factory(
            task=self.task, split=EVAL_SPLIT, scene=(layout, style))
        self._env_scene = (layout, style)
        return self._env

    def _close_env(self) -> None:
        closer = getattr(self._env, "close", None) if self._env is not None else None
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        self._env = None
        self._env_scene = None

    def start_trial(self) -> dict | None:
        """Set up the current trial and return its descriptor, or None if done.

        Setting up a trial means building or reusing a scene env and seeding it, both
        of which touch RoboCasa and can fail. A failure here is scored as a lost trial
        and skipped past rather than raised: the remaining trials are still worth
        running, and a caller that had to handle an exception would have to reimplement
        that decision.
        """
        from .session import seed_episode

        while True:
            entry = self.current()
            if entry is None:
                return None
            _scene, layout, style, seed = entry
            try:
                env = self._scene_env(layout, style)
                self._obs = seed_episode(env, seed)
            except Exception as exc:  # noqa: BLE001
                # The env may be half-built; drop it so the next scene starts clean.
                self._close_env()
                self.trial_steps = 0
                self.segment_index = 0
                self.trial_live = True
                self.end_trial(TRIAL_END_ERROR, error=f"{type(exc).__name__}: {exc}")
                continue
            break
        self.trial_steps = 0
        self.segment_index = 0
        self.trial_live = True
        return self.descriptor()

    def descriptor(self) -> dict | None:
        """What the agent is told about the trial it is currently in.

        SHAPED HERE, not at the wire. Every other reply carries an observation
        directly, so the daemon narrowed each one as it sent it; a descriptor CONTAINS
        one, and the two ops that return a descriptor -- `trial_info` and evaluation
        `reset` -- were each missed, handing out `drawer_obj_pos` and the rest. Doing it
        at the single point descriptors are built is what makes forgetting impossible.

        Through `privileged.agent_view`, which is the daemon's own boundary and not a
        second copy of it. Calling `agent_visible_obs` alone here was the same mistake
        in the other direction: it narrowed correctly and never widened, so an L3
        container answered both these ops with an L1-shaped observation.
        """
        from .privileged import agent_view
        from .session import episode_instruction

        entry = self.current()
        if entry is None or not self.trial_live:
            return None
        scene_i = entry[0]
        return {
            "scene_index": scene_i,
            "trial_index": len([r for r in self.results
                                if r.scene_index == scene_i]),
            "position": self.position,
            "total_trials": self.total_trials,
            "trial_steps": self.trial_steps,
            "max_steps": self.max_steps_per_trial,
            "instruction": (episode_instruction(self._env)
                            if self._env is not None else None),
            "obs": agent_view(self._obs, self._env),
        }

    def end_trial(self, ended_by: str, error: str | None = None) -> TrialRecord:
        """Score the trial where it stands and advance."""
        entry = self.current()
        assert entry is not None, "no trial in progress"
        scene_i = entry[0]
        env = self._env
        # The success predicate is the environment's, and it can raise on a half-built
        # env. An unreadable predicate is a failed trial, never a failed run.
        try:
            success = bool(env._check_success()) if env is not None else False
        except Exception:  # noqa: BLE001
            success = False
        rec = TrialRecord(
            scene_index=scene_i,
            trial_index=len([r for r in self.results if r.scene_index == scene_i]),
            success=success,
            steps=self.trial_steps,
            ended_by=ended_by,
            attempted=self.trial_steps > 0 or ended_by == TRIAL_END_RESET,
            error=error,
        )
        self.results.append(rec)
        self.position += 1
        self.trial_live = False
        return rec

    # -- budget --------------------------------------------------------------
    def steps_left_in_trial(self) -> int | None:
        if self.max_steps_per_trial is None:
            return None
        return max(0, self.max_steps_per_trial - self.trial_steps)

    def charge(self, steps: int) -> None:
        self.steps_used += steps
        self.trial_steps += steps

    # -- result --------------------------------------------------------------
    def summary(self) -> dict:
        n = len(self.results)
        successes = sum(r.success for r in self.results)
        rate = successes / n if n else 0.0
        return {
            "trials": n,
            "planned_trials": self.total_trials,
            "successes": successes,
            "success_rate": rate,
            # Trials the agent actually reached, driven or reset past. Never part of
            # the score -- a trial the agent skipped is a failure like any other -- but
            # it is how a reader tells "solved nothing" from "never got there".
            "trials_attempted": sum(r.attempted for r in self.results),
            "steps_used": self.steps_used,
            "scenes": len({r.scene_index for r in self.results}),
        }

    def finish(self) -> dict:
        """Score any unreached trials as failures and close the env.

        Unreached trials count against the agent: a run sealed part-way has not
        demonstrated the workflow works.
        """
        # Order matters. The live trial is scored first, while its env is still there
        # to be asked. Then the env goes away and the step counter is zeroed, so the
        # trials that were never reached are scored against nothing -- failures with no
        # steps, which is what they are. Scoring them after the live one without
        # closing would ask the PREVIOUS scene's env whether they succeeded.
        if self.trial_live:
            self.end_trial(TRIAL_END_SUBMISSION_BUDGET)
        self._close_env()
        self.trial_steps = 0
        while self.current() is not None:
            self.end_trial(TRIAL_END_SUBMISSION_BUDGET)
        self.finished = True
        return self.summary()

    def per_scene(self) -> dict[int, dict]:
        """Breakdown for the ledger and the verifier. Never returned to the agent:
        per-scene results would let it localise the graded episodes."""
        out: dict[int, dict] = {}
        for r in self.results:
            slot = out.setdefault(r.scene_index, {"trials": 0, "successes": 0})
            slot["trials"] += 1
            slot["successes"] += int(r.success)
        return out
