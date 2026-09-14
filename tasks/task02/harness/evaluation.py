"""Interactive evaluation: one composite trial per Harbor step, one fresh agent each.

WHAT IS BEING MEASURED is a HANDOFF. The development agent's controllers and manual are
the deliverable, and the graded trial is driven by a different agent that has never seen
that session -- it wakes up with the files on disk, an instruction from the simulator, and
nothing else. So the trial measures not "can this agent solve it" but "was the harness
good enough that an agent who only read the manual could".

    step develop   ── practise, write the harness    ── agent A
    step evaluate  ── one trial on the held-out task ── agent B, fresh
                                                        reward = its stage score

One trial per task, one task per step. The step boundary IS the trial boundary: a root
collect hook calls `advance()` between steps (see control.py), so the agent never advances
the evaluation itself. Letting its own `reset` advance would let one agent consume a trial
meant for a later, fresher one.

TRIALS ARE SCORED BY STAGE CREDIT, not by success alone -- composite predicates are
conjunctions and a binary grade collapses every agent onto the same number. See
stages.py, which owns that logic; this module owns only when it is sampled.

A trial ends in one of these ways (the TRIAL_END_* constants below):

  agent_reset      the agent chose to reset while the trial was still running. That is
                   a GIVE UP: the trial is scored where it stands. It is the only ending
                   that is the agent's choice.
  env_success      the environment's success predicate fired.
  env_done         the environment returned done.
  max_steps        the trial hit its own step limit.
  step_ended       the Harbor step ended with the trial still live -- the agent's wall
                   clock ran out, or it simply stopped working. Scored where it stands.
  trial_error      the trial could not be set up or driven. Scored zero, and the run
                   CONTINUES: one composite task that fails to build must not wedge the
                   nine trials behind it, because a partial result beats no result.

Scoring is always read from the simulator through the harness's own code. Neither the
agent nor a controller can assert progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import config as C
from . import stages as S

# RoboCasa's own held-out split. Evaluation always runs here -- it is the test protocol,
# not a knob, which is why no caller may choose it.
EVAL_SPLIT = C.EVAL_SPLIT

TRIAL_END_RESET = "agent_reset"
TRIAL_END_ENV_SUCCESS = "env_success"
TRIAL_END_ENV_DONE = "env_done"
TRIAL_END_MAX_STEPS = "max_steps"
TRIAL_END_STEP_ENDED = "step_ended"
TRIAL_END_ERROR = "trial_error"


@dataclass
class TrialRecord:
    """One graded trial. `score` is the number that reaches the reward; everything else
    is what lets a reader tell a failure apart from a run that never happened."""

    index: int
    task: str
    success: bool
    score: float
    steps: int
    ended_by: str
    stages_at_reset: S.Stages | None = None
    stages_best: S.Stages | None = None
    # Did the agent get this trial at all? True once it drove a step, and also for a
    # trial it reset straight past -- it reached that one and chose to skip it. False
    # only for trials the run never got to, which is the case worth telling apart.
    attempted: bool = False
    error: str | None = None

    def as_payload(self) -> dict:
        """The ledger form. Flat, JSON-safe, and complete enough that the reward can be
        recomputed from the record without the environment that produced it."""
        return {
            "index": self.index,
            "task": self.task,
            "success": bool(self.success),
            "score": float(self.score),
            "steps": int(self.steps),
            "ended_by": self.ended_by,
            "attempted": bool(self.attempted),
            "stages_at_reset": [[n, v] for n, v in (self.stages_at_reset or [])],
            "stages_best": [[n, v] for n, v in (self.stages_best or [])],
            "error": self.error,
        }


@dataclass
class EvaluationRun:
    """One evaluation in progress. Owned by the daemon; the agent drives the live trial
    and the harness advances between them."""

    plan: tuple[tuple[str, int], ...]          # (composite task, trial seed)
    env_factory: Callable[..., Any]
    max_steps_per_trial: int | None = None
    # Called with every TrialRecord the moment it is created; the session writes the
    # ledger record from it.
    #
    # A CALLBACK RATHER THAN THE CALLER DOING IT, because not every trial ends where the
    # caller can see: `start_trial` scores and skips past a task whose scene will not
    # build. The ledger must carry exactly `planned_trials` trial records however trials
    # ended, or the scorer's numerator and denominator come from different runs.
    on_trial_end: Callable[[TrialRecord], None] | None = None

    position: int = 0
    step_index: int = 0
    steps_used: int = 0
    trial_steps: int = 0
    results: list[TrialRecord] = field(default_factory=list)
    finished: bool = False
    # Is a trial actually in progress? `position` cannot answer this: it advances on
    # end_trial, so "trial N ended" and "trial N+1 started" look identical from it.
    # The distinction is what lets reset() know whether it has anything to forfeit.
    trial_live: bool = False
    _env: Any = None
    _obs: Any = None
    # The stage vector as the trial arrived, and the fullest one seen since. Both are
    # needed: the first is the baseline a do-nothing agent would score, the second is
    # what the trial actually reached. See stages.score_trial.
    _stages_reset: S.Stages | None = None
    _stages_best: S.Stages | None = None

    # -- trial lifecycle -----------------------------------------------------
    @property
    def total_trials(self) -> int:
        return len(self.plan)

    def current(self) -> tuple[str, int] | None:
        if self.position >= len(self.plan):
            return None
        return self.plan[self.position]

    def current_task(self) -> str | None:
        entry = self.current()
        return entry[0] if entry else None

    def _close_env(self) -> None:
        closer = getattr(self._env, "close", None) if self._env is not None else None
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        self._env = None

    def start_trial(self) -> dict | None:
        """Set up the current trial and return its descriptor, or None if done.

        Every trial builds a NEW environment, because every trial is a different
        composite task -- there is nothing to reuse, unlike task01 where one env served
        a whole scene's trials. It costs ~20-40 s, which is why the harness does this in
        a collect hook between Harbor steps rather than on any agent's clock.

        A failure here is scored as a lost trial and skipped past rather than raised:
        the remaining trials are still worth running, and a caller that had to handle an
        exception would have to reimplement that decision.
        """
        from .session import seed_episode

        while True:
            entry = self.current()
            if entry is None:
                self._close_env()
                return None
            task, seed = entry
            try:
                self._close_env()
                self._env = self.env_factory(task=task, split=EVAL_SPLIT)
                self._obs = seed_episode(self._env, seed)
            except Exception as exc:  # noqa: BLE001
                # The env may be half-built; drop it so the next task starts clean.
                self._close_env()
                self.trial_steps = 0
                self.step_index = 0
                self.trial_live = True
                self._stages_reset = self._stages_best = None
                self.end_trial(TRIAL_END_ERROR, error=f"{type(exc).__name__}: {exc}")
                continue
            break
        self.trial_steps = 0
        self.step_index = 0
        self.trial_live = True
        # The baseline, sampled before the agent can have touched anything. Without it
        # a do-nothing trial would collect whatever the scene satisfies on arrival --
        # measured at 0.300 across this split.
        self._stages_reset = S.read_stages(task, self._env)
        self._stages_best = self._stages_reset
        return self.descriptor()

    def sample_stages(self) -> None:
        """Take a stage reading and keep it if it is the fullest one yet.

        Called at SEGMENT BOUNDARIES rather than every step. These predicates run contact
        queries and geometry tests, which are far too expensive for a per-step hot path,
        and sampling there would buy nothing for full success: the segment loop already
        tests `env._check_success()` after every step and stops the moment it fires, so a
        solved trial can never be missed. What a coarser sample can miss is a peak in
        PARTIAL progress that was reached and undone inside one segment -- which is a
        conservative error, never a generous one.
        """
        task = self.current_task()
        if task is None or not self.trial_live:
            return
        self._stages_best = S.merge_best(
            self._stages_best, S.read_stages(task, self._env))

    def descriptor(self) -> dict | None:
        """What the agent is told about the trial it is currently in.

        FILTERED HERE, not at the wire. Every other reply carries an observation
        directly, so the daemon narrowed each one as it sent it; a descriptor CONTAINS
        one, and in task01 the two ops that return a descriptor were each missed,
        handing out ground-truth object poses. Doing it at the single point descriptors
        are built is what makes forgetting impossible.

        THE TASK NAME IS NOT IN HERE. The agent is told the instruction the simulator
        produces -- which is what a real operator would say -- but never which RoboCasa
        task it is. A name would let an agent that had seen this file recognise the
        evaluation split, and the split is the one thing task02 must keep private.
        """
        from .session import episode_instruction

        if self.current() is None or not self.trial_live:
            return None
        return {
            "index": self.position,
            "position": self.position,
            "total_trials": self.total_trials,
            "trial_steps": self.trial_steps,
            "max_steps": self.max_steps_per_trial,
            "instruction": (episode_instruction(self._env)
                            if self._env is not None else None),
            "obs": C.agent_visible_obs(self._obs),
        }

    def end_trial(self, ended_by: str, error: str | None = None) -> TrialRecord:
        """Score the trial where it stands and advance.

        The environment's own predicate is asked first and outranks everything: if it
        says the task is solved, the score is 1.0 whatever the stage vector reads.
        """
        entry = self.current()
        assert entry is not None, "no trial in progress"
        task, _seed = entry
        # The success predicate is the environment's, and it can raise on a half-built
        # env. An unreadable predicate is a failed trial, never a failed run.
        try:
            success = bool(self._env._check_success()) if self._env is not None else False
        except Exception:  # noqa: BLE001
            success = False
        # One last reading, so a trial that finished between segments is not graded on a
        # stale sample.
        if self.trial_live:
            self.sample_stages()
        rec = TrialRecord(
            index=self.position,
            task=task,
            success=success,
            score=S.score_trial(self._stages_reset, self._stages_best, success=success),
            steps=self.trial_steps,
            ended_by=ended_by,
            stages_at_reset=self._stages_reset,
            stages_best=self._stages_best,
            attempted=self.trial_steps > 0 or ended_by == TRIAL_END_RESET,
            error=error,
        )
        self.results.append(rec)
        self.position += 1
        self.trial_live = False
        self._stages_reset = self._stages_best = None
        if self.on_trial_end is not None:
            self.on_trial_end(rec)
        return rec

    # -- budget --------------------------------------------------------------
    def steps_left_in_trial(self) -> int | None:
        if self.max_steps_per_trial is None:
            return None
        return max(0, self.max_steps_per_trial - self.trial_steps)

    def charge(self, steps: int) -> None:
        """Evaluation stepping has its own ceiling and never touches the development
        interaction budget. Mixing them would make an agent that drove its graded trials
        thoroughly look like one that had spent its practice."""
        self.steps_used += steps
        self.trial_steps += steps

    # -- result --------------------------------------------------------------
    def summary(self) -> dict:
        n = len(self.results)
        planned = self.total_trials
        return {
            "trials": n,
            "planned_trials": planned,
            "successes": sum(r.success for r in self.results),
            # THE REWARD. Averaged over PLANNED trials, not completed ones: a run that
            # stopped after three trials has not shown the harness works, and dividing
            # by three would score it as though it had.
            "score": (sum(r.score for r in self.results) / planned) if planned else 0.0,
            "success_rate": (sum(r.success for r in self.results) / planned
                             if planned else 0.0),
            # Trials the agent actually reached, driven or reset past. Never part of
            # the score -- a trial the agent skipped is a failure like any other -- but
            # it is how a reader tells "solved nothing" from "never got there".
            "trials_attempted": sum(r.attempted for r in self.results),
            "steps_used": self.steps_used,
        }

    def finish(self) -> dict:
        """Score any unreached trials as failures and close the env.

        Unreached trials count against the run: an evaluation that stopped part-way has
        not demonstrated the harness transfers, and crediting the remainder would reward
        never being asked.
        """
        # Order matters. The live trial is scored first, while its env is still there to
        # be asked. Then the env goes away and the step counter is zeroed, so trials that
        # were never reached are scored against nothing -- zeros with no steps, which is
        # what they are. Scoring them first would ask the PREVIOUS task's env whether
        # they succeeded.
        if self.trial_live:
            self.end_trial(TRIAL_END_STEP_ENDED)
        self._close_env()
        self.trial_steps = 0
        while self.current() is not None:
            self.end_trial(TRIAL_END_STEP_ENDED)
        self.finished = True
        return self.summary()

    def per_task(self) -> dict[str, dict]:
        """Breakdown for the ledger and the verifier. Never returned to the agent:
        naming the tasks would publish the evaluation split."""
        return {
            r.task: {"score": r.score, "success": bool(r.success),
                     "steps": r.steps, "ended_by": r.ended_by}
            for r in self.results
        }
