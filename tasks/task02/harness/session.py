"""Metered session: the authority on what an agent did and what it earned.

Task02 is a HANDOFF task. One agent practises across an activity group's composite tasks
and writes a controller library and a manual; a fresh agent, with no memory of that
session, is then dropped into the group's held-out task and must compose the library to
solve it. The score is how much of that task got done.

Two phases:

  DEVELOPMENT   the agent picks which training task to practise on -- `reset(task=...)`
                over the published training split -- and every env.step is charged
                against a finite interaction budget. Nothing here is scored.
  EVALUATION    one composite trial per Harbor step. The agent may act and may write new
                controllers, but it CANNOT advance to the next trial: the harness does
                that between steps, because the next trial belongs to the next agent.

Design points that matter for trust (CLAUDE.md invariant #1):

  * THE AGENT NEVER HOLDS THE ENV. Everything here runs on the daemon side of the
    socket; the agent's controller is invoked as a callback that receives an observation
    and returns an action. So there is no env object in the agent's process to step
    off-meter, and metering cannot be bypassed by tampering with in-process counters.
  * TWO SEPARATE CURRENCIES. Steps taken during evaluation are NOT charged to the
    interaction budget -- a trial has its own ceiling. Mixing them would make an agent
    that drove its graded trials thoroughly look like one that had spent its practice.
  * THE EVALUATION SPLIT IS NEVER NAMED. Trial descriptors carry the simulator's own
    instruction and no task name, and results are reported to the agent as nothing at
    all. An agent that learned the split could have developed against it.

The env factory is injected so this logic is testable without MuJoCo.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from . import config as C
from . import env as ENV
from . import evaluation as E
from . import ledger as L
from .controller import PROPRIO, Ended

EnvFactory = Callable[..., Any]


def seed_episode(env, seed: int | None):
    """Make the next `reset()` reproducible.

    robosuite sets `self.rng = np.random.default_rng(seed)` ONCE in __init__
    (environments/base.py), and every reset draws layout, style, object instances and
    placements from it. `reset()` itself takes no seed argument -- calling
    `env.reset(seed=...)` raises TypeError -- so per-episode determinism means re-seeding
    that same generator between episodes. This is the documented seed pathway applied per
    episode, not a protocol deviation.
    """
    if seed is not None and hasattr(env, "rng"):
        import numpy as np

        env.rng = np.random.default_rng(int(seed))
    return env.reset()


def episode_instruction(env) -> str | None:
    """The task instruction, which is episode-specific and comes from the simulator.

    `get_ep_meta()["lang"]` interpolates sampled state (which drawer, which fruit), so it
    is only meaningful after reset and cannot be baked into a static task description. It
    is also the ONLY thing that tells an evaluation agent what it has been asked to do.
    """
    try:
        return env.get_ep_meta().get("lang")
    except Exception:  # noqa: BLE001
        return None


class AgentFacingError(RuntimeError):
    """Base for refusals whose message is written FOR the agent and is safe to send it.

    THIS IS A SECURITY BOUNDARY, not a taxonomy. The service returns the message of
    anything under here verbatim and REDACTS every other exception, because an
    unexpected exception's text is made of private material -- above all what the
    simulator says when it cannot build a scene, which names the environment it tried:
    the held-out task. Subclassing this is a claim that the message contains nothing the
    agent may not see, and each one below is phrased with that in mind.
    """


class SessionClosed(AgentFacingError):
    """Raised when anything is attempted after the session was sealed."""


class DevelopmentClosed(AgentFacingError):
    """Raised when development interaction is attempted after end_development().

    Declaring development finished has to MEAN something, or it is just a comment.
    """


class EpisodeOver(AgentFacingError):
    """Raised when an episode (development) or trial (evaluation) is continued after it
    ended. It is gone -- in development, reset() starts the next one; in evaluation the
    harness opens the next trial for the next agent."""


class ConnectionBusy(AgentFacingError):
    """Refused to a second agent connection while one is already open.

    The session is deliberately single-threaded; two clients interleaving on it would
    hand each other's episodes back -- reset(task=X) followed by task_info() is not
    atomic across connections. One connection at a time; sequential reconnects stay
    free."""


class BudgetExhausted(AgentFacingError):
    """Raised when a request would exceed a budget. Never silently truncated: an agent
    must be able to tell 'refused' from 'ran and did nothing'."""


class UnknownTask(ValueError):
    """Raised when the agent asks to practise on something outside the training split.

    Refused loudly rather than ignored. A silent fallback to some default task would
    have the agent developing a controller it believed was for a different fixture, and
    the whole point of the training split is that the agent chooses within it.
    """


@dataclass
class Budgets:
    interaction_steps: int


class _NullRecorder:
    """Recorder-shaped no-op, so every hook below can be called unconditionally.

    __getattr__ rather than a list of stubs: a hook added to DebugRecorder later must not
    be able to crash a run that has debugging switched off.
    """

    enabled = False

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


@dataclass
class SessionState:
    steps_used: int = 0
    episodes: int = 0
    # Steps spent in the CURRENT development episode, so an over-long episode ends and
    # the agent can tell its own segment cap from the harness stopping it.
    episode_steps: int = 0
    # Which training task the current development episode is on.
    current_task: str | None = None
    # Steps charged per training task. Diagnosis only, never scored -- but it is the
    # record of how the agent chose to spend a budget across a set of tasks, which is
    # most of what a reader wants to know about a development phase.
    steps_by_task: dict[str, int] = field(default_factory=dict)
    # Completed episodes per task, and how many the environment's predicate called solved.
    # Separate from `steps_by_task` because spending is not achievement.
    episodes_by_task: dict[str, int] = field(default_factory=dict)
    successes_by_task: dict[str, int] = field(default_factory=dict)


class MeteredSession:
    def __init__(
        self,
        train_tasks: tuple[str, ...],
        budgets: Budgets,
        ledger: L.Ledger,
        env_factory: EnvFactory,
        eval_plan: tuple[tuple[str, int], ...],
        dev_split: str = C.DEV_SPLIT,
        max_episode_steps: int | None = None,
        max_steps_per_trial: int | None = None,
        env_cache_size: int = C.ENV_CACHE_SIZE,
        transcript=None,
        recorder=None,
    ):
        if budgets.interaction_steps < 0:
            raise ValueError("budgets must be non-negative")
        if not train_tasks:
            raise ValueError("the training split must contain at least one task")
        if not eval_plan:
            raise ValueError("the evaluation plan must contain at least one trial")

        self.train_tasks = tuple(train_tasks)
        self.budgets = budgets
        self.ledger = ledger
        self._env_factory = env_factory
        self._eval_plan = tuple(eval_plan)
        self.dev_split = dev_split
        self.max_episode_steps = max_episode_steps
        self.max_steps_per_trial = max_steps_per_trial
        # Human-facing record of the run. Optional, and never read by the scorer.
        self.transcript = transcript
        # Live operator-facing view, disabled unless asked for. Every call on it is a
        # no-op when off, which is why the hooks below are unconditional. Never read by
        # the scorer, and unreachable by the agent.
        self.recorder = recorder or _NullRecorder()

        self.state = SessionState()
        # Development envs, keyed by task name, most-recently-used last. BOUNDED: an env
        # costs ~20-40 s to build and is not small, and the agent may hop between all of
        # its training tasks freely. The bound is on residency, never on the hopping --
        # refusing a hop would make the training split a lie.
        self._dev_envs: "OrderedDict[str, Any]" = OrderedDict()
        self._env_cache_size = max(1, int(env_cache_size))
        self._closed = False
        self._development_ended = False
        self._episode_over = True
        # Steps continue the CURRENT episode, so the last observation has to survive
        # across calls.
        self._last_obs: dict | None = None
        self._accrued_steps = 0
        self._accrued_task: str | None = None
        # Set while an evaluation is open. Its presence changes what `reset` and `step`
        # mean, which is the whole point of the protocol.
        self._eval: E.EvaluationRun | None = None

        self.ledger.append(
            L.KIND_START,
            train_tasks=list(self.train_tasks),
            split=dev_split,
            eval_split=E.EVAL_SPLIT,
            interaction_budget=budgets.interaction_steps,
            planned_trials=len(self._eval_plan),
            max_steps_per_trial=max_steps_per_trial,
            # Which way up the agent's camera frames were. Provenance, not a knob: it
            # changes the pixels a run was scored against, so results from either side of
            # a change are not comparable and the ledger should say which side it is on.
            image_convention=ENV.image_convention(),
        )

    # -- budget accounting ---------------------------------------------------
    @property
    def steps_remaining(self) -> int:
        return max(0, self.budgets.interaction_steps - self.state.steps_used)

    def _charge_steps(self, n: int, task: str | None = None) -> None:
        """`task` overrides the attribution, for a reset paid by the task it resets TO."""
        if n > self.steps_remaining:
            raise BudgetExhausted(
                f"requested {n} steps but only {self.steps_remaining} remain of "
                f"{self.budgets.interaction_steps}"
            )
        self.state.steps_used += n
        # After the check, not before: a refused charge must not inflate the episode.
        self.state.episode_steps += int(n)
        task = task if task is not None else self.state.current_task
        if task is not None:
            self.state.steps_by_task[task] = self.state.steps_by_task.get(task, 0) + n

    def _require_open(self) -> None:
        if self._closed:
            raise SessionClosed("session is closed")

    def _live(self) -> bool:
        """Is there something to act in? A development episode, or an evaluation trial.

        The two lifecycles are tracked separately -- an evaluation does not end the
        development episode -- but the agent sees one answer, which is what lets a single
        loop drive both phases.
        """
        if self._eval is not None:
            return self._eval.trial_live
        return not self._episode_over

    def _require_live_episode(self) -> None:
        if not self._live():
            raise EpisodeOver(
                "this episode has ended; call reset() to start the next one"
            )

    def _require_development_open(self) -> None:
        if self._development_ended:
            raise DevelopmentClosed(
                "development is over (end_development was called); "
                "the harness opens the evaluation for the next phase"
            )

    # -- the training split --------------------------------------------------
    def list_tasks(self) -> list[str]:
        """The training split, in full. Published on purpose -- deciding where to spend a
        finite budget across these is part of the problem."""
        return list(self.train_tasks)

    def _require_train_task(self, task: str) -> str:
        if task not in self.train_tasks:
            raise UnknownTask(
                f"{task!r} is not in the training split. Available tasks: "
                f"{', '.join(self.train_tasks)}"
            )
        return task

    def _dev_env(self, task: str):
        """The development env for `task`, built on demand and cached.

        Least-recently-used eviction. Closing an evicted env matters: MuJoCo contexts and
        the offscreen framebuffer are not reclaimed by the garbage collector alone, and a
        run that hopped across the whole training split would otherwise accumulate one
        live simulator per task.
        """
        env = self._dev_envs.pop(task, None)
        if env is None:
            env = self._env_factory(task=task, split=self.dev_split)
        self._dev_envs[task] = env
        while len(self._dev_envs) > self._env_cache_size:
            _evicted_task, evicted = self._dev_envs.popitem(last=False)
            closer = getattr(evicted, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    pass
        return env

    def _current_env(self):
        """Whichever env the agent is acting in right now."""
        if self._eval is not None:
            return self._eval._env
        task = self.state.current_task
        return self._dev_envs.get(task) if task else None

    # -- development interaction (charged) -----------------------------------
    def reset(self, task: str | None = None, seed: int | None = None) -> dict:
        """Start a new episode.

        During DEVELOPMENT this selects a training task and resets it. `task=None` keeps
        whichever task the previous episode used, so a loop that practises one skill does
        not have to repeat itself; the first reset of a run must name one.

        COSTS ONE INTERACTION STEP, charged to the task being reset TO. A reset is a real
        draw on the simulator -- it builds or re-randomises a kitchen -- and a free one
        makes abandoning an episode cheaper than finishing it, which is the opposite of
        what a composite task should reward.

        During EVALUATION reset means GIVE UP. The trial is scored immediately, in
        whatever state it is in, and nothing else happens: the next trial belongs to the
        next Harbor step and the next agent, and this session will not open it. That is
        the difference from task01, where reset advanced to the agent's own next trial.
        """
        self._require_open()
        if self._eval is not None:
            return self._give_up_trial()
        self._require_development_open()

        if task is None:
            task = self.state.current_task
            if task is None:
                raise UnknownTask(
                    "the first reset must name a training task: reset(task=...). "
                    f"Available tasks: {', '.join(self.train_tasks)}"
                )
        self._require_train_task(task)
        # A reset over a live episode abandons it; journal it before the state moves on,
        # while `episode_steps` still counts only that episode's own stepping.
        self._close_dev_episode("abandoned_by_reset", False)
        # Then charge, attributed to the task being reset TO, and before `episode_steps`
        # is zeroed below so the reset never eats the new episode's horizon. Raises
        # BudgetExhausted on an empty budget, leaving nothing built and no state moved.
        self._charge_steps(1, task=task)
        self.state.current_task = task
        self._accrue_interaction(1)

        env = self._dev_env(task)
        obs = seed_episode(env, seed)
        self.state.episodes += 1
        self.state.episode_steps = 0
        self._episode_over = False
        self._last_obs = obs
        # Reset IS the episode boundary, so this closes the previous episode's video and
        # opens the next one.
        self.recorder.episode_started("development", self.state.episodes, obs)
        return obs

    # -- phase transitions ---------------------------------------------------
    def end_development(self) -> dict:
        """Declare development finished. The agent's own signal that it is ready.

        Deliberately INERT with respect to the simulator: it opens no evaluation, builds
        no environment and touches no scene. All it does is close the development phase
        and record that it happened, so the interaction cost at the moment the agent
        judged its harness ready is a fact in the ledger rather than a claim.

        Entering the scored phase remains the harness's decision -- it happens between
        Harbor steps. Idempotent, so calling it twice is harmless.
        """
        self._require_open()
        if self._development_ended:
            return self._development_summary()
        # First, so the episode's record lands before the dev_end that summarises it.
        self._close_dev_episode("abandoned_by_end_development", False)
        self._flush_interaction()
        self._development_ended = True
        self.ledger.append(
            L.KIND_DEV_END,
            steps_used=self.state.steps_used,
            episodes=self.state.episodes,
            tasks_practised=len(self.state.steps_by_task),
            steps_by_task=dict(self.state.steps_by_task),
            # Headline of what the per-episode records already say, so a reader need not
            # reduce the whole chain. Those records remain the authority.
            tasks_solved=sorted(self.state.successes_by_task),
            episodes_by_task=dict(self.state.episodes_by_task),
            successes_by_task=dict(self.state.successes_by_task),
        )
        return self._development_summary()

    def _development_summary(self) -> dict:
        return {
            "development_ended": True,
            "steps_used": self.state.steps_used,
            "steps_remaining": self.steps_remaining,
            "episodes": self.state.episodes,
            "tasks_practised": sorted(self.state.steps_by_task),
            "tasks_solved": sorted(self.state.successes_by_task),
            "tasks_unsolved": sorted(t for t in self.train_tasks
                                     if t not in self.state.successes_by_task),
        }

    def open_evaluation(self) -> dict:
        """Open the evaluation and set up its first trial. HARNESS ONLY.

        Called from a root collect hook after the development step's agent has finished,
        so entering the scored phase is never the agent's choice. Development is closed
        here whether or not the agent said it was ready -- a run whose agent never called
        `end_development` must still be evaluated, and the ledger already recorded the
        difference.
        """
        self._require_open()
        if self._eval is not None:
            raise RuntimeError("an evaluation is already open")
        if not self._development_ended:
            self.end_development()
        self._eval = E.EvaluationRun(
            plan=self._eval_plan,
            env_factory=self._env_factory,
            max_steps_per_trial=self.max_steps_per_trial,
            on_trial_end=self._record_trial,
        )
        # Development envs are dead weight from here: the evaluation builds its own, and
        # holding a simulator per training task alongside them is what runs a container
        # out of memory.
        self._release_dev_envs()
        first = self._eval.start_trial()
        self._log_trial_start(first)
        self.recorder.episode_started("evaluation", 0, self._eval._obs)
        return {"trial": first, "total_trials": self._eval.total_trials}

    def advance_trial(self) -> dict:
        """Score the live trial and open the next one. HARNESS ONLY.

        This is the STEP BOUNDARY, called from a root collect hook once an evaluation
        step's agent has finished. It is the one place a trial advances, and that is the
        core of task02's protocol: trial N+1 belongs to a fresh agent in step N+1, so an
        agent that could advance could consume a trial meant for a successor -- with its
        own context, which is exactly what the task is measuring the absence of.

        Idempotent at the end: advancing past the last trial finishes the evaluation.
        """
        self._require_open()
        if self._eval is None:
            return {"evaluation_done": True, "reason": "no evaluation is open"}
        if self._eval.trial_live:
            self._end_trial(E.TRIAL_END_STEP_ENDED)
        if self._eval.current() is None:
            return self._finish_evaluation()
        nxt = self._eval.start_trial()
        if nxt is None:
            return self._finish_evaluation()
        self._log_trial_start(nxt)
        self.recorder.episode_started("evaluation", self._eval.position, self._eval._obs)
        return {"trial": nxt, "evaluation_done": False,
                "trials_done": len(self._eval.results),
                "total_trials": self._eval.total_trials}

    def _release_dev_envs(self) -> None:
        while self._dev_envs:
            _task, env = self._dev_envs.popitem()
            closer = getattr(env, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    pass

    def _give_up_trial(self) -> dict:
        """The evaluation half of `reset`: score the live trial where it stands.

        Deliberately does NOT open the next trial -- see advance_trial. What comes back
        says the trial is over and how much of the evaluation remains, and nothing about
        how it scored: telling the agent its own grade would let a run tune against the
        evaluation across steps, through whatever it wrote to disk.
        """
        run = self._eval
        assert run is not None
        if run.trial_live:
            self._end_trial(E.TRIAL_END_RESET)
        return {
            "trial_over": True,
            "trials_done": len(run.results),
            "total_trials": run.total_trials,
        }

    def _log_trial_start(self, descriptor: dict | None) -> None:
        run = self._eval
        if self.transcript is None or run is None or descriptor is None:
            return
        task, seed = run.plan[descriptor["position"]]
        try:
            self.transcript.trial_start(
                trial=descriptor["position"], task=task, seed=seed,
                env=run._env, obs=run._obs)
        except Exception:  # noqa: BLE001
            pass   # the record is diagnostic; it must never break a trial

    def _record_trial(self, rec: E.TrialRecord) -> None:
        """Write one trial to the ledger. Wired as EvaluationRun's `on_trial_end`.

        ONE PER TRIAL, AS THE TRIAL ENDS. That is what lets every step's verifier compute
        the cumulative reward from the ledger as it stands: any step can abort the chain,
        and a run that scored only at the end would produce no reward rather than a
        correct partial one.

        It hangs off the callback rather than off `_end_trial` because trials also end
        inside `EvaluationRun.start_trial`, when a composite scene refuses to build --
        records the scorer needs, or it divides a short numerator by a full denominator.
        """
        self.ledger.append(L.KIND_TRIAL, **rec.as_payload())

    def _end_trial(self, ended_by: str, error: str | None = None):
        """End and score the current trial, and note it for a human reader.

        The ledger record is written by `_record_trial`, through the callback, so that
        every ending -- including a trial whose scene never built -- goes through exactly
        one path.
        """
        run = self._eval
        assert run is not None
        env, obs = run._env, run._obs
        rec = run.end_trial(ended_by, error=error)
        self.recorder.diagnosis(env, obs, ended_by=ended_by, success=rec.success,
                                steps=rec.steps, trial=rec.index)
        self.recorder.event("trial_end", ended_by=ended_by, success=rec.success,
                            steps=rec.steps, trial=rec.index)
        if self.transcript is not None:
            try:
                self.transcript.trial_end(
                    trial=rec.index, ended_by=ended_by, success=rec.success,
                    steps=rec.steps, env=env, obs=obs)
            except Exception:  # noqa: BLE001
                pass
        return rec

    def _finish_evaluation(self) -> dict:
        run = self._eval
        assert run is not None
        # Trials never reached are scored as zeros here. finish() calls end_trial for
        # each, which writes their ledger records, so the ledger always carries exactly
        # `planned_trials` trial records once an evaluation has closed.
        pending = run.total_trials - len(run.results)
        if pending > 0:
            if run.trial_live:
                self._end_trial(E.TRIAL_END_STEP_ENDED)
            run._close_env()
            run.trial_steps = 0
            while run.current() is not None:
                self._end_trial(E.TRIAL_END_STEP_ENDED)
        run.finished = True
        summary = run.summary()
        if self.transcript is not None:
            try:
                self.transcript.summary(summary)
            except Exception:  # noqa: BLE001
                pass
        self.ledger.append(
            L.KIND_RESULT,
            score=summary["score"],
            success_rate=summary["success_rate"],
            trials=summary["trials"],
            planned_trials=summary["planned_trials"],
            successes=summary["successes"],
            trials_attempted=summary["trials_attempted"],
            eval_steps=summary["steps_used"],
            per_task=run.per_task(),
        )
        self._eval = None
        self._eval_summary = summary
        return {"evaluation_done": True, **summary}

    @property
    def evaluating(self) -> bool:
        return self._eval is not None

    @property
    def phase(self) -> str:
        """Which phase the run is in. The agent must always be able to tell, because
        during evaluation `reset` scores the trajectory instead of restarting it."""
        return "evaluation" if self._eval is not None else "development"

    def trial_info(self) -> dict | None:
        """Descriptor of the trial in progress, or None outside evaluation.

        Needed because the harness opens each trial before its agent starts, so the agent
        never sees the reply that opened it and has to ask what it has been dropped into.
        """
        run = self._eval
        if run is None:
            return None
        return run.descriptor()

    # -- observation ---------------------------------------------------------
    def observe(self, spec=None) -> dict:
        """The current observation, without advancing anything. FREE.

        Perception is deliberately an explicit act rather than something that only
        arrives as a side effect of stepping. Looking costs a real robot nothing but
        time, and an agent should not have to spend budget -- or worse, take an action it
        does not want -- merely to see where it is. It matters more here than in task01:
        an evaluation agent wakes with no context at all, and looking is how it finds out
        what it is facing before committing its one-shot trial to an action.

        Free in the interaction sense: no env.step, no charge, no ledger record. It still
        costs wall clock when a re-render is asked for, and the phase clock bounds that.

        `live` says whether there is an episode to act in. Without it a dead episode is
        indistinguishable from a working one: its observation comes back thinner than it
        was asked for -- sometimes empty, sometimes colour with no depth -- and a caller
        that asked for depth meets that as a `KeyError` several frames later. Testing
        `live` is free, and it is the same answer `step` would raise `EpisodeOver` on.
        """
        self._require_open()
        run = self._eval
        if run is not None:
            env, obs = run._env, run._obs
        else:
            env, obs = self._current_env(), self._last_obs
            if env is None or obs is None:
                # Nothing to look at: no episode has been opened, or the last one ended.
                # `observe` must NOT reset to manufacture something -- a reset costs a
                # step, and one taken here would be both unmetered and invisible.
                return {"obs": {}, "instruction": None, "resolution": None,
                        "live": False}
        if obs is None or env is None:
            return {"obs": {}, "instruction": None, "resolution": None, "live": False}

        obs, resolution = ENV.apply_obs_spec(
            env, dict(obs), spec,
            native=C.OBS_RESOLUTION, ceiling=C.OBS_MAX_RESOLUTION)
        return {
            "obs": obs,
            "instruction": episode_instruction(env),
            "resolution": resolution,
            "live": self._live(),
        }

    # -- acting --------------------------------------------------------------
    def step(self, actions, spec=None) -> dict:
        """Apply actions in the current episode or trial. One unit of budget each.

        Identical in both phases -- only the currency differs: development charges the
        interaction budget, evaluation the trial's own ceiling. Actions are applied in
        order and stop at the first one that ends the episode, so the reply's `steps` is
        what was APPLIED rather than what was asked for.
        """
        self._require_open()
        if self._eval is not None:
            if not self._eval.trial_live:
                raise EpisodeOver(
                    "this trial has ended and has already been scored. The harness "
                    "opens the next trial between steps; there is nothing to drive here."
                )
            return self._step_eval(actions, spec)
        self._require_live_episode()
        self._require_development_open()
        return self._step_dev(actions, spec)

    def _step_dev(self, actions, spec) -> dict:
        """Development: charged against the interaction budget, bounded by the episode
        horizon."""
        env = self._current_env()
        obs = self._last_obs if self._last_obs is not None else env.reset()
        applied, ended = 0, None

        for action in actions:
            if (self.max_episode_steps is not None
                    and self.state.episode_steps >= self.max_episode_steps):
                ended = Ended.HORIZON
                break
            if self.steps_remaining <= 0:
                ended = Ended.BUDGET_EXHAUSTED
                break
            obs, _reward, done, _info = env.step(action)
            applied += 1
            self.recorder.frame(obs)
            self._charge_steps(1)
            self._last_obs = obs
            if env._check_success():
                ended = Ended.ENV_SUCCESS
                break
            if done:
                ended = Ended.ENV_DONE
                break

        self._accrue_interaction(applied)
        success = self._safe_success(env)
        if ended is not None:
            self._close_dev_episode(ended, success)
            self._last_obs = None
            self.recorder.diagnosis(env, obs, ended_by=ended,
                                    episode=self.state.episodes)
            self.recorder.event("episode", steps=self.state.episode_steps, ended=ended,
                                success=success, steps_used=self.state.steps_used)
        return self._reply(env, obs, applied, ended, success, spec)

    def _step_eval(self, actions, spec) -> dict:
        """Evaluation: charged to the trial, never to the interaction budget.

        An environment that raises loses this trial rather than the run -- the trials
        behind it belong to other agents.
        """
        run = self._eval
        assert run is not None
        env = run._env
        applied, ended, error = 0, None, None
        obs = run._obs

        try:
            for action in actions:
                if (run.steps_left_in_trial() or 0) - applied <= 0:
                    break
                obs, _reward, done, _info = env.step(action)
                applied += 1
                self.recorder.frame(obs)
                if env._check_success():
                    ended = Ended.ENV_SUCCESS
                    break
                if done:
                    ended = Ended.ENV_DONE
                    break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            ended = Ended.ENV_DONE

        run.charge(applied)
        run._obs = obs
        # THE STAGE SAMPLE, at the granularity partial credit is measured at -- see
        # EvaluationRun.sample_stages for why not every step.
        run.sample_stages()
        success = self._safe_success(env)

        if self.transcript is not None:
            try:
                self.transcript.step(len(run.results), run.step_index,
                                     steps=applied, ended=ended, success=success)
                run.step_index += 1
            except Exception:  # noqa: BLE001
                pass

        if error is not None:
            trial_ended = E.TRIAL_END_ERROR
        elif ended == Ended.ENV_SUCCESS:
            trial_ended = E.TRIAL_END_ENV_SUCCESS
        elif ended == Ended.ENV_DONE:
            trial_ended = E.TRIAL_END_ENV_DONE
        elif (run.steps_left_in_trial() or 0) <= 0:
            trial_ended = E.TRIAL_END_MAX_STEPS
            ended = Ended.HORIZON
        else:
            trial_ended = None
        if trial_ended is not None:
            self._end_trial(trial_ended, error=f"env: {error}" if error else None)
        return self._reply(env, obs, applied, ended, success, spec,
                           over=trial_ended is not None)

    def _reply(self, env, obs, steps, ended, success, spec, over=None) -> dict:
        """What the agent is shown. The harness keeps the raw observation -- narrowing
        what the agent sees must never narrow what the success predicate, the stage
        functions or the recorder check.

        Cameras follow the spec but size does not: re-rendering a closing frame at high
        resolution is a cost with no reader, and `observe()` is there for a deliberate
        look.
        """
        shown, _ = ENV.apply_obs_spec(
            env, obs, PROPRIO if spec is None else replace(spec, width=None, height=None),
            native=C.OBS_RESOLUTION, ceiling=C.OBS_MAX_RESOLUTION)
        return {
            "obs": shown,
            "steps": int(steps),
            "ended": ended,
            "success": bool(success),
            "episode_over": bool(ended is not None if over is None else over),
            "steps_remaining": self.steps_remaining,
        }

    @staticmethod
    def _safe_success(env) -> bool:
        """The env's own success predicate, which can itself raise on a broken env."""
        try:
            return bool(env._check_success()) if env is not None else False
        except Exception:  # noqa: BLE001
            return False

    def _accrue_interaction(self, steps: int) -> None:
        """Accrue charged steps, journalling them in batches.

        A record per step would be tens of thousands per run for a figure that is only
        ever summed and split by task. Flushed whenever the task changes, an episode or
        phase ends, or the accrual reaches LEDGER_FLUSH_STEPS, so both stay exact.
        """
        if steps <= 0:
            return
        if self.state.current_task != self._accrued_task:
            self._flush_interaction()
            self._accrued_task = self.state.current_task
        self._accrued_steps += int(steps)
        if self._accrued_steps >= C.LEDGER_FLUSH_STEPS:
            self._flush_interaction()

    def _flush_interaction(self) -> None:
        if self._accrued_steps > 0:
            self.ledger.append(L.KIND_INTERACT, steps=self._accrued_steps,
                               task=self._accrued_task)
            self._accrued_steps = 0

    def _close_dev_episode(self, ended_by: str, success: bool) -> None:
        """Journal one DEVELOPMENT episode as it ends. Trials have their own record.

        Guarded on `_episode_over`, so the three callers -- an episode-ending step, a
        `reset` that abandons a live episode, and `end_development` with one still open --
        cannot write two records for one episode.

        For the abandonment callers `success=False` holds by construction: `step` reports
        ENV_SUCCESS the moment the predicate fires and that ends the episode, so a live one
        has never satisfied it, and nothing steps in between.
        """
        if self._episode_over or self.state.current_task is None:
            return
        self._episode_over = True
        # Before the episode record, so the ledger reads in the order things happened.
        self._flush_interaction()
        task, success = self.state.current_task, bool(success)
        self.state.episodes_by_task[task] = self.state.episodes_by_task.get(task, 0) + 1
        if success:
            self.state.successes_by_task[task] = \
                self.state.successes_by_task.get(task, 0) + 1
        self.ledger.append(
            L.KIND_EPISODE,
            index=self.state.episodes,
            task=task,
            steps=self.state.episode_steps,
            ended_by=str(ended_by),
            success=success,
        )

    # -- teardown ------------------------------------------------------------
    def close(self, end_reason: str = "harness_seal") -> dict:
        """Seal the ledger. A missing seal means the run was cut short before its results
        were written, and the verifier reads that as an incomplete run.

        `end_reason` records HOW the run ended -- the phase clock expiring, the daemon
        being signalled after the last agent exited. Diagnosis only; it never reaches the
        reward, but without it every run looks alike in the ledger and an infrastructure
        failure is indistinguishable from a genuine zero.
        """
        if self._closed:
            return L.summarize(L.load_verified(self.ledger.path))
        # An evaluation that was opened but never finished must still be SCORED. The
        # harness opens it between steps, so a run whose later agents never start -- a
        # crash, a timeout -- would otherwise seal with trials planned and none recorded.
        if self._eval is not None:
            self._finish_evaluation()
        # An unflushed tail would under-report what development spent.
        self._flush_interaction()
        self._release_dev_envs()
        # record_count includes the seal itself.
        count = sum(1 for _ in L.read_records(self.ledger.path)) + 1
        self.ledger.append(
            L.KIND_SEAL,
            record_count=count,
            # Digest of the human-facing transcript. Recorded so that editing the
            # transcript -- which lives in an agent-writable artifacts dir -- is
            # detectable, even though the transcript is never scored.
            transcript_sha256=(self.transcript.digest()
                               if self.transcript is not None else None),
            steps_used=self.state.steps_used,
            end_reason=end_reason,
        )
        self._closed = True
        self.recorder.close()
        return L.summarize(L.load_verified(self.ledger.path))

    def probe(self, task: str | None = None) -> dict:
        """Build one env and learn its action contract, without charging.

        `env.action_spec` is only valid after the first reset, so the daemon does one free
        reset at startup; otherwise the agent's first `task_info()` would report
        action_dim=None. The contract is uniform across all 317 RoboCasa tasks
        (action_dim 12, horizon 1000, audited in task01), so any training task answers for
        all of them.
        """
        target = self._require_train_task(task or self.train_tasks[0])
        env = self._dev_env(target)
        obs = env.reset()
        self.state.episodes += 1
        low, _ = env.action_spec
        return {
            "action_dim": int(len(low)),
            "obs_keys": sorted(obs.keys()),
        }

    def status(self) -> dict:
        """What the agent is allowed to know about its own run.

        No score, in either phase. During evaluation each trial is driven by a different
        agent, and telling one how it did is a channel between them that the task is
        built to close -- whatever it wrote to disk would carry the grade forward.

        THE STEP BUDGET IS PHASE-LOCAL, and that is a correctness fix rather than a
        presentation choice. In development it is the interaction budget; in evaluation
        it is the OPEN TRIAL's ceiling, which is the only budget an evaluation agent can
        actually spend.

        Reporting the development figure during evaluation was the same channel this
        docstring says is closed, left open by accident: it does not move when a trial
        steps, and it carries whatever the development agent happened to leave behind. A
        development phase that spent 98% of its budget told the graded agent it had 1,007
        steps when its trial allowed 5,000, and the agent sized its whole reconnaissance
        to that number; a frugal one told a different agent 57,394. Either way it is a
        fact about someone else's phase.
        """
        run = self._eval
        if run is None:
            used, remaining = self.state.steps_used, self.steps_remaining
        else:
            used, remaining = run.trial_steps, run.steps_left_in_trial()
        return {
            "phase": self.phase,
            "steps_used": used,
            "steps_remaining": remaining,
            "episodes": self.state.episodes,
            "current_task": self.state.current_task if run is None else None,
            # DEVELOPMENT ONLY, for the same reason the step budget is phase-local: during
            # evaluation this is a fact about someone else's phase, and a channel from the
            # development agent to the graded one. It tells the develop agent nothing new
            # -- every segment result already carries the env's `success` -- only saves it
            # tallying by hand against the readiness check.
            "tasks_solved": (sorted(self.state.successes_by_task) if run is None
                             else None),
            "tasks_unsolved": (sorted(t for t in self.train_tasks
                                      if t not in self.state.successes_by_task)
                               if run is None else None),
            "trials_done": len(run.results) if run is not None else None,
            "total_trials": len(self._eval_plan),
        }
