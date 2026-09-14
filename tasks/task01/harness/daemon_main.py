"""CLI entry point for the metering daemon. Runs as root inside the agent container.

Everything the agent must not see lives here or below: the evaluation plan, its trial
seeds, and the ledger. The module ships only in the root-only /opt/private tree (see
tasks/task01/build_assets.py).

Started by tasks/task01/_template/image/entrypoint.sh before the agent gets control.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from . import config as C
from .service import Service
from .session import Budgets


# All numbers default from config.py, the single place they live. The environment
# variables below let a deployment override a budget for one run without editing it.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--socket", default="/run/rlebench/speedrun.sock")
    ap.add_argument("--ledger", default="/var/lib/rlebench/cost.jsonl")
    ap.add_argument("--transcript", default="/var/lib/rlebench/transcript.jsonl")
    # Harbor collects /logs/artifacts, so the human-facing record and a copy of the
    # ledger are exported there when the run seals.
    ap.add_argument("--artifacts-dir",
                    default=os.environ.get("RLEBENCH_ARTIFACTS",
                                           "/logs/artifacts/speedrun"))
    # Every default below is overridable from task.toml -- see config.env_int for which
    # env channel each knob travels on and who can read it.
    ap.add_argument("--interaction-steps", type=int,
                    default=C.env_int("RLEBENCH_INTERACTION_STEPS",
                                      C.INTERACTION_STEPS))
    ap.add_argument("--submissions", type=int,
                    default=C.env_int("RLEBENCH_SUBMISSIONS", C.SUBMISSIONS))
    # The environment never signals done (create_env sets ignore_done=True), so this
    # cap is what actually ends a trial at the horizon. 0 means "use the configured
    # default"; it must never mean "no cap".
    ap.add_argument("--max-episode-steps", type=int,
                    default=C.env_int("RLEBENCH_MAX_STEPS_PER_TRIAL",
                                      C.env_int("RLEBENCH_MAX_EPISODE_STEPS", 0)),
                    help="per-trial step cap; 0 = RoboCasa's default horizon")
    # NOT from the environment, and there is no env fallback on purpose. See
    # config.eval_overrides: an agent reads its own /proc/self/environ, and `docker exec`
    # hands it whatever [environment.env] set regardless of what entrypoint.sh unset.
    ap.add_argument("--eval-plan", default="",
                    help='evaluation plan, e.g. "1x2,2x2,3x2"; empty = the '
                         'root-only override file, then the default')
    args = ap.parse_args(argv)

    # Read once, here, so a malformed file fails at startup where the message is visible
    # rather than at the first graded trial.
    overrides = C.eval_overrides()
    plan_spec = args.eval_plan or overrides.get("plan", "")
    eval_plan = C.parse_eval_plan(plan_spec) if plan_spec else C.EVAL_PLAN
    # Makes the trial seeds underivable. Empty is the normal case: the seeds are then
    # derived from the task name alone, which is reproducible -- the right property for a
    # benchmark -- and still unreachable, since the agent never sees this module.
    salt = overrides.get("salt") or None

    if not C.is_scoreable(args.task):
        raise SystemExit(
            f"{args.task} is excluded: its success predicate is already true at "
            "reset, so a do-nothing controller would score 100%"
        )

    from .debug import from_env as debug_from_env
    from .env import make_env
    from .eval_video import Recorders, from_env as video_from_env
    from .transcript import Transcript

    # Live operator view. Off unless RLEBENCH_DEBUG says otherwise, and written to a
    # tree the agent cannot traverse (see debug.py). Say so at startup: a debug run that
    # silently wrote nothing would be worse than no debug mode at all.
    # The evaluation-trial videos are on by default and root-only until the verifier
    # exports them (see eval_video.py).
    debug = debug_from_env(ledger_path=args.ledger)
    video = video_from_env()
    print(f"[daemon] debug={'on -> ' + str(debug.root) if debug.enabled else 'off'} "
          f"eval_video={'on -> ' + str(video.root) if video.enabled else 'off'}",
          flush=True)
    recorder = Recorders(debug, video)

    def env_factory(task: str, split: str = "pretrain", scene=None):
        # `scene` pins one (layout, style) pair for an evaluation trial group; it is
        # None for the development env, which uses the full split.
        return make_env(task=task, split=split, scene=scene)

    service = Service(
        task=args.task,
        budgets=Budgets(
            interaction_steps=args.interaction_steps,
            submissions=args.submissions,
        ),
        ledger_path=args.ledger,
        env_factory=env_factory,
        max_episode_steps=args.max_episode_steps or C.MAX_STEPS_PER_TRIAL,
        transcript=Transcript(args.transcript),
        recorder=recorder,
        artifacts_dir=args.artifacts_dir,
        # The (scene, trials) schedule. Plan and salt both came from the root-only file
        # above, never from a variable -- see config.eval_overrides for why an
        # environment channel cannot hold either of them.
        eval_plan_fn=lambda t: C.eval_trials(t, plan=eval_plan, salt=salt),
    )

    # A killed daemon must still seal the ledger: an unsealed ledger is read as a
    # failed run, so dying silently would throw away a legitimate session.
    def _seal_and_exit(signum, _frame):
        try:
            service.seal_if_open()
        finally:
            # os._exit, not sys.exit: sys.exit raises SystemExit in the main thread,
            # which does nothing while that thread is parked in accept(). The daemon
            # would then linger until the next connection -- and the seal hook that
            # sent this signal waits for the process to go away before letting Harbor
            # collect. Sealing and exporting are already done, so there is nothing
            # left to unwind.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _seal_and_exit)

    # Learn the action contract before accepting connections, so the agent's first
    # task_info() is complete. Costs one free reset (~20 s on first construction).
    info = service.warm_up()
    print(f"[daemon] warm-up: action_dim={info['action_dim']} "
          f"instruction={info.get('instruction')!r}", flush=True)

    # The banner is what an operator reads to confirm the run's configuration took, so
    # it reports THIS run's plan rather than the default one.
    print(f"[daemon] task={args.task} socket={args.socket} "
          f"steps={args.interaction_steps} submissions={args.submissions} "
          f"trials={C.plan_trial_count(eval_plan)}",
          flush=True)
    service.serve(args.socket, once=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
