"""CLI entry point for the metering daemon. Runs as root inside the agent container.

Everything the agents must not see lives here or below: the evaluation split, its trial
seeds, the stage function and the ledger. Ships only in the root-only /opt/private tree
(see tasks/task02/build_assets.py).

Started by entrypoint.sh before any agent gets control, and it OUTLIVES THEM ALL: one
daemon serves the development agent and then the evaluation agent. It is the only thing
in the container that remembers the run, which is what makes the handoff measurable --
the agents share a filesystem and a daemon, and nothing else.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from . import config as C
from .service import Service
from .session import Budgets


# Budgets default from config.py and may be overridden per run; the splits may not.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/run/rlebench/toolsmith.sock")
    ap.add_argument("--ledger", default="/var/lib/rlebench/cost.jsonl")
    ap.add_argument("--transcript", default="/var/lib/rlebench/transcript.jsonl")
    # Harbor collects /logs/artifacts, so the human-facing record and a copy of the
    # ledger are exported there when the run seals.
    ap.add_argument("--artifacts-dir",
                    default=os.environ.get("RLEBENCH_ARTIFACTS",
                                           "/logs/artifacts/toolsmith"))
    # Overridable from task.toml -- see config.py for which channel each knob uses.
    ap.add_argument("--interaction-steps", type=int,
                    default=C.env_int("RLEBENCH_INTERACTION_STEPS",
                                      C.INTERACTION_STEPS))
    # The env never signals done, so this cap is what ends a trial. 0 means "the
    # configured default"; it must never mean "no cap".
    ap.add_argument("--max-steps-per-trial", type=int,
                    default=C.env_int("RLEBENCH_MAX_STEPS_PER_TRIAL", 0),
                    help="per-trial step cap; 0 = RoboCasa's default horizon")
    ap.add_argument("--env-cache", type=int,
                    default=C.env_int("RLEBENCH_ENV_CACHE", C.ENV_CACHE_SIZE),
                    help="how many development environments stay resident")

    args = ap.parse_args(argv)

    # Selected from the root-only SPLITS table by RLEBENCH_GROUP, which names the
    # group and nothing else; see config.py.
    train_tasks, eval_tasks = C.TRAIN_TASKS, C.EVAL_TASKS
    # The rule, checked rather than asserted in a comment.
    C.check_split(train=train_tasks, evaluation=eval_tasks)

    from .stages import has_stages

    ungraded = [t for t in eval_tasks if not has_stages(t)]
    if ungraded:
        raise SystemExit(
            f"no stage function for {ungraded}: those trials would collapse to binary "
            "success, which is the failure stage credit exists to avoid. Add them to "
            "tasks/task02/harness/stages.py or drop them from the evaluation split."
        )

    from .debug import from_env as debug_from_env
    from .env import make_env
    from .eval_video import Recorders, from_env as video_from_env
    from .transcript import Transcript

    # Live operator view, off unless RLEBENCH_DEBUG says otherwise, written where the
    # agent cannot traverse (see debug.py). Announced at startup: a debug run that
    # silently wrote nothing would be worse than no debug mode.
    # The evaluation-trial videos are on by default and root-only until the verifier
    # exports them (see eval_video.py).
    debug = debug_from_env(ledger_path=args.ledger)
    video = video_from_env()
    print(f"[daemon] debug={'on -> ' + str(debug.root) if debug.enabled else 'off'} "
          f"eval_video={'on -> ' + str(video.root) if video.enabled else 'off'}",
          flush=True)
    recorder = Recorders(debug, video)

    def env_factory(task: str, split: str = C.DEV_SPLIT):
        # No scene pinning: composite tasks declare EXCLUDE_LAYOUTS / EXCLUDE_STYLES and
        # a pinned pair can be illegal for one. See env.py.
        return make_env(task=task, split=split)

    service = Service(
        train_tasks=train_tasks,
        budgets=Budgets(interaction_steps=args.interaction_steps),
        ledger_path=args.ledger,
        env_factory=env_factory,
        # No salt from the environment: any variable is agent-readable, so a "secret"
        # salt would not be one. Seeds derive from the task names alone, which makes them
        # reproducible -- the right property for a benchmark.
        eval_plan=C.eval_plan(tasks=eval_tasks),
        max_episode_steps=C.MAX_EPISODE_STEPS,
        max_steps_per_trial=args.max_steps_per_trial or C.MAX_STEPS_PER_TRIAL,
        env_cache_size=args.env_cache,
        transcript=Transcript(args.transcript),
        recorder=recorder,
        artifacts_dir=args.artifacts_dir,
    )

    # A killed daemon must still seal: an unsealed ledger says the run did not finish, so
    # dying silently would throw away a legitimate session.
    def _seal_and_exit(signum, _frame):
        try:
            service.seal_if_open()
        finally:
            # os._exit, not sys.exit: SystemExit does nothing while the main thread is
            # parked in accept(), so the daemon would linger until the next connection --
            # and the seal hook waits for it to go away. Sealing is already done.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _seal_and_exit)

    # Learn the action contract before accepting connections, so the first task_info() is
    # complete. Costs one free reset (~20 s). The contract is uniform across all 317
    # RoboCasa tasks, so one training task answers for the graded one too.
    info = service.warm_up()
    print(f"[daemon] warm-up: action_dim={info['action_dim']}", flush=True)

    # What an operator reads to confirm the configuration took. The evaluation TASKS are
    # deliberately not printed: daemon stdout lands in logs someone might later hand to an
    # agent.
    print(f"[daemon] socket={args.socket} train_tasks={len(train_tasks)} "
          f"steps={args.interaction_steps} trials={len(C.eval_plan(tasks=eval_tasks))} "
          f"env_cache={args.env_cache}", flush=True)
    service.serve(args.socket, once=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
