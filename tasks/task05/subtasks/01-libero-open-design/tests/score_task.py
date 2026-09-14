from common import (checkpoint_tensors, fail, freeze_output, guarded_main, per_task_rates,
                    run_rollout, run_training, score_rows, serve_and_evaluate,
                    standard_expected)

SUITE, TASKS, EPISODES, MIN_EPISODES = "libero_10", range(10), 50, 475


def score():
    wall = run_training(1800.0)
    frozen = freeze_output()
    _, parameters = checkpoint_tensors(frozen)
    expected = standard_expected(SUITE, TASKS, EPISODES)

    def evaluate(socket_path, remaining):
        # the official LIBERO initial states, 50 per task; development uses seeded resets
        return run_rollout("standard", socket_path,
                           ["--suite", SUITE, "--tasks", "0-9", "--init", "files",
                            "--episodes", str(EPISODES)],
                           timeout_s=min(remaining, 3600.0), workers=6, gpus=1)

    rows = serve_and_evaluate(evaluate, hard_cap_s=3900.0)
    success, completed = score_rows(rows, expected)
    if success is None or completed < MIN_EPISODES:
        fail(f"evaluation incomplete: {completed}/{len(expected)} episodes")
    return success, {"success_rate": success, "episodes": completed, "parameters": parameters,
                     "per_task": per_task_rates(rows, expected), "replay_wall_s": wall}


if __name__ == "__main__":
    guarded_main(score)
