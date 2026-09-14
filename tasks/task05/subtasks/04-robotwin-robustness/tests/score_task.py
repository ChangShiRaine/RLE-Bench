import json
from pathlib import Path

from common import (checkpoint_tensors, fail, freeze_output, guarded_main, per_task_rates,
                    robotwin_expected, run_rollout, run_training, score_rows,
                    serve_and_evaluate)

EPISODES_BOOK = Path("/tests/robotwin_test_episodes.json")
PER_TASK = 15
MIN_FRACTION = 0.95


def score():
    wall = run_training(1800.0)
    frozen = freeze_output()
    _, parameters = checkpoint_tensors(frozen)
    book = json.loads(EPISODES_BOOK.read_text())
    expected = robotwin_expected(book, PER_TASK)

    def evaluate(socket_path, remaining):
        # frozen solvable seeds (one config, one instruction kind); development uses other seeds
        return run_rollout("robotwin", socket_path,
                           ["--tasks", ",".join(book["tasks"]), "--episodes-file", str(EPISODES_BOOK),
                            "--per-task", str(PER_TASK)],
                           timeout_s=min(remaining, 19800.0), workers=6, gpus=1, simulator="robotwin")
    rows = serve_and_evaluate(evaluate, hard_cap_s=20400.0)
    success, completed = score_rows(rows, expected)
    if success is None or completed < MIN_FRACTION * len(expected):
        fail(f"evaluation incomplete: {completed}/{len(expected)} episodes")
    return success, {"success_rate": success, "episodes": completed, "parameters": parameters,
                     "per_task": per_task_rates(rows, expected), "replay_wall_s": wall}


if __name__ == "__main__":
    guarded_main(score)
