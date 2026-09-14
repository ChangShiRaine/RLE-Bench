import json
from pathlib import Path

from common import (checkpoint_tensors, fail, freeze_output, guarded_main, per_task_rates,
                    plus_expected, run_rollout, run_training, score_rows,
                    serve_and_evaluate)

PLUS_MANIFEST = Path("/tests/plus10_manifest.json")
MIN_EPISODES = 475
FAMILIES = {"language": "language", "light": "light", "table": "table", "tb": "table_background",
            "add": "added_object", "level": "layout", "base": "base"}


def family(perturb: str) -> str:
    head = perturb.split("_")[0]
    return FAMILIES.get("level" if head.startswith("level") else head.rstrip("0123456789"), head)


def per_family_rates(rows_path: Path, manifest: list[dict]) -> dict:
    """Success per perturbation family; missing episodes count as failures."""
    outcomes = {}
    for candidate in (rows_path, Path(str(rows_path) + ".inc")):
        if candidate.is_file():
            for line in candidate.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                    if type(row.get("success")) is bool:
                        outcomes[(row["bddl"], int(row["init_idx"]))] = row["success"]
                except Exception:
                    continue
    totals: dict[str, list] = {}
    for entry in manifest:
        totals.setdefault(family(entry["perturb"]), []).append(
            outcomes.get((entry["bddl"], int(entry["init_idx"])), False))
    return {name: {"success_rate": sum(v) / len(v), "episodes": len(v)}
            for name, v in sorted(totals.items())}


def score():
    wall = run_training(1800.0)
    frozen = freeze_output()
    _, parameters = checkpoint_tensors(frozen)
    manifest = json.loads(PLUS_MANIFEST.read_text())
    expected = plus_expected(manifest)

    def evaluate(socket_path, remaining):
        # 500 LIBERO-plus perturbed episodes of the ten tasks; development sees only seeded standard resets
        return run_rollout("plus", socket_path, ["--plus-manifest", str(PLUS_MANIFEST)],
                           timeout_s=min(remaining, 4200.0), plus=True, workers=6, gpus=1)

    rows = serve_and_evaluate(evaluate, hard_cap_s=4500.0)
    success, completed = score_rows(rows, expected)
    if success is None or completed < MIN_EPISODES:
        fail(f"plus evaluation incomplete: {completed}/{len(expected)} episodes")
    return success, {"plus_success_rate": success, "episodes": completed, "parameters": parameters,
                     "per_family": per_family_rates(rows, manifest),
                     "per_task": per_task_rates(rows, expected), "replay_wall_s": wall}


if __name__ == "__main__":
    guarded_main(score)
