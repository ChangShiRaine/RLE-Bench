"""Stage the pocket-cube task from repository sources."""
from pathlib import Path
import shutil

import build_assets
from tabletop.budgets import INTERACTION_STEPS

HERE = Path(__file__).resolve().parent
SLUG = "04-rubik-cube"
IMAGE = "rlebench-task03-pocket-agent:dev"


def emit():
    destination = HERE / SLUG
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(HERE / "_template/pocket", destination)
    environment = destination / "environment"
    private = environment / "private"
    build_assets.task01_assets.stage(private, build_assets.DAEMON_MODULES)
    build_assets.stage_core(private)
    shutil.copytree(HERE / "tabletop/pocket", private / "harness/tabletop/pocket",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("budgets.py", "reward.py"):
        shutil.copy2(HERE / "tabletop" / name, private / "harness/tabletop" / name)
    shutil.copy2(private / "harness/scoring.py", private / "harness/_pocket_base_scoring.py")
    shutil.copy2(HERE / "tabletop/pocket/experiment_scoring.py", private / "harness/scoring.py")
    public = environment / "public"
    client = public / "opt/rlebench/harness"
    client.mkdir(parents=True)
    for name in build_assets.task01_assets.AGENT_MODULES:
        shutil.copy2(build_assets.task01_assets.SRC / name, client / name)
    for name in ("task.toml", "environment/Dockerfile"):
        path = destination / name
        path.write_text(path.read_text().replace("@@INTERACTION_STEPS@@", str(INTERACTION_STEPS["PocketCube"])))
    (destination / "tests/test.sh").chmod(0o755)
    return destination


if __name__ == "__main__":
    print(emit())
