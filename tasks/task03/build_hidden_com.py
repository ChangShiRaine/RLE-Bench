"""Stage the single-stage hidden center-of-mass task."""
from pathlib import Path
import shutil

from tabletop.budgets import INTERACTION_STEPS

HERE = Path(__file__).resolve().parent
SLUG = "05-hidden-center-of-mass"
IMAGE = "rlebench-task03-hidden-com-agent:dev"
TEMPLATE = HERE / "_template/hidden_com"
SOURCE = HERE / "tabletop/hidden_com"


def emit():
    destination = HERE / SLUG
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(TEMPLATE, destination)
    for name in ("instruction.md", "README.md"):
        path = destination / name
        path.write_text(path.read_text().replace(
            "@@INTERACTION_STEPS@@", f'{INTERACTION_STEPS["HiddenCOM"]:,}'))
    environment = destination / "environment"
    private = environment / "private/harness/tabletop/hidden_com"
    shutil.copytree(SOURCE, private, ignore=shutil.ignore_patterns("__pycache__", "client.py"))
    for name in ("budgets.py", "reward.py"):
        shutil.copy2(HERE / "tabletop" / name, private.parent / name)
    core = environment / "private/rlebench/core"
    core.mkdir(parents=True)
    repo = HERE.parents[1]
    for relative in ("__init__.py", "core/__init__.py", "core/media.py"):
        shutil.copy2(repo / "rlebench" / relative, environment / "private/rlebench" / relative)
    public = environment / "public"
    public.mkdir()
    shutil.copy2(SOURCE / "client.py", public / "client.py")
    for name in ("environment/entrypoint.sh", "tests/test.sh", "solution/solve.sh"):
        path = destination / name
        if path.exists():
            path.chmod(0o755)
    return destination


if __name__ == "__main__":
    print(emit())
