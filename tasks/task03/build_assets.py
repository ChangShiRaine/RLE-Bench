"""Stage a public control client and the private tabletop simulator package."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
TABLETOP_SRC = REPO / "tasks/task03/tabletop"
HOST_PKG = REPO / "tasks/task03/harness"
spec = importlib.util.spec_from_file_location(
    "task01_build_assets", REPO / "tasks/task01/build_assets.py")
task01_assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task01_assets)
AGENT_MODULES = ("__init__.py", "client.py")
DAEMON_MODULES = task01_assets.DAEMON_MODULES
stage_core = task01_assets.stage_core


def stage(dest_root, modules, tabletop=False):
    if tabletop:
        task01_assets.stage(dest_root, modules)
        private = dest_root / "harness"
        shutil.copy2(private / "scoring.py", private / "_tabletop_base_scoring.py")
        shutil.copy2(TABLETOP_SRC / "scoring.py", private / "scoring.py")
        shutil.copytree(TABLETOP_SRC, dest_root / "harness/tabletop",
                        ignore=shutil.ignore_patterns("__pycache__", "hidden_com", "pocket", "client.py"))
    else:
        public = dest_root / "harness"
        public.mkdir(parents=True, exist_ok=True)
        (public / "__init__.py").write_text("")
        shutil.copy2(TABLETOP_SRC / "client.py", public / "client.py")


def check(agent_root):
    files = {p.relative_to(agent_root / "harness").as_posix()
             for p in (agent_root / "harness").rglob("*") if p.is_file()}
    return [] if files == set(AGENT_MODULES) else [f"unexpected public files: {sorted(files)}"]


def stage_host_pkg():
    shutil.rmtree(HOST_PKG, ignore_errors=True)
    temporary = HOST_PKG.parent / "_hostpkg_tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    stage(temporary, DAEMON_MODULES, tabletop=True)
    shutil.copy2(TABLETOP_SRC / "client.py", temporary / "harness/tabletop/client.py")
    shutil.copytree(TABLETOP_SRC / "pocket", temporary / "harness/tabletop/pocket",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.move(str(temporary / "harness"), HOST_PKG)
    shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-pkg", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.host_pkg:
        stage_host_pkg()
    elif args.check:
        problems = check(REPO / "tasks/task03/image/payload_agent")
        if problems:
            raise SystemExit("; ".join(problems))
        print("boundary ok: public client only")
    else:
        parser.error("use --host-pkg or --check")


if __name__ == "__main__":
    main()
