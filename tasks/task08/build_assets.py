"""Assemble the task08 from-scratch design content from the repo.

The agent workspace contains only public scene data and stock component
assets. Simulation metrics, scenarios, base_design, config, oracle data, and all
other evaluation code remain verifier-only.

Run after any change to the task08 harness (assets or code):
    python tasks/task08/build_assets.py

Produces (GITIGNORED, not committed -- run `make task-assets` before
`harbor run`, and after any change to tasks/task08/harness/):
  environment/assets/    public shelf scene/spec + stock components, with no
                         starter chassis and no Python/evaluation package
  tests/harness/       full package for the verifier image
  tests/models/          canonical meshes + stock components (no golden model:
                         task08 is from-scratch)
  solution/payload/      reference robot.xml + controller.py
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from rlebench import taskgen  # noqa: E402

HARNESS_SRC = os.path.join(REPO, "tasks", "task08", "harness")
ASSETS = os.path.join(HARNESS_SRC, "assets")
FRANKA = os.path.join(REPO, "assets", "robots", "franka_emika_panda")
UR5E = os.path.join(REPO, "assets", "robots", "universal_robots_ur5e")
XARM7 = os.path.join(REPO, "assets", "robots", "ufactory_xarm7")
MOBILE_COMPONENTS = os.path.join(ASSETS, "mobile_manipulator_components")
PUBLIC_SCENE = os.path.join(ASSETS, "base_design", "scene.xml")
SHELF_SPEC = os.path.join(ASSETS, "base_design", "shelf_spec.json")
REFERENCE_ROBOT = os.path.join(HERE, "reference", "robot.xml")

# harness/assets/ never ships wholesale to either image: both builds copy only
# named code subpackages out of the harness, and asset files are copied
# explicitly into the destination trees below.
AGENT_MESHDIR = "assets/franka_emika_panda/assets"


_clean_copytree = taskgen.clean_copytree


def _rewrite_meshdir(src_xml: str, dst_xml: str, meshdir: str) -> None:
    tree = ET.parse(src_xml)
    tree.getroot().find("compiler").set("meshdir", meshdir)
    os.makedirs(os.path.dirname(dst_xml), exist_ok=True)
    tree.write(dst_xml)


def build_environment_assets() -> None:
    env_assets = os.path.join(HERE, "environment", "assets")
    if os.path.exists(env_assets):
        shutil.rmtree(env_assets)
    os.makedirs(env_assets)

    # The public shelf scene plus stock components; the agent creates robot.xml.
    shutil.copy(PUBLIC_SCENE, env_assets)
    shutil.copy(SHELF_SPEC, env_assets)
    _clean_copytree(FRANKA, os.path.join(env_assets, "assets", "franka_emika_panda"))
    _clean_copytree(UR5E, os.path.join(
        env_assets, "assets", "universal_robots_ur5e"))
    _clean_copytree(XARM7, os.path.join(
        env_assets, "assets", "ufactory_xarm7"))
    _clean_copytree(
        MOBILE_COMPONENTS,
        os.path.join(env_assets, "assets", "mobile_manipulator_components"))

    # Evaluation implementations are verifier-only. Fail closed if a future
    # build change accidentally introduces executable Python into /workspace.
    shipped_python = []
    for root, _dirs, files in os.walk(env_assets):
        shipped_python.extend(
            os.path.relpath(os.path.join(root, name), env_assets)
            for name in files if name.endswith(".py"))
    if shipped_python:
        raise SystemExit(
            "INVARIANT VIOLATION: Python shipped to agent workspace: "
            + ", ".join(sorted(shipped_python)))
    print(f"environment/assets built ({_du(env_assets)} MB)")


def build_verifier_context() -> None:
    troot = os.path.join(HERE, "tests")
    pkg = os.path.join(troot, "harness")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg)
    for fn in ("__init__.py", "config.py", "thresholds.py"):
        shutil.copy(os.path.join(HARNESS_SRC, fn), pkg)
    for sub in ("metrics", "sim", "base_design"):
        _clean_copytree(os.path.join(HARNESS_SRC, sub),
                        os.path.join(pkg, sub))
    models = os.path.join(troot, "models")
    if os.path.exists(models):
        shutil.rmtree(models)
    # Canonical arm and component assets only: the golden robot stays
    # dev-side, since nothing in the verifier scores against it.
    _clean_copytree(FRANKA, os.path.join(models, "assets", "franka_emika_panda"))
    _clean_copytree(UR5E, os.path.join(
        models, "assets", "universal_robots_ur5e"))
    _clean_copytree(XARM7, os.path.join(
        models, "assets", "ufactory_xarm7"))
    _clean_copytree(
        MOBILE_COMPONENTS,
        os.path.join(models, "assets", "mobile_manipulator_components"))
    print(f"tests/ verifier context built ({_du(troot)} MB)")


def solution_controller_source() -> str:
    # Bundle only the Oracle's numerical helpers into its single-file policy.
    # No harness package or private configuration is imported by the worker.
    helpers = ast.parse(open(os.path.join(HARNESS_SRC, "base_design", "golden_controller.py")).read())
    helpers.body = [node for node in helpers.body
                    if not (isinstance(node, ast.ImportFrom) and node.level)]
    mecanum = ast.parse(open(os.path.join(HARNESS_SRC, "sim", "mecanum.py")).read())
    functions = [node for node in mecanum.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("geometry_from_model", "commanded_wheel_speeds",
                                   "make_base_controller")]
    variants = ast.parse(open(os.path.join(HARNESS_SRC, "sim", "arm_variants.py")).read())
    seeds = {}
    for node in ast.walk(variants):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ArmSpec":
            values = {kw.arg: kw.value for kw in node.keywords}
            seeds[ast.literal_eval(values["name"])] = (
                ast.literal_eval(values["extended_template"]),
                *ast.literal_eval(values["ik_seeds"]))
    return (ast.unparse(helpers) + "\nWHEEL_ORDER = ('FL', 'FR', 'RL', 'RR')\n"
              + "ORACLE_IK_SEEDS = " + repr(seeds) + "\n"
              + "\n".join(ast.unparse(node) for node in functions) + "\n"
              + open(os.path.join(HERE, "solution", "controller.py")).read())


def build_solution_payload() -> None:
    payload = os.path.join(HERE, "solution", "payload")
    if os.path.exists(payload):
        shutil.rmtree(payload)
    os.makedirs(payload)

    # Reference geometry and the standalone shelf controller.
    _rewrite_meshdir(REFERENCE_ROBOT,
                     os.path.join(payload, "robot.xml"), AGENT_MESHDIR)
    with open(os.path.join(payload, "controller.py"), "w") as f:
        f.write(solution_controller_source())
    print("solution/payload built")


def _du(path: str) -> int:
    total = 0
    for root, _d, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total // (1024 * 1024)


def ensure_reference_robot() -> None:
    """Build the profile reference if it is not there yet.

    reference/robot.xml is derived from the golden robot and gitignored, so a
    fresh clone does not have it -- and everything below reads it. conftest
    calls this module directly when a task tree is missing, so the build has
    to be self-sufficient rather than relying on the Makefile ordering.
    """
    if os.path.exists(REFERENCE_ROBOT):
        return
    script = os.path.join(HERE, "dev", "build_profile_reference.py")
    print(f"reference/robot.xml missing; running {os.path.relpath(script, REPO)}")
    subprocess.run([sys.executable, script], cwd=REPO, check=True)


if __name__ == "__main__":
    ensure_reference_robot()
    build_environment_assets()
    build_verifier_context()
    build_solution_payload()
