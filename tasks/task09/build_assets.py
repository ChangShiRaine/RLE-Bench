"""Stage the three task09 GELLO payloads for one Harbor task."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from rlebench import taskgen  # noqa: E402

from harness.variants import VARIANTS  # noqa: E402

HARNESS_SRC = os.path.join(REPO, "tasks", "task09", "harness")
CODESIGN_MODELS = os.path.join(HARNESS_SRC, "assets", "gello_codesign")
VARIANT_NAMES = ("franka", "ur5e", "xarm7")

AGENT_FILES = ("__init__.py", "torques.py", "probe_protocol.py")
AGENT_RENAMES = {
    "balance_codesign_public.py": "balance.py",
    "scenarios_public.py": "scenarios.py",
}
VERIFIER_EXCLUDE = ("*_public.py", "assets")

FORBIDDEN_IN_AGENT = (
    "oracle", "golden", "HIDDEN_SEEDS", "lead_sref", "checkpoint",
    "codesign_battery", "local_provider", "qmat_hold", "qmat_path",
    "qmat_grid", "poke_test", "fixed_hold_configs", "hold_configs",
    "teleop_path", "CodesignEnvelope", "device_buildable",
    "check_validity", "density_report", "link_mass_tol", "payload_max",
    "probe_noise_std", "DENSITY_PRINTED", "DENSITY_ANY",
)


def _public_spec(name: str) -> str:
    variant = VARIANTS[name]
    geometry = json.load(open(os.path.join(HARNESS_SRC, "lead_geometry.json")))[name]
    return f'''\
"""Public constants for the {variant.follower_name} GELLO co-design."""
VARIANT_NAME = {variant.name!r}
FOLLOWER_NAME = {variant.follower_name!r}
FOLLOWER_CHAIN = {variant.follower_chain!r}
FOLLOWER_EE = {variant.follower_ee!r}
N_JOINTS = {variant.n_joints}
LEAD_HOME = {variant.home!r}
WORKSPACE_HALFWIDTH = {variant.workspace_halfwidth!r}
SERVO_TAU_NM = 0.35
SERVO_KP = 8.0
SERVO_KD = 0.4
JOINT_DAMPING = 0.01
JOINT_FRICTIONLOSS = 0.03
JOINT_ARMATURE = 0.002
SERVO_MASS_KG = 0.018
LEAD_CHAIN = {geometry['chain']!r}
LEAD_EE = {geometry['ee']!r}
LEAD_BASE_QUAT = {geometry['base_quat']!r}
STOCK_BODY_MASS = {geometry['stock_mass']!r}
MASS_BUDGET_KG = 2.30
TIMESTEP = 0.002
GRAVITY = (0.0, 0.0, -9.81)
JOINT_NAMES = tuple(f"lead_joint{{i}}" for i in range(1, N_JOINTS + 1))
EE_SITE = "lead_ee_site"
'''


def _assert_public_boundary(root: str) -> None:
    for directory, _dirs, files in os.walk(root):
        for filename in files:
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root)
            text = open(path, "rb").read().decode("utf-8", errors="replace")
            for token in FORBIDDEN_IN_AGENT:
                if token in relative or token in text:
                    raise SystemExit(
                        f"INVARIANT VIOLATION: {token!r} found in {relative}")


def build_environment_assets() -> None:
    root = os.path.join(HERE, "environment", "assets")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    for name in VARIANT_NAMES:
        source = os.path.join(CODESIGN_MODELS, name)
        workdir = os.path.join(root, name)
        os.makedirs(workdir)
        shutil.copytree(os.path.join(source, "meshes"), os.path.join(workdir, "meshes"))
        shutil.copy(os.path.join(source, "lead_bare.xml"),
                    os.path.join(workdir, "lead.xml"))
        scene = open(os.path.join(source, "scene.xml"), encoding="utf-8").read()
        scene = scene.replace("lead_bare.xml", "lead.xml")
        with open(os.path.join(workdir, "scene.xml"), "w", encoding="utf-8") as stream:
            stream.write(scene)

        package = os.path.join(workdir, "harness")
        os.makedirs(package)
        for filename in AGENT_FILES:
            shutil.copy(os.path.join(HARNESS_SRC, filename),
                        os.path.join(package, filename))
        for source_name, destination_name in AGENT_RENAMES.items():
            shutil.copy(os.path.join(HARNESS_SRC, source_name),
                        os.path.join(package, destination_name))
        with open(os.path.join(package, "spec.py"), "w",
                  encoding="utf-8") as stream:
            stream.write(_public_spec(name))
        _assert_public_boundary(workdir)
        print(f"{name} workspace built ({_du(workdir)} KB)")


def build_verifier_context() -> None:
    root = os.path.join(HERE, "tests")
    package = os.path.join(root, "harness")
    if os.path.exists(package):
        shutil.rmtree(package)
    taskgen.clean_copytree(HARNESS_SRC, package, VERIFIER_EXCLUDE)

    models = os.path.join(root, "models", "gello_codesign")
    if os.path.exists(models):
        shutil.rmtree(models)
    for name in VARIANT_NAMES:
        target = os.path.join(models, name)
        os.makedirs(target, exist_ok=True)
        shutil.copy(os.path.join(CODESIGN_MODELS, name, "lead_sref.xml"), target)
        shutil.copytree(os.path.join(CODESIGN_MODELS, name, "meshes"), os.path.join(target, "meshes"))
    print(f"verifier source built ({_du(root)} KB)")


def build_solution_payload() -> None:
    from harness.codesign_oracle import build_reference_submission

    root = os.path.join(HERE, "solution", "payload")
    if os.path.exists(root):
        shutil.rmtree(root)
    for name in VARIANT_NAMES:
        build_reference_submission(os.path.join(root, name), variant=name)
        print(f"{name} Oracle payload built")


def _du(path: str) -> int:
    total = 0
    for directory, _dirs, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(directory, name)) for name in files)
    return total // 1024


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "dev.generate_variant_assets"],
        cwd=REPO, check=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1",
             "PYTHONPATH": os.path.join(REPO, "tasks", "task09")})
    build_environment_assets()
    build_verifier_context()
    build_solution_payload()


if __name__ == "__main__":
    main()
