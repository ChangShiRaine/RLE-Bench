"""Environment health checks: everything a run needs, verdict per item."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from .repo import REPO, robocasa_asset_dir, task05_data_dirs


def _harbor_pin() -> str | None:
    reqs = REPO / "requirements-dev.txt"
    if reqs.exists():
        m = re.search(r"^harbor==(\S+)", reqs.read_text(), re.M)
        if m:
            return m.group(1)
    return None


def checks() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, ok, detail))

    add("uv", shutil.which("uv") is not None, "install: https://docs.astral.sh/uv/")
    venv = REPO / ".venv" / "bin" / "python"
    add(".venv", venv.exists(), "make install")

    docker = shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )
    add("docker daemon", docker, "start docker / join the docker group")

    pin = _harbor_pin()
    actual = None
    if venv.exists():
        proc = subprocess.run(
            [str(venv), "-c", "from importlib.metadata import version; print(version('harbor'))"],
            capture_output=True, text=True,
        )
        actual = proc.stdout.strip() if proc.returncode == 0 else None
    add(
        f"harbor=={pin}", actual == pin and pin is not None,
        f"found {actual or 'none'}; uv pip install --python .venv/bin/python -r requirements-dev.txt",
    )

    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True,
    ) if shutil.which("nvidia-smi") else None
    n_gpu = len(gpus.stdout.split()) if gpus and gpus.returncode == 0 else 0
    add("gpu", n_gpu > 0, f"{n_gpu} visible (task01/02/04/06-gpu need one)")

    asset_dir = os.environ.get("ROBOCASA_ASSET_DIR") or str(robocasa_asset_dir())
    add(
        "RoboCasa dataset", os.path.isfile(os.path.join(asset_dir, "ROBOCASA_ASSET_VERSION")),
        f"{asset_dir} (make sim-robocasa, or export ROBOCASA_ASSET_DIR)",
    )
    add("third_party/", (REPO / "third_party").is_dir(), "make sim-robocasa (task01/02/03)")
    add(
        "HF token", bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "needed only for task01 L2/L3 model fetch",
    )
    for venv_name, use in ((".venv-motiontrack", "task04 suites"), (".venv-robocasa", "task01/02 sim suites")):
        add(venv_name, (REPO / venv_name / "bin" / "python").exists(), use)
    # task05 mounts its demonstration shards and the LIBERO-plus assets by exact path,
    # so a missing directory is a broken run, not a slow one. Unset, a run mounts what
    # `make task05-data` downloaded; the encoder bundles need nothing, `make sim-libero`
    # downloads them at their pins.
    for variable, default in task05_data_dirs().items():
        value = os.environ.get(variable) or str(default)
        present = os.path.isdir(value)
        add(variable, present, value if present else f"{value} (make task05-data, or export {variable})")
    return out


def report() -> int:
    rows = checks()
    hard_failures = 0
    for name, ok, detail in rows:
        mark = "ok " if ok else "MISS"
        print(f"  [{mark}] {name:20s} {detail}")
        if not ok and name in ("uv", ".venv", "docker daemon") or (not ok and name.startswith("harbor==")):
            hard_failures += 1
    return 1 if hard_failures else 0
