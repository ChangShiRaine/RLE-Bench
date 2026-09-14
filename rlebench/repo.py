"""Repo discovery and the task registry.

A task family is a directory under tasks/ carrying a manifest.toml. Harbor
targets resolve to (path, instance) pairs: `-p path` plus an optional
`-i instance` for dataset-style families (task01 levels, task02 groups).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"


def robocasa_asset_dir() -> Path:
    """Where `make sim-robocasa` puts the dataset (same derivation as robocasa.sh)."""
    vendor = Path(os.environ.get("ROBOCASA_VENDOR") or REPO / "third_party")
    return vendor / "robocasa" / "robocasa" / "models" / "assets"


def task05_data_dirs() -> dict[str, Path]:
    """Where `make task05-data` puts what task05 mounts, keyed by the variable that
    points a run somewhere else."""
    root = REPO / "third_party" / "task05"
    return {
        "NANOVLA_SHARDS_L10": root / "shards_l10_128",
        "NANOVLA_LIBERO_PLUS_ASSETS": root / "libero-plus-assets",
        "NANOVLA_ROBOTWIN_SHARDS": root / "shards_rt15_128",
    }


@dataclass(frozen=True)
class Target:
    """One harbor invocation target."""

    name: str  # display name, e.g. task01/L2/01-open-fridge
    path: str  # harbor -p, repo-relative
    instance: str | None = None  # harbor -i, for dataset paths


@dataclass
class Family:
    name: str
    dir: Path
    manifest: dict = field(default_factory=dict)

    @property
    def prepare_commands(self) -> list[list[str]]:
        return [list(c) for c in self.manifest.get("prepare", {}).get("commands", [])]

    @property
    def prepare_env_notes(self) -> list[str]:
        return list(self.manifest.get("prepare", {}).get("env", []))

    @property
    def verify_images(self) -> list[str]:
        return list(self.manifest.get("verify", {}).get("images", []))

    @property
    def verify_paths(self) -> list[str]:
        return list(self.manifest.get("verify", {}).get("paths", []))

    @property
    def run_cfg(self) -> dict:
        return self.manifest.get("run", {})

    def targets(self) -> list[Target]:
        """Every runnable harbor target in this family, from the filesystem."""
        kind = self.manifest.get("task", {}).get("layout", "single")
        if kind == "single":
            return [Target(self.name, f"tasks/{self.name}")]
        if kind == "subtasks":  # task04, task06: one task dir per variant
            out = []
            for d in sorted(self.dir.iterdir()):
                if (d / "task.toml").exists():
                    out.append(Target(f"{self.name}/{d.name}", f"tasks/{self.name}/{d.name}"))
            return out
        if kind == "dataset":  # task02: -p family dir, -i per group
            return [
                Target(f"{self.name}/{d.name}", f"tasks/{self.name}", d.name)
                for d in sorted(self.dir.glob("[0-9][0-9]-*"))
                if (d / "task.toml").exists()
            ]
        if kind == "leveled-dataset":  # task01: -p family/level, -i per subtask
            out = []
            for level in sorted(self.dir.glob("L[0-9]")):
                for d in sorted(level.iterdir()):
                    if (d / "task.toml").exists():
                        out.append(
                            Target(
                                f"{self.name}/{level.name}/{d.name}",
                                f"tasks/{self.name}/{level.name}",
                                d.name,
                            )
                        )
            return out
        raise ValueError(f"{self.name}: unknown layout {kind!r}")


def families() -> dict[str, Family]:
    if not TASKS.is_dir():
        # A non-editable install lands in site-packages, away from tasks/.
        raise SystemExit(
            f"error: no tasks/ next to the rlebench package ({REPO}); "
            "install editably from the checkout: uv pip install -e . (make install)"
        )
    out = {}
    for d in sorted(TASKS.iterdir()):
        mf = d / "manifest.toml"
        if mf.exists():
            out[d.name] = Family(d.name, d, tomllib.loads(mf.read_text()))
    return out


def resolve(name: str) -> list[Target]:
    """A target argument: every target whose name is `name` or lies under it.

    task08 | task01 (15 cells) | task01/L1 (one level) | task01/L1/01-open-fridge
    | task02/06-setting-the-table | task03/01-tower-max-height | task06/rgb-only. A unique
    subtask (rgb-only, 01-tower-max-height) also resolves by its bare name.
    """
    fams = families()
    name = name.rstrip("/")
    fam = fams.get(name.split("/")[0])
    if fam is not None:
        matches = [t for t in fam.targets() if t.name == name or t.name.startswith(name + "/")]
        if matches:
            return matches
        known = ", ".join(t.name for t in fam.targets())
        raise SystemExit(f"error: no target {name!r}; {fam.name} has: {known}")
    bare = [
        target
        for family in fams.values()
        for target in family.targets()
        if target.name.rsplit("/", 1)[-1] == name
    ]
    if len(bare) == 1:
        return bare
    if len(bare) > 1:
        choices = ", ".join(target.name for target in bare)
        raise SystemExit(f"error: ambiguous subtask {name!r}; choose: {choices}")
    raise SystemExit(f"error: unknown task {name!r} (families: {', '.join(fams)})")


def harbor_bin() -> str:
    venv = REPO / ".venv" / "bin" / "harbor"
    if venv.exists():
        return str(venv)
    from shutil import which

    found = which("harbor")
    if not found:
        raise SystemExit("error: harbor not found (.venv/bin/harbor or PATH); run `make install`")
    return found


def read_task_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())
