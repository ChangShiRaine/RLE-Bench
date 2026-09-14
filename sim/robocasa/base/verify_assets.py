"""Fail loudly if the RoboCasa asset dataset is missing, stale, or incomplete.

The asset tree is a separately versioned read-only dataset mounted over
models/assets, so it can drift out of step with the code image. Every failure
mode here is otherwise SILENT or produces a confusing error deep inside MuJoCo:

  - dataset not mounted      -> the image's empty mount point is used, and
                                reset() dies on a missing mesh
  - stale dataset version    -> different scenes/objects than intended, NO error
  - partial asset download   -> fails only on the seed that samples the gap
  - mounted a subdirectory   -> the repo-tracked registry files under fixtures/
                                and scenes/ are obscured

Run at container start, before anything trusts the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Every top-level entry the complete tree must have. `fixtures` and the four
# downloaded families are the bulk; `scenes`/`arenas`/`box_links`/
# `novel_instructions` ship with the repo and are the ones a naive
# "mount only the heavy subdirs" layout would obscure.
REQUIRED_DIRS = [
    "objects",
    "textures",
    "generative_textures",
    "fixtures",
    "scenes",
    "arenas",
    "box_links",
]

REQUIRED_OBJECT_FAMILIES = ["objaverse", "aigen_objs", "lightwheel"]

# Canary files. generative_textures/wall/tex075.png is here for a reason: the
# fixture registry YAMLs reference generative-texture paths directly, so a build
# that "optimizes away" tex_generative passes every import check and then dies at
# reset() with "Error opening file '.../generative_textures/wall/tex075.png'".
CANARY_FILES = [
    "generative_textures/wall/tex075.png",
    "fixtures/fixture_registry/wall.yaml",
    "fixtures/fixture_registry/counter.yaml",
]

VERSION_MARKER = "ROBOCASA_ASSET_VERSION"


def fail(msg: str) -> None:
    print(f"[verify_assets] FAIL: {msg}", file=sys.stderr)


def main() -> int:
    assets = Path(
        os.environ.get(
            "ROBOCASA_ASSETS", "/opt/src/robocasa/robocasa/models/assets"
        )
    )
    expected_version = os.environ.get("ROBOCASA_ASSET_VERSION")
    problems: list[str] = []

    if not assets.is_dir():
        fail(f"{assets} does not exist")
        return 1

    entries = list(assets.iterdir())
    if not entries:
        fail(
            f"{assets} is EMPTY -- the asset dataset is not mounted. "
            "Mount the SquashFS over it (see the task's "
            "environment/docker-compose.yaml and `sim/robocasa/robocasa.sh assets mount`)."
        )
        return 1

    for name in REQUIRED_DIRS:
        if not (assets / name).is_dir():
            problems.append(f"missing top-level directory: {name}/")

    for family in REQUIRED_OBJECT_FAMILIES:
        if not (assets / "objects" / family).is_dir():
            problems.append(f"missing object family: objects/{family}/")

    for rel in CANARY_FILES:
        if not (assets / rel).is_file():
            problems.append(f"missing canary file: {rel}")

    # Version skew is the failure mode with no natural symptom, so treat a
    # missing or mismatched marker as fatal rather than a warning.
    marker = assets / VERSION_MARKER
    if expected_version:
        if not marker.is_file():
            problems.append(
                f"no {VERSION_MARKER} in the dataset; cannot confirm it is "
                f"{expected_version}"
            )
        else:
            found = marker.read_text().strip()
            if found != expected_version:
                problems.append(
                    f"asset version mismatch: image expects "
                    f"{expected_version!r}, mounted dataset is {found!r}"
                )

    if problems:
        for p in problems:
            fail(p)
        return 1

    version = marker.read_text().strip() if marker.is_file() else "<unversioned>"
    print(f"[verify_assets] OK: {assets} version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
