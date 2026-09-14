#!/usr/bin/env python3
"""Validate the external assets of the nanoVLA subtasks.

    validate_assets.py <NN-slug>             the directories that subtask mounts
    validate_assets.py bundle NAME PATH      an HF_HOME directory before it is baked into images
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LIBERO10_FRAMES = 101469
LIBERO10_TASKS = 10
DINOV2_MODELS = {
    "models--facebook--dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
}
BASE_MODELS = DINOV2_MODELS | {
    "models--google--flan-t5-small": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
    "models--google--siglip-base-patch16-224": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    "models--google--siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "models--openai--clip-vit-base-patch16": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
    "models--sentence-transformers--all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
OPEN_MODELS = BASE_MODELS | {
    "models--facebook--dinov2-giant": "611a9d42f2335e0f921f1e313ad3c1b7178d206d",
    "models--facebook--dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "models--google--flan-t5-base": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    "models--google--siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
}
BUNDLES = {"base": BASE_MODELS, "dinov2": DINOV2_MODELS, "open": OPEN_MODELS}
# Which perturbation assets each subtask needs, and which run on RoboTwin rather
# than LIBERO. `02` is the only one scored on the LIBERO-plus variants.
SUBTASK_PLUS = {"01-libero-open-design": False, "02-libero-robustness": True,
                "03-robotwin-open-design": False, "04-robotwin-robustness": False}
ROBOTWIN_SUBTASKS = {"03-robotwin-open-design", "04-robotwin-robustness"}
ROBOTWIN_FRAMES, ROBOTWIN_TASKS, ROBOTWIN_EPISODES = 135519, 15, 700


def require_file(path: Path, expected_size: int | None = None) -> None:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"wrong size for {path}: {path.stat().st_size} != {expected_size}")


def validate_shards(path: Path, *, frames: int = LIBERO10_FRAMES, tasks: int = LIBERO10_TASKS) -> None:
    meta_path = path / "meta.json"
    require_file(meta_path)
    meta = json.loads(meta_path.read_text())
    expected = {"n_frames": frames, "n_tasks": tasks, "img_size": 128,
                "state_dim": 8, "act_dim": 7, "suite": "libero_10"}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{meta_path}: {key}={meta.get(key)!r}, expected {value!r}")
    require_file(path / "agent.u8", frames * 128 * 128 * 3)
    require_file(path / "wrist.u8", frames * 128 * 128 * 3)
    require_file(path / "state.f32", frames * 8 * 4)
    require_file(path / "actions.f32", frames * 7 * 4)
    require_file(path / "ep_end.i32", frames * 4)
    require_file(path / "task_idx.u8", frames)
    for name in ("norm_stats.json", "task_strs.json", "task_tokens.json", "vocab.json"):
        require_file(path / name)
    if len(json.loads((path / "task_strs.json").read_text())) != tasks:
        raise ValueError(f"{path}: task_strs.json must list {tasks} tasks")
    if any(path.glob("emb_[vt]_*")):
        raise ValueError(f"shard directory must not contain tower caches: {path}")


def validate_robotwin_shards(path: Path) -> None:
    meta_path = path / "meta.json"
    require_file(meta_path)
    meta = json.loads(meta_path.read_text())
    expected = {"n_frames": ROBOTWIN_FRAMES, "n_episodes": ROBOTWIN_EPISODES, "n_tasks": ROBOTWIN_TASKS,
                "img_size": 128, "state_dim": 16, "act_dim": 14, "suite": "robotwin"}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{meta_path}: {key}={meta.get(key)!r}, expected {value!r}")
    n = ROBOTWIN_FRAMES
    require_file(path / "agent.u8", n * 128 * 128 * 3)
    require_file(path / "wrist.u8", n * 128 * 128 * 3)
    require_file(path / "state.f32", n * 16 * 4)
    require_file(path / "actions.f32", n * 14 * 4)
    require_file(path / "ep_end.i32", n * 4)
    require_file(path / "task_idx.u8", n)
    require_file(path / "episode_id.i32", n * 4)
    for name in ("norm_stats.json", "task_strs.json", "task_names.json", "instructions.json"):
        require_file(path / name)
    if len(json.loads((path / "instructions.json").read_text())) != ROBOTWIN_EPISODES:
        raise ValueError(f"{path}: instructions.json must list {ROBOTWIN_EPISODES} episodes")


def validate_hf(path: Path, allowed: dict[str, str]) -> None:
    hub = path / "hub"
    if not hub.is_dir():
        raise ValueError(f"missing Hugging Face hub directory: {hub}")
    present = {entry.name for entry in hub.iterdir() if entry.is_dir() and
               entry.name.startswith("models--")}
    missing = set(allowed) - present
    unexpected = present - set(allowed)
    if missing:
        raise ValueError(f"HF bundle missing approved models: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"HF bundle contains non-approved models: {sorted(unexpected)}")
    bundle_root = path.resolve()
    for model_name, revision in allowed.items():
        model = hub / model_name
        if model.is_symlink():
            raise ValueError(f"HF model directory must not be a symlink: {model}")
        ref = model / "refs" / "main"
        require_file(ref)
        if ref.read_text().strip() != revision:
            raise ValueError(f"HF model revision mismatch: {model_name}")
        snapshots = model / "snapshots"
        present_revisions = {
            entry.name for entry in snapshots.iterdir()
            if entry.is_dir() and not entry.is_symlink()
        } if snapshots.is_dir() else set()
        if present_revisions != {revision}:
            raise ValueError(
                f"HF model snapshots for {model_name}: {sorted(present_revisions)}, "
                f"expected only {revision}"
            )
    for entry in path.rglob("*"):
        if not entry.is_symlink():
            continue
        try:
            entry.resolve(strict=True).relative_to(bundle_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"HF bundle has broken or escaping symlink: {entry}") from exc


def validate_plus_assets(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"missing LIBERO-plus assets directory: {path}")
    for relative in (
        "scenes/lights/study_light_sync_modified_308.xml",
        "scenes/lights/tabletop_light_sync_modified_308.xml",
        "scenes/lights/floor_light_sync_modified_0.xml",
        "textures/cream-plaster.png",
    ):
        require_file(path / relative)


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"set {name} to an absolute host path")
    path = Path(value).expanduser().resolve()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=sorted(SUBTASK_PLUS) + ["bundle"])
    parser.add_argument("bundle", nargs="?", choices=sorted(BUNDLES))
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    if args.target == "bundle":
        if not args.bundle or not args.path:
            parser.error("bundle NAME PATH")
        validate_hf(args.path.expanduser().resolve(), BUNDLES[args.bundle])
        print(f"nanoVLA bundle {args.bundle} valid: {args.path}")
        return
    if args.target in ROBOTWIN_SUBTASKS:
        validate_robotwin_shards(env_path("NANOVLA_ROBOTWIN_SHARDS"))
    else:
        validate_shards(env_path("NANOVLA_SHARDS_L10"))
    if SUBTASK_PLUS[args.target]:
        validate_plus_assets(env_path("NANOVLA_LIBERO_PLUS_ASSETS"))
    print(f"nanoVLA assets valid for {args.target}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
