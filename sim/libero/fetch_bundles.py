#!/usr/bin/env python3
"""Materialise a nanoVLA encoder bundle from the Hub, pinned to the validator's commits.

    fetch_bundles.py base|dinov2|open DIRECTORY

The bundle is an offline `HF_HOME` holding exactly the models `validate_assets.py`
approves for that bundle, each at exactly one snapshot -- the commit recorded there.
That file stays the single source of truth: this script imports its allowlist rather
than repeating it, so a pin can never drift between what is downloaded and what is
accepted. Re-running is cheap: the hub cache skips files it already has, and the
directory is pruned to the allowlist before it is handed to the validator.

Weights for frameworks nothing in the task loads (TensorFlow, Flax, ONNX) are skipped;
everything the transformers loaders read is fetched.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "tasks" / "task05" / "harness" / "validate_assets.py"
# What a tower actually loads: the config, one set of weights, and the tokenizer or
# preprocessor files beside them. Model cards, TensorFlow/Flax/ONNX exports and the
# rest of a repo stay out, which is what keeps the open bundle to tens of GB.
ALLOW = [
    "config.json", "preprocessor_config.json",
    "*.safetensors", "*.safetensors.index.json",
    "pytorch_model.bin", "pytorch_model-*.bin", "pytorch_model.bin.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "spiece.model", "vocab.json", "vocab.txt", "merges.txt",
]


def allowlist() -> dict[str, dict[str, str]]:
    spec = importlib.util.spec_from_file_location("validate_assets", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BUNDLES


def repo_id(cache_name: str) -> str:
    owner, _, model = cache_name[len("models--"):].partition("--")
    return f"{owner}/{model}"


def drop_redundant_weights(snapshot: Path) -> None:
    """Keep one weight format. Repos that ship both formats would otherwise double the
    bundle; the ones that only ship `pytorch_model.bin` (CLIP at its pinned commit)
    keep it."""
    if not (snapshot / "model.safetensors").exists():
        return
    for name in ("pytorch_model.bin", "pytorch_model.bin.index.json"):
        link = snapshot / name
        if not link.exists() and not link.is_symlink():
            continue
        blob = link.resolve() if link.is_symlink() else None
        link.unlink()
        if blob is not None and blob.exists():
            blob.unlink()


def fetch(bundle: str, root: Path) -> None:
    from huggingface_hub import snapshot_download

    models = allowlist()[bundle]
    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    for cache_name, revision in sorted(models.items()):
        name = repo_id(cache_name)
        print(f"[{bundle}] {name}@{revision[:12]}", flush=True)
        snapshot_download(name, revision=revision, cache_dir=str(hub),
                          allow_patterns=ALLOW)
        # Downloading by commit leaves no branch ref behind, and the validator reads
        # refs/main to prove the bundle holds the pinned revision and not a moving tag.
        ref = hub / cache_name / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision)
        # One snapshot per model: an older download of the same repo would otherwise
        # sit beside the pinned one and the validator would reject the bundle.
        snapshots = hub / cache_name / "snapshots"
        for entry in snapshots.iterdir() if snapshots.is_dir() else ():
            if entry.name != revision:
                shutil.rmtree(entry, ignore_errors=True)
        drop_redundant_weights(snapshots / revision)
    for entry in hub.iterdir():
        if entry.name.startswith("models--") and entry.name not in models:
            print(f"[{bundle}] removing non-approved {entry.name}", flush=True)
            shutil.rmtree(entry, ignore_errors=True)
    # Cache bookkeeping, not bundle content: it would ride into the image and the
    # validator walks every symlink under the root.
    shutil.rmtree(hub / ".locks", ignore_errors=True)
    (hub / "CACHEDIR.TAG").unlink(missing_ok=True)
    for cache_name in models:
        shutil.rmtree(hub / cache_name / "trees", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", choices=sorted(allowlist()))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    try:
        fetch(args.bundle, root)
    except ImportError:
        sys.exit("huggingface_hub is required: uv pip install huggingface_hub")
    print(f"nanoVLA bundle {args.bundle} materialised at {root}")


if __name__ == "__main__":
    main()
