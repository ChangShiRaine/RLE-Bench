#!/usr/bin/env python3
"""Download what the task05 cells mount into third_party/task05 (`make task05-data`).

    fetch_data.py [--dest DIR]

shards_l10_128/ and shards_rt15_128/ come from a dataset commit whose SHA256SUMS lists
every file; libero-plus-assets/ is LIBERO-plus's archive plus the one texture it omits.
Everything is checked by sha256, and DEST/.pins marks a complete fetch, so a rerun with
the same pins only validates. `rlebench run` mounts DEST unless NANOVLA_* say otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "tasks" / "task05" / "harness" / "validate_assets.py"

SHARDS_REPO = "RLE-Bench/task05"
SHARDS_REVISION = "fb5820f94bae3d793aea44d19625371ba80414aa"
SHARDS = ("shards_l10_128", "shards_rt15_128")

PLUS_REPO = "Sylvest/LIBERO-plus"
PLUS_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
PLUS_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
PLUS_PREFIX = "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/"
# Named by LIBERO-plus's envs/textures.py, absent from its archive; LIBERO has it at LIBERO_SHA.
TEXTURE = "textures/cream-plaster.png"
TEXTURE_URL = ("https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/"
               "8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/assets/" + TEXTURE)
TEXTURE_SHA256 = "5f989033f089158b2c8f0f1f7881f6922b018a5b154558b2edcc9f41a21ab983"

PINS = "\n".join((SHARDS_REPO, SHARDS_REVISION, PLUS_REVISION, PLUS_SHA256, TEXTURE_SHA256)) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path.name}: sha256 {actual}, pinned {expected}")


def extract(archive: Path, dest: Path, prefix: str) -> None:
    """Unpack the members under `prefix` into `dest`, without the prefix."""
    root = dest.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename
            if name.startswith("__MACOSX/") or prefix.startswith(name):
                continue
            if not name.startswith(prefix):
                raise ValueError(f"{archive.name}: {name} is outside {prefix}")
            target = (dest / name[len(prefix):]).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"{archive.name}: {name} escapes {dest}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1 << 24)


def readable(tree: Path) -> None:
    """The cells read the mounts as uids 1000, 65533 and 65534, whatever the umask was."""
    tree.chmod(0o755)
    for path in tree.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def fetch_shards(dest: Path, work: Path) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    sums = Path(hf_hub_download(SHARDS_REPO, "SHA256SUMS", repo_type="dataset",
                                revision=SHARDS_REVISION, local_dir=work)).read_text()
    wanted = {name: digest for digest, name in (line.split(maxsplit=1) for line in sums.splitlines())
              if name.split("/")[0] in SHARDS}
    stale = sorted(name for name, digest in wanted.items()
                   if not ((dest / name).is_file() and sha256(dest / name) == digest))
    print(f"shards: {len(wanted) - len(stale)} of {len(wanted)} files current", flush=True)
    if stale:
        snapshot_download(SHARDS_REPO, repo_type="dataset", revision=SHARDS_REVISION,
                          allow_patterns=stale, local_dir=work)
        for name in stale:
            check(work / name, wanted[name])
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            os.replace(work / name, dest / name)
    for shard in SHARDS:
        for path in (dest / shard).iterdir():
            if f"{shard}/{path.name}" not in wanted:
                path.unlink()
        readable(dest / shard)


def fetch_plus_assets(dest: Path, work: Path) -> None:
    from huggingface_hub import hf_hub_download

    print("libero-plus-assets: downloading", flush=True)
    archive = Path(hf_hub_download(PLUS_REPO, "assets.zip", repo_type="dataset",
                                   revision=PLUS_REVISION, local_dir=work))
    check(archive, PLUS_SHA256)
    staged = work / "libero-plus-assets"
    extract(archive, staged, PLUS_PREFIX)
    archive.unlink()
    with urllib.request.urlopen(TEXTURE_URL, timeout=120) as source, (staged / TEXTURE).open("wb") as sink:
        shutil.copyfileobj(source, sink)
    check(staged / TEXTURE, TEXTURE_SHA256)
    readable(staged)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(staged, dest)


def validate(dest: Path) -> None:
    env = dict(os.environ, NANOVLA_SHARDS_L10=str(dest / "shards_l10_128"),
               NANOVLA_ROBOTWIN_SHARDS=str(dest / "shards_rt15_128"),
               NANOVLA_LIBERO_PLUS_ASSETS=str(dest / "libero-plus-assets"))
    for subtask in ("02-libero-robustness", "03-robotwin-open-design"):
        subprocess.run([sys.executable, str(VALIDATOR), subtask], env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=REPO / "third_party" / "task05")
    dest = parser.parse_args(argv).dest.resolve()
    stamp = dest / ".pins"
    if stamp.is_file() and stamp.read_text() == PINS:
        try:
            validate(dest)
            return
        except subprocess.CalledProcessError:
            print("third_party/task05 is at the pins but incomplete; fetching again", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    stamp.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=dest, prefix=".fetch-") as work:
            fetch_shards(dest, Path(work))
            fetch_plus_assets(dest / "libero-plus-assets", Path(work))
    except ImportError:
        sys.exit("huggingface_hub is required: make install")
    try:
        validate(dest)
    except subprocess.CalledProcessError:
        sys.exit(f"{dest} failed validation")
    stamp.write_text(PINS)


if __name__ == "__main__":
    main()
