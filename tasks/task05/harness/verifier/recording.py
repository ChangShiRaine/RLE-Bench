"""Scored-episode videos for the verifier's media (rlebench.core.media).

The rollout driver names the first RECORDED episodes of every task and the workers film
the ones they run. `assemble` keeps every successful clip as <task>__<i>.mp4 and joins
one clip per task -- its first success, else its first -- into rollouts.mp4. Evidence
only: nothing here can raise into a rollout or change a row.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

RECORDED = 5
CLIPS = ".clips"
VIDEO = "rollouts.mp4"


def designate(groups: dict[str, list[tuple]]) -> dict[str, str]:
    """{episode key: clip name} for the first RECORDED episodes of each group, in order."""
    clips = {}
    for group, keys in groups.items():
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(group))
        for i, key in enumerate(keys[:RECORDED]):
            clips[json.dumps(list(key))] = f"{label}__{i}"
    return clips


def recorder(args, key: tuple, every: int):
    """A Recorder for this episode when it is designated, else None."""
    name = getattr(args, "clips", {}).get(json.dumps(list(key)))
    if name is None:
        return None
    try:
        from rlebench.core import media
        return media.Recorder(Path(args.record) / CLIPS / f"{name}.mp4", every=every) if media.enabled() else None
    except Exception:
        return None


def choose(clips: dict[str, str], outcomes: dict[str, bool], filmed: set[str]) -> list[str]:
    """One filmed clip per group, in designation order: its first success, else its first."""
    groups: dict[str, list[tuple[str, bool]]] = {}
    for key, name in clips.items():
        if name in filmed:
            groups.setdefault(name.rsplit("__", 1)[0], []).append((name, outcomes.get(key, False)))
    return [next((name for name, ok in entries if ok), entries[0][0]) for entries in groups.values()]


def assemble(root: str, clips: dict[str, str], outcomes: dict[str, bool]) -> None:
    try:
        from rlebench.core import media
        session, folder = media.Media(Path(root)), Path(root) / CLIPS
        filmed = {path.stem for path in folder.glob("*.mp4")}
        chosen = choose(clips, outcomes, filmed) if media.ffmpeg_exe() else []
        if chosen:
            listing = folder / "concat.txt"
            listing.write_text("".join(f"file '{folder / name}.mp4'\n" for name in chosen))
            joined = subprocess.run([media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                                     "-safe", "0", "-i", str(listing), "-c", "copy", str(session.path(VIDEO))],
                                    capture_output=True, timeout=600)
            if joined.returncode == 0:
                session.files.append(VIDEO)
            else:
                session.skip(VIDEO, joined.stderr.decode(errors="replace")[-300:])
        else:
            session.skip(VIDEO, "no episode filmed" if media.ffmpeg_exe() else "ffmpeg unavailable")
        for key, name in clips.items():
            if name in filmed and outcomes.get(key):
                (folder / f"{name}.mp4").replace(session.path(f"{name}.mp4"))
                session.files.append(f"{name}.mp4")
        shutil.rmtree(folder, ignore_errors=True)
        session.close()
        print(f"media: {len(session.files)} videos, one clip per task in {VIDEO}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"media skipped: {exc}", flush=True)
