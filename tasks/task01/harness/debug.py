"""Live, operator-facing view of a run in progress. ROOT ONLY.

Without it a multi-hour run is a black box until it ends -- and development worst of all,
since `Transcript` is wired only into the evaluation path, so the phase where the agent
spends most of its time would leave no record.

This writes, as the run happens: what phase it is in and what it has spent, every
operation the agent issued, what the simulator looked like as per-episode video, the
the code it has written to disk, and the reward the run would currently score.

WHY THIS IS SAFE TO WRITE, AND WHY IT GOES WHERE IT DOES
--------------------------------------------------------
Most of what is here is ground truth the agent must never see: the reward breakdown
discloses the weights and the success threshold, and the per-episode diagnosis carries
object and fixture poses -- exactly what the task withholds so the drawer has to be
perceived rather than looked up.

Ownership cannot protect it. In debug mode `AGENT_UID` is the host user's own uid, so
the agent and the person reading the output are the SAME uid, and no chmod can tell them
apart.

Directory traversal can. The debug tree is bind-mounted at /opt/private/debug, and
/opt/private is root:root 0700, so the agent has no `+x` on the parent and cannot enter
it whatever the files inside are owned by -- verified as the agent user:

    /opt/private/debug      ls -> Permission denied, cat -> Permission denied
    /logs/artifacts/debug   agent writes freely

That is the same mechanism that already hides the simulator and the scorer, and
check_isolation.sh asserts it. So there is no safe/unsafe split to maintain here: the
whole tree is privileged, and the agent can reach none of it.

Everything is best-effort. Debug output must never be able to fail a run.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

# Camera order in the tiled frame: the two agent views that show the scene, then the
# wrist. Left-to-right in one image so a single video shows everything at once.
TILED_CAMERAS = (
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
)

# Views are tiled at this size whatever the cameras were baked at. The debug recorder
# holds an episode's frames in memory until the next reset, so the tile size sets that
# cost.
TILE_VIEW = 256


def match_owner(paths, reference) -> None:
    """Give `paths` to whoever owns `reference`, and open their modes.

    Both the debug tree and the artifacts export are bind mounts written by a root
    daemon, so anything left root-owned lands root-owned on the HOST -- where the
    person who started the run, an ordinary user, then cannot read it, and where Harbor
    cannot move it into the job directory (archiving fails with EACCES after a
    completely successful run).

    Best-effort per path: failing to chown a convenience file must never fail a run.
    A missing reference means "leave ownership alone" -- outside a container there is
    nobody to hand the files to, and the writer already owns them.
    """
    if reference is None:
        return
    try:
        ref = os.stat(reference)
    except OSError:
        return
    for path in paths:
        try:
            os.chown(path, ref.st_uid, ref.st_gid)
            os.chmod(path, 0o755 if Path(path).is_dir() else 0o644)
        except OSError:
            pass


def _find_owner_ref():
    """A path inside the container that carries the HOST user's identity.

    Harbor's bind mounts do: `/logs/agent` stats as the host user's uid and gid inside
    the container exactly as it does outside. Everything the daemon writes is chowned to
    match, or it lands root-owned on the host and the person watching cannot read it.
    """
    for candidate in ("/logs/agent", "/logs/artifacts", "/logs", "/workspace"):
        path = Path(candidate)
        if path.exists():
            return path
    return None


def tile_frame(obs: dict | None):
    """The three camera views side by side, upright, as one uint8 image.

    NO FLIP: frames arrive upright, because env.make_env sets robosuite's image
    convention before any env is built. What this video shows is therefore exactly what
    the agent saw, which is the point of watching it. Returns None if the observation
    carries no images, which is the case for the fake envs the tests use.
    """
    if not obs:
        return None
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None

    from .env import resample

    views = []
    for key in TILED_CAMERAS:
        arr = obs.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            continue
        arr = arr[:, :, :3].astype("uint8")
        if max(arr.shape[0], arr.shape[1]) > TILE_VIEW:
            arr = resample(arr, TILE_VIEW, TILE_VIEW)
        views.append(arr)
    if not views:
        return None
    height = max(v.shape[0] for v in views)
    padded = [
        v if v.shape[0] == height
        else np.pad(v, ((0, height - v.shape[0]), (0, 0), (0, 0)))
        for v in views
    ]
    return np.concatenate(padded, axis=1)


SEGMENT_SECONDS = 10
MAX_BUFFER_BYTES = 64 * 1024 * 1024


class DebugRecorder:
    """Disabled by default, and a no-op in that state.

    Call sites stay unconditional -- there is no `if debug:` threaded through the
    session -- so the instrumented paths read the same whether or not anyone is
    watching.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        enabled: bool = False,
        every_n_steps: int = 2,
        fps: int = 10,
        workspace: str | os.PathLike[str] = "/workspace",
        ledger_path: str | os.PathLike[str] | None = None,
        owner_ref: str | os.PathLike[str] | None = None,
    ):
        self.enabled = bool(enabled and root)
        self.root = Path(root) if root else None
        self.every_n_steps = max(1, int(every_n_steps))
        self.fps = int(fps)
        self.workspace = Path(workspace)
        self.ledger_path = Path(ledger_path) if ledger_path else None

        self._frames: list = []
        self._buffer_bytes = 0
        self._segment = 0
        self._episode = 0
        self._phase = "development"
        self._step_in_episode = 0
        self._reward_cache: dict | None = None
        self._video_count = 0
        # Whose files these should be. Getting this wrong is not cosmetic: the daemon
        # is root, so without it the whole tree lands root-owned on the HOST and the
        # person who asked for the debug output cannot read their own run.
        #
        # Neither the debug root nor its parent will do: Docker creates a missing
        # bind-mount source as root, and the container-side parent is /opt/private,
        # root:root 0700 by design -- stat'ing either one chowns the tree back to root.
        #
        # Harbor's own bind mounts carry the host user's uid AND gid into the
        # container (measured: /logs/agent is 3566665:2626 inside, the same as
        # outside), so they are the reliable answer to "who is watching this run".
        # Chowning to that uid is safe even though it is also the agent's uid: what
        # keeps the agent out of here is the 0700 parent it cannot traverse, never
        # ownership.
        self._owner_ref = owner_ref or _find_owner_ref()

        if not self.enabled:
            # The bind mount is declared unconditionally (compose has no conditionals),
            # so Docker creates the host directory even with debugging off. Left alone
            # it is a root-owned empty dir in the job directory that the person who ran
            # the job cannot even rmdir. Hand it over and write nothing into it.
            if self.root and self.root.is_dir():
                match_owner([self.root], self._owner_ref)
            return
        try:
            for sub in ("", "episodes", "workspace"):
                (self.root / sub).mkdir(parents=True, exist_ok=True)
            match_owner([self.root, self.root / "episodes",
                         self.root / "workspace"], self._owner_ref)
        except OSError as exc:
            print(f"[debug] disabled: cannot write {self.root}: {exc}", flush=True)
            self.enabled = False

    # -- writing helpers -----------------------------------------------------
    def _append(self, name: str, record: dict) -> None:
        path = self.root / name
        existed = path.exists()
        with open(path, "a") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        if not existed:
            match_owner([path], self._owner_ref)

    def _write(self, name: str, payload: dict) -> None:
        path = self.root / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        # Atomic replace: a reader tailing this must never catch a half-written file.
        os.replace(tmp, path)
        match_owner([path], self._owner_ref)

    # -- lifecycle -----------------------------------------------------------
    def episode_started(self, phase: str, index: int, obs: dict | None = None) -> None:
        """A reset happened. Reset is the episode boundary, so the previous episode's
        video is flushed here and a new buffer starts."""
        if not self.enabled:
            return
        try:
            self.flush_episode()
            self._phase = phase
            self._episode = int(index)
            self._segment = 0
            self._step_in_episode = 0
            self.frame(obs)
            self._snapshot_workspace()
        except Exception as exc:  # noqa: BLE001
            self._complain(exc)

    def frame(self, obs: dict | None) -> None:
        """Buffer one frame, subsampled. Costs no rendering: the images are already in
        the observation the harness holds."""
        if not self.enabled:
            return
        try:
            self._step_in_episode += 1
            if (self._step_in_episode - 1) % self.every_n_steps:
                return
            img = tile_frame(obs)
            if img is not None:
                if self._frames and self._buffer_bytes + img.nbytes > MAX_BUFFER_BYTES:
                    self.flush_episode()
                self._frames.append(img)
                self._buffer_bytes += img.nbytes
                if (len(self._frames) >= max(1, self.fps * SEGMENT_SECONDS)
                        or self._buffer_bytes >= MAX_BUFFER_BYTES):
                    self.flush_episode()
        except Exception as exc:  # noqa: BLE001
            self._complain(exc)

    def flush_episode(self) -> None:
        """Write one bounded segment, also flushing partial segments at reset/close."""
        if not self.enabled or not self._frames:
            return
        frames, self._frames = self._frames, []
        self._buffer_bytes = 0
        suffix = f"_part{self._segment:04d}" if self._segment else ""
        name = f"{self._phase}_ep{self._episode:04d}{suffix}.mp4"
        self._segment += 1
        path = self.root / "episodes" / name
        try:
            import imageio

            imageio.mimwrite(path, frames, fps=self.fps, macro_block_size=None)
            match_owner([path], self._owner_ref)
            self._video_count += 1
            print(f"[debug] wrote {path} ({len(frames)} frames)", flush=True)
        except Exception as exc:  # noqa: BLE001
            self._complain(exc)

    def event(self, op: str, **fields: Any) -> None:
        """One line per agent operation, appended as it happens."""
        if not self.enabled:
            return
        try:
            self._append("events.jsonl", {"op": op, "phase": self._phase,
                                          "episode": self._episode, **fields})
        except Exception as exc:  # noqa: BLE001
            self._complain(exc)

    def status(self, status: dict, *, recompute_reward: bool = False) -> None:
        """Rewrite the live status file.

        `recompute_reward` is off by default because scoring re-reads the whole ledger,
        which is thousands of records by mid-run; the reward is refreshed at episode
        boundaries and cached in between.
        """
        if not self.enabled:
            return
        try:
            if recompute_reward or self._reward_cache is None:
                self._reward_cache = self._score()
            self._write("status.json", {
                **status,
                "episode": self._episode,
                "step_in_episode": self._step_in_episode,
                "videos_written": self._video_count,
                "reward_now": self._reward_cache,
            })
        except Exception as exc:  # noqa: BLE001
            self._complain(exc)

    def diagnosis(self, env, obs: dict | None, **fields: Any) -> None:
        """What the ENVIRONMENT says -- object poses, fixture state, success predicate.

        The privileged half, and the reason this tree is unreachable by the agent.
        """
        if not self.enabled:
            return
        try:
            from .transcript import diagnose

            self._append("diagnosis.jsonl", {
                "phase": self._phase, "episode": self._episode,
                **fields, "diagnosis": diagnose(env, obs),
            })
        except Exception as exc:  # noqa: BLE001
            self._complain(exc)

    def close(self) -> None:
        if not self.enabled:
            return
        self.flush_episode()
        self._snapshot_workspace(final=True)

    # -- internals -----------------------------------------------------------
    def _score(self) -> dict | None:
        if not self.ledger_path or not self.ledger_path.exists():
            return None
        try:
            from . import scoring

            return scoring.score_path(self.ledger_path)
        except Exception:  # noqa: BLE001
            return None

    # An agent may create a venv, clone a repo or pip-install into its workspace, any
    # of which holds tens of thousands of .py files -- and this runs once per episode.
    _SKIP_DIRS = frozenset({
        "__pycache__", ".git", ".venv", "venv", "env", "node_modules",
        "site-packages", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints",
    })
    _MAX_SNAPSHOT_FILES = 200

    def _workspace_sources(self) -> tuple[list[Path], bool]:
        """Every .py the agent has written, and whether the list was truncated.

        RECURSIVE, because agents organise differently: one wrote to `/workspace/*.py`
        and was captured, the next used `/workspace/dev/` and left an empty snapshot --
        which reads as "wrote no code at all" and is badly misleading.

        `os.walk` rather than `rglob` so a venv or a clone is pruned BEFORE being
        descended into; rglob would walk the whole tree first and only then discard it.
        """
        found: list[Path] = []
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in sorted(dirs)
                       if d not in self._SKIP_DIRS and not d.startswith(".")]
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                found.append(Path(root) / name)
                if len(found) >= self._MAX_SNAPSHOT_FILES:
                    return found, True
        return found, False

    def _snapshot_workspace(self, final: bool = False) -> None:
        """Copy whatever the agent has written, so a controller can be read at the
        moment it was in use rather than only in its final form."""
        if not self.workspace.is_dir():
            return
        label = "final" if final else f"ep{self._episode:04d}"
        dest = self.root / "workspace" / label
        try:
            sources, truncated = self._workspace_sources()
            if not sources:
                return                  # nothing written yet; leave no empty dir
            dest.mkdir(parents=True, exist_ok=True)
            copied = [dest]
            for src in sources:
                # Mirror the layout, so `dev/push.py` and `push.py` stay distinct.
                target = dest / src.relative_to(self.workspace)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                copied += [target.parent, target]
            if truncated:
                # Never a silent cap: a snapshot that quietly dropped files would read
                # as the agent having written only these.
                print(f"[debug] controller snapshot capped at "
                      f"{self._MAX_SNAPSHOT_FILES} files", flush=True)
            match_owner(copied, self._owner_ref)
        except Exception as exc:  # noqa: BLE001
            self._complain(exc)

    def _complain(self, exc: Exception) -> None:
        print(f"[debug] {type(exc).__name__}: {exc}", flush=True)


def from_env(ledger_path=None, workspace="/workspace") -> DebugRecorder:
    """Build the recorder the daemon uses, from the environment.

    RLEBENCH_DEBUG turns it on; RLEBENCH_DEBUG_DIR says where. Off unless asked for,
    so a normal run writes exactly what it wrote before.
    """
    flag = os.environ.get("RLEBENCH_DEBUG", "").strip().lower()
    enabled = flag in ("1", "true", "yes", "on")
    return DebugRecorder(
        root=os.environ.get("RLEBENCH_DEBUG_DIR", "/opt/private/debug"),
        enabled=enabled,
        every_n_steps=int(os.environ.get("RLEBENCH_DEBUG_EVERY", "2") or 2),
        fps=int(os.environ.get("RLEBENCH_DEBUG_FPS", "10") or 10),
        workspace=workspace,
        ledger_path=ledger_path,
    )
