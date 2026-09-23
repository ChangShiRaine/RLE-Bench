"""Isolated CLOSED-LOOP execution of untrusted task11 policies.

Fresh subprocess, submission-only sys.path, scrubbed env, uid drop to
nobody under root, /tests chmod 700 by test.sh. The loop is interactive:
the parent steps physics and exchanges one message per control tick.

Wire protocol (pipes, parent <-> child):
  parent -> child : 8-byte big-endian length + npz payload
                    (kind 0 = reset {spec_json, seed}, 1 = act {obs...},
                     2 = shutdown)
  child  -> parent: exactly 65 bytes — 1 tag byte (0 ok / 1 fault) +
                    8 float64 ctrl values
The parent never unpickles agent bytes: replies are fixed-size raw floats,
validated for shape and finiteness upstream. Timeouts (startup and per-act
hang caps) are enforced on the PARENT's clock; a dead child yields None
forever and the episode runs on with held ctrl.

The child's stdout is re-pointed at stderr before any agent code runs, so
print() cannot corrupt the protocol stream.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from io import BytesIO

import numpy as np

from . import scene, spec

STARTUP_TIMEOUT_S = 120.0    # spawn + imports (torch headroom) + reset()
KIND_RESET, KIND_ACT, KIND_END = 0, 1, 2
REPLY_LEN = 1 + 8 * spec.N_CTRL

_RUNNER = r'''
import importlib.util, json, os, struct, sys
import numpy as np

# protect the protocol stream: agent prints go to stderr
_out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr
_in = os.fdopen(os.dup(0), "rb")

N_CTRL = %(n_ctrl)d
REPLY_LEN = 1 + 8 * N_CTRL


def _read_msg():
    head = _in.read(8)
    if len(head) < 8:
        return None
    n = struct.unpack(">Q", head)[0]
    buf = b""
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    from io import BytesIO
    return np.load(BytesIO(buf), allow_pickle=False)


def _reply(tag, ctrl=None):
    vals = np.zeros(N_CTRL) if ctrl is None else \
        np.asarray(ctrl, dtype=np.float64).reshape(-1)[:N_CTRL]
    if vals.shape[0] < N_CTRL:
        vals = np.concatenate([vals, np.zeros(N_CTRL - vals.shape[0])])
    _out.write(bytes([tag]) + vals.astype(">f8").tobytes())
    _out.flush()


def main():
    sub_dir = sys.argv[1]
    sys.path.insert(0, sub_dir)
    mspec = importlib.util.spec_from_file_location(
        "policy", os.path.join(sub_dir, "policy.py"))
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    policy = mod.make_policy()
    _reply(0)

    while True:
        msg = _read_msg()
        if msg is None:
            return
        kind = int(msg["_kind"])
        if kind == 2:
            return
        if kind == 0:
            try:
                cell = json.loads(bytes(msg["spec_json"]).decode())
                policy.reset(cell, int(msg["seed"]))
                _reply(0)
            except Exception:
                import traceback; traceback.print_exc()
                _reply(1)
            continue
        obs = {}
        for k in msg.files:
            if k == "_kind":
                continue
            v = msg[k]
            obs[k] = float(v) if v.ndim == 0 else v
        try:
            ctrl = np.asarray(policy.act(obs), dtype=np.float64).reshape(-1)
            if ctrl.shape != (N_CTRL,):
                raise ValueError(f"ctrl shape {ctrl.shape}")
            _reply(0, ctrl)
        except Exception:
            import traceback; traceback.print_exc()
            _reply(1)


main()
''' % {"n_ctrl": spec.N_CTRL}


SANDBOX_UID = 65534


def package_problem(path: str) -> str | None:
    """Why a submitted package is unusable (links, size, file count), or None.
    Walks without following links, before anything is copied."""
    total = count = 0
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            p = os.path.join(root, name)
            if os.path.islink(p):
                return "policy package must not contain symbolic links"
            if name in files:
                count += 1
                total += os.lstat(p).st_size
        if count > spec.POLICY_MAX_FILES:
            return f"policy package has more than {spec.POLICY_MAX_FILES} files"
        if total > spec.POLICY_MAX_BYTES:
            return (f"policy package exceeds {spec.POLICY_MAX_BYTES >> 20} "
                    "MiB")
    return None


def kill_sandboxed() -> int:
    """SIGKILL every process running as the sandbox uid (daemons a policy may
    have forked). Only acts when the verifier runs as root."""
    if os.geteuid() != 0:
        return 0
    killed = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                uid = next(int(line.split()[1]) for line in f
                           if line.startswith("Uid:"))
            if uid == SANDBOX_UID:
                os.kill(int(pid), 9)
                killed += 1
        except (OSError, StopIteration, ValueError):
            pass
    return killed


def _drop_privileges():
    """Pre-exec: become nobody so /tests (chmod 700) is unreadable."""
    os.setgroups([])
    os.setgid(SANDBOX_UID)
    os.setuid(SANDBOX_UID)


def _world_readable(path: str) -> None:
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, os.stat(p).st_mode | stat.S_IROTH | stat.S_IRGRP)
    os.chmod(path, 0o755)


def _pack(kind: int, arrays: dict) -> bytes:
    buf = BytesIO()
    np.savez(buf, _kind=np.array(kind), **arrays)
    payload = buf.getvalue()
    return struct.pack(">Q", len(payload)) + payload


class PolicyLoadError(RuntimeError):
    """The submitted module or factory could not be loaded."""


class PolicyProcess:
    """Sandboxed submission implementing the runtime Policy interface.

    act() returns the child's ctrl array, or None on any fault (bad reply,
    hang, dead child) — the runtime holds the last valid ctrl and counts it.
    """

    def __init__(self, submission_dir: str,
                 act_timeout_s: float = spec.ACT_HANG_CAP_S,
                 startup_timeout_s: float = STARTUP_TIMEOUT_S):
        self.submission_dir = os.path.abspath(submission_dir)
        self.act_timeout_s = act_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.dead: str | None = None
        self.loaded = False
        self.load_wall_s = 0.0
        # Parent-observed submission time. The episode budget deliberately
        # excludes MuJoCo stepping, sensor rendering, and verifier video.
        self.reset_wall_s = 0.0
        self.act_wall_s = 0.0
        self.n_acts = 0
        self.work = tempfile.mkdtemp(prefix="task11_")
        os.chmod(self.work, 0o755)

        if not os.path.exists(os.path.join(self.submission_dir,
                                           "policy.py")):
            self.dead = "policy.py missing"
            self.proc = None
            return
        problem = package_problem(self.submission_dir)
        if problem:
            self.dead, self.proc = problem, None
            return
        staged = os.path.join(self.work, "policy")
        shutil.copytree(self.submission_dir, staged, symlinks=True)
        self.submission_dir = staged
        _world_readable(staged)

        self.cell_mjb = os.path.join(self.work, "cell.mjb")
        scene.save_cell_mjb(self.cell_mjb)
        os.chmod(self.cell_mjb, 0o644)
        runner = os.path.join(self.work, "runner.py")
        with open(runner, "w") as f:
            f.write(_RUNNER)
        os.chmod(runner, 0o644)
        self._stderr_path = os.path.join(self.work, "stderr.log")
        self._stderr_f = open(self._stderr_path, "wb")
        os.chmod(self._stderr_path, 0o666)

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": self.work, "TMPDIR": self.work,
            "MPLCONFIGDIR": self.work, "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"),
        }
        preexec = _drop_privileges if os.geteuid() == 0 else None
        self.proc = subprocess.Popen(
            [sys.executable, runner, self.submission_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr_f, env=env, cwd=self.work,
            preexec_fn=preexec)

    def check_loaded(self) -> None:
        """Wait for module import and make_policy(), before starting physics."""
        if self.loaded:
            return
        t0 = time.monotonic()
        try:
            if not self.dead and self._read_reply(self.startup_timeout_s) is not None:
                self.loaded = True
                return
        finally:
            self.load_wall_s += time.monotonic() - t0
        raise PolicyLoadError(f"policy load failed: {self.dead}\n{self.stderr_tail()}")

    # ------------------------------------------------------------- protocol
    def _send(self, kind: int, arrays: dict) -> bool:
        try:
            self.proc.stdin.write(_pack(kind, arrays))
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            self._kill("child pipe closed")
            return False

    def _read_reply(self, timeout_s: float) -> np.ndarray | None:
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        buf = b""
        while len(buf) < REPLY_LEN:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill(f"reply timeout after {timeout_s:.0f}s")
                return None
            r, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not r:
                continue
            chunk = os.read(fd, REPLY_LEN - len(buf))
            if not chunk:
                self._kill("child exited")
                return None
            buf += chunk
        if buf[0] != 0:
            return None      # child-reported fault (episode continues)
        return np.frombuffer(buf[1:], dtype=">f8").astype(np.float64)

    def _kill(self, reason: str) -> None:
        if self.dead is None:
            self.dead = reason
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    # ------------------------------------------------------------ interface
    def reset(self, cell_spec: dict, seed: int) -> None:
        if self.dead:
            return
        self.check_loaded()
        # Charge preflight import time to the episode's first reset.
        self.reset_wall_s += self.load_wall_s
        self.load_wall_s = 0.0
        cell = dict(cell_spec)
        cell["model_file"] = self.cell_mjb
        spec_json = json.dumps(cell).encode()
        t0 = time.monotonic()
        try:
            if not self._send(KIND_RESET, {
                    "spec_json": np.frombuffer(spec_json, dtype=np.uint8),
                    "seed": np.array(int(seed))}):
                return
            self._read_reply(self.startup_timeout_s)
        finally:
            self.reset_wall_s += time.monotonic() - t0

    def act(self, obs: dict):
        if self.dead:
            return None
        arrays = {}
        for k, v in obs.items():
            arrays[k] = np.asarray(v)
        t0 = time.monotonic()
        try:
            if not self._send(KIND_ACT, arrays):
                return None
            return self._read_reply(self.act_timeout_s)
        finally:
            self.act_wall_s += time.monotonic() - t0
            self.n_acts += 1

    @property
    def policy_wall_s(self) -> float:
        """Cumulative parent-observed time spent waiting on submission code."""
        return self.reset_wall_s + self.act_wall_s

    def stderr_tail(self, n: int = 2000) -> str:
        try:
            with open(self._stderr_path, "rb") as f:
                return f.read()[-n:].decode(errors="replace")
        except (OSError, AttributeError):
            return ""

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._send(KIND_END, {})
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self._kill("closed")
        if self.proc is not None:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        if getattr(self, "_stderr_f", None) is not None:
            self._stderr_f.close()
        kill_sandboxed()                 # no forked daemon outlives it
        shutil.rmtree(self.work, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
