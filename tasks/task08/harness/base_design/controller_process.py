"""Bounded exchange of observations and actions with an untrusted controller."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np

from .. import config
from . import controller_worker


class ControllerFault(RuntimeError):
    pass


class ControllerProcess:
    def __init__(self, source: str, model, context: dict):
        if os.geteuid() != 0:
            raise ControllerFault("controller isolation requires the root verifier container")
        # Fail closed if the verifier was launched without its permission boundary.
        for directory in ("/tests", "/logs/verifier"):
            if not Path(directory).is_dir() or os.stat(directory).st_mode & 0o077:
                raise ControllerFault(f"private verifier directory is not protected: {directory}")
        self.work = Path(tempfile.mkdtemp(prefix="task08-controller-"))
        self.proc = None
        self.count = len(context["actuator_names"])
        try:
            raw = Path(source).read_bytes()
            if len(raw) > config.PICK_CONTROLLER_SOURCE_BYTES:
                raise ControllerFault("controller.py exceeds the source size limit")
            (self.work / "controller.py").write_bytes(raw)
            shutil.copyfile(Path(controller_worker.__file__),
                            self.work / "worker.py")
            mujoco.mj_saveModel(model, str(self.work / "model.mjb"))
            for path in self.work.iterdir():
                path.chmod(0o444)
            os.chown(self.work, 65534, 65534)
            self.work.chmod(0o700)
            env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.work),
                   "TMPDIR": str(self.work), "PYTHONHASHSEED": "0",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "MUJOCO_GL": "osmesa"}
            self.proc = subprocess.Popen(
                [sys.executable, "-s", "-P", str(self.work / "worker.py"),
                 str(self.work), str(config.PICK_CONTROLLER_CPU_SECONDS)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env, cwd="/",
                start_new_session=True)
            os.set_blocking(self.proc.stdin.fileno(), False)
            os.set_blocking(self.proc.stdout.fileno(), False)
            self._exchange(context, config.PICK_CONTROLLER_STARTUP_SECONDS)
        except Exception:
            self.close()
            raise

    def _exchange(self, message: dict, timeout: float) -> np.ndarray:
        payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
        outgoing = memoryview(struct.pack(">I", len(payload)) + payload)
        incoming = bytearray()
        expected = 1 + 8 * self.count
        deadline = time.monotonic() + timeout  # hang protection, not a reward metric
        try:
            while outgoing or len(incoming) < expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ControllerFault("controller response timeout")
                reads, writes, _ = select.select(
                    [self.proc.stdout], [self.proc.stdin] if outgoing else [],
                    [], remaining)
                if writes:
                    outgoing = outgoing[os.write(self.proc.stdin.fileno(), outgoing):]
                if reads:
                    chunk = os.read(self.proc.stdout.fileno(), expected - len(incoming))
                    if not chunk:
                        raise ControllerFault("controller exited or returned an invalid action")
                    incoming.extend(chunk)
            values = np.frombuffer(incoming[1:], dtype=">f8").astype(float)
            if incoming[0] or not np.isfinite(values).all():
                raise ControllerFault("controller returned an invalid action")
            return values
        except (OSError, ValueError) as error:
            raise ControllerFault(str(error)) from error

    def act(self, observation: dict) -> np.ndarray:
        return self._exchange(observation, config.PICK_CONTROLLER_REPLY_SECONDS)

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.proc.wait(timeout=5)
            self.proc.stdin.close()
            self.proc.stdout.close()
        shutil.rmtree(self.work, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
