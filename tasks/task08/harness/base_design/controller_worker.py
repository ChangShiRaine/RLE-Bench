"""Pipe worker: loaded before privileges are dropped; never imports the harness."""
import ctypes
import importlib.util
import json
import os
import random
import resource
import struct
import sys

import mujoco  # noqa: F401 — preload the public runtime before limiting processes
import numpy as np


def main():
    work, cpu_seconds = sys.argv[1], int(sys.argv[2])
    if mujoco.__version__ != "3.5.0":
        raise RuntimeError("unexpected MuJoCo runtime")
    output = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    # The parent creates this directory for this worker alone. Neither the
    # verifier tree nor its model/state is accessible after the uid change.
    os.chdir(work)
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    if ctypes.CDLL(None).prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        raise RuntimeError("could not disable privilege escalation")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    sys.path.insert(0, work)
    random.seed(0)
    np.random.seed(0)
    spec = importlib.util.spec_from_file_location("controller", "controller.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = module
    spec.loader.exec_module(module)
    policy = module.make_controller()
    count = None
    while True:
        header = sys.stdin.buffer.read(4)
        if not header:
            return
        size = struct.unpack(">I", header)[0]
        if size > 1024 * 1024:
            raise ValueError("oversized request")
        message = json.loads(sys.stdin.buffer.read(size))
        if count is None:
            count = len(message["actuator_names"])
            message["model_file"] = os.path.join(work, "model.mjb")
            policy.reset(message, message["seed"])
            control = np.zeros(count)
        else:
            control = np.asarray(policy.act(message), dtype=float)
        if control.shape != (count,) or not np.isfinite(control).all():
            raise ValueError("act() must return one finite value per actuator")
        output.write(b"\0" + control.astype(">f8").tobytes())


if __name__ == "__main__":
    main()
