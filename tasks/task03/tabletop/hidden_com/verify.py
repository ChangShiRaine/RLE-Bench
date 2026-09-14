"""Read only the privileged simulator's submission record."""
import json
import os
from pathlib import Path
import socket
import stat

from .config import CASES, MAX_STEPS, SOCKET, STATE
from ..reward import reward


def trial_score(record, index, case):
    empty = dict(reward=0.0, correct=0, submitted=0, control_steps=0)
    if not isinstance(record, dict):
        return empty
    steps = record.get("steps")
    valid = (record.get("closed") is True and type(record.get("trial")) is int
             and record["trial"] == index+1 and type(steps) is int
             and 0 <= steps <= MAX_STEPS and record.get("quadrant") in ("A", "B", "C", "D"))
    if not valid:
        return empty
    submitted = record.get("answer") in ("A", "B", "C", "D")
    correct = submitted and record["answer"] == record["quadrant"]
    return dict(reward=round(reward(float(correct)), 12),
                correct=int(correct), submitted=int(submitted),
                control_steps=steps)


def score(record, cases=CASES):
    records = record.get("trials", []) if isinstance(record, dict) else []
    if not isinstance(records, list):
        records = []
    trials = [trial_score(records[i] if i < len(records) else None, i, case)
              for i, case in enumerate(cases)]
    result = dict(reward=sum(t["reward"] for t in trials) / len(cases),
                  correct=sum(t["correct"] for t in trials),
                  submitted=sum(t["submitted"] for t in trials),
                  control_steps=sum(t["control_steps"] for t in trials))
    for i, trial in enumerate(trials, 1):
        result.update({f"trial_{i}_{key}": value for key, value in trial.items()})
    return result


def main():
    if os.geteuid() != 0:
        raise PermissionError("verifier must run as root")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(120)
            connection.connect(SOCKET)
            connection.sendall(b'{"op":"finalize"}\n')
            with connection.makefile("rb") as stream:
                response = json.loads(stream.readline(4096))
            if not response.get("ok"):
                raise RuntimeError("failed to finalize")
    except (OSError, ValueError):
        # A previously committed submission remains authoritative after a daemon exit.
        pass
    path = Path(STATE)
    metadata = path.lstat()
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o600 or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError("invalid submission record ownership or mode")
    print(json.dumps(score(json.loads(path.read_text()))))


if __name__ == "__main__":
    main()
