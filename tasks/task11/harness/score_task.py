"""Task 11 verifier entry point (/tests/score_task.py).

Stages the untrusted policy package, reruns it on the hidden seeds, and
publishes reward.json (headline + checkpoint fields), report.json and the
episode videos. Everything is first written to a root-only directory; only
after every sandboxed process is gone is /logs/verifier cleared of anything
a policy may have planted and the outputs copied in.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

ARTIFACTS = os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts")
STAGING = os.environ.get("RLEBENCH_STAGING", "/tmp/submission")
VERIFIER_DIR = os.environ.get("RLEBENCH_VERIFIER_DIR", "/logs/verifier")
OUTPUTS = ("reward.json", "report.json", "media")
REWARD_KEYS = ("utilization", "throughput_ppm", "pick_success_rate",
               "pick_place_success_rate", "floor_drop_rate", "bin_full_rate",
               "n_packed", "floor_drops", "damage_events", "bin_hits",
               "policy_faults")


def _dump(path: str, obj) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str, allow_nan=False)


def _publish(private: str) -> None:
    """Replace the verifier outputs with the private ones, never following
    links a policy may have planted."""
    from harness.sandbox import kill_sandboxed
    kill_sandboxed()
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    for name in OUTPUTS:
        path = os.path.join(VERIFIER_DIR, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.unlink(path)
    for name in OUTPUTS:
        src = os.path.join(private, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(VERIFIER_DIR, name))
        elif os.path.isfile(src):
            shutil.copyfile(src, os.path.join(VERIFIER_DIR, name),
                            follow_symlinks=False)


def _media(private: str):
    try:
        from rlebench.core.media import Media
        return Media.verifier(private)
    except Exception as e:  # noqa: BLE001 — media is never load-bearing
        print(f"media disabled: {e}")
        return None


def main() -> int:
    private = tempfile.mkdtemp(prefix="verifier_")        # mode 0700
    try:
        shutil.rmtree(STAGING, ignore_errors=True)
        source = os.path.join(ARTIFACTS, "policy")
        problem = None
        if os.path.isdir(source) and not os.path.islink(source):
            from harness.sandbox import package_problem
            problem = package_problem(source)
            if problem is None:
                shutil.copytree(source, STAGING, symlinks=True)
        os.makedirs(STAGING, exist_ok=True)
        from harness.scorer import score_submission
        media = _media(private)
        report = score_submission(STAGING, media=media)
        if problem:
            report["errors"].insert(0, problem)
        reward = {"reward": float(report["reward"]),
                  "raw_total": float(report["raw_total"]),
                  "gated": int(report["gated"])}
        reward.update({k: float(report[k]) for k in REWARD_KEYS if k in report})
        reward.update({f"gate_{k}": int(v) for k, v in report["gate"].items()})
        _dump(os.path.join(private, "reward.json"), reward)
        _dump(os.path.join(private, "report.json"), report)
        if media is not None:
            media.report()
        _publish(private)
        print(json.dumps(reward, indent=2))
        return 0
    except Exception:  # noqa: BLE001 — infrastructure error, not a low score
        traceback.print_exc()
        shutil.rmtree(private, ignore_errors=True)
        private = tempfile.mkdtemp(prefix="verifier_")
        _dump(os.path.join(private, "reward.json"),
              {"reward": 0.0, "harness_error": 1})
        _publish(private)
        return 1


if __name__ == "__main__":
    sys.exit(main())
