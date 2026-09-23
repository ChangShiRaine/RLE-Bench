"""Verifier scoring for task11 (verifier-only).

score_submission(submission_dir) reruns the untrusted policy package through
the sandbox on the hidden evaluation seeds — one fresh subprocess per
episode — and scores harness instrumentation only; nothing the agent
self-reports is used. The reward formula is documented in config.py.

Gate: the policy loads, answers the contract on a smoke episode with few
faults, the simulation stays stable, and a repeated smoke run is
bit-identical. A failed gate caps the reward at GATE_CAP.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

from rlebench.core.media import VIDEO_FPS

from . import config, runtime, sandbox, spec

_SEEDS_PATH = os.path.join(os.path.dirname(__file__), "eval_seeds.json")

# Verifier-only key: the policy gets a deterministic RNG seed without learning
# the world seed that reproduces the hidden box stream.
_POLICY_SEED_KEY = bytes.fromhex(
    "5b1f0c7e93a24d6f8e21c4b7a9d03e6f"
    "17c8e2a95d4b3f60a1e7c29b84d6f3a2")
EVAL_VIDEO_EVERY = round(spec.CTRL_HZ / VIDEO_FPS)


def _policy_seed(world_seed: int) -> int:
    digest = hmac.new(_POLICY_SEED_KEY, str(int(world_seed)).encode(),
                      hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _load_seeds(seeds=None):
    if seeds is None:
        with open(_SEEDS_PATH) as f:
            seeds = json.load(f)
    return int(seeds["smoke"]), [int(s) for s in seeds["eval"]]


def episode_score(m: dict) -> float:
    u = min(m["utilization"] / config.UTIL_CAP, 1.0)
    t = min(m["throughput_ppm"] / config.TP_CAP_PPM, 1.0)
    pp = m["pick_place_success_rate"] * min(
        m["n_packed"] / config.PP_MIN_PACKED, 1.0)
    full = bool(m["bin_full"]) and m["floor_drops"] == 0
    s = (config.W_UTIL * u + config.W_TP * t * u + config.W_PP * pp
         + config.W_FULL * full * u
         - config.W_FLOOR * m["floor_drops"]
         - config.W_DAMAGE * m["damage_events"]
         - config.W_BIN_HIT * m["bin_hits"])
    return max(0.0, min(1.0, s))


def _run_sandboxed(submission_dir: str, seed: int, budget_t: float,
                   video=None, wall_budget_s: float = spec.EPISODE_WALL_BUDGET_S):
    start = time.monotonic()
    with sandbox.PolicyProcess(submission_dir) as p:
        p.check_loaded()
        log = runtime.run_episode(
            p, seed, budget_t=budget_t, render=True,
            policy_seed=_policy_seed(seed),
            wall_budget_s=wall_budget_s, video=video,
            video_every=EVAL_VIDEO_EVERY)
        diag = {"policy_wall_s": round(p.policy_wall_s, 3),
                "n_acts": p.n_acts, "sandbox_dead": p.dead,
                "sandbox_stderr_tail": p.stderr_tail(2000)}
    diag["episode_wall_s"] = round(time.monotonic() - start, 3)
    return log, diag


def _failed(report: dict, reason: str) -> dict:
    report["errors"].append(reason)
    report.update(reward=0.0, raw_total=0.0, gated=True,
                  gate={"load": False, "contract": False, "stable": False,
                        "determinism": False})
    return report


def score_submission(submission_dir: str, seeds: dict | None = None,
                     media=None) -> dict:
    """``media``: optional rlebench.core.media.Media; each scored episode is
    recorded as ep<i>.mp4. Recording never affects the score."""
    smoke_seed, eval_seeds = _load_seeds(seeds)
    report: dict = {"errors": [], "episodes": []}
    if not os.path.isfile(os.path.join(submission_dir, "policy.py")):
        return _failed(report, "policy.py missing")
    problem = sandbox.package_problem(submission_dir)
    if problem:
        return _failed(report, problem)

    try:
        smoke1, diag1 = _run_sandboxed(submission_dir, smoke_seed,
                                       config.SMOKE_T,
                                       wall_budget_s=spec.SMOKE_WALL_BUDGET_S)
    except sandbox.PolicyLoadError as exc:
        return _failed(report, str(exc))
    smoke2, diag2 = _run_sandboxed(submission_dir, smoke_seed, config.SMOKE_T,
                                   wall_budget_s=spec.SMOKE_WALL_BUDGET_S)
    report["smoke"] = [
        {"ticks": s.ticks, "policy_faults": s.policy_faults,
         "fault_reasons": s.fault_reasons, "aborted": s.aborted, **d}
        for s, d in ((smoke1, diag1), (smoke2, diag2))]
    gate = {
        "load": diag1["sandbox_dead"] is None
        or smoke1.policy_faults < smoke1.ticks,
        "contract": smoke1.policy_faults <= 0.10 * max(smoke1.ticks, 1),
        "stable": smoke1.aborted is None,
        "determinism": smoke1.digest == smoke2.digest,
    }
    for i, d in enumerate((diag1, diag2)):
        if d["sandbox_dead"] is not None or d["sandbox_stderr_tail"].strip():
            report["errors"].append(
                f"smoke {i}: {d['sandbox_dead']} :: "
                f"{d['sandbox_stderr_tail'][-500:]}")
    if not gate["determinism"]:
        report["errors"].append("smoke episodes diverged: the policy must be "
                                "deterministic (seeded RNG, no wall clock)")

    scores = []
    for i, seed in enumerate(eval_seeds):
        video = media.video(f"ep{i}.mp4") if media is not None else None
        log, diag = _run_sandboxed(submission_dir, seed, spec.EPISODE_T,
                                   video=video)
        if media is not None:
            media.finish(video)
        m = log.metrics()
        m.pop("penalty_events")      # hidden-scene poses stay private
        m["episode_score"] = round(episode_score(m), 4)
        m.update(diag)
        report["episodes"].append(m)
        scores.append(m["episode_score"])
        if diag["sandbox_dead"] is not None:
            report["errors"].append(f"eval {i}: {diag['sandbox_dead']} :: "
                                    f"{diag['sandbox_stderr_tail'][-500:]}")

    total = sum(scores) / len(scores) if scores else 0.0
    gate_ok = all(gate.values())
    eps = report["episodes"]

    def mean(k):
        return round(sum(float(e[k]) for e in eps) / len(eps), 4) \
            if eps else 0.0

    report.update(
        reward=round(total if gate_ok else min(total, config.GATE_CAP), 4),
        raw_total=round(total, 4), gated=not gate_ok, gate=gate,
        utilization=mean("utilization"),
        throughput_ppm=mean("throughput_ppm"),
        pick_success_rate=mean("pick_success_rate"),
        pick_place_success_rate=mean("pick_place_success_rate"),
        floor_drop_rate=mean("floor_drop_rate"),
        bin_full_rate=mean("bin_full"),
        n_packed=sum(e["n_packed"] for e in eps),
        floor_drops=sum(e["floor_drops"] for e in eps),
        damage_events=sum(e["damage_events"] for e in eps),
        bin_hits=sum(e["bin_hits"] for e in eps),
        policy_faults=sum(e["policy_faults"] for e in eps),
    )
    return report
