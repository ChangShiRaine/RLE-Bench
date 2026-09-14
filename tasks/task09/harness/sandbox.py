"""Run the untrusted trim.py in an isolated subprocess and return only plain
arrays.

The scoring process never imports agent Python: module-level code in trim.py
would otherwise share an interpreter with the scorer. All agent code runs in
a child that shares no state with the parent and returns pose proposals or feedforward samples; the parent re-clips and re-simulates every torque with its own pristine
state. A wall-clock timeout bounds runaway code (the job degrades to a
failure result).
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os

import numpy as np

from .probe_protocol import CHOSEN_POSES

DEFAULT_TIMEOUT_S = 120.0


def _load(submission_dir: str, filename: str, modname: str):
    path = os.path.join(submission_dir, filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_trim_rows(trim, qmat, nv: int) -> np.ndarray:
    """Sample a trim callable on a query matrix; bad rows degrade to zeros."""
    rows = []
    for qq in qmat:
        try:
            t = np.asarray(trim(np.asarray(qq, dtype=float).copy()),
                           dtype=float).reshape(-1)
            rows.append(t if (t.shape[0] == nv and np.all(np.isfinite(t)))
                        else np.zeros(nv))
        except Exception:
            rows.append(np.zeros(nv))
    return np.asarray(rows, dtype=float)


def _worker(q_out, submission_dir: str, lead_xml: str, jobs: dict) -> None:
    """jobs["plan_probe"]: {seed: sample, lower, upper} -> 15 proposed poses.
    jobs["ctrim"]: {key: qmat} sampled on the nominal feedforward.
    jobs["cadapt"]: {seed: dict(probe=..., qmats={key: qmat})}, adapt() per
    instance then sample the adapted feedforward; a failing instance
    degrades to None."""
    result: dict = {}
    try:
        from .scenarios import compose_lead
        mod = _load(submission_dir, "trim.py", "agent_trim")
        if "plan_probe" in jobs:
            plans = {}
            for key, request in jobs["plan_probe"].items():
                try:
                    model, _ = compose_lead(lead_xml, seed=0, perturb=False)
                    sample = tuple(np.asarray(q, dtype=float).copy() for q in request["sample"])
                    plan = np.asarray(mod.plan_probe(
                        model, sample, np.asarray(request["lower"]).copy(),
                        np.asarray(request["upper"]).copy()), dtype=float)
                    if plan.shape != (CHOSEN_POSES, model.nv) or not np.all(np.isfinite(plan)):
                        raise ValueError("invalid probe plan shape or values")
                    plans[key] = dict(configs=plan)
                except Exception as exc:
                    plans[key] = dict(error=str(exc))
            result["plan_probe"] = plans
        if "ctrim" in jobs:
            model, _ = compose_lead(lead_xml, seed=0, perturb=False)
            trim = mod.make_trim(model)
            result["ctrim"] = {key: _eval_trim_rows(trim, qmat, model.nv)
                               for key, qmat in jobs["ctrim"].items()}
        if "cadapt" in jobs:
            model, _ = compose_lead(lead_xml, seed=0, perturb=False)
            out = {}
            for key, spec in jobs["cadapt"].items():
                try:
                    probe = [(np.asarray(a, dtype=float),
                              np.asarray(b, dtype=float))
                             for a, b in spec["probe"]]
                    params = mod.adapt(model, probe)
                    trim2 = mod.make_trim_adapted(model, params)
                    out[key] = {k: _eval_trim_rows(trim2, qmat, model.nv)
                                for k, qmat in spec["qmats"].items()}
                except Exception:
                    out[key] = None
            result["cadapt"] = out
        q_out.put(("ok", result))
    except Exception as e:  # missing file, un-importable module, etc.
        q_out.put(("err", repr(e)))


def _context():
    """fork gives the child a copy-on-write snapshot, so agent code mutating
    its own module globals cannot reach this process; spawn elsewhere."""
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


def run_jobs(submission_dir: str, lead_xml: str | None, jobs: dict,
             timeout: float = DEFAULT_TIMEOUT_S) -> dict | None:
    """Execute `jobs` in an isolated subprocess. Returns the result dict, or
    None if the child errored, timed out, or crashed."""
    if not jobs:
        return {}
    ctx = _context()
    q_out = ctx.Queue()
    proc = ctx.Process(target=_worker,
                       args=(q_out, submission_dir, lead_xml, jobs))
    proc.start()
    try:
        status, payload = q_out.get(timeout=timeout)
    except Exception:
        status, payload = "err", "timeout"
    finally:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
    return payload if status == "ok" else None
