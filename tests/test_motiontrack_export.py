"""The submission boundary: checkpoint -> policy.onnx -> evaluator.

The evaluator must be able to run a policy without importing a line of the
submission's code, and the model-size budget must be measured from the artifact
rather than from anything the submission says about itself.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from harness import evaluator, export, spec  # noqa: E402


def make_actor(hidden=64, history=1):
    return torch.nn.Sequential(
        torch.nn.Linear(spec.OBS_DIM * history, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, spec.N_ACTIONS))


def test_export_round_trips_through_onnxruntime(tmp_path):
    torch.manual_seed(0)
    actor = make_actor()
    mean = torch.randn(spec.OBS_DIM)
    std = torch.rand(spec.OBS_DIM) + 0.5
    export.export(actor, str(tmp_path), obs_mean=mean, obs_std=std)

    policy = evaluator.OnnxPolicy(str(tmp_path))
    obs = np.random.default_rng(0).normal(size=spec.OBS_DIM)
    want = actor(((torch.tensor(obs).float() - mean) / std)[None]).detach().numpy()[0]
    assert np.abs(policy.act(obs) - want).max() < 1e-5
    assert policy.act(obs).shape == (spec.N_ACTIONS,)


def test_normaliser_is_baked_into_the_graph(tmp_path):
    """A normaliser left outside the graph would have to be described to the
    evaluator and trusted. Shifting the statistics must change the output."""
    torch.manual_seed(1)
    actor = make_actor()
    obs = np.zeros(spec.OBS_DIM)

    export.export(actor, str(tmp_path), obs_mean=torch.zeros(spec.OBS_DIM),
                  obs_std=torch.ones(spec.OBS_DIM))
    plain = evaluator.OnnxPolicy(str(tmp_path)).act(obs)

    export.export(actor, str(tmp_path), obs_mean=torch.full((spec.OBS_DIM,), 3.0),
                  obs_std=torch.full((spec.OBS_DIM,), 2.0))
    shifted = evaluator.OnnxPolicy(str(tmp_path)).act(obs)
    assert not np.allclose(plain, shifted)


def test_parameter_count_comes_from_the_graph(tmp_path):
    actor = make_actor(hidden=64)
    meta = export.export(actor, str(tmp_path))
    torch_params = sum(p.numel() for p in actor.parameters())
    # The graph also carries the normaliser's mean and std.
    assert meta["n_params"] == torch_params + 2 * spec.OBS_DIM
    assert meta["n_params"] == export.count_parameters(
        os.path.join(tmp_path, spec.POLICY_FILE))


def test_a_bigger_network_counts_bigger(tmp_path):
    small = export.export(make_actor(hidden=32), str(tmp_path))["n_params"]
    big = export.export(make_actor(hidden=512), str(tmp_path))["n_params"]
    assert big > small * 4


def test_param_budget_is_enforceable_from_the_artifact(tmp_path):
    """A submission cannot spend its way past the cap by lying in the metadata."""
    meta = export.export(make_actor(hidden=64), str(tmp_path))
    assert meta["n_params"] < spec.MAX_PARAMS

    path = os.path.join(tmp_path, spec.META_FILE)
    with open(path) as f:
        claimed = json.load(f)
    claimed["n_params"] = 1
    with open(path, "w") as f:
        json.dump(claimed, f)
    counted = export.count_parameters(os.path.join(tmp_path, spec.POLICY_FILE))
    assert counted == meta["n_params"], "the graph, not the claim, is the budget"


def test_history_stacking_widens_the_input(tmp_path):
    meta = export.export(make_actor(history=4), str(tmp_path), obs_history=4)
    assert meta["obs_dim"] == 4 * spec.OBS_DIM
    policy = evaluator.OnnxPolicy(str(tmp_path))
    assert policy.history == 4
    assert policy.act(np.zeros(4 * spec.OBS_DIM)).shape == (spec.N_ACTIONS,)


def test_absurd_history_is_rejected(tmp_path):
    export.export(make_actor(), str(tmp_path), obs_history=1)
    path = os.path.join(tmp_path, spec.META_FILE)
    with open(path) as f:
        meta = json.load(f)
    meta["obs_history_length"] = spec.MAX_HISTORY + 1
    with open(path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError):
        evaluator.OnnxPolicy(str(tmp_path))


def test_export_leaves_a_gpu_model_where_it_was(tmp_path):
    """Every submission trains on the GPU and exports mid-run; export must not
    quietly move the caller's model to CPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    actor = make_actor().cuda()
    export.export(actor, str(tmp_path))
    assert next(actor.parameters()).is_cuda


@pytest.mark.skipif(
    not os.path.exists("third_party/motiontrack/motions/dance1_subject2.npz"),
    reason="run `make sim-motiontrack` first")
def test_an_untrained_policy_scores_near_the_floor(tmp_path):
    from harness import motion, robot

    export.export(make_actor(), str(tmp_path))
    policy = evaluator.OnnxPolicy(str(tmp_path))
    clip = motion.Motion.load("third_party/motiontrack/motions/dance1_subject2.npz")
    log = evaluator.run_episode(robot.build("third_party/motiontrack/unitree_g1"),
                                clip, policy, seed=11, history=policy.history)
    assert log.summary()["tracking_score"] < 0.1


def test_validate_accepts_a_well_formed_submission(tmp_path):
    export.export(make_actor(), str(tmp_path))
    assert export.validate(str(tmp_path)) == []


def test_validate_rejects_an_oversized_network(tmp_path):
    big = torch.nn.Sequential(
        torch.nn.Linear(spec.OBS_DIM, 4000), torch.nn.ELU(),
        torch.nn.Linear(4000, 3000), torch.nn.ELU(),
        torch.nn.Linear(3000, spec.N_ACTIONS))
    meta = export.export(big, str(tmp_path))
    assert meta["n_params"] > spec.MAX_PARAMS
    assert any("budget" in p for p in export.validate(str(tmp_path)))


@pytest.mark.parametrize("break_it, expect", [
    (lambda d: os.remove(os.path.join(d, spec.POLICY_FILE)), "missing"),
    (lambda d: open(os.path.join(d, spec.POLICY_FILE), "wb").write(b"junk"),
     "will not load"),
    (lambda d: os.remove(os.path.join(d, spec.META_FILE)), "unreadable"),
])
def test_validate_rejects_broken_submissions(tmp_path, break_it, expect):
    export.export(make_actor(), str(tmp_path))
    break_it(str(tmp_path))
    problems = export.validate(str(tmp_path))
    assert problems and any(expect in p for p in problems)


def test_validate_rejects_a_wrong_action_dimension(tmp_path):
    export.export(torch.nn.Linear(spec.OBS_DIM, 7), str(tmp_path))
    assert any("shape" in p for p in export.validate(str(tmp_path)))


def test_validate_rejects_an_absurd_history(tmp_path):
    export.export(make_actor(), str(tmp_path))
    path = os.path.join(tmp_path, spec.META_FILE)
    meta = json.load(open(path))
    meta["obs_history_length"] = spec.MAX_HISTORY + 1
    json.dump(meta, open(path, "w"))
    assert any("obs_history_length" in p for p in export.validate(str(tmp_path)))


def test_validate_rejects_a_policy_that_returns_nan(tmp_path):
    actor = make_actor()
    with torch.no_grad():
        actor[0].weight.fill_(float("nan"))
    export.export(actor, str(tmp_path))
    assert any("non-finite" in p for p in export.validate(str(tmp_path)))


def test_submission_is_a_single_self_contained_file(tmp_path):
    """The artifact is copied between containers; weights split into a
    policy.onnx.data sidecar would arrive as a weightless graph whose parameter
    count still reads correctly. Validate a directory holding ONLY the two
    contract files."""
    import shutil

    source = tmp_path / "src"
    export.export(make_actor(hidden=256), str(source))
    assert sorted(os.listdir(source)) == sorted([spec.POLICY_FILE, spec.META_FILE])

    handoff = tmp_path / "handoff"
    handoff.mkdir()
    for name in (spec.POLICY_FILE, spec.META_FILE):
        shutil.copy(source / name, handoff)
    assert export.validate(str(handoff)) == []

    obs = np.random.default_rng(0).normal(size=spec.OBS_DIM)
    assert np.isfinite(evaluator.OnnxPolicy(str(handoff)).act(obs)).all()
