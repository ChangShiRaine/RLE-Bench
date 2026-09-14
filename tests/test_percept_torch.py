"""Dev tests for the rgb-depth-model-training subtask learned path: contract, harness, smoke training.

The full GPU training run happens in the Harbor container; here we lock the
machinery — tensor contract round-trip, deterministic CPU inference,
parameter-limit enforcement, epoch control, and a tiny learning smoke test.
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest
import torch

from harness import episodes, spec, torch_contract, train_harness
from harness.oracle_cnn import PoseNet


@pytest.fixture(scope="module")
def frame():
    return episodes.single_frame(11)


def test_pack_observation_contract(frame):
    x = torch_contract.pack_observation(frame.obs["rgb"], frame.obs["depth"])
    assert x.shape == (5, spec.IMG_H, spec.IMG_W)
    assert x.dtype == torch.float32
    assert 0.0 <= float(x[:3].min()) and float(x[:3].max()) <= 1.0
    assert torch.isfinite(x).all(), "NaN depth must be zeroed, mask carries it"
    nan_frac = np.mean(np.isnan(frame.obs["depth"]))
    mask_frac = 1.0 - float(x[4].mean())
    assert abs(nan_frac - mask_frac) < 1e-6


def test_pose_encode_decode_roundtrip():
    for pose in [(0.1, -0.05, 0.7), (-0.2, 0.15, -3.0), (0.0, 0.0, 3.14)]:
        vec = torch.cat((torch_contract.encode_pose(*pose), torch.tensor([0., 1., 0.])))
        x, y, th, shape_id = torch_contract.decode_output(vec)
        assert shape_id == 1
        assert x == pytest.approx(pose[0], abs=1e-6)
        assert y == pytest.approx(pose[1], abs=1e-6)
        assert spec.wrap_angle(th - pose[2]) == pytest.approx(0.0, abs=1e-5)


def test_fixed_logit_order_and_invalid_outputs():
    for shape in range(3):
        v = torch.tensor([0., 0., 1., 0., 0., 0., 0.])
        v[4 + shape] = 1
        assert torch_contract.decode_output(v)[3] == shape
    assert torch_contract.decode_output(torch.tensor([0., 0., 1., 0., 0., 0., 0.]))[3] == 0
    for values in ([0.] * 4, [0.] * 8, [0.] * 6 + [float("nan")]):
        with pytest.raises(ValueError):
            torch_contract.decode_output(torch.tensor(values))


def test_save_load_infer_deterministic(frame, tmp_path):
    torch.manual_seed(0)
    model = PoseNet().eval()
    path = str(tmp_path / torch_contract.MODEL_FILENAME)
    torch_contract.save_model(model, path)
    loaded = torch_contract.load_model(path)
    p1 = torch_contract.infer(loaded, frame.obs["rgb"], frame.obs["depth"])
    p2 = torch_contract.infer(loaded, frame.obs["rgb"], frame.obs["depth"])
    assert p1 == p2
    # scripted module matches the eager one
    with torch.no_grad():
        eager = model(torch_contract.pack_observation(
            frame.obs["rgb"], frame.obs["depth"])[None])[0]
    assert np.allclose(torch_contract.decode_output(eager), p1, atol=1e-5)


def test_parameter_limit_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(spec, "MODEL_MAX_PARAMETERS", 1)
    with pytest.raises(ValueError, match="model has"):
        torch_contract.save_model(
            PoseNet(), str(tmp_path / torch_contract.MODEL_FILENAME))


def test_sandbox_independently_enforces_parameter_limit(
        monkeypatch, frame, tmp_path):
    from harness import sandbox

    path = str(tmp_path / torch_contract.MODEL_FILENAME)
    frozen = torch.jit.freeze(torch.jit.script(PoseNet().eval()))
    assert list(frozen.parameters()) == []
    torch.jit.save(frozen, path)
    monkeypatch.setattr(spec, "MODEL_MAX_PARAMETERS", 1)
    result = sandbox.run_job(
        "torch", [frame.obs], ("rgb", "depth"), model_path=path)
    assert not result.ok
    assert "model parameter limit exceeded" in result.error


def test_fit_uses_requested_epochs(tmp_path):
    data = str(tmp_path / "data")
    train_harness.generate_dataset(data, seeds=(11,), frames_per_seed=4,
                                   episode_frames=8)
    model = PoseNet()
    report = train_harness.fit(model, data, epochs=2, batch_size=4, seed=0)
    assert report["epochs"] == 2
    assert report["batches"] == 6
    assert report["parameter_count"] == torch_contract.parameter_count(model)
    assert report["parameter_limit"] == spec.MODEL_MAX_PARAMETERS


def test_fit_trains_shape_head_from_caller_labels(tmp_path):
    data = str(tmp_path / "data")
    train_harness.generate_dataset(data, seeds=(11,), frames_per_seed=2,
                                   episode_frames=2)
    torch.manual_seed(0)
    model = PoseNet()
    before = model.shape_head.weight.detach().clone()
    report = train_harness.fit(model, data, epochs=2, batch_size=2,
                               shape_labeler=lambda rgb, depth: 1, device="cpu")
    assert report["shape_labeled_frames"] == 4
    assert not torch.equal(before, model.shape_head.weight)
    assert model.shape_head.weight.grad is not None


def test_reference_pseudo_labels_use_observations(frame):
    from harness.reference_labels import label_shape
    for shape_id, shape in enumerate(spec.BLOCK_SHAPES):
        example = episodes.single_frame(11, shapes=(shape,))
        assert label_shape(example.obs["rgb"], example.obs["depth"]) == shape_id
    assert label_shape(frame.obs["rgb"], np.full_like(frame.obs["depth"], np.nan)) == -1


@pytest.mark.heavy  # 20-epoch CPU training run
def test_smoke_train_learns(tmp_path):
    """A tiny CPU run must beat the untrained net — locks the whole path."""
    data = str(tmp_path / "data")
    train_harness.generate_dataset(data, seeds=(11, 23), frames_per_seed=10,
                                   episode_frames=15)
    val = [episodes.single_frame(s) for s in (901, 902, 903, 904)]

    def median_err(model):
        errs = []
        scripted_path = str(tmp_path / "m.pt")
        torch_contract.save_model(model, scripted_path)
        m = torch_contract.load_model(scripted_path)
        for f in val:
            x, y, th, shape_id = torch_contract.infer(m, f.obs["rgb"], f.obs["depth"])
            errs.append(np.hypot(x - f.gt[0], y - f.gt[1]))
        return float(np.median(errs))

    torch.manual_seed(0)
    untrained = PoseNet()
    e_untrained = median_err(untrained)

    torch.manual_seed(0)
    model = PoseNet()
    train_harness.fit(model, data, epochs=20, batch_size=8, lr=1e-3,
                      seed=0)
    e_trained = median_err(model)
    # CI load varies, so assert learning happened rather than a tight figure
    assert e_trained < e_untrained, \
        f"training didn't help: {e_trained:.3f} vs {e_untrained:.3f}"
    assert e_trained < 0.25, \
        f"smoke-train should approach mean-pose level, got {e_trained:.3f}"
