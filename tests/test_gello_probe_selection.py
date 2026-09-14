"""The chosen-pose protocol spends exactly one sample plus fifteen holds."""
from pathlib import Path

import numpy as np
import pytest

from harness import codesign, spec
from harness.codesign_scorer import Submission, _sandbox_for_arm
from harness.codesign_oracle import CODESIGN_MODELS

ARM = str(Path(CODESIGN_MODELS) / 'franka/lead_sref.xml')


def _submission(tmp_path, planner):
    (tmp_path / 'trim.py').write_text('''import numpy as np

def make_trim(model):
    return lambda q: np.zeros(model.nv)

def adapt(model, probe):
    assert len(probe) == 16
    return len(probe)

def make_trim_adapted(model, params):
    return lambda q: np.full(model.nv, params)

''' + planner)
    return Submission(lead_xml=ARM, has_trim=True, has_adapt=True)


def _run(tmp_path, monkeypatch, planner):
    holds = []
    def hold(model, data, q, **kwargs):
        holds.append(np.asarray(q).copy())
        data.qpos[:] = q
    monkeypatch.setattr(codesign, 'hold_sim', hold)
    sub = _submission(tmp_path, planner)
    errors = []
    result = _sandbox_for_arm(str(tmp_path), ARM, sub, codesign.CodesignEnvelope(),
                              [129], errors, want_grid=False)
    return result, holds, errors


def test_selected_poses_and_exact_budget(tmp_path, monkeypatch):
    planner = '''def plan_probe(model, sample, lower, upper):
    # Receives only one observation and uses it to choose a batch.
    assert len(sample) == 2
    assert sample[0].shape == sample[1].shape == (model.nv,)
    return np.tile((lower + upper)/2 + .01, (15, 1))
'''
    result, holds, errors = _run(tmp_path, monkeypatch, planner)
    assert not errors and len(holds) == 16
    np.testing.assert_array_equal(holds[0], spec.LEAD_HOME)
    np.testing.assert_allclose(holds[1:], np.tile(np.asarray(spec.LEAD_HOME) + .01, (15, 1)))
    np.testing.assert_array_equal(result['cadapt']['129']['hold::129'], np.full((12, 7), 16))


@pytest.mark.parametrize('expression', [
    'np.tile(sample[0], (16, 1))',
    'np.tile(sample[0], (14, 1))',
    'np.zeros((15, model.nv + 1))',
    'np.full((15, model.nv), np.nan)',
    'np.full((15, model.nv), np.inf)',
    'np.tile(upper + 1, (15, 1))',
])
def test_invalid_batch_consumes_no_additional_measurements(tmp_path, monkeypatch, expression):
    planner = 'def plan_probe(model, sample, lower, upper):\n    return ' + expression + '\n'
    result, holds, errors = _run(tmp_path, monkeypatch, planner)
    assert len(holds) == 1
    assert errors and 'probe plan rejected' in errors[0]
    assert not result['cadapt']
    assert 'hold::129' in result['ctrim']  # Nominal fallback remains available.


def test_missing_planner_has_no_random_fallback(tmp_path, monkeypatch):
    result, holds, errors = _run(tmp_path, monkeypatch, '')
    assert len(holds) == 1 and errors and not result['cadapt']


def test_split_noise_matches_single_batch():
    env = codesign.CodesignEnvelope()
    model, data, _ = codesign.compose_instance(ARM, 129, env)
    q = np.tile(spec.LEAD_HOME, (16, 1))
    full = codesign.run_probe(model, data, 129, env, q)
    split = codesign.run_probe(model, data, 129, env, q[:1])
    split += codesign.run_probe(model, data, 129, env, q[1:], noise_offset=1)
    np.testing.assert_array_equal(full, split)
    with pytest.raises(ValueError, match='budget'):
        codesign.run_probe(model, data, 129, env, q, noise_offset=1)
