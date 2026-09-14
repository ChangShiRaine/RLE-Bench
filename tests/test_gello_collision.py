"""Analytic interference fixtures and validity scoring integration."""
import itertools

import mujoco
import numpy as np
import pytest

from harness import collision, spec
from harness.codesign_checkpoints import CHECKPOINTS_CD, evaluate_codesign
from harness.codesign_oracle import CODESIGN_MODELS
from harness.scenarios import compose_lead


def _ring_xml():
    vertices, faces = [], []
    cube = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    triangles = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                          [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    for half, pos in [([.01, .05, .05], [-.04, 0, 0]),
                      ([.01, .05, .05], [.04, 0, 0]),
                      ([.03, .01, .05], [0, -.04, 0]),
                      ([.03, .01, .05], [0, .04, 0])]:
        faces.extend(triangles + len(vertices))
        vertices.extend(cube * half + pos)
    v = ' '.join(map(str, np.asarray(vertices).ravel()))
    f = ' '.join(map(str, np.asarray(faces).ravel()))
    return f'''<mujoco><asset><mesh name="ring" vertex="{v}" face="{f}"/></asset>
    <worldbody><geom name="stock" type="mesh" mesh="ring" contype="0" conaffinity="0"/>
    <body name="moving"><joint type="slide" axis="1 0 0"/>
    <geom name="addition" type="sphere" size=".01" contype="0" conaffinity="0"/>
    </body></worldbody><contact><exclude body1="world" body2="moving"/></contact></mujoco>'''


@pytest.fixture
def ring(monkeypatch):
    monkeypatch.setattr(spec, 'LEAD_GEOMETRY', {'supplied_geoms': {'stock': 'ring'},
                                              'stock_geoms': {}})
    return mujoco.MjModel.from_xml_string(_ring_xml())


def test_concave_hole_and_crossing_without_mujoco_contacts(ring, monkeypatch):
    # No contact simulation, convex distance, or dynamics may run here.
    def forbidden(*args, **kwargs):
        raise AssertionError('contact dynamics used')
    for name in ('mj_step', 'mj_forward', 'mj_collision', 'mj_geomDistance'):
        monkeypatch.setattr(mujoco, name, forbidden)
    masks = ring.geom_contype.copy(), ring.geom_conaffinity.copy()
    paths = [(1, [[0.0], [.008]]), (2, [[.0], [.035], [.1]]), (3, [[.1]])]
    result = collision.motion_clearance(ring, paths)
    assert result['n_checks'] == 3 and result['n_failed'] == 1
    assert not result['problems']
    assert [t['ok'] for t in result['trajectories']] == [True, False, True]
    witness = result['trajectories'][1]['collisions'][0]
    assert witness['sample'] == 1
    assert witness['geoms'] == ['stock', 'addition']
    assert collision.motion_clearance(ring, paths) == result
    np.testing.assert_array_equal(ring.geom_contype, masks[0])
    np.testing.assert_array_equal(ring.geom_conaffinity, masks[1])


def test_same_rigid_component_mounting_is_excluded(monkeypatch):
    monkeypatch.setattr(spec, 'LEAD_GEOMETRY', {'supplied_geoms': {'stock': None},
                                              'stock_geoms': {}})
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint/><geom name="stock" size=".02"/>
        <body><geom name="new_mount" size=".01"/></body>
      </body></worldbody></mujoco>''')
    result = collision.motion_clearance(model, [(1, [[0.]])])
    assert result['ok'] and result['n_pairs'] == 0


def test_unnamed_and_added_added_pairs_are_checked(monkeypatch):
    monkeypatch.setattr(spec, 'LEAD_GEOMETRY', {'supplied_geoms': {}, 'stock_geoms': {}})
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom size=".02"/><body><joint/><geom size=".02"/></body>
      </worldbody></mujoco>''')
    result = collision.motion_clearance(model, [(1, [[0.]])])
    assert result['n_failed'] == 1
    assert result['trajectories'][0]['collisions'][0]['geom_ids'] == [0, 1]


def test_uncheckable_shape_loses_only_motion_credit(monkeypatch):
    monkeypatch.setattr(spec, 'LEAD_GEOMETRY', {'supplied_geoms': {}, 'stock_geoms': {}})
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/><body><joint/><geom size=".02"/></body>
      </worldbody></mujoco>''')
    report = collision.motion_clearance(model, [(1, [[0.]])])
    assert not report['ok'] and report['n_failed'] == 1 and report['problems']


def test_checkpoint_is_fraction_of_clear_paths_and_not_gate():
    score = evaluate_codesign({'motion_clearance': {'n_checks': 3, 'n_failed': 1}},
                             1, {}, {})
    assert score['C1.motion_clearance'] == pytest.approx(2 / 3)
    checkpoint = next(c for c in CHECKPOINTS_CD if c.id == 'C1.motion_clearance')
    assert not checkpoint.gate
    assert checkpoint.weight == .10
    assert sum(c.weight for c in CHECKPOINTS_CD if c.stage == "validity") == pytest.approx(.10)
    assert sum(c.weight for c in CHECKPOINTS_CD) == pytest.approx(1)


def test_reference_has_no_new_interference():
    model, _ = compose_lead(str(CODESIGN_MODELS) + '/franka/lead_sref.xml', perturb=False)
    report = collision.motion_clearance(model)
    assert report['ok'] and report['n_failed'] == 0
    assert len(report['trajectories']) == 3
    assert all(t['samples'] == 241 for t in report['trajectories'])
