"""Check the assembled GELLO against the STL mounting holes.

    python -m dev.check_assembly --arm all

This checks kinematic seating, not collision clearance over the joint range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .assemble import ARMS, ASSETS, REFERENCE_POSES, MM, build
from .mesh_features import circles, load_stl


def check(directory, arm='franka_fer'):
    table, order = ARMS[arm]
    count = len(order)
    path = Path(build(arm, str(directory / f'{arm}.xml')))
    first = path.read_bytes(), path.with_name('servo_single.stl').read_bytes()
    build(arm, str(path))
    assert first == (path.read_bytes(), path.with_name('servo_single.stl').read_bytes())
    root = ET.parse(path).getroot()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nq == count + 1

    def cad_pose(name):
        if name.startswith('servo'):
            name = 'visual_' + name
        geom = model.geom(name)
        node = root.find(f'.//geom[@name="{name}"]')
        local = np.zeros(9)
        mujoco.mju_quat2Mat(local, np.fromstring(node.get('quat', '1 0 0 0'), sep=' '))
        body = geom.bodyid[0]
        parent = data.xmat[body].reshape(3, 3)
        return (parent @ local.reshape(3, 3),
                data.xpos[body] + parent @ np.fromstring(node.get('pos', '0 0 0'), sep=' '))

    def world(name, point):
        rotation, position = cad_pose(name)
        return position + rotation @ (np.asarray(point) * MM)

    errors, gaps = [], []
    horn_faces = 0

    def horn_face(source, center, axis):
        nonlocal horn_faces
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        found = circles(str(source), r_min=1, r_max=1.45, max_residual=.01)
        centers = set()
        for feature in found:
            delta = feature['centre'] - center
            if (abs(float(feature['axis'] @ axis)) > .999
                    and abs(float(delta @ axis)) < .03
                    and abs(np.linalg.norm(delta) - 6) < .03):
                centers.add(tuple(np.round(feature['centre'], 2)))
        assert len(centers) >= 3, (source, 'horn face', center, centers)
        horn_faces += 1

    def bolts(servo, printed, source, pattern, expected_gap):
        found = [c for c in circles(str(source), r_min=1, r_max=1.45, max_residual=.01)]
        rotation, _ = cad_pose(printed)
        axis = cad_pose(servo)[0][:, 2]
        found = [c for c in found if abs(float((rotation @ c['axis']) @ axis)) > .999]
        centers = np.array([world(printed, c['centre']) for c in found])
        assert len(centers), source
        for point in pattern:
            delta = centers - world(servo, point)
            closest = delta[np.argmin(np.linalg.norm(delta, axis=1))] / MM
            gap = abs(float(closest @ axis))
            error = np.linalg.norm(closest - (closest @ axis) * axis)
            assert error < .03, (servo, point, 'lateral', error)
            assert abs(gap - expected_gap) < .03, (servo, point, 'gap', gap)
            errors.append(float(error))
            gaps.append(float(gap))

    for i, parent in enumerate(('base',) + order[:-1], start=1):
        reverse = table[order[i-1]]['prox_kind'] == 'case'
        printed = 'printed_base' if i == 1 else f'printed_link{i-1}'
        if reverse:
            parent, printed = order[i-1], f'printed_link{i}'
        clamped = table[order[i-1]]['prox_kind'] == 'fork'
        pattern = ([(x, 7.5, z) for x in (-8, 8) for z in (-19.5, 3.5)]
                   if clamped else [(x, y, -19.5) for x in (-8, 8) for y in (-22.5, 7.5)])
        if arm == 'xarm7' and parent == 'L4':
            # All four bore rims are resolved on the back of this 2.3 mm plate.
            pattern = [(x, y, z - 2.3) for x, y, z in pattern]
        bolts(f'servo{i}', printed, Path(ASSETS) / arm / f'{parent}.STL',
              pattern, .25 if clamped else 0)
        expected_parent = model.body('lead_base' if i == 1 else f'lead_link{i-1}').id
        if reverse:
            expected_parent = model.body(f'lead_link{i}').id
        assert model.geom(f'servo{i}').bodyid[0] == expected_parent

    for i, part in enumerate(order):
        spec = table[part]
        source = Path(ASSETS) / arm / spec.get('mesh', f'{part}.STL')
        center = np.asarray(spec['prox'])
        axis = np.asarray(spec['prox_axis'])
        if spec['prox_kind'] == 'case':
            parent = table[order[i-1]]
            source = Path(ASSETS) / arm / f'{order[i-1]}.STL'
            axis = np.asarray(parent['dist_axis'])
            center = np.asarray(parent['dist']) - 14.5 * axis
        offsets = (-14.75, 14.75) if spec['prox_kind'] == 'fork' else (0,)
        for offset in offsets:
            horn_face(source, center + offset * axis, axis)
    horn_face(Path(ASSETS) / 'gripper/trigger.STL', np.array([0, 8, 0]), (0, 1, 0))

    # Three exposed screw holes fix the grip case; the fourth corner is open.
    bolts('servo_trigger', f'printed_link{count}', Path(ASSETS) / 'gripper/handle.STL',
          [(8, -22.5, -19.5), (8, 7.5, -19.5), (-8, 7.5, -19.5)], 0)
    assert model.geom('servo_trigger').bodyid[0] == model.body(f'lead_link{count}').id

    for pose in (np.zeros(count+1), np.asarray(REFERENCE_POSES[arm]),
                 np.array([17, -43, 21, 115, -32, 138, 61, 20])[:count+1]):
        data.qpos[:] = np.deg2rad(pose)
        mujoco.mj_forward(model, data)
        for i, part in enumerate(order, start=1):
            spec = table[part]
            reverse = spec['prox_kind'] == 'case'
            midpoint = -19.5 if reverse else -8 if spec['prox_kind'] == 'fork' else 6.5
            np.testing.assert_allclose(world(f'servo{i}', (0, 0, midpoint)),
                                       world(f'printed_link{i}', spec['prox']), atol=1e-6)
            a = cad_pose(f'servo{i}')[0][:, 2]
            b = cad_pose(f'printed_link{i}')[0] @ np.asarray(spec['prox_axis'])
            if reverse:
                b = -b
                np.testing.assert_allclose(world(f'servo{i}', (0, 0, -8)),
                                           world(f'printed_link{i-1}', table[order[i-2]]['dist']),
                                           atol=1e-6)
            np.testing.assert_allclose(a, b, atol=2e-6)
        np.testing.assert_allclose(world('servo_trigger', (0, 0, 6.5)),
                                   world('printed_trigger', (0, 8, 0)), atol=1e-6)
    for i in range(count+1):
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)
        servo = f'servo{i+1}' if i < count else 'servo_trigger'
        before = data.geom_xpos[model.geom(servo).id].copy()
        orientation = data.geom_xmat[model.geom(servo).id].copy()
        data.qpos[i] = .37
        mujoco.mj_forward(model, data)
        if i < count and table[order[i]]['prox_kind'] == 'case':
            # A reversed motor's case rotates with the child around a fixed horn axis.
            assert not np.allclose(data.geom_xmat[model.geom(servo).id], orientation)
            continue
        np.testing.assert_allclose(data.geom_xpos[model.geom(servo).id], before, atol=1e-12)
        np.testing.assert_allclose(data.geom_xmat[model.geom(servo).id], orientation, atol=1e-12)

    # Flat mounts cannot contain the optional rear idler penetrating the plate.
    single = load_stl(path.with_name('servo_single.stl'))
    assert single[:, :, 2].min() >= -19.5001
    assert single[:, :, 2].max() == 6.5
    return dict(arm=arm, dof=model.nq, checked_case_screws=len(errors), checked_horn_faces=horn_faces,
                max_lateral_error_mm=max(errors), max_mount_gap_mm=max(gaps),
                shaft_poses_checked=3, case_ownership_checks=count+1, deterministic=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(ARMS) + ('all',), default='franka_fer')
    args = parser.parse_args()
    with TemporaryDirectory(prefix='gello-assembly-') as temporary:
        for arm in ARMS if args.arm == 'all' else (args.arm,):
            print(json.dumps(check(Path(temporary), arm), indent=2))


if __name__ == '__main__':
    main()
