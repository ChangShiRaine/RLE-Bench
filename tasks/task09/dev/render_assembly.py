"""Render the measured GELLO assembly and its mating interfaces.

    MUJOCO_GL=egl python -m dev.render_assembly --arm ur5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .assemble import ARMS, REFERENCE_POSES, build

COLORS = [(0.58, 0.65, 0.75), (1.0, 0.55, 0.08), (0.2, 0.7, 0.3),
          (0.9, 0.25, 0.25), (0.2, 0.45, 0.9), (0.95, 0.8, 0.1),
          (0.65, 0.3, 0.75), (0.1, 0.75, 0.75)]


def load_model(path):
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((Path(path).resolve().parent / compiler.get('meshdir', '')).resolve()))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='900', offheight='900')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5',
                  diffuse='.65 .65 .65', specular='.2 .2 .2')
    ET.SubElement(root.find('asset'), 'texture', type='skybox', builtin='gradient',
                  rgb1='.95 .96 .98', rgb2='.82 .85 .89', width='512', height='3072')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def sheet(renderer, model, data, views, path, columns=2, size=700):
    canvas = Image.new('RGB', (size * columns, (size + 32) * ((len(views)+columns-1)//columns)), 'white')
    draw = ImageDraw.Draw(canvas)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    for i, (label, center, span, azimuth, elevation, visible) in enumerate(views):
        rgba = model.geom_rgba.copy()
        if visible is not None:
            for g in range(model.ngeom):
                if model.geom(g).name not in visible:
                    model.geom_rgba[g, 3] = 0
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = center
        cam.orthographic = 1
        cam.distance = span / (2 * np.tan(np.deg2rad(model.vis.global_.fovy / 2)))
        cam.azimuth, cam.elevation = azimuth, elevation
        renderer.update_scene(data, camera=cam, scene_option=options)
        tile = Image.fromarray(renderer.render()).resize((size, size))
        model.geom_rgba[:] = rgba
        x, y = (i % columns) * size, (i // columns) * (size + 32)
        canvas.paste(tile, (x, y + 32))
        draw.text((x + 12, y + 10), label, fill='black')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(ARMS), default='franka_fer')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--q', type=float, nargs='+',
                        help='Arm joint angles in degrees')
    args = parser.parse_args()
    count = len(ARMS[args.arm][1])
    if args.q is None:
        args.q = REFERENCE_POSES[args.arm][:-1]
    if len(args.q) != count:
        parser.error(f'{args.arm} requires {count} arm joint angles')
    if args.out is None:
        args.out = Path('renders') if args.arm == 'franka_fer' else Path('renders') / args.arm
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(build(args.arm, str(args.out / f'{args.arm}.xml')))
    data = mujoco.MjData(model)
    data.qpos[:count] = np.deg2rad(args.q)
    mujoco.mj_forward(model, data)
    for i, color in enumerate(COLORS[:count+1]):
        model.geom('printed_base' if i == 0 else f'printed_link{i}').rgba[:] = [*color, 1]
    model.geom(f'printed_link{count}').rgba[:] = [*COLORS[-1], 1]
    model.geom('printed_trigger').rgba[:] = [.1, .55, .6, 1]
    points = []
    for g in range(model.ngeom):
        mesh = model.geom_dataid[g]
        start, vertices = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points.append(model.mesh_vert[start:start+vertices] @ data.geom_xmat[g].reshape(3, 3).T
                      + data.geom_xpos[g])
    points = np.vstack(points)
    center = (points.min(0) + points.max(0)) / 2
    span = np.linalg.norm(np.ptp(points, axis=0)) * 1.12
    front, side, angled = (90, 0, 60) if args.arm == 'xarm7' else (0, 90, 135)
    views = [(label, center, span, az, el, None) for label, az, el in
             [('Front', front, 0), ('Side', side, 0), ('Three quarter', angled, -20), ('Top', 90, -90)]]
    with mujoco.Renderer(model, height=900, width=900) as renderer:
        sheet(renderer, model, data, views, args.out / '14_corrected_assembly.png')
        joints = []
        for i in range(1, count+1):
            parent = 'printed_base' if i == 1 else f'printed_link{i-1}'
            visible = {parent, f'printed_link{i}', f'servo{i}'}
            joint = model.joint(f'lead_joint{i}').id
            for suffix, az in [('A', 135), ('B', -45)]:
                joints.append((f'J{i} / {suffix}', data.xanchor[joint], .15,
                               az, -20, visible))
        sheet(renderer, model, data, joints, args.out / '15_corrected_joints.png', size=450)
        grip = [(label, data.xpos[model.body(f'lead_link{count}').id], .2, az, el,
                 {f'printed_link{count}', 'printed_trigger', 'servo_trigger',
                  f'servo{count}', f'printed_link{count-1}'})
                for label, az, el in [('Grip A', 135, -20), ('Grip B', -45, -20)]]
        sheet(renderer, model, data, grip, args.out / '16_corrected_grip.png')
        for g in range(model.ngeom):
            if model.geom(g).name.startswith('printed_'):
                model.geom_rgba[g] = [.88, .84, .71, 1]
        sheet(renderer, model, data, [views[0], views[2]], args.out / '17_reference_pose.png')
    print('q (degrees):', args.q)
    for i in range(1, count+1):
        print(f'J{i}:', np.round(data.xanchor[model.joint(f'lead_joint{i}').id] * 1000, 2))
    print(args.out / '14_corrected_assembly.png')


if __name__ == '__main__':
    main()
