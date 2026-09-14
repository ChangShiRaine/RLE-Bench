"""Tilt-mounted pocket-cube variant with one public front camera."""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .scene import PocketCube

BASE_POSITIONS = ((-.40, .20, .86), (.40, .20, .86))
BASE_ROTATIONS = tuple(Rotation.from_euler('z', yaw) * Rotation.from_euler('y', -45, degrees=True)
                       for yaw in (0, np.pi))
CAMERA_POSITION = np.array([0., -.85, 1.40])
CAMERA_TARGET = np.array([0., 0., 1.055])
CAMERAS = ('front',)


class FrontPocketCube(PocketCube):
    approach_qpos = np.array([
        [-.9885232701565028, .7752956989407416, .23709192702087103, -2.4424220082144803,
         2.505069867115093, 2.4201439055468854, 2.116077456596067],
        [.3932303167604584, .7571527556234873, .20914707157905102, -2.4926345453604757,
         -2.531359452612828, 2.1966602089664073, 2.269144961918102],
    ])

    def __init__(self, **kwargs):
        kwargs['camera_names'] = list(CAMERAS)
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        root = self.model.root
        for site in root.iter('site'):
            site.set('rgba', '0 0 0 0')
        for geom in list(self.model.worldbody.findall('geom')):
            if geom.get('type') == 'cylinder' and geom.get('size') == '0.105 0.0125':
                self.model.worldbody.remove(geom)
        for parent in root.iter():
            for cam in list(parent.findall('camera')):
                parent.remove(cam)
        for i, (pos, rotation) in enumerate(zip(BASE_POSITIONS, BASE_ROTATIONS)):
            body = root.find(f".//body[@name='{self.robots[i].robot_model.root_body}']")
            body.set('pos', ' '.join(map(str, pos)))
            body.set('quat', ' '.join(map(str, rotation.as_quat(scalar_first=True))))
            ET.SubElement(self.model.worldbody, 'geom', name=f'tilt_mount_{i}', type='box',
                          pos=f'{pos[0]} {pos[1]} .805', size='.105 .105 .055',
                          rgba='.18 .22 .26 1', group='1')
        z = CAMERA_POSITION - CAMERA_TARGET
        z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z)
        x /= np.linalg.norm(x)
        quat = Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat(scalar_first=True)
        ET.SubElement(self.model.worldbody, 'camera', name='front',
                      pos=' '.join(map(str, CAMERA_POSITION)),
                      quat=' '.join(map(str, quat)), fovy='42')
