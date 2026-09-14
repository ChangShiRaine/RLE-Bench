"""RGB camera frames for observations and recordings."""
from PIL import Image


def render(env, camera, size=512):
    return Image.fromarray(env.sim.render(camera_name=camera, width=size, height=size)[::-1])
