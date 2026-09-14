"""Getting an observation onto disk, so it can be looked at.

A PERCEPTION PRIMITIVE is a pure function of an observation:

    obs  ->  a fact about this episode

Where the pan is, whether the gripper closed on something, whether the arm is still
moving. Nothing here implements one -- they are yours to write. This module exists for
the one thing a primitive cannot do for itself: an image has to be a file before anyone
can look at it.

There is no pretrained detector in this image and no network, so open-vocabulary
perception is either a rule you fit yourself or your own reading of a saved frame. The
second needs `save_view`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_view(obs: dict, directory: str) -> dict[str, str]:
    """Write every image in `obs` as PNG and every depth map as `.npy`.

    Returns {key: path}. Images are already the right way up. Depth stays a raw array
    because it is metres to be measured, not a picture to be looked at.
    """
    from PIL import Image

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for key, value in obs.items():
        if key.endswith("_image"):
            path = out / f"{key}.png"
            Image.fromarray(np.asarray(value, dtype=np.uint8)).save(path)
        elif key.endswith("_depth"):
            path = out / f"{key}.npy"
            np.save(path, np.asarray(value, dtype=np.float32))
        else:
            continue
        written[key] = str(path)
    return written
