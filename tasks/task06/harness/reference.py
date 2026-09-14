"""Reference and probe submission assembly for the task06 family.

Code-task submissions contain only the estimator and the private helper
package it needs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

_PKG_FILES = ("spec.py", "footprint.py", "sensor.py",
              "oracle_cv.py", "oracle_geom.py", "oracle_fused.py", "oracle_unknown.py",
              "baselines.py", "reference_labels.py")

_ENTRY = {
    "a": "from rbp.oracle_cv import make_estimator\n",
    "b": "from rbp.oracle_fused import make_estimator\n",
    "d": "from rbp.oracle_fused import make_estimator\n",
}

_PROBE_ENTRY = {
    "naive_color": ("from rbp.baselines import NaiveColorEstimator\n"
                    "def make_estimator():\n    return NaiveColorEstimator()\n"),
    "raw_icp": ("from rbp.baselines import RawICPEstimator\n"
                "def make_estimator():\n    return RawICPEstimator()\n"),
    "last_pose": ("from rbp.baselines import LastPoseEstimator\n"
                  "def make_estimator():\n    return LastPoseEstimator()\n"),
}


def _stage_package(entry: str, out_dir: str) -> Path:
    out = Path(out_dir)
    pkg = out / "rbp"
    pkg.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    for name in _PKG_FILES:
        shutil.copy(src / name, pkg / name)
    (pkg / "__init__.py").write_text("")
    adapter = '''
from rbp.spec import BLOCK_SHAPES
class Estimator:
    def __init__(self): self.inner = _make_pose_estimator()
    def reset(self): self.inner.reset()
    def update(self, **obs):
        pose = self.inner.update(**obs)
        shape = getattr(self.inner, "shape", "tshape")
        shape_id = BLOCK_SHAPES.index(shape) if shape in BLOCK_SHAPES else 0
        return (*pose, shape_id)
def make_estimator(): return Estimator()
'''
    entry = entry.replace("import make_estimator", "import make_estimator as _make_pose_estimator")
    entry = entry.replace("def make_estimator():", "def _make_pose_estimator():")
    (out / "estimator.py").write_text(entry + adapter)
    return out


def build_probe_submission(kind: str, out_dir: str) -> Path:
    if kind not in _PROBE_ENTRY:
        raise ValueError(f"unknown probe {kind!r}")
    return _stage_package(_PROBE_ENTRY[kind], out_dir)


def build_reference_submission(variant: str, out_dir: str) -> Path:
    """Assemble a self-contained code reference submission."""
    if variant not in _ENTRY:
        raise ValueError(f"no code reference for variant {variant!r}")
    return _stage_package(
        "# Reference estimator (self-contained package)\n" + _ENTRY[variant],
        out_dir)
