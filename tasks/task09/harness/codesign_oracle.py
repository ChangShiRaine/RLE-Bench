"""Build the pinned task09 oracle submission from generated variant assets."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import types

CODESIGN_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets", "gello_codesign")


def reference_trim_source(variant: str | None = None) -> str:
    """Return the selected variant's portable calibration module."""
    from . import spec as gspec
    return (Path(CODESIGN_MODELS) / (variant or gspec.VARIANT_NAME)
            / "trim_oracle.py").read_text()


def reference_trim_module(variant: str | None = None) -> types.ModuleType:
    """The shipped trim.py text, importable."""
    mod = types.ModuleType("codesign_reference_trim")
    exec(compile(reference_trim_source(variant), "trim.py", "exec"), mod.__dict__)
    return mod


def build_reference_submission(outdir: str, lead_xml: str | None = None,
                               variant: str | None = None) -> str:
    """Write the full oracle submission, optionally substituting its hardware."""
    from . import spec as gspec
    variant = variant or gspec.VARIANT_NAME
    model_dir = os.path.join(CODESIGN_MODELS, variant)
    lead_src = lead_xml or os.path.join(model_dir, "lead_oracle.xml")
    os.makedirs(outdir, exist_ok=True)
    shutil.copy(lead_src, os.path.join(outdir, "lead.xml"))
    mesh_src = os.path.join(os.path.dirname(lead_src), "meshes")
    if not os.path.isdir(mesh_src):
        mesh_src = os.path.join(model_dir, "meshes")
    shutil.copytree(mesh_src, os.path.join(outdir, "meshes"), dirs_exist_ok=True)
    (Path(outdir) / "trim.py").write_text(reference_trim_source(variant))
    return outdir
