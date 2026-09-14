"""Verifier-only realism checks for task09."""

import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

import numpy as np

from harness.codesign import (CodesignEnvelope, compose_instance,
                                      device_buildable)
from harness.codesign_oracle import CODESIGN_MODELS
from harness.scenarios import compose_lead
from harness.validity import density_report


ORACLE = CODESIGN_MODELS + "/franka/lead_oracle.xml"


def test_nearly_massless_printed_material_is_rejected(tmp_path):
    shutil.copytree(Path(ORACLE).parent / "meshes", tmp_path / "meshes")
    tree = ET.parse(ORACLE)
    target = None
    for geom in tree.getroot().iter("geom"):
        if geom.get("name") == "printed_link4":
            target = geom
            break
    assert target is not None
    target.set("mass", "0.001")
    path = tmp_path / "near_empty_print.xml"
    tree.write(path)

    report = density_report(str(path))
    assert not report["ok"]
    assert any("too low" in problem for problem in report["problems"])


def test_disconnected_printed_pieces_are_rejected(tmp_path):
    shutil.copytree(Path(ORACLE).parent / "meshes", tmp_path / "meshes")
    tree = ET.parse(ORACLE)
    parent = None
    target = None
    for body in tree.getroot().iter("body"):
        candidate = body.find("./geom[@name='printed_link2']")
        if candidate is not None:
            parent = body
            target = candidate
            break
    assert parent is not None and target is not None

    target.attrib.pop("fromto", None)
    target.attrib.pop("mesh", None)
    target.set("type", "sphere")
    target.set("size", "0.02")
    target.set("pos", "0 0 0")
    ET.SubElement(
        parent,
        "geom",
        name="printed_link2_endpoint",
        type="sphere",
        size="0.02",
        pos=parent.find("body").get("pos"),
        mass="0.01",
    )
    path = tmp_path / "disconnected_print.xml"
    tree.write(path)

    report = device_buildable(str(path))
    assert not report["ok"]
    assert any("not continuous" in problem for problem in report["problems"])


def test_physical_copies_vary_link_masses_independently():
    nominal, _ = compose_lead(ORACLE, seed=0, perturb=False)
    instance, _, _ = compose_instance(
        ORACLE, seed=311, env=CodesignEnvelope(), payload=False)

    ratios = []
    for index in range(1, 8):
        body_name = f"lead_link{index}"
        body_id = nominal.body(body_name).id
        ratios.append(instance.body_mass[body_id] / nominal.body_mass[body_id])

    assert np.ptp(ratios) > 0.01
