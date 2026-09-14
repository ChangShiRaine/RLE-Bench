#!/usr/bin/env python3
"""Build task08's aluminum-profile reference from the golden robot.

The Franka, wheels, battery, actuators, and interfaces are retained. Only the
monolithic chassis and mounting puck are replaced by members cut from the
public 4040 stock profiles.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tasks" / "task08" / "harness" / "assets" / "golden" / "robot.xml"
OUTPUT = ROOT / "tasks" / "task08" / "reference" / "robot.xml"
CATALOG = (
    ROOT / "tasks" / "task08" / "harness" / "assets" / "mobile_manipulator_components"
    / "component_catalog.json"
)


def _profile_mass(profile: str, length_m: float) -> float:
    catalog = json.loads(CATALOG.read_text())
    densities = {
        row["name"]: row["linear_density_kg_m"]
        for row in catalog["aluminum_profiles"]
    }
    return densities[profile] * length_m


def _geom(name: str, profile: str, size: tuple[float, float, float],
          pos: tuple[float, float, float], length_m: float) -> ET.Element:
    return ET.Element(
        "geom",
        name=name,
        type="box",
        size=" ".join(f"{value:.6g}" for value in size),
        pos=" ".join(f"{value:.6g}" for value in pos),
        mass=f"{_profile_mass(profile, length_m):.6g}",
        rgba="0.72 0.74 0.78 1",
    )


def _geom_between(name: str, start: tuple[float, float, float],
                  end: tuple[float, float, float]) -> ET.Element:
    """Return a 4040 box profile whose local x-axis joins two frame nodes."""
    vector = tuple(b - a for a, b in zip(start, end))
    length = math.sqrt(sum(value * value for value in vector))
    direction = tuple(value / length for value in vector)
    midpoint = tuple((a + b) / 2 for a, b in zip(start, end))
    dot = direction[0]
    if dot < -1.0 + 1e-12:
        quat = (0.0, 0.0, 0.0, 1.0)
    else:
        qw = math.sqrt((1.0 + dot) / 2.0)
        scale = 1.0 / (2.0 * qw)
        quat = (qw, 0.0, -direction[2] * scale,
                direction[1] * scale)
    geom = _geom(name, "4040", (length / 2, 0.02, 0.02),
                 midpoint, length)
    geom.set("quat", " ".join(f"{value:.9g}" for value in quat))
    return geom


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "task08 aluminum profile reference")
    root.find("compiler").set(
        "meshdir",
        os.path.relpath(ROOT / "assets" / "robots"
                        / "franka_emika_panda" / "assets", OUTPUT.parent))
    base = root.find(".//body[@name='base']")
    if base is None:
        raise RuntimeError("golden robot has no base body")

    for child in list(base):
        if child.tag == "geom" and child.get("name") in {
                "chassis", "arm_mount"}:
            base.remove(child)

    # Sparse three-dimensional space frame. The 4040 axle members pass
    # exactly through all four drive-joint centers; perimeter rails, diagonal
    # braces, and a compact pedestal carry the battery and Franka without a
    # solid or densely packed deck.
    members = [
        # Upper perimeter and arm crossmembers (z = 85 mm).
        _geom("profile_4040_upper_left", "4040", (0.245, 0.02, 0.02),
              (0.0, 0.188, 0.085), 0.49),
        _geom("profile_4040_upper_right", "4040", (0.245, 0.02, 0.02),
              (0.0, -0.188, 0.085), 0.49),
        _geom("profile_4040_upper_front", "4040", (0.02, 0.188, 0.02),
              (0.245, 0.0, 0.085), 0.376),
        _geom("profile_4040_upper_rear", "4040", (0.02, 0.188, 0.02),
              (-0.245, 0.0, 0.085), 0.376),
        _geom("profile_4040_arm_cross_front", "4040", (0.02, 0.188, 0.02),
              (0.06, 0.0, 0.085), 0.376),
        _geom("profile_4040_arm_cross_rear", "4040", (0.02, 0.188, 0.02),
              (-0.06, 0.0, 0.085), 0.376),
        # Lower ladder frame at wheel-center height.
        _geom("profile_4040_lower_left", "4040", (0.203, 0.02, 0.02),
              (0.0, 0.188, 0.0), 0.406),
        _geom("profile_4040_lower_right", "4040", (0.203, 0.02, 0.02),
              (0.0, -0.188, 0.0), 0.406),
        _geom("profile_4040_axle_front", "4040", (0.02, 0.243, 0.02),
              (0.203, 0.0, 0.0), 0.486),
        _geom("profile_4040_axle_rear", "4040", (0.02, 0.243, 0.02),
              (-0.203, 0.0, 0.0), 0.486),
        # A middle perimeter turns the two ladders into a torsion box while
        # leaving the center open for components and cabling.
        _geom("profile_4040_mid_left", "4040", (0.245, 0.02, 0.02),
              (0.0, 0.188, 0.04), 0.49),
        _geom("profile_4040_mid_right", "4040", (0.245, 0.02, 0.02),
              (0.0, -0.188, 0.04), 0.49),
        _geom("profile_4040_mid_front", "4040", (0.02, 0.188, 0.02),
              (0.245, 0.0, 0.04), 0.376),
        _geom("profile_4040_mid_rear", "4040", (0.02, 0.188, 0.02),
              (-0.245, 0.0, 0.04), 0.376),
        # Two low rails form a physical battery cradle, not ballast blocks.
        _geom("profile_4040_battery_cradle_left", "4040",
              (0.183, 0.02, 0.02), (0.0, 0.11, -0.02), 0.366),
        _geom("profile_4040_battery_cradle_right", "4040",
              (0.183, 0.02, 0.02), (0.0, -0.11, -0.02), 0.366),
    ]
    # X braces on both side faces resist pitch and carry arm overturning loads.
    for y in (-0.188, 0.188):
        members.append(_geom_between(
            f"profile_4040_side_brace_a_{y:+.3f}",
            (-0.203, y, 0.0), (0.245, y, 0.085)))
        members.append(_geom_between(
            f"profile_4040_side_brace_b_{y:+.3f}",
            (0.203, y, 0.0), (-0.245, y, 0.085)))
    # Alternating front/rear face diagonals balance lateral stiffness and mass.
    members.append(_geom_between(
        "profile_4040_front_face_brace",
        (0.203, -0.243, 0.0), (0.245, 0.188, 0.085)))
    members.append(_geom_between(
        "profile_4040_rear_face_brace",
        (-0.203, 0.243, 0.0), (-0.245, -0.188, 0.085)))
    for x in (-0.203, 0.203):
        for y in (-0.188, 0.188):
            members.append(_geom(
                f"profile_4040_riser_{x:+.3f}_{y:+.3f}", "4040",
                (0.02, 0.02, 0.0425), (x, y, 0.0425), 0.085))
    # Four posts and three short rails form a compact arm pedestal whose top
    # mounting plane is 285 mm above the base origin.
    for x in (-0.06, 0.06):
        for y in (-0.06, 0.06):
            members.append(_geom(
                f"profile_4040_pedestal_{x:+.2f}_{y:+.2f}", "4040",
                (0.02, 0.02, 0.07), (x, y, 0.175), 0.14))
    members.extend([
        _geom("profile_4040_pedestal_front", "4040", (0.02, 0.08, 0.02),
              (0.06, 0.0, 0.265), 0.16),
        _geom("profile_4040_pedestal_rear", "4040", (0.02, 0.08, 0.02),
              (-0.06, 0.0, 0.265), 0.16),
        _geom("profile_4040_pedestal_bridge", "4040", (0.06, 0.02, 0.02),
              (0.0, 0.0, 0.265), 0.12),
    ])
    insert_at = next(
        index + 1 for index, child in enumerate(base)
        if child.tag == "joint" and child.get("name") == "base_free")
    for member in reversed(members):
        base.insert(insert_at, member)

    for suffix in ("FL", "FR", "RL", "RR"):
        wheel = base.find(f"body[@name='wheel_{suffix}']")
        x = 0.203 if suffix[0] == "F" else -0.203
        y = 0.243 if suffix[1] == "L" else -0.243
        wheel.set("pos", f"{x:.3f} {y:.3f} 0")

    base.find("site[@name='arm_mount_site']").set("pos", "0 0 0.285")
    base.find("body[@name='arm_link0']").set("pos", "0 0 0.285")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
