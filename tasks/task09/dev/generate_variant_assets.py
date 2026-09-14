"""Generate portable task09 models from measured GELLO CAD assemblies."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import linprog

from . import assemble, servo_proxy
from harness.variants import VARIANTS, GelloVariant

ROOT = str(Path(__file__).resolve().parents[1] / "harness/assets/gello_codesign")
CW_DENSITY = 8000.0
ARM_NAMES = {"franka": "franka_fer", "ur5e": "ur5", "xarm7": "xarm7"}
# Effective density under the pinned MuJoCo CAD inertia approximation.
BARE_DENSITY = 1000.0
SREF_DENSITY = 400.0
ORACLE_DENSITY = 248.0

def _vec(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _workspace_grid(variant: GelloVariant, levels: int = 3) -> np.ndarray:
    home = np.asarray(variant.home)
    halfwidth = np.asarray(variant.workspace_halfwidth)
    axes = [np.asarray([home[0]])]
    axes.extend(home[i] + np.linspace(-halfwidth[i], halfwidth[i], levels)
                for i in range(1, variant.n_joints))
    return np.asarray(list(itertools.product(*axes)), dtype=float)


def _cheb_fit(y: np.ndarray, x: np.ndarray, slope_min: float = -20.0):
    def error(slope):
        residual = y - slope * x
        return 0.5 * (residual.max() - residual.min())

    lo, hi = slope_min, 0.0
    for _ in range(60):
        one = lo + (hi - lo) / 3.0
        two = hi - (hi - lo) / 3.0
        if error(one) < error(two):
            hi = two
        else:
            lo = one
    slope = 0.5 * (lo + hi)
    if error(0.0) <= error(slope) + 1e-12:
        slope = 0.0
    residual = y - slope * x
    return slope, 0.5 * (residual.max() + residual.min())


def _fit_springs(path: str, variant: GelloVariant) -> dict[int, tuple[float, float]]:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    grid = _workspace_grid(variant)
    gravity = []
    for q in grid:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        gravity.append(np.asarray(data.qfrc_bias).copy())
    gravity = np.asarray(gravity)
    springs = {}
    for index in range(variant.n_joints):
        slope, intercept = _cheb_fit(gravity[:, index], grid[:, index])
        stiffness = -slope
        if stiffness < 1e-9 and abs(intercept) < 5e-3:
            continue
        if stiffness < 0.01:
            stiffness = 0.01
            shifted = gravity[:, index] + stiffness * grid[:, index]
            intercept = 0.5 * (shifted.max() + shifted.min())
        springs[index + 1] = (stiffness, intercept / stiffness)
    return springs


def _counterweight_radius(mass: float) -> float:
    return (3.0 * mass / (4.0 * math.pi * CW_DENSITY)) ** (1.0 / 3.0)



def _write(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def _counterweight_fit(path, variant):
    """Minimax gravity fit in mass/first-moment and linear-spring variables."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    bodies = [model.body(f"lead_link{i}").id
              for i in range(2, variant.n_joints)]
    grid = _workspace_grid(variant)
    n, nb = variant.n_joints, len(bodies)
    # Four variables per CW: mass and its three first moments in body frame.
    A = np.zeros((len(grid) * n, 4 * nb + 2 * n + 1))
    gravity = np.zeros(len(grid) * n)
    for row, q in enumerate(grid):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        sl = slice(row*n, (row+1)*n)
        gravity[sl] = data.qfrc_bias
        for bindex, body in enumerate(bodies):
            p, R = data.xpos[body], data.xmat[body].reshape(3, 3)
            jp, jr = np.zeros((3, n)), np.zeros((3, n))
            mujoco.mj_jac(model, data, jp, jr, p, body)
            origin = 9.81 * jp[2]
            A[sl, 4*bindex] = origin
            for axis in range(3):
                mujoco.mj_jac(model, data, jp, jr, p + R[:, axis], body)
                A[sl, 4*bindex+axis+1] = 9.81 * jp[2] - origin
        A[sl, 4*nb:4*nb+n] = np.diag(q)
        A[sl, 4*nb+n:4*nb+2*n] = -np.eye(n)
    A[:, -1] = -1
    negative = -A.copy()
    negative[:, -1] = -1
    extra, rhs = [], []
    for b in range(nb):
        for axis in range(1, 4):
            for sign in (-1, 1):
                row = np.zeros(A.shape[1])
                row[4*b] = -0.12
                row[4*b+axis] = sign
                extra.append(row)
                rhs.append(0)
    row = np.zeros(A.shape[1])
    row[:4*nb:4] = 1
    extra.append(row)
    rhs.append(2.30 - mujoco.mj_getTotalmass(model))
    c = np.zeros(A.shape[1])
    c[-1] = 1
    c[:4*nb:4] = 0.005
    bounds = [(0, None) if i % 4 == 0 else (None, None)
              for i in range(4*nb)]
    bounds += [(0, 20)]*n + [(None, None)]*n + [(0, None)]
    fit = linprog(c, A_ub=np.vstack([A, negative, extra]),
                  b_ub=np.concatenate([-gravity, gravity, rhs]),
                  bounds=bounds, method="highs")
    if not fit.success:
        raise RuntimeError(fit.message)
    return {i: (float(fit.x[4*b]), fit.x[4*b+1:4*b+4] / fit.x[4*b])
            for b, i in enumerate(range(2, n)) if fit.x[4*b] > 1e-5}


def _template(variant, outdir):
    arm = ARM_NAMES[variant.name]
    with tempfile.TemporaryDirectory(prefix="gello_assembly_") as temporary:
        preview = Path(temporary) / "preview.xml"
        assemble.build(arm, str(preview))
        root = ET.parse(preview).getroot()
        compiler = root.find("compiler")
        source_dir = Path(compiler.get("meshdir"))
        meshes = outdir / "meshes"
        if meshes.exists():
            shutil.rmtree(meshes)
        meshes.mkdir(exist_ok=True)
        shutil.copyfile(Path(assemble.ASSETS) / "LICENSE", meshes / "LICENSE")
        shutil.copyfile(Path(__file__).resolve().parents[3] / "LICENSE",
                        meshes / "SERVO_LICENSE.txt")
        (meshes / "SERVO_SOURCE.txt").write_text(
            "Original RLE-Bench generic micro-servo visuals, MIT.\n"
            "Procedurally generated from elementary solids; no manufacturer CAD.\n"
            "Visual meshes have zero mass; fixed inertial proxies carry the motor physics.\n")
        for mesh in root.findall("./asset/mesh"):
            source = source_dir / mesh.get("file")
            filename = mesh.get("name") + ".stl"
            shutil.copyfile(source, meshes / filename)
            mesh.set("file", filename)
            # Match the accepted assembly's mesh integration. The newer
            # convex mode crashes MuJoCo 3.5.0 on the UR5 multi-shell STL.
            mesh.set("inertia", "legacy")
        compiler.set("meshdir", "meshes")
    root.set("model", f"gello_{variant.name}")
    option = root.find("option")
    option.set("iterations", "100")
    option.set("ls_iterations", "50")
    n = variant.n_joints
    ref = np.deg2rad(assemble.REFERENCE_POSES[arm][:-1])
    for i, body in enumerate(root.findall(".//body[@name]")):
        name = body.get("name")
        if not name.startswith("lead_link"):
            continue
        j = int(name.removeprefix("lead_link")) - 1
        q = np.fromstring(body.get("quat"), sep=" ")
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, q)
        angle = ref[j] - variant.home[j]
        c, s = np.cos(angle), np.sin(angle)
        rotation = rotation.reshape(3, 3) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        body.set("quat", _vec(assemble._quat(rotation)))
        joint = body.find("joint")
        joint.set("range", _vec(variant.follower_chain[j]["range"]))
    # Keep the trigger at its accepted reference pose, outside the arm DoFs.
    trigger = root.find(".//body[@name='lead_trigger']")
    trigger.remove(trigger.find("joint"))
    last = root.find(f".//body[@name='lead_link{n}']")
    handle = ET.SubElement(last, "body", name="lead_handle")
    for element in list(last):
        if (element is handle or element.tag == "joint"
                or element.get("name") in ("servo_trigger", "visual_servo_trigger")):
            continue
        last.remove(element)
        handle.append(element)
    last_geom = handle.find(f"geom[@name='printed_link{n}']")
    last_geom.set("name", "printed_handle")
    root.remove(root.find("keyframe"))
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="home", qpos=_vec(variant.home), ctrl=_vec(variant.home))
    actuators = ET.SubElement(root, "actuator")
    for i, entry in enumerate(variant.follower_chain, 1):
        ET.SubElement(actuators, "position", name=f"lead_actuator{i}",
                      joint=f"lead_joint{i}", kp="8", kv="0.4",
                      forcerange="-0.35 0.35", ctrlrange=_vec(entry["range"]))
    return root


def _set_density(root, density):
    for geom in root.iter("geom"):
        if geom.get("name", "").startswith("printed_"):
            # The handle/trigger remain grabbable even with light links.
            rho = max(400, density) if geom.get("name") in ("printed_handle", "printed_trigger") else density
            geom.set("density", str(rho))


def _add_springs(root, springs):
    for index, (stiffness, reference) in springs.items():
        joint = root.find(f".//joint[@name='lead_joint{index}']")
        joint.set("stiffness", f"{stiffness:.12g}")
        joint.set("springref", f"{reference:.12g}")


def _contract(root, path, variant):
    """Independent verifier/public kinematic and stock-mass contract."""
    model = mujoco.MjModel.from_xml_path(str(path))
    chain = []
    for i in range(1, variant.n_joints + 1):
        b = model.body(f"lead_link{i}").id
        j = model.joint(f"lead_joint{i}").id
        chain.append(dict(pos=model.body_pos[b].tolist(), quat=model.body_quat[b].tolist(),
                          axis=model.jnt_axis[j].tolist()))
    base = model.body("lead_base").id
    ee = model.site("lead_ee_site").id
    stock = {}
    stock_geoms = {}
    volumes = {}
    supplied_meshes = {}
    for geom in root.iter("geom"):
        if geom.get("name", "").startswith(("servo", "visual_servo")):
            g = model.geom(geom.get("name")).id
            bname = model.body(int(model.geom_bodyid[g])).name
            mass = float(geom.get("mass"))
            stock[bname] = stock.get(bname, 0) + mass
            stock_geoms[geom.get("name")] = dict(body=bname,
                pos=model.geom_pos[g].tolist(), quat=model.geom_quat[g].tolist(),
                mesh=geom.get("mesh"), type=geom.get("type"), mass=mass,
                size=model.geom_size[g].tolist())
            if geom.get("name").startswith("servo"):
                visual = root.find(f".//geom[@name='visual_{geom.get('name')}']")
                stock_geoms[geom.get("name")]["rbound"] = servo_proxy.PHYSICS[visual.get("mesh")]["rbound"]
    # Unit-density integration matches the declared mesh inertia mode/scale.
    for mesh in root.findall("./asset/mesh"):
        standalone = ET.Element("mujoco")
        ET.SubElement(standalone, "compiler", meshdir=str(path.parent / "meshes"))
        asset = ET.SubElement(standalone, "asset")
        asset.append(ET.fromstring(ET.tostring(mesh)))
        body = ET.SubElement(ET.SubElement(standalone, "worldbody"), "body")
        ET.SubElement(body, "geom", type="mesh", mesh=mesh.get("name"), density="1")
        unit = mujoco.MjModel.from_xml_string(ET.tostring(standalone, encoding="unicode"))
        digest = hashlib.sha256((path.parent/"meshes"/mesh.get("file")).read_bytes()).hexdigest()
        volumes[digest] = float(unit.body_mass[1])
        supplied_meshes[mesh.get("name")] = digest
    return dict(chain=chain, base_quat=model.body_quat[base].tolist(),
                ee=dict(pos=model.site_pos[ee].tolist(), quat=model.site_quat[ee].tolist()),
                stock_mass=stock, stock_geoms=stock_geoms, mesh_volumes=volumes,
                supplied_meshes=supplied_meshes,
                supplied_geoms={g.get("name"): g.get("mesh")
                                for g in root.iter("geom") if g.get("name")})


def _pinned_design(root, name, kind):
    recipe = json.loads((Path(__file__).parent / "data/reference_design.json").read_text())[name][kind]
    for joint, attributes in recipe["springs"].items():
        root.find(f".//joint[@name='{joint}']").attrib.update(attributes)
    for body, geoms in recipe["geoms"].items():
        parent = root.find(f".//body[@name='{body}']")
        for attributes in geoms:
            ET.SubElement(parent, "geom", **attributes)


def generate_variant(name, refit=False):
    variant = VARIANTS[name]
    outdir = Path(ROOT) / name
    outdir.mkdir(parents=True, exist_ok=True)
    root = _template(variant, outdir)
    _set_density(root, BARE_DENSITY)
    bare = outdir / "lead_bare.xml"
    _write(root, bare)
    contract = _contract(root, bare, variant)
    _set_density(root, SREF_DENSITY)
    sref = outdir / "lead_sref.xml"
    _write(root, sref)
    if refit:
        _add_springs(root, _fit_springs(str(sref), variant))
    else:
        _pinned_design(root, name, "sref")
    _write(root, sref)
    oracle = outdir / "lead_oracle.xml"
    pinned = Path(__file__).parent / "data/oracle" / name
    if refit:
        root = ET.parse(bare).getroot()
        _set_density(root, ORACLE_DENSITY)
        _write(root, oracle)
        cws = _counterweight_fit(oracle, variant)
        model = mujoco.MjModel.from_xml_path(str(oracle))
        for index, (mass, position) in cws.items():
            body = root.find(f".//body[@name='lead_link{index}']")
            geom = model.geom(f"printed_link{index}").id
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            vertices = (model.mesh_vert[start:start+count] @ rotation.reshape(3, 3).T
                        + model.geom_pos[geom])
            anchor = vertices[np.argmin(np.linalg.norm(vertices-position, axis=1))]
            radius = _counterweight_radius(mass)
            if np.linalg.norm(anchor-position) > radius:
                ET.SubElement(body, "geom", name=f"printed_cw_support_{index}",
                              type="capsule", fromto=_vec(np.r_[anchor, position]),
                              size="0.003", density=str(ORACLE_DENSITY),
                              rgba="0.93 0.92 0.88 1")
            ET.SubElement(body, "geom", name=f"counterweight_{index}_0", type="sphere",
                          size=f"{radius:.12g}", pos=_vec(position),
                          mass=f"{mass:.12g}", rgba="0.35 0.35 0.4 1")
        _write(root, oracle)
        _add_springs(root, _fit_springs(str(oracle), variant))
        _write(root, oracle)
    else:
        shutil.copyfile(pinned / "lead.xml", oracle)
    shutil.copyfile(pinned / "trim.py", outdir / "trim_oracle.py")
    scene = ET.Element("mujoco", model=f"gello_{name}_scene")
    ET.SubElement(scene, "include", file="lead_bare.xml")
    ET.SubElement(scene, "statistic", center="0 0 0.3", extent="1.0")
    _write(scene, outdir / "scene.xml")
    mass = mujoco.mj_getTotalmass(mujoco.MjModel.from_xml_path(str(oracle)))
    print(f"{name}: oracle mass={mass:.4f} kg")
    return contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS)+("all",), default="all")
    parser.add_argument("--refit", action="store_true",
                        help="re-optimize instead of using the pinned reference recipe")
    args = parser.parse_args()
    contract_path = Path(__file__).resolve().parents[1] / "harness/lead_geometry.json"
    contracts = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    for name in VARIANTS if args.variant == "all" else (args.variant,):
        contracts[name] = generate_variant(name, refit=args.refit)
    contract_path.write_text(json.dumps(contracts, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
