"""Task09 device and task constants for the follower selected by
RLEBENCH_GELLO_VARIANT. Ships in the verifier image only; the agent gets a
per-variant public spec written by build_assets.py."""
import json
from pathlib import Path

from .variants import selected_variant

VARIANT = selected_variant()
VARIANT_NAME = VARIANT.name
N_JOINTS = VARIANT.n_joints
LEAD_HOME = VARIANT.home
WORKSPACE_HALFWIDTH = VARIANT.workspace_halfwidth
JOINT_NAMES = VARIANT.joint_names
EE_SITE = "lead_ee_site"

# --- stock hardware, pinned on every submitted model ---------------------------
SERVO_TAU_NM = 0.35               # rated servo torque cap
SERVO_KP = 8.0                    # position-servo P gain, Nm/rad
SERVO_KD = 0.4                    # position-servo D gain, Nm/(rad/s)
JOINT_DAMPING = 0.01              # Nm/(rad/s)
JOINT_FRICTIONLOSS = 0.03         # Nm dry friction
JOINT_ARMATURE = 0.002            # kg m^2 reflected servo inertia

# --- budgets -------------------------------------------------------------------
MASS_BUDGET_KG = 2.30             # whole device incl. counterweights
DENSITY_PRINTED_MIN = 100.0       # kg/m^3 floor for printed_* geoms
DENSITY_PRINTED_MAX = 2000.0      # kg/m^3 cap for printed_* geoms
DENSITY_ANY_MAX = 9000.0          # kg/m^3 cap for anything (steel)

# --- physics pins ----------------------------------------------------------------
TIMESTEP = 0.002
GRAVITY = (0.0, 0.0, -9.81)

# --- measured leader geometry and stock-mass contract ----------------------------
LEAD_GEOMETRY = json.loads(
    Path(__file__).with_name("lead_geometry.json").read_text())[VARIANT_NAME]
LEAD_CHAIN = LEAD_GEOMETRY["chain"]
LEAD_EE = LEAD_GEOMETRY["ee"]
LEAD_BASE_QUAT = LEAD_GEOMETRY["base_quat"]
STOCK_BODY_MASS = LEAD_GEOMETRY["stock_mass"]
