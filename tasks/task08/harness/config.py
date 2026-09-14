"""Task08 configuration, shared by host validation and the verifier."""

from . import thresholds as _thresholds

REACH_BEYOND_MIN_M = _thresholds.REACH_BEYOND_MIN_M
SSM_THRESHOLD_M = _thresholds.SSM_THRESHOLD_M
SSM_SAT_M = _thresholds.SSM_SAT_M
FASM_SAT = _thresholds.FASM_SAT
D2_MAX_LIFTOFF_EVENTS = _thresholds.D2_MAX_LIFTOFF_EVENTS
D2_MAX_LIFTOFF_DURATION_S = _thresholds.D2_MAX_LIFTOFF_DURATION_S
TRACKING_RMS_TOL = _thresholds.TRACKING_RMS_TOL
S5_TRACKING_GATE = _thresholds.S5_TRACKING_GATE

# --- seeds -----------------------------------------------------------------
DESIGN_SEEDS = (11, 23, 37)      # calibration only
HIDDEN_SEEDS = (101, 211, 307)   # scoring battery: held out

# --- soft resource targets ----------------------------------------------------
FOOTPRINT_BUDGET_M = 0.56        # preferred x and y extent at rest
MASS_BUDGET_KG = 60.0            # preferred assembled mass without payload
# Common-base efficiency excludes verifier-supplied arm + adapter mass.
BASE_MASS_BUDGET_KG = 38.0
PROFILE_LENGTH_BUDGET_M = 12.0  # preferred recognized profile length
# The design-efficiency checkpoint uses normalized remaining budget directly;
# exceeding a target reduces credit but never invalidates the design.
# Each arm must keep essentially all of its own reach.
REACH_PRESERVED_FRAC = 0.99      # of the arm's canonical reach
# Public shelf layout and required/extra robustness loads.
PICK_SHELF_FACE_X_M = 0.40
PICK_PAYLOAD_KG = 1.0
PICK_EXTRA_PAYLOAD_KG = 1.0

# --- battery pack ------------------------------------------------------------
BATTERY_MASS_KG = 10.0           # spec mass of the pack
BATTERY_MASS_RTOL = 0.02
BATTERY_SIZE_HALF_M = (0.14, 0.09, 0.04)   # box half-extents (size attr)
BATTERY_SIZE_RTOL = 0.05

# --- stability thresholds ------------------------------------------------------
# Scenario A reports a STATIC margin, so a sample only counts once the base
# has actually come to rest after an arm move. On compliant mecanum rollers
# the swing can outlast the dwell; sampling it would grade a transient.
SCENARIO_A_QUIET_V = 0.02        # m/s, base linear speed
SCENARIO_A_QUIET_W = 0.05        # rad/s, base angular speed

ROLL_LIMIT_DEG = 10.0

# --- actuator caps (anti-cheese: no monster motors) ---------------------------
WHEEL_KV_MAX = 12.0
WHEEL_FORCERANGE_MAX = 15.0
WHEEL_CTRLRANGE_MAX = 45.0

GATE_CAP = 0.15                  # stage-1 failure caps the total here

# Validity checkpoints score continuously: a constraint gives full credit
# while it holds and decays to zero once the violation reaches this multiple
# of the limit, so a hair over the line costs a hair. Only the two runnable
# gates (model loads, interface/DoF layout) still cap the total.
CONSTRAINT_BAND = 1.5

# Submitted shelf-controller protocol. Physics remains at TIMESTEP=0.002.
PICK_CONTROL_DT = 0.02
PICK_EPISODE_SECONDS = 30.0
PICK_START_POSITION = (-0.60, 0.0, 0.08)
PICK_START_QUATERNION = (1.0, 0.0, 0.0, 0.0)
PICK_CONTROLLER_SEED = 0
PICK_CONTROLLER_CPU_SECONDS = 60
PICK_CONTROLLER_STARTUP_SECONDS = 30.0
PICK_CONTROLLER_REPLY_SECONDS = 2.0
PICK_CONTROLLER_SOURCE_BYTES = 1024 * 1024
ROBOT_SOURCE_BYTES = 4 * 1024 * 1024
