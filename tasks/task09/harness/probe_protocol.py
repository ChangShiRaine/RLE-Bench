"""Instance calibration contract for trim.py.

Implement plan_probe(model, sample, lower, upper):
  model: nominal device, without the unknown instance mass changes or load.
  sample: (q_target, q_measured) from one 1.2-second hold at home.
  lower, upper: allowed joint-angle bounds in radians.
  return: an array of exactly 15 additional target poses, shape (15, model.nv).

Each measurement starts at its target pose with zero velocity and runs for
1.2 seconds with feedforward off. The fixed position servo applies
clip(8.0 * (q_target - q) - 0.4 * qvel, -0.35, 0.35) Nm per joint.
The returned joint angles include independent zero-mean Gaussian encoder
noise with standard deviation 0.005 rad.

The complete batch must contain finite angles within the supplied workspace
and the model's joint limits. An invalid batch is rejected without running
any of its holds, and no retry or additional probes are provided.

The evaluator measures these 15 poses, then calls adapt(model, probe) with
16 (q_target, q_measured) pairs: the initial sample followed by the chosen
poses in their returned order. make_trim_adapted(model, params) returns the
pose-to-torque function. make_trim(model) remains the uncalibrated function.
Selection and adaptation can run in separate processes; do not rely on
shared Python globals. The planner cannot request measurements interactively.
Implementations without plan_probe receive no adapted measurements; nominal
evaluation remains.
"""
CHOSEN_POSES = 15
TOTAL_MEASUREMENTS = 16
