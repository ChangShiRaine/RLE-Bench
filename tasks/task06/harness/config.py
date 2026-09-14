"""Kernel scoring configuration for the task06 family (verifier-only).

Each group's worst-five mean errors contribute a pose score. Translation and
rotation use the same uncapped Gaussian kernel as task04::

    exp(-error**2 / std**2)

The denominator is the variance.  Therefore only exactly zero error receives
full credit; there is no saturation plateau and no calibrated pass threshold.
"""

# Load is the only gate.  It carries no positive credit; if it fails the
# aggregate is capped at zero.
GATE_CAP = 0.0

STAGE_A_WEIGHT = 0.3
STAGE_B_WEIGHT = 0.7
STAGE_A_GROUPS = 10
POSE_WORST_FRAMES = 5
EFFICIENCY_MAX_DEDUCTION = 0.2
EFFICIENCY_FULL_HZ = 10.0
EFFICIENCY_ZERO_HZ = 1.0

# Kernel standard deviations.  These set the error scale, not a full-credit
# threshold: any nonzero error scores strictly below 1.
POSE_TRANS_STD_M = 0.01
POSE_ROT_STD_RAD = 0.1
