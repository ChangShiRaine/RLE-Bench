"""Per-follower task09 scoring bars selected at verifier start."""
from __future__ import annotations

from . import spec as gspec

HIDDEN_SEEDS = (129, 223, 311)   # seed % 3 -> payload level low/mid/high
REWARD_HEADROOM = 3.0

_BARS = {
    "franka": dict(
        H1_RESIDUAL_BAR=0.02,
        H2_DROOP_BAR=0.0212,
        H3_EFFORT_BAR=0.2825,
        S1_RATIO_BAR=0.7816,
        S2_DROOP_BAR=0.0045,
        S2_EFFORT_BAR=0.0961,
        B1_DROOP_BAR=0.0035,
        B2_EFFORT_BAR=0.0999,
        B3_HEADROOM_BAR=0.094,
        B4_POKE_BAR=0.0112,
        SREF_PASSIVE_DROOP=0.0249,
    ),
    "ur5e": dict(
        H1_RESIDUAL_BAR=0.0329,
        H2_DROOP_BAR=0.0143,
        H3_EFFORT_BAR=0.2197,
        S1_RATIO_BAR=0.8077,
        S2_DROOP_BAR=0.0014,
        S2_EFFORT_BAR=0.0664,
        B1_DROOP_BAR=0.0011,
        B2_EFFORT_BAR=0.0681,
        B3_HEADROOM_BAR=0.0576,
        B4_POKE_BAR=0.0122,
        SREF_PASSIVE_DROOP=0.0167,
    ),
    "xarm7": dict(
        H1_RESIDUAL_BAR=0.0206,
        H2_DROOP_BAR=0.0143,
        H3_EFFORT_BAR=0.2181,
        S1_RATIO_BAR=0.7221,
        S2_DROOP_BAR=0.0019,
        S2_EFFORT_BAR=0.0654,
        B1_DROOP_BAR=0.0013,
        B2_EFFORT_BAR=0.0616,
        B3_HEADROOM_BAR=0.0777,
        B4_POKE_BAR=0.0031,
        SREF_PASSIVE_DROOP=0.0181,
    ),
}

globals().update(_BARS[gspec.VARIANT_NAME])
