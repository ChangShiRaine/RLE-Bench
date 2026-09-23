"""Scoring thresholds for task11 (verifier-only).

Force and speed limits are pinned by dev.calibrate against the reference
packers; weights and caps are design choices. No seeds live here (see
eval_seeds.json).

Reward, per episode, clamped to [0, 1]:
    u     = min(utilization / UTIL_CAP, 1)
    t     = min(throughput_ppm / TP_CAP_PPM, 1)
    pp    = pick_place_success_rate * min(n_packed / PP_MIN_PACKED, 1)
    full  = bin declared full AND zero floor drops
    score = W_UTIL*u + W_TP*t*u + W_PP*pp + W_FULL*full*u
            - W_FLOOR*floor_drops - W_DAMAGE*damage_events - W_BIN_HIT*bin_hits
Mean over evaluation episodes; a failed gate caps the total at GATE_CAP.
"""

GATE_CAP = 0.1

W_UTIL = 0.40
W_TP = 0.25
W_PP = 0.20
W_FULL = 0.15
UTIL_CAP = 0.70          # a dense, stable mixed-carton load
TP_CAP_PPM = 7.0         # boxes packed per minute
PP_MIN_PACKED = 10       # the success rate pays in full from this many packs

W_FLOOR = 0.05           # per box that reaches the floor
W_DAMAGE = 0.04          # per damaged box (crush or impact)
W_BIN_HIT = 0.03         # per tool strike on the tote

# Sustained (>= DMG_TICKS control ticks) contact force on a box above this is
# a crush; a released box that makes contact moving faster than
# DROP_SPEED_LIMIT (about a 7 cm free fall) is an impact.
DAMAGE_FORCE_N = 310.0
DMG_TICKS = 2
DROP_SPEED_LIMIT = 1.2

# Tool-tote contact above this for BIN_HIT_TICKS ticks is a strike; it must
# stay below for BIN_HIT_REARM_TICKS before another counts.
BIN_HIT_FORCE_N = 150.0
BIN_HIT_TICKS = 2
BIN_HIT_REARM_TICKS = 5

SMOKE_T = 20.0

# Reference anchors on the evaluation seeds (regression guards).
GOLDEN_REF_SCORE = 0.7149
BASELINE_REF_SCORE = 0.6047      # in the verifier image (host GL: 0.614)
