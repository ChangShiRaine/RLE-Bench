"""Private, deterministic experiment and scoring configuration."""
from dataclasses import dataclass

from ..budgets import INTERACTION_STEPS

CONTROL_HZ = 20
MAX_STEPS = INTERACTION_STEPS["HiddenCOM"]
MAX_BATCH = 200
SOCKET = "/run/rlebench/hidden-com.sock"
STATE = "/var/lib/rlebench/hidden-com.json"


@dataclass(frozen=True)
class Case:
    seed: int
    shell_mass: float
    ballast_mass: float
    offset_xy: tuple[float, float]
    ballast_half: tuple[float, float, float]
    oracle_steps: int = 240


CASES = (
    Case(7007, 0.15, 0.30, (0.040, 0.040), (0.015, 0.015, 0.015)),
    Case(7018, 0.24, 0.34, (0.045, 0.035), (0.010, 0.020, 0.012)),
    Case(7009, 0.10, 0.25, (0.030, 0.045), (0.022, 0.010, 0.016)),
)
