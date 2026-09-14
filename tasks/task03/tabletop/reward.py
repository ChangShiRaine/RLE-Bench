"""Task03 rewards from normalized task quality."""
import math


def reward(quality):
    if not math.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError("quality must be in [0, 1]")
    return quality


def apply(result, quality):
    value = reward(quality)
    result.update(task_score=100 * value, outcome_score=value,
                  efficiency_score=0.0, reward=round(value, 6))
    return result
