"""Reference-only visual pseudo-labels; no renderer identity metadata."""
from . import spec
from .oracle_geom import top_points
from .oracle_unknown import fit_unknown_shape, MAX_SHAPE_COST


def label_shape(rgb, depth):
    _, cost, shape = fit_unknown_shape(top_points(depth))
    if shape is None or cost > MAX_SHAPE_COST:
        return -1
    return spec.BLOCK_SHAPES.index(shape)
