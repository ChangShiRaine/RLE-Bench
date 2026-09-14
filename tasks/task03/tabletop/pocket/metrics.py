"""Eight-corner legality; solved states are equivalent under rigid rotation."""

from itertools import permutations, product

import numpy as np

# Face order and handedness follow the conventional URFDLB cubie representation.
NORMALS = dict(U=(0, 0, 1), R=(1, 0, 0), F=(0, -1, 0),
               D=(0, 0, -1), L=(-1, 0, 0), B=(0, 1, 0))
CORNERS = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
SLOTS = CORNERS
POSITIONS = np.array([np.sum([NORMALS[c] for c in slot], axis=0) for slot in SLOTS])


def _rotations():
    out = []
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.eye(3, dtype=int)[:, perm] * signs
            if round(np.linalg.det(rotation)) == 1:
                out.append(rotation)
    return np.array(out)


ROTATIONS = _rotations()
ALIGNMENT_TOLERANCE = np.deg2rad(4)


def inspect_cube(rotations, tolerance=ALIGNMENT_TOLERANCE):
    rotations = np.asarray(rotations, dtype=float)
    invalid = {"legal": False, "solved": False, "matched": 0}
    if rotations.shape != (8, 3, 3) or not np.isfinite(rotations).all():
        return invalid
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6)):
        return invalid
    traces = np.einsum("bij,kij->bk", rotations, ROTATIONS)
    nearest = traces.argmax(axis=1)
    errors = np.arccos(np.clip((traces[np.arange(8), nearest]-1)/2, -1, 1))
    if errors.max() > tolerance:
        return invalid
    snapped = ROTATIONS[nearest]
    positions = np.einsum("bij,bj->bi", snapped, POSITIONS)
    lookup = {tuple(p): i for i, p in enumerate(POSITIONS)}
    destinations = [lookup.get(tuple(p), -1) for p in positions]
    if sorted(destinations) != list(range(8)):
        return invalid
    twists, normals, transformed = [], [], []
    for slot, rotation, destination in zip(SLOTS, snapped, destinations):
        source = [NORMALS[c] for c in slot]
        target = [NORMALS[c] for c in SLOTS[destination]]
        actual = [tuple(rotation @ n) for n in source]
        if set(actual) != set(target):
            return invalid
        twists.append(target.index(actual[0]))
        normals.extend(source)
        transformed.extend(actual)
    if sum(twists) % 3:
        return invalid
    # There are no edge pieces and therefore no corner/edge parity constraint.
    targets = np.einsum("kij,bj->kbi", ROTATIONS, normals)
    matched = int(np.all(targets == np.asarray(transformed), axis=2).sum(axis=1).max())
    return {"legal": True, "solved": matched == 24, "matched": matched,
            "alignment_error_rad": float(errors.max())}


def apply_moves(rotations, moves):
    result = np.array(rotations, dtype=float, copy=True)
    for axis, sign, turns in moves:
        direction = np.eye(3, dtype=int)[axis] * sign
        cross = np.array([[0, -direction[2], direction[1]],
                          [direction[2], 0, -direction[0]],
                          [-direction[1], direction[0], 0]])
        quarter = np.outer(direction, direction) + cross
        turn = np.linalg.matrix_power(quarter, turns % 4)
        selected = np.einsum("bij,bj->bi", result, POSITIONS)[:, axis] * sign > .5
        result[selected] = turn @ result[selected]
    return result


def scramble(seed, length=20):
    """Deterministic legal moves, with consecutive turns on different axes."""
    rng = np.random.default_rng(seed)
    moves, previous = [], -1
    for _ in range(length):
        axis = int(rng.choice([a for a in range(3) if a != previous]))
        moves.append((axis, int(rng.choice((-1, 1))), int(rng.choice((-1, 1, 2)))))
        previous = axis
    return tuple(moves)
