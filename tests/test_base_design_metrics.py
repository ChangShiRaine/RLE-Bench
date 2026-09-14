"""Analytic unit tests for the stability metrics — the highest-leverage tests.

Each asserts a hand-computed value on a trivial geometry, so a wrong metric
fails immediately without needing a full sim. Pure math only (no MuJoCo model);
sim-based cross-checks live in test_base_design_metrics_sim.py.
"""
import numpy as np
import pytest

SQUARE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
SQUARE3 = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a), np.cos(a)]])


# ---------------------------------------------------------------- SSM / SSF

def test_ssm_centroid_of_unit_square():
    from harness.metrics.ssm import static_stability_margin
    assert static_stability_margin(np.array([0.0, 0.0]), SQUARE) == pytest.approx(0.5)


def test_ssm_off_center():
    from harness.metrics.ssm import static_stability_margin
    assert static_stability_margin(np.array([0.3, 0.1]), SQUARE) == pytest.approx(0.2)


def test_ssm_outside_is_negative():
    from harness.metrics.ssm import static_stability_margin
    assert static_stability_margin(np.array([1.0, 0.0]), SQUARE) < 0


def test_ssm_degenerate_support_is_unstable():
    from harness.metrics.ssm import static_stability_margin
    line = np.array([[-0.5, 0.0], [0.5, 0.0]])
    assert static_stability_margin(np.array([0.0, 0.0]), line) == float("-inf")


def test_ssf_formula():
    from harness.metrics.ssf import static_stability_factor
    assert static_stability_factor(0.5, 0.25) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        static_stability_factor(0.5, 0.0)


# ------------------------------------------------------------------- FASM

def test_fasm_vertical_force_symmetric_positive():
    """Downward force through the centroid of a symmetric polygon: FASM > 0 and
    equal margins across symmetric edges (square: same min from any edge)."""
    from harness.metrics.fasm import force_angle_stability
    com = np.array([0.0, 0.0, 0.4])
    f = np.array([0.0, 0.0, -600.0])
    m = force_angle_stability(com, SQUARE3, f)
    assert m > 0
    # symmetry: mirroring the (centered) system changes nothing
    mirrored = SQUARE3 * np.array([-1, 1, 1])
    assert force_angle_stability(com, mirrored[::-1], f) == pytest.approx(m)


def test_fasm_tilted_force_crosses_zero():
    """Force tilted far past an edge -> FASM <= 0 (tipping)."""
    from harness.metrics.fasm import force_angle_stability
    com = np.array([0.0, 0.0, 0.4])
    f = np.array([2000.0, 0.0, -300.0])  # strong lateral push
    assert force_angle_stability(com, SQUARE3, f) <= 0


def test_fasm_margin_decreases_as_force_tilts():
    from harness.metrics.fasm import force_angle_stability
    com = np.array([0.0, 0.0, 0.4])
    margins = [force_angle_stability(com, SQUARE3, np.array([fx, 0.0, -600.0]))
               for fx in (0.0, 200.0, 400.0, 600.0)]
    assert all(a > b for a, b in zip(margins, margins[1:]))


def test_fasm_rotation_invariance():
    """FASM is frame-independent: rotating polygon+CoM+force together on a
    slope changes nothing. This is what makes it valid off flat ground."""
    from harness.metrics.fasm import force_angle_stability
    com = np.array([0.1, -0.05, 0.4])
    f = np.array([150.0, 80.0, -600.0])
    m_flat = force_angle_stability(com, SQUARE3, f)
    R = rot_x(25)
    m_slope = force_angle_stability(R @ com, SQUARE3 @ R.T, R @ f)
    assert m_slope == pytest.approx(m_flat, rel=1e-9)


def test_fasm_on_slope_gravity_tips_at_steep_angle():
    """Square support tilted about x; CoM 0.4 above the plane center; pure
    gravity. Line of action exits the downhill edge when tan(a) > 0.5/0.4."""
    from harness.metrics.fasm import force_angle_stability
    g = np.array([0.0, 0.0, -600.0])
    com_local = np.array([0.0, 0.0, 0.4])
    shallow = rot_x(25)   # tan 25 = 0.47 -> shift 0.19 < 0.5: stable
    steep = rot_x(55)     # tan 55 = 1.43 -> shift 0.57 > 0.5: tips
    assert force_angle_stability(shallow @ com_local, SQUARE3 @ shallow.T, g) > 0
    assert force_angle_stability(steep @ com_local, SQUARE3 @ steep.T, g) < 0


def test_fasm_knife_edge_is_marginal():
    """Two contact points = a tip-over line: vertical force through it is
    exactly marginal (0), never positive."""
    from harness.metrics.fasm import force_angle_stability
    line = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    com = np.array([0.0, 0.0, 0.4])
    m = force_angle_stability(com, line, np.array([0.0, 0.0, -600.0]))
    assert m == pytest.approx(0.0, abs=1e-9)


def test_fasm_no_support_is_unstable():
    from harness.metrics.fasm import force_angle_stability
    single = np.array([[0.0, 0.0, 0.0]])
    assert force_angle_stability(np.array([0, 0, 0.4]), single,
                                 np.array([0, 0, -600.0])) == float("-inf")


# --------------------------------------------------------- support polygon

def test_support_polygon_ccw_hull():
    from harness.metrics.support_polygon import support_polygon
    pts = np.array([[0.2, 0.25, 0], [-0.2, 0.25, 0], [-0.2, -0.25, 0],
                    [0.2, -0.25, 0], [0.0, 0.0, 0]])  # interior point dropped
    poly = support_polygon(pts)
    assert len(poly) == 4
    # shoelace > 0 => CCW
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area == pytest.approx(0.2, abs=1e-9)


def test_support_polygon_collinear_returns_extremes():
    from harness.metrics.support_polygon import support_polygon
    pts = np.array([[x, 0.0, 0.0] for x in (-0.3, 0.1, 0.4)])
    poly = support_polygon(pts)
    assert len(poly) == 2
    assert sorted(poly[:, 0].tolist()) == pytest.approx([-0.3, 0.4])
