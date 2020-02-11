import numpy as np
from euclid.shape_classes.circle import Circle
from hypothesis import given
from hypothesis.strategies import floats


@given(floats(0.1, 1000))
def test_perimeter(r):
    C = 2 * np.pi * r
    circle = Circle(1)
    circle.radius = r
    assert circle.perimeter == C
    assert circle.circumference == C


@given(floats(0.1, 1000))
def test_area(r):
    A = np.pi * r**2
    circle = Circle(1)
    circle.radius = r
    assert circle.area == A


@given(floats(0.1, 1000))
def test_iq(r):
    circle = Circle(r)
    assert circle.iq == 1


@given(floats(0.1, 1000))
def test_eccentricity(r):
    circle = Circle(r)
    assert circle.eccentricity == 0


@given(floats(0.1, 1000))
def test_inertia_tensor(r):
    circle = Circle(r)
    assert np.all(np.asarray(circle.planar_moments_inertia) >= 0)

    expected = np.pi / 4 * r**4
    np.testing.assert_allclose(circle.planar_moments_inertia[:2], expected)
    np.testing.assert_allclose(circle.planar_moments_inertia[2], 0)
    assert np.isclose(circle.polar_moment_inertia, expected*2)
