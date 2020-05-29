import pytest
import numpy as np
from coxeter.shape_classes.circle import Circle
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays


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


@given(floats(0.1, 1000),
       arrays(np.float64, (3, ), elements=floats(-10, 10, width=64),
              unique=True))
def test_moment_inertia(r, center):
    circle = Circle(r)
    assert np.all(np.asarray(circle.planar_moments_inertia) >= 0)

    circle.center = center
    area = np.pi * r**2
    expected = [np.pi / 4 * r**4] * 3
    expected[0] += area * center[0]**2
    expected[1] += area * center[1]**2
    expected[2] = area * center[0] * center[1]
    np.testing.assert_allclose(circle.planar_moments_inertia[:2], expected[:2])
    np.testing.assert_allclose(circle.planar_moments_inertia[2], expected[2])
    assert circle.polar_moment_inertia == pytest.approx(sum(expected[:2]))


def test_center():
    """Test getting and setting the center."""
    circle = Circle(1)
    assert all(circle.center == (0, 0, 0))

    center = (1, 1, 1)
    circle.center = center
    assert all(circle.center == center)
