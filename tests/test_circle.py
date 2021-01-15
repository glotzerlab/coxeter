import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pytest import approx

from coxeter.shapes.circle import Circle


@given(floats(0.1, 1000))
def test_perimeter(r):
    circle = Circle(1)
    circle.radius = r
    assert circle.perimeter == 2 * np.pi * r
    assert circle.circumference == 2 * np.pi * r


@given(floats(0.1, 1000))
def test_perimeter_setter(perimeter):
    """Test setting the perimeter."""
    circle = Circle(1)
    circle.perimeter = perimeter
    assert circle.radius == perimeter / (2 * np.pi)
    assert circle.perimeter == approx(perimeter)
    circle.perimeter = perimeter + 1
    assert circle.radius == (perimeter + 1) / (2 * np.pi)
    assert circle.perimeter == approx(perimeter + 1)
    circle.circumference = perimeter
    assert circle.radius == perimeter / (2 * np.pi)
    assert circle.circumference == approx(perimeter)
    circle.circumference = perimeter + 1
    assert circle.radius == (perimeter + 1) / (2 * np.pi)
    assert circle.circumference == approx(perimeter + 1)


@given(floats(0.1, 1000))
def test_area(r):
    circle = Circle(1)
    circle.radius = r
    assert circle.area == np.pi * r ** 2


@given(floats(0.1, 1000))
def test_set_area(area):
    """Test setting the area."""
    circle = Circle(1)
    circle.area = area
    assert circle.area == approx(area)
    assert circle.radius == approx((area / np.pi) ** 0.5)


@given(floats(0.1, 1000))
def test_iq(r):
    circle = Circle(r)
    assert circle.iq == 1


@given(floats(0.1, 1000))
def test_eccentricity(r):
    circle = Circle(r)
    assert circle.eccentricity == 0


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_moment_inertia(r, center):
    circle = Circle(r)
    assert np.all(np.asarray(circle.planar_moments_inertia) >= 0)

    circle.center = center
    area = np.pi * r ** 2
    expected = [np.pi / 4 * r ** 4] * 3
    expected[0] += area * center[0] ** 2
    expected[1] += area * center[1] ** 2
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


@given(floats(0.1, 1000))
def test_radius_getter_setter(r):
    """Test getting and setting the radius."""
    circle = Circle(r)
    assert circle.radius == r
    circle.radius = r + 1
    assert circle.radius == r + 1


def test_invalid_radius():
    with pytest.raises(ValueError):
        Circle(-1)


def test_invalid_radius_setter():
    circle = Circle(1)
    with pytest.raises(ValueError):
        circle.radius = -1
