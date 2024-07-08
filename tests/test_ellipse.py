# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pytest import approx

from conftest import (
    _test_get_set_minimal_bounding_sphere_radius,
    assert_distance_to_surface_2d,
    sphere_isclose,
)
from coxeter.shapes import Circle, Ellipse


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_a_b_getter_setter(a, b):
    """Test getter and setter for a and b."""
    ellipse = Ellipse(a, b)
    assert ellipse.a == a
    assert ellipse.b == b
    ellipse.a = a + 1
    ellipse.b = b + 1
    assert ellipse.a == approx(a + 1)
    assert ellipse.b == approx(b + 1)


@given(floats(-1000, -1))
def test_invalid_a_b_setter(a):
    """Test setting invalid a, b values."""
    ellipse = Ellipse(1, 1)
    with pytest.raises(ValueError):
        ellipse.a = a
    with pytest.raises(ValueError):
        ellipse.b = a


@given(floats(-1000, -1), floats(0.1, 1000))
def test_invalid_a_b_constructor(a, b):
    """Test invalid a, b values in constructor."""
    with pytest.raises(ValueError):
        Ellipse(a, b)
    with pytest.raises(ValueError):
        Ellipse(b, a)
    with pytest.raises(ValueError):
        Ellipse(a, a)


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_perimeter(a, b):
    """Check surface area against an approximate formula."""
    # Uses Ramanujan's approximation for the circumference of an ellipse:
    # https://en.wikipedia.org/wiki/Ellipse#Circumference
    ellipse = Ellipse(a, b)
    b, a = sorted([a, b])
    h = (a - b) ** 2 / (a + b) ** 2
    approx_perimeter = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    assert ellipse.perimeter == approx(approx_perimeter, rel=1e-3)
    assert ellipse.circumference == approx(approx_perimeter, rel=1e-3)


@given(floats(0.1, 1000))
def test_perimeter_setter(value):
    """Test perimeter and circumference getter and setter."""
    ellipse = Ellipse(1, 2)
    perimeter_old = ellipse.perimeter
    circumference_old = ellipse.circumference
    ellipse.circumference = value
    assert ellipse.circumference == approx(value)
    assert ellipse.a == approx(1 * value / circumference_old)
    assert ellipse.b == approx(2 * value / circumference_old)
    ellipse = Ellipse(1, 2)
    ellipse.perimeter = value
    assert ellipse.perimeter == approx(value)
    assert ellipse.a == approx(1 * value / perimeter_old)
    assert ellipse.b == approx(2 * value / perimeter_old)


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_area_getter(a, b):
    ellipse = Ellipse(1, 1)
    ellipse.a = a
    ellipse.b = b
    assert ellipse.area == approx(np.pi * a * b)


@given(floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000))
def test_set_area(a, b, area):
    """Test setting the area."""
    ellipse = Ellipse(a, b)
    original_area = ellipse.area
    ellipse.area = area
    assert ellipse.area == approx(area)

    # Reset to original area
    ellipse.area = original_area
    assert ellipse.area == approx(original_area)
    assert ellipse.a == approx(a)
    assert ellipse.b == approx(b)


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_iq(a, b):
    ellipse = Ellipse(a, b)
    assert ellipse.iq <= 1


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_eccentricity(a, b):
    ellipse = Ellipse(a, b)
    assert ellipse.eccentricity >= 0


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_eccentricity_ratio(a, k):
    b = a * k
    ellipse = Ellipse(a, b)
    b, a = sorted([a, b])
    expected = np.sqrt(1 - b**2 / a**2)
    assert ellipse.eccentricity == approx(expected)


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_moment_inertia(a, b, center):
    ellipse = Ellipse(a, b)
    assert np.all(np.asarray(ellipse.planar_moments_inertia) >= 0)

    # We must set the center after construction so that the inertia tensor
    # calculation is not shifted away from the origin.
    ellipse.center = center
    area = ellipse.area
    expected = [np.pi / 4 * a * b**3, np.pi / 4 * a**3 * b, 0]
    expected[0] += area * center[0] ** 2
    expected[1] += area * center[1] ** 2
    expected[2] = area * center[0] * center[1]
    np.testing.assert_allclose(ellipse.planar_moments_inertia[:2], expected[:2])
    np.testing.assert_allclose(ellipse.planar_moments_inertia[2], expected[2])
    assert ellipse.polar_moment_inertia == approx(sum(expected[:2]))


def test_center():
    """Test getting and setting the center."""
    ellipse = Ellipse(1, 2)
    np.testing.assert_allclose(ellipse.center, (0, 0, 0))

    center = (1, 1, 1)
    ellipse.center = center
    np.testing.assert_allclose(ellipse.center, center)


@settings(deadline=500)
@given(floats(0.1, 10), floats(0.1, 10))
def test_distance_to_surface(a, b):
    """Test consistent volume and area for shape distance of an ellipse."""
    theta = np.linspace(0, 2 * np.pi, 50000)
    ellipse = Ellipse(a, b)
    distance = ellipse.distance_to_surface(theta)
    assert_distance_to_surface_2d(ellipse, theta, distance)


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_minimal_bounding_circle(a, b, center):
    ellipse = Ellipse(a, b, center)
    bounding_circle = ellipse.minimal_bounding_circle
    assert sphere_isclose(bounding_circle, Circle(max(a, b), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_minimal_centered_bounding_circle(a, b, center):
    ellipse = Ellipse(a, b, center)
    bounding_circle = ellipse.minimal_centered_bounding_circle
    assert sphere_isclose(bounding_circle, Circle(max(a, b), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_maximal_bounded_circle(a, b, center):
    ellipse = Ellipse(a, b, center)
    bounded_circle = ellipse.maximal_bounded_circle
    assert sphere_isclose(bounded_circle, Circle(min(a, b), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_maximal_centered_bounded_circle(a, b, center):
    ellipse = Ellipse(a, b, center)
    bounded_circle = ellipse.maximal_centered_bounded_circle
    assert sphere_isclose(bounded_circle, Circle(min(a, b), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_get_set_minimal_bounding_ellipse_radius(a, b, center):
    _test_get_set_minimal_bounding_sphere_radius(Ellipse(a, b, center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_get_set_minimal_centered_bounding_ellipse_radius(a, b, center):
    _test_get_set_minimal_bounding_sphere_radius(Ellipse(a, b, center), True)


def test_inertia_tensor():
    """Test the inertia tensor calculation."""
    ellipse = Ellipse(1, 2)
    ellipse.center = (0, 0, 0)
    assert np.sum(ellipse.inertia_tensor > 1e-6) == 1
    assert ellipse.inertia_tensor[2, 2] == approx(5 * np.pi / 2)


@given(
    floats(0.1, 10),
    floats(0.1, 10),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_is_inside(x, y, center):
    a, b = 1, 2
    ellipse = Ellipse(a, b, center)
    assert ellipse.is_inside([x, y, 0] + center).squeeze() == np.all(
        np.array([x / a, y / b]) <= 1
    )


def test_repr():
    ellipse = Ellipse(1, 2, [1, 2, 0])
    assert str(ellipse), str(eval(repr(ellipse)))
