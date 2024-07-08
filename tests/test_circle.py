# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pytest import approx

from conftest import (
    _test_get_set_minimal_bounding_sphere_radius,
    assert_distance_to_surface_2d,
    sphere_isclose,
)
from coxeter.shapes import Circle


@given(floats(0.1, 1000))
def test_perimeter(r):
    circle = Circle(1)
    circle.radius = r
    assert circle.perimeter == approx(2 * np.pi * r)
    assert circle.circumference == approx(2 * np.pi * r)


@given(floats(0.1, 1000))
def test_perimeter_setter(perimeter):
    """Test setting the perimeter."""
    circle = Circle(1)
    circle.perimeter = perimeter
    assert circle.radius == approx(perimeter / (2 * np.pi))
    assert circle.perimeter == approx(perimeter)
    circle.perimeter = perimeter + 1
    assert circle.radius == approx((perimeter + 1) / (2 * np.pi))
    assert circle.perimeter == approx(perimeter + 1)
    circle.circumference = perimeter
    assert circle.radius == approx(perimeter / (2 * np.pi))
    assert circle.circumference == approx(perimeter)
    circle.circumference = perimeter + 1
    assert circle.radius == approx((perimeter + 1) / (2 * np.pi))
    assert circle.circumference == approx(perimeter + 1)


@given(floats(0.1, 1000))
def test_area(r):
    circle = Circle(1)
    circle.radius = r
    assert circle.area == approx(np.pi * r**2)


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
    area = np.pi * r**2
    expected = [np.pi / 4 * r**4] * 3
    expected[0] += area * center[0] ** 2
    expected[1] += area * center[1] ** 2
    expected[2] = area * center[0] * center[1]
    np.testing.assert_allclose(circle.planar_moments_inertia[:2], expected[:2])
    np.testing.assert_allclose(circle.planar_moments_inertia[2], expected[2])
    assert circle.polar_moment_inertia == pytest.approx(sum(expected[:2]))


def test_center():
    """Test getting and setting the center."""
    circle = Circle(1)
    np.testing.assert_allclose(circle.center, (0, 0, 0))

    center = (1, 1, 1)
    circle.center = center
    np.testing.assert_allclose(circle.center, center)


@given(floats(0.1, 1000))
def test_radius_getter_setter(r):
    """Test getting and setting the radius."""
    circle = Circle(r)
    assert circle.radius == approx(r)
    circle.radius = r + 1
    assert circle.radius == approx(r + 1)


def test_invalid_radius():
    with pytest.raises(ValueError):
        Circle(-1)


def test_invalid_radius_setter():
    circle = Circle(1)
    with pytest.raises(ValueError):
        circle.radius = -1


@given(floats(0.1, 10))
def test_distance_to_surface(r):
    """Test calculating the shape distance."""
    theta = np.linspace(0, 2 * np.pi, 10000)
    circle = Circle(r)
    distance = circle.distance_to_surface(theta)
    assert_distance_to_surface_2d(circle, theta, distance)


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_minimal_bounding_circle(r, center):
    circ = Circle(r, center)
    assert sphere_isclose(circ.minimal_centered_bounding_circle, circ)


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_minimal_centered_bounding_circle(r, center):
    circ = Circle(r, center)
    assert sphere_isclose(circ.minimal_centered_bounding_circle, circ)


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_get_set_minimal_bounding_circle_radius(r, center):
    _test_get_set_minimal_bounding_sphere_radius(Circle(r, center))


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_get_set_minimal_centered_bounding_circle_radius(r, center):
    _test_get_set_minimal_bounding_sphere_radius(Circle(r, center), True)


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_maximal_bounded_circle(r, center):
    circ = Circle(r, center)
    assert sphere_isclose(circ.maximal_centered_bounded_circle, circ)


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_maximal_centered_bounded_circle(r, center):
    circ = Circle(r, center)
    assert sphere_isclose(circ.maximal_centered_bounded_circle, circ)


@given(
    floats(0.1, 10),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_is_inside(x, center):
    circle = Circle(1, center)
    assert circle.is_inside([x, 0, 0] + center).squeeze() == (x <= 1)


def test_inertia_tensor():
    """Test the inertia tensor calculation."""
    circle = Circle(1)
    circle.center = (0, 0, 0)
    assert np.sum(circle.inertia_tensor > 1e-6) == 1
    assert circle.inertia_tensor[2, 2] == approx(np.pi / 2)


def test_repr():
    circle = Circle(1, [1, 2, 0])
    assert str(circle), str(eval(repr(circle)))
