# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pytest import approx

from conftest import _test_get_set_minimal_bounding_sphere_radius, sphere_isclose
from coxeter.shapes import Ellipsoid, Sphere
from coxeter.shapes.utils import translate_inertia_tensor


@given(floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000))
def test_a_b_c_getter_setter(a, b, c):
    """Test getter and setter for a, b, and c."""
    ellipsoid = Ellipsoid(a, b, c)
    assert ellipsoid.a == a
    assert ellipsoid.b == b
    assert ellipsoid.c == c
    ellipsoid.a = a + 1
    ellipsoid.b = b + 1
    ellipsoid.c = c + 1
    assert ellipsoid.a == a + 1
    assert ellipsoid.b == b + 1
    assert ellipsoid.c == c + 1


@given(floats(-1000, -1))
def test_invalid_a_b_c_setter(a):
    """Test setting invalid a, b, c values."""
    # a is invalid
    ellipsoid = Ellipsoid(1, 1, 1)
    with pytest.raises(ValueError):
        ellipsoid.a = a
    with pytest.raises(ValueError):
        ellipsoid.b = a
    with pytest.raises(ValueError):
        ellipsoid.c = a


@given(floats(-1000, -1), floats(0.1, 1000), floats(0.1, 1000))
def test_invalid_a_b_c_constructor(a, b, c):
    """Test invalid a, b, c values in constructor."""
    with pytest.raises(ValueError):
        Ellipsoid(a, b, c)
    with pytest.raises(ValueError):
        Ellipsoid(c, a, b)
    with pytest.raises(ValueError):
        Ellipsoid(b, c, a)


@given(floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000))
def test_surface_area(a, b, c):
    """Check surface area against an approximate formula."""
    # Approximation from:
    # https://en.wikipedia.org/wiki/Ellipsoid#Approximate_formula
    p = 1.6075
    approx_surface = (
        4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3) ** (1 / p)
    )

    ellipsoid = Ellipsoid(a, b, c)
    assert ellipsoid.surface_area == approx(approx_surface, rel=0.015)


@given(floats(0.1, 1000))
def test_set_surface_area(value):
    """Test setting the surface area."""
    ellipsoid = Ellipsoid(1, 2, 3)
    area_old = ellipsoid.surface_area
    ellipsoid.surface_area = value
    assert ellipsoid.surface_area == approx(value)
    assert ellipsoid.a == approx(1 * np.sqrt(value / area_old))
    assert ellipsoid.b == approx(2 * np.sqrt(value / area_old))
    assert ellipsoid.c == approx(3 * np.sqrt(value / area_old))


@given(floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000))
def test_volume(a, b, c):
    ellipsoid = Ellipsoid(1, 1, 1)
    ellipsoid.a = a
    ellipsoid.b = b
    ellipsoid.c = c
    assert ellipsoid.volume == approx(4 / 3 * np.pi * a * b * c)


@given(floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000))
def test_set_volume(a, b, c, volume):
    """Test setting the volume."""
    ellipsoid = Ellipsoid(a, b, c)
    original_volume = ellipsoid.volume
    ellipsoid.volume = volume
    assert ellipsoid.volume == approx(volume)

    # Reset to original volume
    ellipsoid.volume = original_volume
    assert ellipsoid.volume == approx(original_volume)
    assert ellipsoid.a == approx(a)
    assert ellipsoid.b == approx(b)
    assert ellipsoid.c == approx(c)


def test_earth():
    """Approximate Earth's volume and surface area."""
    # Uses data (in meters) from GRS 80:
    # https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980
    a = b = 6378137
    c = 6356752.314
    earth_volume = 1.08e21  # or so, m^3
    earth_surface = 510.1e12  # or so, m^2
    earth = Ellipsoid(a, b, c)
    assert earth.volume == approx(earth_volume, rel=0.01)
    assert earth.surface_area == approx(earth_surface, rel=0.01)


@given(floats(0.1, 1000))
def test_iq(r):
    sphere = Ellipsoid(r, r, r)
    assert sphere.iq == approx(1)


@given(floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000))
def test_iq_symmetry(a, b, c):
    ellipsoid1 = Ellipsoid(a, b, c)
    ellipsoid2 = Ellipsoid(c, a, b)
    ellipsoid3 = Ellipsoid(b, c, a)
    assert ellipsoid1.iq == approx(ellipsoid2.iq)
    assert ellipsoid1.iq == approx(ellipsoid3.iq)
    assert ellipsoid2.iq == approx(ellipsoid3.iq)


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_inertia_tensor(a, b, c, center):
    # First just test a sphere.
    ellipsoid = Ellipsoid(a, a, a)
    assert np.all(ellipsoid.inertia_tensor >= 0)

    volume = ellipsoid.volume
    expected = [2 / 5 * volume * a**2] * 3
    np.testing.assert_allclose(np.diag(ellipsoid.inertia_tensor), expected)

    ellipsoid.center = center
    expected = translate_inertia_tensor(center, np.diag(expected), volume)
    np.testing.assert_allclose(ellipsoid.inertia_tensor, expected)


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_is_inside(a, b, c, center):
    ellipsoid = Ellipsoid(a, b, c, center)
    # Add a small fudge factor to avoid floating point error.
    points_inside = (1 - 1e-6) * np.array(
        [[a, 0, 0], [-a, 0, 0], [0, b, 0], [0, -b, 0], [0, 0, c], [0, 0, -c]]
    )
    assert all(ellipsoid.is_inside(points_inside + center))
    assert all(ellipsoid.is_inside(points_inside / 2 + center))
    assert not any(ellipsoid.is_inside(points_inside * 1.1 + center))


def test_center():
    """Test getting and setting the center."""
    ellipsoid = Ellipsoid(1, 2, 3)
    np.testing.assert_allclose(ellipsoid.center, (0, 0, 0))

    center = (1, 1, 1)
    ellipsoid.center = center
    np.testing.assert_allclose(ellipsoid.center, center)


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_minimal_bounding_sphere(a, b, c, center):
    ellipsoid = Ellipsoid(a, b, c, center)
    bounding_sphere = ellipsoid.minimal_bounding_sphere
    sphere_isclose(bounding_sphere, Sphere(max(a, b, c), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_minimal_centered_bounding_sphere(a, b, c, center):
    ellipsoid = Ellipsoid(a, b, c, center)
    bounding_sphere = ellipsoid.minimal_centered_bounding_sphere
    sphere_isclose(bounding_sphere, Sphere(max(a, b, c), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_maximal_bounded_sphere(a, b, c, center):
    ellipsoid = Ellipsoid(a, b, c, center)
    bounded_sphere = ellipsoid.maximal_bounded_sphere
    sphere_isclose(bounded_sphere, Sphere(min(a, b, c), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_maximal_centered_bounded_sphere(a, b, c, center):
    ellipsoid = Ellipsoid(a, b, c, center)
    bounded_sphere = ellipsoid.maximal_centered_bounded_sphere
    sphere_isclose(bounded_sphere, Sphere(min(a, b, c), center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_get_set_minimal_bounding_circle_radius(a, b, c, center):
    _test_get_set_minimal_bounding_sphere_radius(Ellipsoid(a, b, c, center))


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_get_set_minimal_centered_bounding_circle_radius(a, b, c, center):
    _test_get_set_minimal_bounding_sphere_radius(Ellipsoid(a, b, c, center), True)


def test_repr():
    ellipsoid = Ellipsoid(1, 2, 3, [1, 2, 3])
    assert str(ellipsoid), str(eval(repr(ellipsoid)))


@given(floats(0.1, 1000), floats(0.1, 1000), floats(0.1, 1000))
def test_to_hoomd(a, b, c):
    ellipsoid = Ellipsoid(a, b, c)
    dict_keys = ["a", "b", "c", "centroid", "volume", "moment_inertia"]
    dict_vals = [
        ellipsoid.a,
        ellipsoid.b,
        ellipsoid.c,
        [0, 0, 0],
        ellipsoid.volume,
        ellipsoid.inertia_tensor,
    ]
    hoomd_dict = ellipsoid.to_hoomd()
    for key, val in zip(dict_keys, dict_vals):
        assert np.allclose(hoomd_dict[key], val), f"{key}"
