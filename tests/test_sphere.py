import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pytest import approx

from coxeter.shapes.sphere import Sphere
from coxeter.shapes.utils import translate_inertia_tensor


@given(floats(0.1, 1000))
def test_radius_getter_setter(r):
    sphere = Sphere(r)
    assert sphere.radius == r
    sphere.radius = r + 1
    assert sphere.radius == r + 1


@given(floats(-1000, -1))
def test_invalid_radius_setter(r):
    """Test setting an invalid Volume."""
    with pytest.raises(ValueError):
        Sphere(r)


@given(floats(0.1, 1000))
def test_surface_area(r):
    sphere = Sphere(1)
    sphere.radius = r
    assert sphere.surface_area == 4 * np.pi * r ** 2


@given(floats(0.1, 1000))
def test_volume(r):
    sphere = Sphere(1)
    sphere.radius = r
    assert sphere.volume == 4 / 3 * np.pi * r ** 3


@given(floats(-1000, -1))
def test_invalid_volume_setter(volume):
    """Test setting an invalid Volume."""
    sphere = Sphere(1)
    with pytest.raises(ValueError):
        sphere.volume = volume


@given(floats(0.1, 1000))
def test_set_volume(volume):
    """Test setting the volume."""
    sphere = Sphere(1)
    sphere.volume = volume
    assert sphere.volume == approx(volume)
    assert sphere.radius == approx((3 * volume / (4 * np.pi)) ** (1 / 3))


@given(floats(0.1, 1000))
def test_area_getter_setter(area):
    """Test setting the volume."""
    sphere = Sphere(1)
    sphere.surface_area = area
    assert sphere.surface_area == approx(area)
    assert sphere.radius == approx(np.sqrt(area / (4 * np.pi)))


@given(floats(-1000, -1))
def test_invalid_area_setter(area):
    """Test setting an invalid Volume."""
    sphere = Sphere(1)
    with pytest.raises(ValueError):
        sphere.surface_area = area


@given(floats(0.1, 1000))
def test_iq(r):
    sphere = Sphere(r)
    assert sphere.iq == 1


@given(
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_inertia_tensor(r, center):
    sphere = Sphere(r)
    assert np.all(sphere.inertia_tensor >= 0)

    volume = 4 / 3 * np.pi * r ** 3
    expected = [2 / 5 * volume * r ** 2] * 3
    np.testing.assert_allclose(np.diag(sphere.inertia_tensor), expected)

    sphere.center = center
    expected = translate_inertia_tensor(center, np.diag(expected), volume)
    np.testing.assert_allclose(sphere.inertia_tensor, expected)


@given(
    floats(0.1, 10),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_is_inside(x, center):
    sphere = Sphere(1, center)
    assert sphere.is_inside([x, 0, 0] + center).squeeze() == (x <= 1)


def test_center():
    """Test getting and setting the center."""
    sphere = Sphere(1)
    assert all(sphere.center == (0, 0, 0))

    center = (1, 1, 1)
    sphere.center = center
    assert all(sphere.center == center)


def test_form_factor():
    """Validate the form factor of a sphere.

    At the moment this is primarily a regression test, and should be expanded for more
    rigorous validation.
    """
    q = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 2, 3],
            [-2, 4, -5.2],
        ],
        dtype=np.float,
    )

    sphere = Sphere(0.5)
    np.testing.assert_allclose(
        sphere.compute_form_factor_amplitude(q),
        [
            0.52359878,
            0.51062514,
            0.51062514,
            0.47307465,
            0.51062514,
            0.51062514,
            0.36181941,
            0.11702976,
        ],
        atol=1e-7,
    )

    sphere.center = [1, 1, 1]
    np.testing.assert_allclose(
        sphere.compute_form_factor_amplitude(q),
        [
            0.52359878 + 0.0j,
            0.27589194 - 0.42967624j,
            0.27589194 + 0.42967624j,
            -0.19686852 - 0.43016557j,
            0.27589194 - 0.42967624j,
            0.27589194 - 0.42967624j,
            0.34740824 + 0.10109795j,
            -0.11683018 - 0.00683151j,
        ],
        atol=1e-7,
    )
