import numpy as np
from coxeter.shape_classes.sphere import Sphere
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from coxeter.shape_classes.utils import translate_inertia_tensor


@given(floats(0.1, 1000))
def test_surface_area(r):
    sphere = Sphere(1)
    sphere.radius = r
    assert sphere.surface_area == 4 * np.pi * r**2


@given(floats(0.1, 1000))
def test_volume(r):
    sphere = Sphere(1)
    sphere.radius = r
    assert sphere.volume == 4 / 3 * np.pi * r**3


@given(floats(0.1, 1000))
def test_iq(r):
    sphere = Sphere(r)
    assert sphere.iq == 1


@given(floats(0.1, 1000),
       arrays(np.float64, (3, ), elements=floats(-10, 10, width=64),
              unique=True))
def test_inertia_tensor(r, center):
    sphere = Sphere(r)
    assert np.all(sphere.inertia_tensor >= 0)

    volume = 4 / 3 * np.pi * r**3
    expected = [2 / 5 * volume * r**2] * 3
    np.testing.assert_allclose(np.diag(sphere.inertia_tensor), expected)

    sphere.center = center
    expected = translate_inertia_tensor(center, np.diag(expected), volume)
    np.testing.assert_allclose(sphere.inertia_tensor, expected)


@given(floats(0.1, 10))
def test_is_inside(r):
    sphere = Sphere(1)
    assert sphere.is_inside([r, 0, 0]).squeeze() == (r <= 1)


def test_center():
    """Test getting and setting the center."""
    sphere = Sphere(1)
    assert all(sphere.center == (0, 0, 0))

    center = (1, 1, 1)
    sphere.center = center
    assert all(sphere.center == center)
