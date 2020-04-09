import numpy as np
from coxeter.shape_classes.sphere import Sphere
from hypothesis import given
from hypothesis.strategies import floats


@given(floats(0.1, 1000))
def test_surface_area(r):
    S = 4 * np.pi * r**2
    sphere = Sphere(1)
    sphere.radius = r
    assert sphere.surface_area == S


@given(floats(0.1, 1000))
def test_volume(r):
    V = 4/3 * np.pi * r**3
    sphere = Sphere(1)
    sphere.radius = r
    assert sphere.volume == V


@given(floats(0.1, 1000))
def test_iq(r):
    sphere = Sphere(r)
    assert sphere.iq == 1


@given(floats(0.1, 1000))
def test_inertia_tensor(r):
    sphere = Sphere(r)
    assert np.all(sphere.inertia_tensor >= 0)

    expected = 2/5 * sphere.volume * r**2
    np.testing.assert_allclose(np.diag(sphere.inertia_tensor), expected)


@given(floats(0.1, 10))
def test_is_inside(r):
    sphere = Sphere(1)
    assert sphere.is_inside([r, 0, 0]).squeeze() == (r <= 1)
