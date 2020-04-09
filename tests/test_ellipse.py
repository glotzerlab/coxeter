import pytest
import numpy as np
from coxeter.shape_classes.ellipse import Ellipse
from hypothesis import given
from hypothesis.strategies import floats


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_perimeter(a, b):
    """Check surface area against an approximate formula."""
    # Uses Ramanujan's approximation for the circumference of an ellipse:
    # https://en.wikipedia.org/wiki/Ellipse#Circumference
    ellipse = Ellipse(a, b)
    b, a = sorted([a, b])
    h = (a-b)**2 / (a+b)**2
    approx_perimeter = np.pi * (a+b) * (1 + 3*h/(10 + np.sqrt(4 - 3*h)))
    assert ellipse.perimeter == pytest.approx(approx_perimeter, rel=1e-3)
    assert ellipse.circumference == pytest.approx(approx_perimeter, rel=1e-3)


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_area(a, b):
    A = np.pi * a * b
    ellipse = Ellipse(1, 1)
    ellipse.a = a
    ellipse.b = b
    assert ellipse.area == A


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
    assert ellipse.eccentricity == pytest.approx(expected)


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_inertia_tensor(a, b):
    ellipse = Ellipse(a, b)
    assert np.all(np.asarray(ellipse.planar_moments_inertia) >= 0)

    expected = [np.pi / 4 * a * b**3, np.pi / 4 * a**3 * b]
    np.testing.assert_allclose(ellipse.planar_moments_inertia[:2], expected)
    np.testing.assert_allclose(ellipse.planar_moments_inertia[2], 0)
    assert ellipse.polar_moment_inertia == pytest.approx(sum(expected))
