import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pytest import approx

from conftest import assert_shape_kernel_2d
from coxeter.shapes.ellipse import Ellipse


@given(floats(0.1, 1000), floats(0.1, 1000))
def test_a_b_getter_setter(a, b):
    """Test getter and setter for a and b."""
    ellipse = Ellipse(a, b)
    assert ellipse.a == a
    assert ellipse.b == b
    ellipse.a = a + 1
    ellipse.b = b + 1
    assert ellipse.a == a + 1
    assert ellipse.b == b + 1


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
    assert ellipse.perimeter == pytest.approx(approx_perimeter, rel=1e-3)
    assert ellipse.circumference == pytest.approx(approx_perimeter, rel=1e-3)


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
    assert ellipse.area == np.pi * a * b


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
    expected = np.sqrt(1 - b ** 2 / a ** 2)
    assert ellipse.eccentricity == pytest.approx(expected)


@given(
    floats(0.1, 1000),
    floats(0.1, 1000),
    arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True),
)
def test_inertia_tensor(a, b, center):
    ellipse = Ellipse(a, b)
    assert np.all(np.asarray(ellipse.planar_moments_inertia) >= 0)

    ellipse.center = center
    area = ellipse.area
    expected = [np.pi / 4 * a * b ** 3, np.pi / 4 * a ** 3 * b, 0]
    expected[0] += area * center[0] ** 2
    expected[1] += area * center[1] ** 2
    expected[2] = area * center[0] * center[1]
    np.testing.assert_allclose(ellipse.planar_moments_inertia[:2], expected[:2])
    np.testing.assert_allclose(ellipse.planar_moments_inertia[2], expected[2])
    assert ellipse.polar_moment_inertia == pytest.approx(sum(expected[:2]))


def test_center():
    """Test getting and setting the center."""
    ellipse = Ellipse(1, 2)
    assert all(ellipse.center == (0, 0, 0))

    center = (1, 1, 1)
    ellipse.center = center
    assert all(ellipse.center == center)


@given(floats(0.1, 10), floats(0.1, 10))
def test_shape_kernel(a, b):
    """Test consistent volume and area for shape kernel of an ellipse."""
    theta = np.linspace(0, 2 * np.pi, 50000)
    ellipse = Ellipse(a, b)
    kernel = ellipse.shape_kernel(theta)
    assert_shape_kernel_2d(ellipse, theta, kernel)
