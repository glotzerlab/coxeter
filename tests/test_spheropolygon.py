import pytest
import numpy as np
from euclid.shape_classes.spheropolygon import ConvexSpheropolygon


def get_square_points():
    return np.asarray([[0, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0]])


@pytest.fixture
def square_points():
    return get_square_points()


@pytest.fixture
def unit_rounded_square():
    return ConvexSpheropolygon(get_square_points(), 1)


def test_area(unit_rounded_square):
    """Test area calculation."""
    shape = unit_rounded_square
    area = 1 + 4 + np.pi
    assert shape.signed_area == area
    assert shape.area == area

    # Ensure that area is signed.
    shape.reorder_verts(True)
    assert shape.signed_area == -area
    assert shape.area == area


def test_center(square_points, unit_rounded_square):
    """Test centering the polygon."""
    square = unit_rounded_square
    assert np.all(square.center == np.mean(square_points, axis=0))
    square.center = [0, 0, 0]
    assert np.all(square.center == [0, 0, 0])


def test_nonplanar(square_points):
    """Ensure that nonplanar vertices raise an error."""

    with pytest.raises(ValueError):
        square_points[0, 2] += 1
        ConvexSpheropolygon(square_points, 1)
