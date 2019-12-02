import pytest
import numpy as np
import numpy.testing as npt
from euclid.shape_classes.polygon import Polygon


def get_square_points():
    return np.asarray([[0, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0]])


@pytest.fixture
def square_points():
    return get_square_points()


@pytest.fixture
def square():
    return Polygon(get_square_points())


def test_reordering(square_points, square):
    """Test that vertices can be reordered appropriately."""
    npt.assert_equal(square.vertices, square_points)

    square.reorder_verts(True)
    # We need the roll because the algorithm attempts to minimize unexpected
    # vertex shuffling by keeping the original 0 vertex in place.
    reordered_points = np.roll(np.flip(square_points, axis=0), shift=1, axis=0)
    npt.assert_equal(square.vertices, reordered_points)

    # Original vertices are clockwise, so they'll be flipped on construction if
    # we specify the normal.
    square = Polygon(square_points, normal=[0, 0, 1])
    npt.assert_equal(square.vertices, reordered_points)

    square.reorder_verts(True)
    npt.assert_equal(square.vertices, square_points)


def test_area(square_points):
    """Test area calculation."""
    # Shift to ensure that the negative areas are subtracted as needed.
    points = np.asarray(square_points) + 2
    square = Polygon(points)
    assert square.signed_area == 1
    assert square.area == 1

    # Ensure that area is signed.
    square.reorder_verts(True)
    assert square.signed_area == -1
    assert square.area == 1


def test_set_area(square):
    """Test setting area."""
    square.area = 2
    assert np.isclose(square.area, 2)


def test_center(square, square_points):
    """Test centering the polygon."""
    assert np.all(square.center == np.mean(square_points, axis=0))
    square.center = [0, 0, 0]
    assert np.all(square.center == [0, 0, 0])


def test_moment_inertia(square):
    """Test moment of inertia calculation."""
    square.center = [0, 0, 0]
    assert np.allclose(square.planar_moments_inertia, (1/12, 1/12, 0))
    assert np.isclose(square.polar_moment_inertia, 1/6)
