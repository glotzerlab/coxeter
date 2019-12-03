import pytest
import numpy as np
import numpy.testing as npt
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


@pytest.fixture
def ones():
    return np.ones((4, 2))


def test_2d_verts(square_points):
    """Try creating object with 2D vertices."""
    square_points = square_points[:, :2]
    ConvexSpheropolygon(square_points, 1)


def test_duplicate_points(square_points):
    """Ensure that running with any duplicate points produces a warning."""
    square_points = np.vstack((square_points, square_points[[0]]))
    with pytest.raises(ValueError):
        ConvexSpheropolygon(square_points, 1)


def test_identical_points(ones):
    """Ensure that running with identical points produces an error."""
    with pytest.raises(ValueError):
        ConvexSpheropolygon(ones, 1)


def test_reordering(square_points, unit_rounded_square):
    """Test that vertices can be reordered appropriately."""
    npt.assert_equal(unit_rounded_square.vertices, square_points)

    unit_rounded_square.reorder_verts(True)
    # We need the roll because the algorithm attempts to minimize unexpected
    # vertex shuffling by keeping the original 0 vertex in place.
    reordered_points = np.roll(np.flip(square_points, axis=0), shift=1, axis=0)
    npt.assert_equal(unit_rounded_square.vertices, reordered_points)

    # Original vertices are clockwise, so they'll be flipped on construction if
    # we specify the normal.
    new_square = ConvexSpheropolygon(square_points, 1, normal=[0, 0, 1])
    npt.assert_equal(new_square.vertices, reordered_points)

    new_square.reorder_verts(True)
    npt.assert_equal(new_square.vertices, square_points)


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
