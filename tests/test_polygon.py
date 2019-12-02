import numpy as np
import numpy.testing as npt
from euclid.shape_classes.polygon import Polygon


def test_reordering():
    """Test that vertices can be reordered appropriately."""
    points = [[0, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]
    square = Polygon(points)
    npt.assert_equal(square.vertices, points)

    square.reorder_verts(True)
    # We need the roll because the algorithm attempts to minimize unexpected
    # vertex shuffling by keeping the original 0 vertex in place.
    reordered_points = np.roll(np.flip(points, axis=0), shift=1, axis=0)
    npt.assert_equal(square.vertices, reordered_points)

    # Original vertices are clockwise, so they'll be flipped on construction if
    # we specify the normal.
    square = Polygon(points, normal=[0, 0, 1])
    npt.assert_equal(square.vertices, reordered_points)

    square.reorder_verts(True)
    npt.assert_equal(square.vertices, points)


def test_area():
    """Test area calculation."""
    points = [[0, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]
    # Shift to ensure that the negative areas are subtracted as needed.
    points = np.asarray(points) + 2
    square = Polygon(points)
    assert square.signed_area == 1
    assert square.area == 1

    # Ensure that area is signed.
    square.reorder_verts(True)
    assert square.signed_area == -1
    assert square.area == 1


def test_center():
    """Test centering the polygon."""
    points = [[0, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]
    square = Polygon(points)
    assert np.all(square.center == np.mean(points, axis=0))
    square.center = [0, 0, 0]
    assert np.all(square.center == [0, 0, 0])


def test_moment_inertia():
    """Test moment of inertia calculation."""
    points = [[0, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]
    square = Polygon(points)
    square.center = [0, 0, 0]
    assert np.allclose(square.planar_moments_inertia, (1/12, 1/12, 0))
    assert np.isclose(square.polar_moment_inertia, 1/6)
