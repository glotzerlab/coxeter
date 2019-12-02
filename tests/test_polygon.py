import numpy as np
import numpy.testing as npt
from euclid.shape_classes.polygon import Polygon


def test_reordering():
    """Test that vertices can be reordered appropriately."""
    # Original vertices are clockwise, so they'll be flipped on construction.
    points = [[0, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]
    square = Polygon(points)
    # We need the roll because the algorithm attempts to minimize unexpected
    # vertex shuffling by keeping the original 0 vertex in place.
    npt.assert_equal(square.vertices,
                     np.roll(np.flip(points, axis=0), shift=1, axis=0))

    square.reorder_verts(True)
    npt.assert_equal(square.vertices, points)


def test_area():
    """Test area calculation."""
    # Original vertices are clockwise, so they'll be flipped on construction.
    points = [[0, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]
    square = Polygon(points)
    assert square.area == 1
