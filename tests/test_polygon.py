import pytest
import numpy as np
import numpy.testing as npt
from euclid.shape_classes.polygon import Polygon


@pytest.fixture
def square():
    points = [[0, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]
    return Polygon(points)


def test_reordering(square):
    """Test that vertices can be reordered appropriately."""
    original_verts = square.vertices
    square.reorder_verts(True)
    npt.assert_equal(square.vertices, original_verts)

    square.reorder_verts()
    npt.assert_equal(square.vertices,
                     np.roll(np.flip(original_verts, axis=0), shift=1, axis=0))
