import pytest
import numpy as np
from euclid.shape_classes.polyhedron import Polyhedron


# Need to declare this outside the fixture so that it can be used in multiple
# fixtures (pytest does not allow fixtures to be called).
def get_cube_points():
    return np.asarray([[0, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1],
                       [1, 0, 1]])


@pytest.fixture
def cube_points():
    return get_cube_points()


@pytest.fixture
def cube():
    return Polyhedron(get_cube_points())


def test_surface_area(cube):
    """Test surface area calculation."""
    assert cube.surface_area == 6
