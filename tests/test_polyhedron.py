import pytest
import numpy as np
from euclid.shape_classes.polyhedron import Polyhedron
from euclid.polyhedron import ConvexPolyhedron
from hypothesis import given, example, assume
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays


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


def get_oriented_cube_facets():
    return np.array([[0, 1, 2, 3],  # Bottom face
                     [4, 7, 6, 5],  # Top face
                     [0, 3, 7, 4],  # Left face
                     [1, 5, 6, 2],  # Right face
                     [3, 2, 6, 7],  # Front face
                     [0, 4, 5, 1]])  # Back face


def get_oriented_cube_normals():
    return np.asarray([[0, 0, -1],
                       [0, 0, 1],
                       [0, -1, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [-1, 0, 0]])


@pytest.fixture
def cube_points():
    return get_cube_points()


@pytest.fixture
def cube():
    return Polyhedron(get_cube_points())


def test_surface_area(cube):
    """Test surface area calculation."""
    assert cube.surface_area == 6


def test_volume():
    cube = Polyhedron(get_cube_points(), facets=get_oriented_cube_facets(),
                      normals=get_oriented_cube_normals())
    assert cube.volume == 1


def test_merge_facets():
    """Test that coplanar facets can be correctly merged."""
    cube = Polyhedron(get_cube_points())
    cube.merge_facets()
    assert len(cube.facets) == 6


@given(arrays(np.float64, (3, ), floats(-10, 10, width=64)))
def test_volume_center_shift(new_center):
    """Make sure that moving the center doesn't affect the volume."""
    cube = Polyhedron(get_cube_points())
    cube.merge_facets()
    cube.sort_facets()
    cube.center = new_center
    assert np.isclose(cube.volume, 1)

def test_facet_alignment():
    """Make sure that facets are constructed correctly given vertices."""
    cube = Polyhedron(get_cube_points())
    cube.merge_facets()
    cube.sort_facets()

    def facet_to_string(facet):
        # Convenience function to create a string of vertex ids, which is the
        # easiest way to test for sequences that are cyclically equal.
        return ''.join([str(c) for c in facet])

    reference_facets = []
    for facet in get_oriented_cube_facets():
        reference_facets.append(facet_to_string(facet)*2)

    assert len(cube.facets) == len(reference_facets)

    for facet in cube.facets:
        str_facet = facet_to_string(facet)
        assert any([str_facet in ref for ref in reference_facets])
