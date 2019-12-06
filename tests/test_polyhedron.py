import pytest
import numpy as np
from euclid.shape_classes.polyhedron import Polyhedron
from scipy.spatial import ConvexHull
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


# @given(arrays(np.float64, (5, 3), floats(-10, 10, width=64), unique=True))
# @example(points=np.array([[1.        , 1.0005    , 1.0015    ],
                          # [1.0025    , 1.0045    , 1.0055    ],
                          # [1.0065    , 1.00750053, 1.12304688],
                          # [1.15429688, 1.15625   , 1.16210938],
                          # [1.00390625, 1.20373154, 1.24704742]]))
# @example(points=np.array([[1.0351543 , 1.        , 1.0005    ],
                       # [1.0015    , 1.0025    , 1.0035    ],
                       # [1.0045    , 1.0055    , 1.00683504],
                       # [1.00781354, 1.00859256, 1.08362384],
                       # [1.00976737, 1.24999805, 1.25050087]]))
# @example(points=np.array([[0.00000000e+00, 5.00000000e-04, 1.50000000e-03],
                          # [2.50000000e-03, 3.50000000e-03, 4.50000000e-03],
                          # [5.50000000e-03, 1.95258026e-02, 1.24999455e+00],
                          # [1.25050000e+00, 9.76017761e-03, 3.90619553e-01],
                          # [8.12499794e+00, 8.12692097e+00, 8.33251408e+00]]))
# def test_convex_volume(points):
    # """Check the volumes of various convex sets."""
    # assume(points.size == np.unique(np.round(points, 3)).size)
    # hull = ConvexHull(points)
    # assume(hull.volume > 1)
    # verts = points[hull.vertices]
    # poly = Polyhedron(verts)
    # assert np.isclose(hull.volume, poly.volume)


@given(arrays(np.float64, (5, 3), floats(1, 5, width=64), unique=True))
def test_convex_surface_area(points):
    """Check the surface areas of various convex sets."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = Polyhedron(verts)
    assert np.isclose(hull.area, poly.surface_area)
