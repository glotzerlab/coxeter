import pytest
import numpy as np
from euclid.shape_classes.polyhedron import Polyhedron
from euclid.polyhedron import ConvexPolyhedron
from scipy.spatial import ConvexHull, Delaunay
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from euclid.damasceno import SHAPES


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


@pytest.fixture
def oriented_cube():
    return Polyhedron(get_cube_points(), get_oriented_cube_facets(),
                      get_oriented_cube_normals())


def test_surface_area(cube):
    """Test surface area calculation."""
    assert cube.surface_area == 6


def test_volume():
    cube = Polyhedron(get_cube_points(), facets=get_oriented_cube_facets(),
                      normals=get_oriented_cube_normals())
    assert cube.volume == 1


def test_merge_facets(cube):
    """Test that coplanar facets can be correctly merged."""
    cube.merge_facets()
    assert len(cube.facets) == 6


@given(new_center=arrays(np.float64, (3, ), floats(-10, 10, width=64)))
def test_volume_center_shift(cube, new_center):
    """Make sure that moving the center doesn't affect the volume."""
    cube.merge_facets()
    cube.sort_facets()
    cube.center = new_center
    assert np.isclose(cube.volume, 1)


def test_facet_alignment(cube):
    """Make sure that facets are constructed correctly given vertices."""
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


@given(arrays(np.float64, (5, 3), floats(-10, 10, width=64), unique=True))
def test_convex_volume(points):
    """Check the volumes of various convex sets."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = Polyhedron(verts)
    poly.merge_facets()
    poly.sort_facets()

    assert np.isclose(hull.volume, poly.volume)


@given(arrays(np.float64, (5, 3), floats(-10, 10, width=64), unique=True))
def test_convex_surface_area(points):
    """Check the surface areas of various convex sets."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = Polyhedron(verts)
    poly.merge_facets()
    poly.sort_facets()
    assert np.isclose(hull.area, poly.surface_area)


def compute_inertia_mc(vertices, num_samples=1e7):
    """Use Monte Carlo integration to compute the moment of inertia."""
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)

    points = np.random.rand(int(num_samples), 3)*(maxs-mins)+mins

    hull = Delaunay(vertices)
    inside = hull.find_simplex(points) >= 0

    Ixx = np.mean(points[inside][:, 1]**2 + points[inside][:, 2]**2)
    Iyy = np.mean(points[inside][:, 0]**2 + points[inside][:, 2]**2)
    Izz = np.mean(points[inside][:, 0]**2 + points[inside][:, 1]**2)
    Ixy = np.mean(-points[inside][:, 0] * points[inside][:, 1]**2)
    Ixz = np.mean(-points[inside][:, 0] * points[inside][:, 2]**2)
    Iyz = np.mean(-points[inside][:, 1] * points[inside][:, 2]**2)

    poly = Polyhedron(vertices)
    poly.merge_facets()
    poly.sort_facets()

    inertia_tensor = np.array([[Ixx, Ixy, Ixz],
                               [Ixy,   Iyy, Iyz],
                               [Ixz,   Iyz,   Izz]]) * poly.volume

    return inertia_tensor


def test_moment_inertia(cube):
    cube.merge_facets()
    cube.sort_facets()
    assert np.allclose(cube.inertia_tensor, np.diag([1/6]*3))


def test_volume2():
    for i in range(1, len(SHAPES)-1):
        shape = SHAPES[i]
        poly_new = Polyhedron(shape.vertices)
        poly_new.merge_facets()
        poly_new.sort_facets()
        poly_old = ConvexPolyhedron(shape.vertices)
        if not np.allclose(poly_new.volume, poly_old.getVolume(), atol=1e-3):
            print("Not good for shape ", shape.Name,
                  ", values were {} and {}".format(
                      poly_new.volume, poly_old.getVolume(), atol=1e-3))
        else:
            print("Good for shape ", shape.Name)


def test_moment_inertia_damasceno_shapes():
    for i in range(1, len(SHAPES)-1):
        shape = SHAPES[i]
        poly = Polyhedron(shape.vertices)
        poly.merge_facets()
        poly.sort_facets()
        assert np.allclose(poly.inertia_tensor,
                           compute_inertia_mc(poly.vertices - poly.center),
                           atol=1e-2)
