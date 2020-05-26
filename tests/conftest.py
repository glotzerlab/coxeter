import pytest
import numpy as np
from coxeter.shape_classes import Polyhedron
from coxeter.shape_classes import ConvexPolyhedron
from coxeter.shape_classes import ConvexSpheropolyhedron
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


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


def make_sphero_cube(radius=0):
    return ConvexSpheropolyhedron(get_cube_points(), radius)


@pytest.fixture
def cube_points():
    return get_cube_points()


@pytest.fixture
def convex_cube():
    return ConvexPolyhedron(get_cube_points())


@pytest.fixture
def oriented_cube():
    return Polyhedron(get_cube_points(), get_oriented_cube_facets())


@pytest.fixture
def unoriented_cube():
    """A cube with the facets disordered (but still provided)."""
    facets = get_oriented_cube_facets()
    for facet in facets:
        np.random.shuffle(facet)
    poly = Polyhedron(get_cube_points(), facets, facets_are_convex=True)
    poly.sort_facets()
    return poly


@pytest.fixture
def cube(request):
    return request.getfixturevalue(request.param)


def get_valid_hull(points, min_hull_area=1e-2):
    """To avoid issues from floating point error, we require any test that
    computes a convex hull from a random set of points to successfully build a
    hull, and the hull must have a reasonable finite area.

    Args:
        points (np.array):
            The points to compute a hull for.

    Returns:
        hull (scipy.spatial.ConvexHull) or False:
            A ConvexHull if the construction succeeded, otherwise False.
        min_hull_area (float):
            The minimum size of the hull required.
    """
    try:
        hull = ConvexHull(points)
    except QhullError:
        return False
    else:
        # Avoid cases where numerical imprecision make tests fail.
        if hull.volume > min_hull_area:
            return hull
        else:
            return False
