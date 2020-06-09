import numpy as np
import pytest
from hypothesis.strategies import builds, floats, integers
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from coxeter.shape_classes import ConvexPolyhedron, ConvexSpheropolyhedron, Polyhedron


# Need to declare this outside the fixture so that it can be used in multiple
# fixtures (pytest does not allow fixtures to be called).
def get_cube_points():
    return np.asarray(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ]
    )


def get_oriented_cube_faces():
    return np.array(
        [
            [0, 1, 2, 3],  # Bottom face
            [4, 7, 6, 5],  # Top face
            [0, 3, 7, 4],  # Left face
            [1, 5, 6, 2],  # Right face
            [3, 2, 6, 7],  # Front face
            [0, 4, 5, 1],
        ]
    )  # Back face


def get_oriented_cube_normals():
    return np.asarray(
        [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0]]
    )


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
    return Polyhedron(get_cube_points(), get_oriented_cube_faces())


@pytest.fixture
def unoriented_cube():
    """Get a cube with the faces out of order on construction."""
    faces = get_oriented_cube_faces()
    for face in faces:
        np.random.shuffle(face)
    poly = Polyhedron(get_cube_points(), faces, faces_are_convex=True)
    poly.sort_faces()
    return poly


@pytest.fixture
def cube(request):
    return request.getfixturevalue(request.param)


def get_valid_hull(points, min_hull_area=1e-2):
    """Get a convex hull that adheres to our requirements.

    To avoid issues from floating point error, we require any test that
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


def points_from_ellipsoid_surface(a, b, c=0, n=10):
    """Sample points on an ellipsoid.

    The ellipsoid is given by the equation :math:`x^2/a^2 + y^2/b^2 + z^2/c^2 = 1`.

    Args:
        a (float):
            The semi-major axis along x.
        b (float):
            The semi-major axis along y.
        c (float, optional):
            The semi-major axis along z. If it is ``0``, the returned array is an array
            of 2D points on the surface of an ellipse.
        n (int, optional):
            The number of points on the surface of the ellipsoid (ellipse).

    Returns:
        :math:`(N, 3)` or :math:`(N, 2)` :class:`numpy.ndarray`: The points.
    """
    points = []
    points = np.zeros((n, 3))
    points[:, 0] = np.random.normal(0, a, n)
    points[:, 1] = np.random.normal(0, b, n)
    if c > 0:
        points[:, 2] = np.random.normal(0, c, n)
    ds = np.linalg.norm(points / [a, b, c], axis=-1)
    points /= ds[:, np.newaxis]
    return points if c else points[:, :2]


def polyhedron_from_ellipsoid_surface(a, b, c, n):
    """Generate a polyhedron from points on an ellipsoidal surface.

    See :func:`points_from_ellipsoid_surface` for the parameters.
    """
    return ConvexPolyhedron(points_from_ellipsoid_surface(a, b, c, n))


EllipsoidSurfaceStrategy = builds(
    points_from_ellipsoid_surface,
    floats(0.1, 5),
    floats(0.1, 5),
    floats(0.1, 5),
    integers(5, 15),
)


EllipseSurfaceStrategy = builds(
    points_from_ellipsoid_surface, floats(0.1, 5), floats(0.1, 5), n=integers(5, 15)
)
