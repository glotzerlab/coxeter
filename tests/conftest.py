import os

import numpy as np
import pytest
import rowan
from hypothesis import settings
from hypothesis.strategies import builds, floats, integers
from scipy.spatial import ConvexHull

from coxeter.families import DOI_SHAPE_REPOSITORIES, PlatonicFamily, RegularNGonFamily
from coxeter.shapes import ConvexPolyhedron, ConvexSpheropolyhedron, Polyhedron, Shape2D

# Avoid deadlines on CI. coxeter prioritizes correctness over speed, and while
# it may be optimized in the future it is currently much more annoying to have
# tests fail due to deadline issues
settings.register_profile("ci", deadline=None)
if os.getenv("CI", "false") == "true" or os.getenv("CIRCLECI", "false") == "true":
    settings.load_profile("ci")


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
    return Polyhedron(get_cube_points(), get_oriented_cube_faces().tolist())


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


@pytest.fixture
def tetrahedron():
    tet = PlatonicFamily.get_shape("Tetrahedron")
    tet.volume = 1
    return tet


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
    ds = np.linalg.norm(points / [a, b, c if c else 1], axis=-1)
    points /= ds[:, np.newaxis]
    return points if c else points[:, :2]


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


def assert_distance_to_surface_2d(shape, angles, computed_distance):
    """Check correctness of 2d shape distance implementations."""
    xy = np.array(
        [computed_distance * np.cos(angles), computed_distance * np.sin(angles)]
    )
    xy = np.transpose(xy)
    hull = ConvexHull(xy)

    # Test the area
    assert np.isclose(shape.area, hull.volume)

    # Test the circumference
    assert np.isclose(shape.perimeter, hull.area)


def quaternion_from_axis_angle(x, y, z, theta):
    """Generate a quaternion from axis [x, y, z] and angle theta."""
    if x == y == z == 0:
        return np.array([1, 0, 0, 0])
    axis = np.array([x, y, z])
    axis /= np.linalg.norm(axis)
    return rowan.from_axis_angle(axis, theta)


Random3DRotationStrategy = builds(
    quaternion_from_axis_angle,
    floats(-1, 1, allow_nan=False),
    floats(-1, 1, allow_nan=False),
    floats(-1, 1, allow_nan=False),
    floats(0, 2 * np.pi, allow_nan=False),
).filter(lambda quat: not np.isnan(quat).any())

Random2DRotationStrategy = builds(
    quaternion_from_axis_angle,
    floats(0, 0, allow_nan=False),
    floats(0, 0, allow_nan=False),
    floats(-1, 1, allow_nan=False),
    floats(0, 2 * np.pi, allow_nan=False),
).filter(lambda quat: not np.isnan(quat).any())


def sphere_isclose(c1, c2, *args, **kwargs):
    """Check if two spheres are almost equal.

    Works for both circles and spheres. All args and kwargs are forwarded to
    np.isclose and np.allclose.
    """
    return np.isclose(c1.radius, c2.radius, *args, **kwargs) and np.allclose(
        c1.center, c2.center, *args, **kwargs
    )


def platonic_solids():
    """Generate platonic solids."""
    for shape_name in PlatonicFamily.data:
        yield PlatonicFamily.get_shape(shape_name)


# A convenient mark decorator that also includes names for the polyhedra.
# Assumes that the argument name is "poly".
_platonic_shape_names = PlatonicFamily.data.keys()
named_platonic_mark = pytest.mark.parametrize(
    argnames="poly",
    argvalues=[PlatonicFamily.get_shape(name) for name in _platonic_shape_names],
    ids=_platonic_shape_names,
)


def regular_polygons(n=10):
    """Generate regular polygons."""
    for i in range(3, n + 1):
        yield RegularNGonFamily.get_shape(i)


def _test_get_set_minimal_bounding_sphere_radius(shape, centered=False):
    """Test getting and setting the minimal bounding circle radius.

    This function will work for any shape in two or three dimensions based on
    the generic base class APIs, so it can be called in other pytest tests.
    """
    base_attr = "minimal" + ("_centered_" if centered else "_")
    sphere_type = "circle" if isinstance(shape, Shape2D) else "sphere"
    attr = base_attr + "bounding_" + sphere_type

    bounding_sphere = getattr(shape, attr)
    bounding_sphere_radius = getattr(shape, attr + "_radius")

    assert np.isclose(bounding_sphere_radius, bounding_sphere.radius)
    setattr(shape, attr + "_radius", bounding_sphere_radius * 2)
    assert np.isclose(getattr(shape, attr).radius, bounding_sphere_radius * 2)


# Generate the shapes from :cite:`Damasceno2012a`. Use the raw shape dicts to
# allow excluding subsets for different tests.
_damasceno_data = DOI_SHAPE_REPOSITORIES["10.1126/science.1220869"][0].data
_damasceno_shape_names = _damasceno_data.keys()
named_damasceno_shapes_mark = pytest.mark.parametrize(
    argnames="shape",
    argvalues=[_damasceno_data[shape_id] for shape_id in _damasceno_shape_names],
    ids=[
        f"{shape_id}: {_damasceno_data[shape_id]['name']}"
        for shape_id in _damasceno_shape_names
    ],
)
