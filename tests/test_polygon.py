import pytest
import numpy as np
import numpy.testing as npt
import rowan
from euclid.shape_classes.polygon import Polygon
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from hypothesis import given, example, assume
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays


# Need to declare this outside the fixture so that it can be used in multiple
# fixtures (pytest does not allow fixtures to be called).
def get_square_points():
    return np.asarray([[0, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0]])


@pytest.fixture
def square_points():
    return get_square_points()


@pytest.fixture
def square():
    return Polygon(get_square_points())


@pytest.fixture
def ones():
    return np.ones((4, 2))


def test_2d_verts(square_points):
    """Try creating object with 2D vertices."""
    square_points = square_points[:, :2]
    Polygon(square_points)


def test_duplicate_points(square_points):
    """Ensure that running with any duplicate points produces a warning."""
    square_points = np.vstack((square_points, square_points[[0]]))
    with pytest.raises(ValueError):
        Polygon(square_points)


def test_identical_points(ones):
    """Ensure that running with identical points produces an error."""
    with pytest.raises(ValueError):
        Polygon(ones)


def test_reordering(square_points, square):
    """Test that vertices can be reordered appropriately."""
    npt.assert_equal(square.vertices, square_points)

    square.reorder_verts(True)
    # We need the roll because the algorithm attempts to minimize unexpected
    # vertex shuffling by keeping the original 0 vertex in place.
    reordered_points = np.roll(np.flip(square_points, axis=0), shift=1, axis=0)
    npt.assert_equal(square.vertices, reordered_points)

    # Original vertices are clockwise, so they'll be flipped on construction if
    # we specify the normal.
    square = Polygon(square_points, normal=[0, 0, 1])
    npt.assert_equal(square.vertices, reordered_points)

    square.reorder_verts(True)
    npt.assert_equal(square.vertices, square_points)


def test_area(square_points):
    """Test area calculation."""
    # Shift to ensure that the negative areas are subtracted as needed.
    points = np.asarray(square_points) + 2
    square = Polygon(points)
    assert square.signed_area == 1
    assert square.area == 1

    # Ensure that area is signed.
    square.reorder_verts(True)
    assert square.signed_area == -1
    assert square.area == 1


def test_set_area(square):
    """Test setting area."""
    square.area = 2
    assert np.isclose(square.area, 2)


def test_center(square, square_points):
    """Test centering the polygon."""
    assert np.all(square.center == np.mean(square_points, axis=0))
    square.center = [0, 0, 0]
    assert np.all(square.center == [0, 0, 0])


def test_moment_inertia(square):
    """Test moment of inertia calculation."""
    square.center = [0, 0, 0]
    assert np.allclose(square.planar_moments_inertia, (1/12, 1/12, 0))
    assert np.isclose(square.polar_moment_inertia, 1/6)


def test_nonplanar(square_points):
    """Ensure that nonplanar vertices raise an error."""
    with pytest.raises(ValueError):
        square_points[0, 2] += 1
        Polygon(square_points)


@given(arrays(np.float64, (4, 2), floats(1, 5, width=64), unique=True))
@example(np.array([[1, 1],
                   [1, 1.00041707],
                   [2.78722762, 1],
                   [2.72755193, 1.32128906]]))
def test_reordering_convex(points):
    """Test that vertices can be reordered appropriately."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = Polygon(verts)
    assert np.all(poly.vertices[:, :2] == verts)


@given(arrays(np.float64, (4, 2), floats(-5, 5, width=64), unique=True))
@example(np.array([[1, 1],
                   [1, 1.00041707],
                   [2.78722762, 1],
                   [2.72755193, 1.32128906]]))
def test_convex_area(points):
    """Check the areas of various convex sets."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = Polygon(verts)
    assert np.isclose(hull.volume, poly.area)


@given(random_quat=arrays(np.float64, (4, ), floats(-1, 1, width=64)))
def test_rotation_signed_area(random_quat, square_points):
    """Ensure that rotating does not change the signed area."""
    assume(not np.all(random_quat == 0))
    random_quat = rowan.normalize(random_quat)
    rotated_points = rowan.rotate(random_quat, square_points)
    poly = Polygon(rotated_points)
    assert np.isclose(poly.signed_area, 1)

    poly.reorder_verts(clockwise=True)
    assert np.isclose(poly.signed_area, -1)


@given(arrays(np.float64, (4, 2), floats(-5, 5, width=64), unique=True))
def test_set_convex_area(points):
    """Test setting area of arbitrary convex sets."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = Polygon(verts)
    original_area = poly.area
    poly.area *= 2
    assert np.isclose(poly.area, 2*original_area)


def test_triangulate(square):
    triangles = [tri for tri in square._triangulation()]
    assert len(triangles) == 2

    all_vertices = [tuple(vertex) for triangle in triangles for vertex in
                    triangle]
    assert len(set(all_vertices)) == 4

    assert not np.all(np.asarray(triangles[0]) == np.asarray(triangles[1]))


def test_circumsphere_radius_regular_polygon():
    from geometry import get_unit_area_ngon
    import miniball
    for i in range(3, 10):
        vertices = get_unit_area_ngon(i)
        rmax = np.max(np.linalg.norm(vertices, axis=-1))
        C, r2 = miniball.get_bounding_ball(vertices)
        assert np.isclose(rmax, np.sqrt(r2))
        assert np.allclose(C, 0)


@given(arrays(np.float64, (3, 2), floats(-5, 5, width=64), unique=True))
def test_circumsphere_radius_random_hull(points):
    import miniball

    try:
        hull = ConvexHull(points)
    except QhullError:
        assume(False)
    else:
        # Avoid cases where numerical imprecision make tests fail.
        assume(hull.volume > 1e-1)

    vertices = points[hull.vertices]
    poly = Polygon(vertices)
    poly.center = [0, 0, 0]
    print(poly.vertices)
    print(np.mean(poly.vertices))

    # For an arbitrary convex polygon, the furthest vertex from the origin is
    # an upper bound on the bounding sphere radius, but need not be the radius
    # because the ball need not be centered at the centroid.
    rmax = np.max(np.linalg.norm(poly.vertices, axis=-1))
    C, r2 = miniball.get_bounding_ball(vertices)
    assert np.sqrt(r2) <= rmax

    C, r2 = miniball.get_bounding_ball(poly.vertices)
    assert np.sqrt(r2) <= rmax
