import numpy as np
import numpy.testing as npt
import pytest
import rowan
from hypothesis import assume, example, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from scipy.spatial import ConvexHull

from conftest import EllipseSurfaceStrategy
from coxeter.shape_classes import ConvexSpheropolygon


def get_square_points():
    return np.asarray([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])


@pytest.fixture
def square_points():
    return get_square_points()


@pytest.fixture
def unit_rounded_square():
    return ConvexSpheropolygon(get_square_points(), 1)


@pytest.fixture
def ones():
    return np.ones((4, 2))


def test_2d_verts(square_points):
    """Try creating object with 2D vertices."""
    square_points = square_points[:, :2]
    ConvexSpheropolygon(square_points, 1)


def test_duplicate_points(square_points):
    """Ensure that running with any duplicate points produces a warning."""
    square_points = np.vstack((square_points, square_points[[0]]))
    with pytest.raises(ValueError):
        ConvexSpheropolygon(square_points, 1)


def test_identical_points(ones):
    """Ensure that running with identical points produces an error."""
    with pytest.raises(ValueError):
        ConvexSpheropolygon(ones, 1)


def test_reordering(square_points, unit_rounded_square):
    """Test that vertices can be reordered appropriately."""
    npt.assert_equal(unit_rounded_square.vertices, square_points)

    unit_rounded_square.reorder_verts(True)
    # We need the roll because the algorithm attempts to minimize unexpected
    # vertex shuffling by keeping the original 0 vertex in place.
    reordered_points = np.roll(np.flip(square_points, axis=0), shift=1, axis=0)
    npt.assert_equal(unit_rounded_square.vertices, reordered_points)

    # Original vertices are clockwise, so they'll be flipped on construction if
    # we specify the normal.
    new_square = ConvexSpheropolygon(square_points, 1, normal=[0, 0, 1])
    npt.assert_equal(new_square.vertices, reordered_points)

    new_square.reorder_verts(True)
    npt.assert_equal(new_square.vertices, square_points)


def test_area(unit_rounded_square):
    """Test area calculation."""
    shape = unit_rounded_square
    area = 1 + 4 + np.pi
    assert shape.signed_area == area
    assert shape.area == area

    # Ensure that area is signed.
    shape.reorder_verts(True)
    assert shape.signed_area == -area
    assert shape.area == area


def test_center(square_points, unit_rounded_square):
    """Test centering the polygon."""
    square = unit_rounded_square
    assert np.all(square.center == np.mean(square_points, axis=0))
    square.center = [0, 0, 0]
    assert np.all(square.center == [0, 0, 0])


def test_nonplanar(square_points):
    """Ensure that nonplanar vertices raise an error."""
    with pytest.raises(ValueError):
        square_points[0, 2] += 1
        ConvexSpheropolygon(square_points, 1)


@given(EllipseSurfaceStrategy)
def test_reordering_convex(points):
    """Test that vertices can be reordered appropriately."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = ConvexSpheropolygon(verts, radius=1)
    assert np.all(poly.vertices[:, :2] == verts)


@given(EllipseSurfaceStrategy)
def test_convex_area(points):
    """Check the areas of various convex sets."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    r = 1
    poly = ConvexSpheropolygon(verts, radius=r)

    cap_area = np.pi * r * r
    edge_area = np.sum(np.linalg.norm(verts - np.roll(verts, 1, 0), axis=1), axis=0)
    assert np.isclose(hull.volume + edge_area + cap_area, poly.area)


def test_convex_signed_area(square_points):
    """Ensure that rotating does not change the signed area."""

    @given(random_quat=arrays(np.float64, (4,), elements=floats(-1, 1, width=64)))
    @example(
        random_quat=np.array(
            [0.00000000e00, 2.22044605e-16, 2.60771169e-08, 2.60771169e-08]
        )
    )
    def testfun(random_quat):
        assume(not np.all(random_quat == 0))
        random_quat = rowan.normalize(random_quat)
        rotated_points = rowan.rotate(random_quat, square_points)
        r = 1
        poly = ConvexSpheropolygon(rotated_points, radius=r)

        hull = ConvexHull(square_points[:, :2])

        cap_area = np.pi * r * r
        edge_area = np.sum(
            np.linalg.norm(square_points - np.roll(square_points, 1, 0), axis=1), axis=0
        )
        sphero_area = cap_area + edge_area
        assert np.isclose(poly.signed_area, hull.volume + sphero_area)

        poly.reorder_verts(clockwise=True)
        assert np.isclose(poly.signed_area, -hull.volume - sphero_area)

    testfun()
