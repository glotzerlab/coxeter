import numpy as np
import numpy.testing as npt
import pytest
import rowan
from hypothesis import assume, example, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pytest import approx
from scipy.spatial import ConvexHull

from conftest import assert_shape_kernel_2d, EllipseSurfaceStrategy
from coxeter.families import RegularNGonFamily
from coxeter.shapes.convex_polygon import ConvexPolygon
from coxeter.shapes.polygon import Polygon


def polygon_from_hull(verts):
    """Generate a polygon from a hull if possible.

    This function tries to generate a polygon from a hull, and returns False if
    it fails so that Hypothesis can simply assume(False) if the hull is nearly
    degenerate.
    """
    try:
        poly = Polygon(verts)
    except AssertionError:
        # Don't worry about failures caused by bad hulls that fail the simple
        # polygon test.
        return False
    return poly


# Need to declare this outside the fixture so that it can be used in multiple
# fixtures (pytest does not allow fixtures to be called).
def get_square_points():
    return np.asarray([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])


@pytest.fixture
def square_points():
    return get_square_points()


@pytest.fixture
def square():
    return Polygon(get_square_points())


@pytest.fixture
def convex_square():
    return ConvexPolygon(get_square_points())


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
    # we specify the normal. Note that we MUST use the Convexpolygon class,
    # since Polygon will not sort by default.
    square = ConvexPolygon(square_points, normal=[0, 0, 1])
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
    # First test the default values.
    square.center = (0, 0, 0)
    assert np.allclose(square.planar_moments_inertia, (1 / 12, 1 / 12, 0))
    assert np.isclose(square.polar_moment_inertia, 1 / 6)

    # Use hypothesis to validate the simple parallel axis theorem.
    @given(arrays(np.float64, (3,), elements=floats(-5, 5, width=64), unique=True))
    def testfun(center):
        # Just move in the plane.
        center[2] = 0
        square.center = center

        assert np.isclose(
            square.polar_moment_inertia, 1 / 6 + square.area * np.dot(center, center)
        )

    testfun()


def test_inertia_tensor(square):
    """Test the inertia tensor calculation."""
    square.center = (0, 0, 0)
    assert np.sum(square.inertia_tensor > 1e-6) == 1
    assert square.inertia_tensor[2, 2] == 1 / 6

    # Validate yz plane.
    rotation = rowan.from_axis_angle([0, 1, 0], np.pi / 2)
    rotated_verts = rowan.rotate(rotation, square.vertices)
    rotated_square = ConvexPolygon(rotated_verts)
    assert np.sum(rotated_square.inertia_tensor > 1e-6) == 1
    assert rotated_square.inertia_tensor[0, 0] == 1 / 6

    # Validate xz plane.
    rotation = rowan.from_axis_angle([1, 0, 0], np.pi / 2)
    rotated_verts = rowan.rotate(rotation, square.vertices)
    rotated_square = ConvexPolygon(rotated_verts)
    assert np.sum(rotated_square.inertia_tensor > 1e-6) == 1
    assert rotated_square.inertia_tensor[1, 1] == 1 / 6

    # Validate translation along each axis.
    delta = 2
    area = square.area
    for i in range(3):
        translation = [0] * 3
        translation[i] = delta
        translated_verts = square.vertices + translation
        translated_square = ConvexPolygon(translated_verts)
        offdiagonal_tensor = translated_square.inertia_tensor.copy()
        diag_indices = np.diag_indices(3)
        offdiagonal_tensor[diag_indices] = 0
        assert np.sum(offdiagonal_tensor > 1e-6) == 0
        expected_diagonals = [0, 0, 1 / 6]
        for j in range(3):
            if i != j:
                expected_diagonals[j] += area * delta * delta
        assert np.allclose(
            np.diag(translated_square.inertia_tensor), expected_diagonals
        )


def test_nonplanar(square_points):
    """Ensure that nonplanar vertices raise an error."""
    with pytest.raises(ValueError):
        square_points[0, 2] += 1
        Polygon(square_points)


@settings(deadline=500)
@given(EllipseSurfaceStrategy)
@example(np.array([[1, 1], [1, 1.00041707], [2.78722762, 1], [2.72755193, 1.32128906]]))
def test_reordering_convex(points):
    """Test that vertices can be reordered appropriately."""
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    poly = polygon_from_hull(points[hull.vertices])
    assert np.all(poly.vertices[:, :2] == verts)


@settings(deadline=500)
@given(EllipseSurfaceStrategy)
@example(np.array([[1, 1], [1, 1.00041707], [2.78722762, 1], [2.72755193, 1.32128906]]))
def test_convex_area(points):
    """Check the areas of various convex sets."""
    hull = ConvexHull(points)
    poly = polygon_from_hull(points[hull.vertices])
    assert np.isclose(hull.volume, poly.area)


@given(random_quat=arrays(np.float64, (4,), elements=floats(-1, 1, width=64)))
def test_rotation_signed_area(random_quat):
    """Ensure that rotating does not change the signed area."""
    assume(not np.all(random_quat == 0))
    random_quat = rowan.normalize(random_quat)
    rotated_points = rowan.rotate(random_quat, get_square_points())
    poly = Polygon(rotated_points)
    assert np.isclose(poly.signed_area, 1)

    poly.reorder_verts(clockwise=True)
    assert np.isclose(poly.signed_area, -1)


@settings(deadline=500)
@given(EllipseSurfaceStrategy)
def test_set_convex_area(points):
    """Test setting area of arbitrary convex sets."""
    hull = ConvexHull(points)
    poly = polygon_from_hull(points[hull.vertices])
    original_area = poly.area
    poly.area *= 2
    assert np.isclose(poly.area, 2 * original_area)


def test_triangulate(square):
    triangles = [tri for tri in square._triangulation()]
    assert len(triangles) == 2

    all_vertices = [tuple(vertex) for triangle in triangles for vertex in triangle]
    assert len(set(all_vertices)) == 4

    assert not np.all(np.asarray(triangles[0]) == np.asarray(triangles[1]))


def test_bounding_circle_radius_regular_polygon():
    family = RegularNGonFamily()
    for i in range(3, 10):
        vertices = family.make_vertices(i)
        rmax = np.max(np.linalg.norm(vertices, axis=-1))

        poly = Polygon(vertices)
        circle = poly.bounding_circle

        assert np.isclose(rmax, circle.radius)
        assert np.allclose(circle.center, 0)


@given(EllipseSurfaceStrategy)
def test_bounding_circle_radius_random_hull(points):
    hull = ConvexHull(points)
    poly = Polygon(points[hull.vertices])

    # For an arbitrary convex polygon, the furthest vertex from the origin is
    # an upper bound on the bounding sphere radius, but need not be the radius
    # because the ball need not be centered at the centroid.
    rmax = np.max(np.linalg.norm(poly.vertices, axis=-1))
    circle = poly.bounding_circle
    assert circle.radius <= rmax + 1e-6

    poly.center = [0, 0, 0]
    circle = poly.bounding_circle
    assert circle.radius <= rmax + 1e-6


@settings(deadline=500)
@given(
    points=EllipseSurfaceStrategy,
    rotation=arrays(np.float64, (4,), elements=floats(-1, 1, width=64)),
)
def test_bounding_circle_radius_random_hull_rotation(points, rotation):
    """Test that rotating vertices does not change the bounding radius."""
    assume(not np.all(rotation == 0))

    hull = ConvexHull(points)
    poly = Polygon(points[hull.vertices])

    rotation = rowan.normalize(rotation)
    rotated_vertices = rowan.rotate(rotation, poly.vertices)
    poly_rotated = Polygon(rotated_vertices)

    circle = poly.bounding_circle
    rotated_circle = poly_rotated.bounding_circle
    assert np.isclose(circle.radius, rotated_circle.radius)


def test_circumcircle():
    family = RegularNGonFamily()
    for i in range(3, 10):
        vertices = family.make_vertices(i)
        rmax = np.max(np.linalg.norm(vertices, axis=-1))

        poly = Polygon(vertices)
        circle = poly.circumcircle

        assert np.isclose(rmax, circle.radius)
        assert np.allclose(circle.center, 0)


def test_incircle_from_center(convex_square):
    circle = convex_square.incircle_from_center
    assert np.all(circle.center == convex_square.center)
    assert circle.radius == 0.5


def test_form_factor(square):
    """Validate the form factor of a polygon.

    At the moment this is primarily a regression test, and should be expanded for more
    rigorous validation.
    """
    square.center = (0, 0, 0)

    ks = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 2, 3],
            [-2, 4, -5.2],
        ],
        dtype=np.float,
    )

    ampl = [
        1.0,
        0.95885108,
        0.95885108,
        0.84147098,
        0.95885108,
        1.0,
        0.80684536,
        0.3825737,
    ]
    np.testing.assert_allclose(square.compute_form_factor_amplitude(ks), ampl)

    # Form factor should be invariant to shifts along the normal.
    square.center = [0, 0, -1]
    np.testing.assert_allclose(square.compute_form_factor_amplitude(ks), ampl)

    # Form factor should be invariant to polygon "direction" (the line integral
    # direction change should cause a sign flip that cancels the normal flip).
    new_square = Polygon(square.vertices[::-1, :])
    np.testing.assert_allclose(new_square.compute_form_factor_amplitude(ks), ampl)


@pytest.mark.parametrize("num_sides", range(3, 6))
def test_perimeter(num_sides):
    """Test the polygon perimeter calculation."""

    def unit_area_regular_n_gon_side_length(n):
        r"""Compute the side length of a unit-area regular polygon analytically.

        The area of regular n-gon is given by
        :math:`\frac{s^2 n}{4 \tan\left(\frac{180}{n}\right)}`. This function sets
        the area to 1 and inverts that formula.
        """
        return np.sqrt((4 * np.tan(np.pi / n)) / n)

    poly = RegularNGonFamily.get_shape(num_sides)
    assert np.isclose(
        num_sides * unit_area_regular_n_gon_side_length(num_sides), poly.perimeter
    )


@given(floats(0.1, 1000))
def test_set_perimeter(value):
    """Test the perimeter and circumference setter."""
    square = RegularNGonFamily.get_shape(4)
    square.perimeter = value
    assert square.perimeter == approx(value)
    assert square.vertices == approx(
        RegularNGonFamily.get_shape(4).vertices
        * (value / RegularNGonFamily.get_shape(4).perimeter)
    )


@pytest.mark.parametrize("num_sides", range(3, 10))
def test_convex_polygon_shape_kernel_unit_area_ngon(num_sides):
    """Check shape kernel consistency with perimeter and area."""
    theta = np.linspace(0, 2 * np.pi, 1000000)
    shape = RegularNGonFamily.get_shape(num_sides)
    kernel = shape.shape_kernel(theta)
    assert_shape_kernel_2d(shape, theta, kernel)


@pytest.mark.parametrize("num_sides", range(3, 10))
def test_nonregular_convex_polygon_shape_kernel_unit_area_ngon(num_sides):
    """Check shape kernel consistency with perimeter and area."""
    theta = np.linspace(0, 2 * np.pi, 1000000)
    shape = RegularNGonFamily.get_shape(num_sides)
    verts = shape.vertices[:, :2]
    # shift making shape a nonregular polygon
    verts[0, 1] = verts[0, 1] + 0.2
    shape = ConvexPolygon(verts)
    kernel = shape.shape_kernel(theta)
    assert_shape_kernel_2d(shape, theta, kernel)


def test_kernel_values_for_square():
    """Check shape kernel of a square with infinite slopes."""
    verts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    theta = np.linspace(0, 2 * np.pi, 1000000)
    shape = ConvexPolygon(verts)
    kernel = shape.shape_kernel(theta)
    assert_shape_kernel_2d(shape, theta, kernel)


def test_kernel_values_for_square_triangle():
    """Check shape kernel of a triangle with infinite slopes."""
    verts = np.array([[1, 1], [-1, 0], [1, -1]])
    theta = np.linspace(0, 2 * np.pi, 1000000)
    shape = ConvexPolygon(verts)
    kernel = shape.shape_kernel(theta)
    assert_shape_kernel_2d(shape, theta, kernel)
