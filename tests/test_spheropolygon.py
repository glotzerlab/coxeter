# Copyright (c) 2015-2025 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest
import rowan
from hypothesis import assume, example, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, tuples
from pytest import approx
from scipy.spatial import ConvexHull

from conftest import (
    EllipseSurfaceStrategy,
    assert_distance_to_surface_2d,
    regular_polygons,
)
from coxeter.shapes import ConvexSpheropolygon


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


def test_radius_getter_setter(square_points):
    """Test getting and setting the radius."""

    @given(r=floats(0.1, 1000))
    def testfun(r):
        square_points_2d = square_points[:, :2]
        convexspheropolygon = ConvexSpheropolygon(square_points_2d, r)
        assert convexspheropolygon.radius == approx(r)
        convexspheropolygon.radius = r + 1
        assert convexspheropolygon.radius == approx(r + 1)

    testfun()


def test_invalid_radius_constructor(square_points):
    """Test invalid radius values in constructor."""

    @given(r=floats(-1000, -1))
    def testfun(r):
        square_points_2d = square_points[:, :2]
        with pytest.raises(ValueError):
            ConvexSpheropolygon(square_points_2d, r)

    testfun()


def test_invalid_radius_setter(square_points):
    """Test setting invalid radius values."""

    @given(r=floats(-1000, -1))
    def testfun(r):
        square_points_2d = square_points[:, :2]
        spheropolygon = ConvexSpheropolygon(square_points_2d, 1)
        with pytest.raises(ValueError):
            spheropolygon.radius = r

    testfun()


def test_duplicate_points(square_points):
    """Ensure that running with any duplicate points produces a warning."""
    square_points = np.vstack((square_points, square_points[[0]]))
    with pytest.raises(ValueError):
        ConvexSpheropolygon(square_points, 1)


def test_identical_points(ones):
    """Ensure that running with identical points produces an error."""
    with pytest.raises(ValueError):
        ConvexSpheropolygon(ones, 1)


def test_area(unit_rounded_square):
    """Test area calculation."""
    shape = unit_rounded_square
    area = 1 + 4 + np.pi
    assert shape.signed_area == approx(area)
    assert shape.area == approx(area)


def test_area_getter_setter(unit_rounded_square):
    """Test setting the area."""

    @given(area=floats(0.1, 1000))
    def testfun(area):
        unit_rounded_square.area = area
        assert unit_rounded_square.signed_area == approx(area)
        assert unit_rounded_square.area == approx(area)

        # Reset to original area
        original_area = 1 + 4 + np.pi
        unit_rounded_square.area = original_area
        assert unit_rounded_square.signed_area == approx(original_area)
        assert unit_rounded_square.area == approx(original_area)

    testfun()


def test_nonplanar(square_points):
    """Ensure that nonplanar vertices raise an error."""
    with pytest.raises(ValueError):
        square_points[0, 2] += 1
        ConvexSpheropolygon(square_points, 1)


@settings(deadline=500)
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
        assume(not np.allclose(random_quat, 0))
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

    testfun()


def test_sphero_square_perimeter(unit_rounded_square):
    """Test calculating the perimeter of a spheropolygon."""
    assert unit_rounded_square.perimeter == approx(4 + 2 * np.pi)


def test_perimeter_setter(unit_rounded_square):
    """Test setting the perimeter."""

    @given(perimeter=floats(0.1, 1000))
    def testfun(perimeter):
        unit_rounded_square.perimeter = perimeter
        assert unit_rounded_square.perimeter == approx(perimeter)

        # Reset to original perimeter
        original_perimeter = 4 + 2 * np.pi
        unit_rounded_square.perimeter = original_perimeter
        assert unit_rounded_square.perimeter == approx(original_perimeter)
        assert unit_rounded_square.radius == approx(1.0)

    testfun()


@pytest.mark.parametrize("shape", regular_polygons())
def test_distance_to_surface_regular_ngons(shape):
    """Make sure shape distance works for regular ngons."""
    verts = shape.vertices[:, :2]

    @given(floats(0.1, 10), tuples(floats(-1.0, 1.0), floats(-1.0, 1.0)))
    def testfun(rounding_radius, vertex_shift):
        theta = np.linspace(0, 2 * np.pi, 10000)
        shape = ConvexSpheropolygon(verts + np.asarray(vertex_shift), rounding_radius)
        distance = shape.distance_to_surface(theta)
        assert_distance_to_surface_2d(shape, theta, distance)

    testfun()


def test_inertia(unit_rounded_square):
    """None of the inertia calculations are implemented for this class."""
    with pytest.raises(NotImplementedError):
        unit_rounded_square.planar_moments_inertia
    with pytest.raises(NotImplementedError):
        unit_rounded_square.polar_moment_inertia
    with pytest.raises(NotImplementedError):
        unit_rounded_square.inertia_tensor


def test_repr(unit_rounded_square):
    assert str(unit_rounded_square), str(eval(repr(unit_rounded_square)))


def test_to_hoomd(unit_rounded_square):
    """Test hoomd JSON calculation."""
    shape = unit_rounded_square
    dict_keys = ["vertices", "centroid", "sweep_radius", "area"]
    dict_vals = [
        shape.vertices,
        [0, 0, 0],
        1,
        shape.area,
    ]
    hoomd_dict = shape.to_hoomd()
    for key, val in zip(dict_keys, dict_vals):
        assert np.allclose(hoomd_dict[key], val), f"{key}"





def test_shortest_distance_convex():
    tri_verts = np.array([[0, 0.5], [-0.25*np.sqrt(3), -0.25], [0.25*np.sqrt(3), -0.25]])
    triangle = ConvexSpheropolygon(vertices=tri_verts, radius = 0.25)

    test_points = np.array([[3.5,3.25,0], [3,3.75,0], [3,3.25,0], [3,3,1], [3.25,3.5, -1], [3.5,3.75,1], [3-0.25*np.sqrt(3), 2.65,0], [3,4,-1]])

    distances = triangle.shortest_distance_to_surface(test_points, translation_vector=np.array([3,3,0]))
    displacements = triangle.shortest_displacement_to_surface(test_points, translation_vector=np.array([3,3,0]))

    true_distances = np.array([0.0580127018, 0, 0, 1, 1, 1.0463612304, 0, 1.0307764064])
    true_displacements = np.array([[-0.0502404735, -0.0290063509, 0], [0,0,0], [0,0,0], [0,0,-1], [0,0,1], [-0.2667468244, -0.1540063509, -1], [0,0,0], [0,-0.25,1]])

    np.testing.assert_allclose(distances, true_distances)
    np.testing.assert_allclose(displacements, true_displacements)



def test_shortest_distance_general():
    #Creating a random convex spheropolygon
    # np.random.seed(3)
    random_angles = np.random.rand(10)*2*np.pi #angles
    sorted_angles = np.sort(random_angles)
    random_dist = np.random.rand(1)*10 #from origin
    radius = np.random.rand(1)*5 

    vertices = np.zeros((10,2))
    vertices[:,0] = random_dist * np.cos(sorted_angles) #x
    vertices[:,1] = random_dist * np.sin(sorted_angles) #y

    poly = ConvexSpheropolygon(vertices=vertices, radius=radius, normal=[0,0,1])

    points2d = np.random.rand(100,2)*20-10
    points3d = np.random.rand(150, 3)*20 -10

    distances2d = poly.shortest_distance_to_surface(points2d)
    distances3d = poly.shortest_distance_to_surface(points3d)
    displacements2d = poly.shortest_displacement_to_surface(points2d)
    displacements3d = poly.shortest_displacement_to_surface(points3d)

    np.testing.assert_allclose(distances2d, np.linalg.norm(displacements2d, axis=1))
    np.testing.assert_allclose(distances3d, np.linalg.norm(displacements3d, axis=1))
    
    edges_90 = np.cross(poly._polygon.edge_vectors, poly.normal) #point outwards (10, 3)
    upper_bounds = np.sum(edges_90*poly.vertices, axis=1) #(10,)

    def scipy_closest_point(point, edges_90, upper_bounds):
        from scipy.optimize import LinearConstraint, minimize

        tri_min_point = minimize(
            fun=lambda pt: np.linalg.norm(pt - point),  # Function to optimize
            x0=np.zeros(3),  # Initial guess
            constraints=[LinearConstraint(np.append(edges_90, [[0,0,1], [0,0,-1]], axis=0), -np.inf, np.append(upper_bounds, [0,0]))],
            tol=1e-10
            )

        distance = np.linalg.norm(tri_min_point.x - point)
        displacement = tri_min_point.x - point

        return distance, displacement
    
    #--- 2D ---
    scipy_distances2d = []
    scipy_displacements2d = []
    for point in points2d:
        point = np.append(point, [0])
        
        scipy_dist2d, scipy_displace2d = scipy_closest_point(point, edges_90, upper_bounds)
        scipy_distances2d.append(scipy_dist2d)
        scipy_displacements2d.append(scipy_displace2d)
    
    #--- 3D ---
    scipy_distances3d = []
    scipy_displacements3d = []
    for point in points3d:

        scipy_dist3d, scipy_displace3d = scipy_closest_point(point, edges_90, upper_bounds)

        inplane_disp = scipy_displace3d - (scipy_displace3d @ poly.normal)*poly.normal
        inplane_dist = np.linalg.norm(inplane_disp)
        if inplane_dist < radius:
            subtract_vector = inplane_disp
        else:
            subtract_vector = (radius * inplane_disp/ np.linalg.norm(inplane_disp))

        scipy_displace3d = scipy_displace3d - subtract_vector
        scipy_displacements3d.append(scipy_displace3d)
        scipy_distances3d.append(np.linalg.norm(scipy_displace3d))

    scipy_distances2d = np.asarray(scipy_distances2d) - radius
    is_zero2d = scipy_distances2d < 0
    scipy_distances2d[is_zero2d] = 0

    scipy_displacements2d = np.asarray(scipy_displacements2d) 
    scipy_displacements2d = scipy_displacements2d - (radius * scipy_displacements2d / np.expand_dims(np.linalg.norm(scipy_displacements2d, axis=1),axis=1))
    scipy_displacements2d[is_zero2d] = np.array([0,0,0])

    np.testing.assert_allclose(distances2d, scipy_distances2d, atol=2e-8)
    np.testing.assert_allclose(displacements2d, scipy_displacements2d, atol=2e-5)
    np.testing.assert_allclose(distances3d, scipy_distances3d, atol=2e-8)
    np.testing.assert_allclose(displacements3d, scipy_displacements3d, atol=2e-5)
