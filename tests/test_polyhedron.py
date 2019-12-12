import pytest
import numpy as np
from euclid.shape_classes.convex_polyhedron import ConvexPolyhedron
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError
from hypothesis import given, assume
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from euclid.damasceno import SHAPES
import os
from conftest import get_oriented_cube_facets, get_oriented_cube_normals


def test_normal_detection(convex_cube):
    detected_normals = set([tuple(n) for n in convex_cube.normals])
    expected_normals = set([tuple(n) for n in get_oriented_cube_normals()])
    assert detected_normals == expected_normals


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
def test_surface_area(cube):
    """Test surface area calculation."""
    assert cube.surface_area == 6


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
def test_volume(cube):
    assert cube.volume == 1


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
def test_set_volume(cube):
    """Test setting volume."""
    cube.volume = 2
    assert np.isclose(cube.volume, 2)


def test_merge_facets(convex_cube):
    """Test that coplanar facets can be correctly merged."""
    assert len(convex_cube.facets) == 6


@given(arrays(np.float64, (5, 3), floats(-10, 10, width=64), unique=True))
def test_convex_volume(points):
    """Check the volumes of various convex sets."""
    try:
        hull = ConvexHull(points)
    except QhullError:
        assume(False)
    else:
        # Avoid cases where numerical imprecision make tests fail.
        assume(hull.volume > 1e-6)
    verts = points[hull.vertices]
    poly = ConvexPolyhedron(verts)

    assert np.isclose(hull.volume, poly.volume)


@given(arrays(np.float64, (5, 3), floats(-10, 10, width=64), unique=True))
def test_convex_surface_area(points):
    """Check the surface areas of various convex sets."""
    try:
        hull = ConvexHull(points)
    except QhullError:
        assume(False)
    else:
        # Avoid cases where numerical imprecision make tests fail.
        assume(hull.area > 1e-6)
    verts = points[hull.vertices]
    poly = ConvexPolyhedron(verts)
    assert np.isclose(hull.area, poly.surface_area)


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
@given(new_center=arrays(np.float64, (3, ), floats(-10, 10, width=64)))
def test_volume_center_shift(cube, new_center):
    """Make sure that moving the center doesn't affect the volume."""
    cube.center = new_center
    assert np.isclose(cube.volume, 1)


def test_facet_alignment(convex_cube):
    """Make sure that facets are constructed correctly given vertices."""
    def facet_to_string(facet):
        # Convenience function to create a string of vertex ids, which is the
        # easiest way to test for sequences that are cyclically equal.
        return ''.join([str(c) for c in facet])

    reference_facets = []
    for facet in get_oriented_cube_facets():
        reference_facets.append(facet_to_string(facet)*2)

    assert len(convex_cube.facets) == len(reference_facets)

    for facet in convex_cube.facets:
        str_facet = facet_to_string(facet)
        assert any([str_facet in ref for ref in reference_facets])


def compute_inertia_mc(vertices, num_samples=1e6):
    """Use Monte Carlo integration to compute the moment of inertia."""
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)

    points = np.random.rand(int(num_samples), 3)*(maxs-mins)+mins

    hull = Delaunay(vertices)
    inside = hull.find_simplex(points) >= 0

    Ixx = np.mean(points[inside][:, 1]**2 + points[inside][:, 2]**2)
    Iyy = np.mean(points[inside][:, 0]**2 + points[inside][:, 2]**2)
    Izz = np.mean(points[inside][:, 0]**2 + points[inside][:, 1]**2)
    Ixy = np.mean(-points[inside][:, 0] * points[inside][:, 1])
    Ixz = np.mean(-points[inside][:, 0] * points[inside][:, 2])
    Iyz = np.mean(-points[inside][:, 1] * points[inside][:, 2])

    poly = ConvexPolyhedron(vertices)

    inertia_tensor = np.array([[Ixx, Ixy, Ixz],
                               [Ixy,   Iyy, Iyz],
                               [Ixz,   Iyz,   Izz]]) * poly.volume

    return inertia_tensor


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
def test_moment_inertia(cube):
    assert np.allclose(cube.inertia_tensor, np.diag([1/6]*3))


def test_volume_damasceno_shapes():
    for shape in SHAPES:
        if shape.Name in ('RESERVED', 'Sphere'):
            continue
        poly = ConvexPolyhedron(shape.vertices)
        hull = ConvexHull(shape.vertices)
        assert np.isclose(poly.volume, hull.volume)


@pytest.mark.skip("Need test data")
def test_circumsphere_radius():
    pass


# This test is a bit slow (a couple of minutes), so skip running it locally.
@pytest.mark.skipif(os.getenv('CI', 'false') != 'true' and
                    os.getenv('CIRCLECI', 'false') != 'true',
                    reason="Test is too slow to run during rapid development")
def test_moment_inertia_damasceno_shapes():
    # These shapes pass the test for a sufficiently high number of samples, but
    # the number is too high to be worth running them regularly.
    bad_shapes = [
        'Augmented Truncated Dodecahedron',
        'Deltoidal Hexecontahedron',
        'Disdyakis Triacontahedron',
        'Truncated Dodecahedron',
        'Truncated Icosidodecahedron',
        'Metabiaugmented Truncated Dodecahedron',
        'Pentagonal Hexecontahedron',
        'Paragyrate Diminished Rhombicosidodecahedron',
        'Square Cupola',
        'Triaugmented Truncated Dodecahedron',
        'Parabiaugmented Truncated Dodecahedron',
    ]
    np.random.seed(0)
    for shape in SHAPES:
        if shape.Name in ['RESERVED', 'Sphere'] + bad_shapes:
            continue

        poly = ConvexPolyhedron(shape.vertices)
        num_samples = 1000
        accept = False
        # Loop over different sampling rates to minimize the test runtime.
        while num_samples < 1e8:
            try:
                euclid_result = poly.inertia_tensor
                mc_result = compute_inertia_mc(shape.vertices, num_samples)
                assert np.allclose(euclid_result, mc_result, atol=1e-1)
                accept = True
                break
            except AssertionError:
                num_samples *= 10
                continue
        if not accept:
            raise AssertionError("The test failed for shape {}.\nMC Result: "
                                 "\n{}\neuclid result: \n{}".format(
                                     shape.Name, mc_result, euclid_result
                                 ))


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
def test_iq(cube):
    assert cube.iq == 36*np.pi*cube.volume**2/cube.surface_area**3


def test_dihedrals():
    known_shapes = {
        'Tetrahedron': np.arccos(1/3),
        'Cube': np.pi/2,
        'Octahedron': np.pi - np.arccos(1/3),
        'Dodecahedron':  np.pi - np.arctan(2),
        'Icosahedron': np.pi - np.arccos(np.sqrt(5)/3),
    }
    for shape in SHAPES:
        if shape.Name in known_shapes:
            poly = ConvexPolyhedron(shape.vertices)
            # The dodecahedron in SHAPES needs a slightly more expansive merge
            # to get all the facets joined.
            poly.merge_facets(rtol=1e-4)
            for i in range(poly.num_facets):
                for j in poly.neighbors[i]:
                    assert np.isclose(poly.get_dihedral(i, j),
                                      known_shapes[shape.Name])


def test_curvature():
    """Regression test against values computed with older method."""
    known_shapes = {
        'Cube': 0.75,
        'Dodecahedron': 1.3215637405498626,
        'Icosahedron': 0.8710482367460449,
        'Octahedron': 0.5877400099213849,
        'Tetrahedron': 0.4561299069097583,
    }

    for shape in SHAPES:
        if shape.Name in known_shapes:
            assert np.isclose(
                ConvexPolyhedron(shape.vertices).mean_curvature,
                known_shapes[shape.Name])


@pytest.mark.skip("Need test data")
def test_curvature_exact():
    """Curvature test based on explicit calculation."""
    pass


@pytest.mark.skip("Need test data")
def test_nonconvex_polyhedron():
    pass


@pytest.mark.skip("Need test data")
def test_nonconvex_polyhedron_with_nonconvex_polygon_face():
    pass


@pytest.mark.skip("Need test data")
def test_tau():
    pass


@pytest.mark.skip("Need test data")
def test_asphericity():
    pass
