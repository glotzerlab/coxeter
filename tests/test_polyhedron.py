import pytest
import numpy as np
from coxeter.shape_classes.convex_polyhedron import ConvexPolyhedron
from coxeter.shape_classes.sphere import Sphere
from coxeter.shape_families import PlatonicFamily, family_from_doi
from scipy.spatial import ConvexHull
from hypothesis import given, assume
from hypothesis.strategies import floats, integers
from hypothesis.extra.numpy import arrays
import os
from conftest import get_oriented_cube_faces, get_oriented_cube_normals
from utils import compute_inertia_mc
import rowan
from coxeter.shape_classes.utils import (translate_inertia_tensor,
                                         rotate_order2_tensor)
from conftest import get_valid_hull


def polyhedron_from_hull(verts):
    """Try to generate a polyhedron from a hull, and fail gracefully (in the
    context of Hypothesis) if the hull is nearly degenerate."""
    try:
        poly = ConvexPolyhedron(verts)
    except ValueError as e:
        # Don't worry about failures caused by bad hulls.
        allowed_errors = [
            'The provided vertices do not form a convex polygon.',
            'Not all vertices are coplanar.'
        ]
        if any([ae in str(e) for ae in allowed_errors]):
            return False
        else:
            raise e
    return poly


def damasceno_shapes():
    """For efficiency, we don't construct all the shape classes, but rather
    just yield the raw shape dicts."""
    family = family_from_doi('10.1126/science.1220869')[0]
    for shape_data in family.data.values():
        yield shape_data


def platonic_solids():
    family = PlatonicFamily()
    for shape_name in family.data:
        yield family(shape_name)


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


def test_merge_faces(convex_cube):
    """Test that coplanar faces can be correctly merged."""
    assert len(convex_cube.faces) == 6


@given(arrays(np.float64, (5, 3), elements=floats(-10, 10, width=64),
              unique=True))
def test_convex_volume(points):
    """Check the volumes of various convex sets."""
    hull = get_valid_hull(points)
    assume(hull)

    poly = polyhedron_from_hull(points[hull.vertices])
    assume(poly)
    assert np.isclose(hull.volume, poly.volume)


@given(arrays(np.float64, (5, 3), elements=floats(-10, 10, width=64),
              unique=True))
def test_convex_surface_area(points):
    """Check the surface areas of various convex sets."""
    hull = get_valid_hull(points)
    assume(hull)

    poly = polyhedron_from_hull(points[hull.vertices])
    assume(poly)
    assert np.isclose(hull.area, poly.surface_area)


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
def test_volume_center_shift(cube):
    """Make sure that moving the center doesn't affect the volume."""
    # Use a nested function to avoid warnings from hypothesis. In this case, it
    # is safe to reuse the cube fixture.
    # See https://github.com/HypothesisWorks/hypothesis/issues/377
    @given(new_center=arrays(np.float64, (3, ),
                             elements=floats(-10, 10, width=64)))
    def testfun(new_center):
        cube.center = new_center
        assert np.isclose(cube.volume, 1)
    testfun()


def test_face_alignment(convex_cube):
    """Make sure that faces are constructed correctly given vertices."""
    def face_to_string(face):
        # Convenience function to create a string of vertex ids, which is the
        # easiest way to test for sequences that are cyclically equal.
        return ''.join([str(c) for c in face])

    reference_faces = []
    for face in get_oriented_cube_faces():
        reference_faces.append(face_to_string(face)*2)

    assert len(convex_cube.faces) == len(reference_faces)

    for face in convex_cube.faces:
        str_face = face_to_string(face)
        assert any([str_face in ref for ref in reference_faces])


@pytest.mark.parametrize('cube',
                         ['convex_cube', 'oriented_cube', 'unoriented_cube'],
                         indirect=True)
def test_moment_inertia(cube):
    cube.center = (0, 0, 0)
    assert np.allclose(cube.inertia_tensor, np.diag([1/6]*3))


@pytest.mark.parametrize('shape', damasceno_shapes())
def test_volume_damasceno_shapes(shape):
    if shape['name'] in ('RESERVED', 'Sphere'):
        return
    vertices = shape['vertices']
    poly = ConvexPolyhedron(vertices)
    hull = ConvexHull(vertices)
    assert np.isclose(poly.volume, hull.volume)


# This test is a bit slow (a couple of minutes), so skip running it locally.
@pytest.mark.skipif(os.getenv('CI', 'false') != 'true' and
                    os.getenv('CIRCLECI', 'false') != 'true',
                    reason="Test is too slow to run during rapid development")
@pytest.mark.parametrize('shape', damasceno_shapes())
def test_moment_inertia_damasceno_shapes(shape):
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
    if shape['name'] in ['RESERVED', 'Sphere'] + bad_shapes:
        return

    np.random.seed(0)
    poly = ConvexPolyhedron(shape['vertices'])
    num_samples = 1000
    accept = False
    # Loop over different sampling rates to minimize the test runtime.
    while num_samples < 1e8:
        try:
            coxeter_result = poly.inertia_tensor
            mc_result = compute_inertia_mc(shape['vertices'], num_samples)
            assert np.allclose(coxeter_result, mc_result, atol=1e-1)
            accept = True
            break
        except AssertionError:
            num_samples *= 10
            continue
    if not accept:
        raise AssertionError("The test failed for shape {}.\nMC Result: "
                             "\n{}\ncoxeter result: \n{}".format(
                                 shape['name'], mc_result, coxeter_result
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
    family = PlatonicFamily()
    for name, dihedral in known_shapes.items():
        poly = family(name)
        # The dodecahedron needs a more expansive merge to get all the
        # faces joined.
        if name == 'Dodecahedron':
            poly.merge_faces(rtol=1)
        for i in range(poly.num_faces):
            for j in poly.neighbors[i]:
                assert np.isclose(poly.get_dihedral(i, j), dihedral)


def test_curvature():
    """Regression test against values computed with older method."""
    known_shapes = {
        'Cube': 0.75,
        'Dodecahedron': 0.6703242780091758,
        'Icosahedron': 0.6715997848012972,
        'Octahedron': 0.75518565565632,
        'Tetrahedron': 0.9303430847680867,
    }

    family = PlatonicFamily()
    for name, curvature in known_shapes.items():
        poly = family(name)
        if name == 'Dodecahedron':
            poly.merge_faces(rtol=1)
        assert np.isclose(
            poly.mean_curvature, curvature)


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


@pytest.mark.parametrize('poly', platonic_solids())
def test_circumsphere_platonic(poly):
    center, radius = poly.circumsphere

    # Ensure polyhedron is centered, then compute distances.
    poly.center = [0, 0, 0]
    r2 = np.sum(poly.vertices**2, axis=1)

    assert np.allclose(r2, radius*radius)


def test_circumsphere_from_center():
    """Check that all points outside this circumsphere are also outside the
    polyhedron. Note that this is a necessary but not sufficient condition for
    correctness."""
    # Building convex polyhedra is the slowest part of this test, so rather
    # than testing all shapes every time we test a random subset each time the
    # test runs. To further speed the tests, we build all convex polyhedra
    # ahead of time. Each set of random points is tested against a different
    # random polyhedron.
    #
    # Use a nested function to avoid warnings from hypothesis. While the shape
    # does get modified inside the testfun, it's simply being recentered each
    # time, which is not destructive since it can be overwritten in subsequent
    # calls.
    # See https://github.com/HypothesisWorks/hypothesis/issues/377
    import random
    family = family_from_doi('10.1126/science.1220869')[0]
    shapes = [ConvexPolyhedron(s['vertices']) for s in
              random.sample([s for s in family.data.values() if
                             len(s['vertices'])], len(family.data)//5)]

    @given(center=arrays(np.float64, (3, ), elements=floats(-10, 10, width=64),
                         unique=True),
           points=arrays(np.float64, (50, 3), elements=floats(-1, 1, width=64),
                         unique=True),
           shape_index=integers(0, len(shapes)-1))
    def testfun(center, points, shape_index):
        poly = shapes[shape_index]
        poly.center = center

        centroid, radius = poly.circumsphere_from_center
        sphere = Sphere(radius)

        scaled_points = points*radius
        points_outside = np.logical_not(sphere.is_inside(scaled_points))

        # Verify that all points outside the circumsphere are also outside the
        # polyhedron.
        shifted_points = scaled_points + centroid
        assert not np.any(np.logical_and(
            points_outside,
            poly.is_inside(shifted_points)))

    testfun()


@pytest.mark.parametrize('poly', platonic_solids())
def test_bounding_sphere_platonic(poly):
    center, radius = poly.bounding_sphere

    # Ensure polyhedron is centered, then compute distances.
    poly.center = [0, 0, 0]
    r2 = np.sum(poly.vertices**2, axis=1)

    assert np.allclose(r2, radius*radius, rtol=1e-4)


def test_inside_boundaries(convex_cube):
    assert np.all(convex_cube.is_inside(convex_cube.vertices))
    convex_cube.center = [0, 0, 0]
    assert np.all(convex_cube.is_inside(convex_cube.vertices * 0.99))
    assert not np.any(convex_cube.is_inside(convex_cube.vertices * 1.01))


def test_inside(convex_cube):
    # Use a nested function to avoid warnings from hypothesis. In this case, it
    # is safe to reuse the convex cube.
    # See https://github.com/HypothesisWorks/hypothesis/issues/377
    @given(arrays(np.float64, (100, 3), elements=floats(-10, 10, width=64),
                  unique=True))
    def testfun(test_points):
        expected = np.all(np.logical_and(test_points >= 0, test_points <= 1),
                          axis=1)
        actual = convex_cube.is_inside(test_points)
        assert np.all(expected == actual)
    testfun()


@given(arrays(np.float64, (5, 3), elements=floats(-10, 10, width=64),
              unique=True),
       arrays(np.float64, (100, 3), elements=floats(0, 1, width=64),
              unique=True))
def test_insphere_from_center_convex_hulls(points, test_points):
    hull = get_valid_hull(points)
    assume(hull)
    verts = points[hull.vertices]
    poly = ConvexPolyhedron(verts)
    center, radius = poly.insphere_from_center
    assert poly.is_inside(center)
    poly.center = [0, 0, 0]
    insphere = Sphere(radius)
    test_points -= np.mean(test_points, axis=0)
    test_points *= radius * 3
    points_in_sphere = insphere.is_inside(test_points)
    points_in_poly = poly.is_inside(test_points)
    assert np.all(points_in_sphere <= points_in_poly)
    assert insphere.volume < poly.volume


@given(arrays(np.float64, (4, 3), elements=floats(-10, 10, width=64),
              unique=True))
def test_rotate_inertia(points):
    # Use the input as noise rather than the base points to avoid precision and
    # degenerate cases provided by hypothesis.
    tet = PlatonicFamily()('Tetrahedron')
    vertices = tet.vertices + points

    rotation = rowan.random.rand()
    shape = ConvexPolyhedron(vertices)
    rotated_shape = ConvexPolyhedron(rowan.rotate(rotation, vertices))

    mat = rowan.to_matrix(rotation)
    rotated_inertia = rotate_order2_tensor(mat, shape.inertia_tensor)

    assert np.allclose(rotated_inertia, rotated_shape.inertia_tensor)


# Use a small range of translations to ensure that the Delaunay triangulation
# used by the MC calculation will not break.
@given(arrays(np.float64, (3, ), elements=floats(-0.2, 0.2, width=64),
              unique=True))
def test_translate_inertia(translation):
    shape = PlatonicFamily()('Cube')
    # Choose a volume > 1 to test the volume scaling, but don't choose one
    # that's too large because the uncentered polyhedral calculation has
    # massive error without fixing that.
    shape.volume = 2
    shape.center = (0, 0, 0)

    translated_shape = ConvexPolyhedron(shape.vertices + translation)

    translated_inertia = translate_inertia_tensor(
        translation, shape.inertia_tensor, shape.volume)
    mc_tensor = compute_inertia_mc(translated_shape.vertices, 1e4)

    assert np.allclose(translated_inertia,
                       translated_shape._compute_inertia_tensor(False),
                       atol=2e-1,
                       rtol=2e-1)
    assert np.allclose(mc_tensor,
                       translated_shape._compute_inertia_tensor(False),
                       atol=2e-1,
                       rtol=2e-1)

    assert np.allclose(mc_tensor,
                       translated_inertia,
                       atol=1e-2,
                       rtol=1e-2)
    assert np.allclose(mc_tensor,
                       translated_shape.inertia_tensor,
                       atol=1e-2,
                       rtol=1e-2)
