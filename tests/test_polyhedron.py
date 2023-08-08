# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.

import numpy as np
import pytest
import rowan
from hypothesis import assume, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers
from pytest import approx
from scipy.spatial import ConvexHull

from conftest import (
    EllipsoidSurfaceStrategy,
    Random3DRotationStrategy,
    _test_get_set_minimal_bounding_sphere_radius,
    combine_marks,
    get_oriented_cube_faces,
    get_oriented_cube_normals,
    is_not_ci_circleci,
    named_archimedean_mark,
    named_catalan_mark,
    named_damasceno_shapes_mark,
    named_platonic_mark,
    named_solids_mark,
    sphere_isclose,
)
from coxeter.families import DOI_SHAPE_REPOSITORIES, PlatonicFamily
from coxeter.shapes import ConvexPolyhedron, Polyhedron
from coxeter.shapes.utils import rotate_order2_tensor, translate_inertia_tensor
from utils import compute_centroid_mc, compute_inertia_mc


def test_normal_detection(convex_cube):
    detected_normals = [tuple(n) for n in convex_cube.normals]
    expected_normals = [tuple(n) for n in get_oriented_cube_normals()]
    for dn in detected_normals:
        # Each detected normal should be identical to exactly one expected
        # normal. No ordering is guaranteed, so we have to check them all.
        assert sum(np.allclose(dn, en) for en in expected_normals) == 1


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_surface_area(cube):
    """Test surface area calculation."""
    assert cube.surface_area == approx(6)


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_set_surface_area(cube):
    """Test surface area calculation."""
    cube_old = cube
    cube.surface_area = 4
    assert cube.surface_area == approx(4)
    assert cube.vertices == approx(cube_old.vertices * (4 / cube_old.surface_area))


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_volume(cube):
    assert cube.volume == approx(1)


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_set_volume(cube):
    """Test setting volume."""
    cube.volume = 2
    assert np.isclose(cube.volume, 2)


def test_merge_faces(convex_cube):
    """Test that coplanar faces can be correctly merged."""
    assert len(convex_cube.faces) == 6


@settings(deadline=500)
@given(EllipsoidSurfaceStrategy)
def test_convex_volume(points):
    """Check the volumes of various convex sets."""
    hull = ConvexHull(points)
    poly = ConvexPolyhedron(hull.points[hull.vertices])
    assert np.isclose(hull.volume, poly.volume)


@settings(deadline=500)
@given(EllipsoidSurfaceStrategy)
def test_convex_surface_area(points):
    """Check the surface areas of various convex sets."""
    hull = ConvexHull(points)
    poly = ConvexPolyhedron(points[hull.vertices])
    assert np.isclose(hull.area, poly.surface_area)


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_volume_center_shift(cube):
    """Make sure that moving the center doesn't affect the volume."""

    # Use a nested function to avoid warnings from hypothesis. In this case, it
    # is safe to reuse the cube fixture.
    # See https://github.com/HypothesisWorks/hypothesis/issues/377
    @given(new_center=arrays(np.float64, (3,), elements=floats(-10, 10, width=64)))
    def testfun(new_center):
        cube.center = new_center
        assert np.isclose(cube.volume, 1)

    testfun()


def test_face_alignment(convex_cube):
    """Make sure that faces are constructed correctly given vertices."""

    def face_to_string(face):
        # Convenience function to create a string of vertex ids, which is the
        # easiest way to test for sequences that are cyclically equal.
        return "".join([str(c) for c in face])

    reference_faces = []
    for face in get_oriented_cube_faces():
        reference_faces.append(face_to_string(face) * 2)

    assert len(convex_cube.faces) == len(reference_faces)

    for face in convex_cube.faces:
        str_face = face_to_string(face)
        assert any([str_face in ref for ref in reference_faces])


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_moment_inertia(cube):
    cube.center = (0, 0, 0)
    assert np.allclose(cube.inertia_tensor, np.diag([1 / 6] * 3))


@named_damasceno_shapes_mark
def test_volume_damasceno_shapes(shape):
    if shape["name"] in ("RESERVED", "Sphere"):
        return
    vertices = shape["vertices"]
    poly = ConvexPolyhedron(vertices)
    hull = ConvexHull(vertices)
    assert np.isclose(poly.volume, hull.volume)


@pytest.mark.skipif(
    is_not_ci_circleci(), reason="Test is too slow to run during rapid development"
)
@settings(deadline=1000)
@named_damasceno_shapes_mark
@given(v_test=floats(0, 10))
def test_set_volume_damasceno_shapes(shape, v_test):
    if shape["name"] in ("RESERVED", "Sphere"):
        return
    if v_test == 0:
        return
    vertices = shape["vertices"]
    poly = ConvexPolyhedron(vertices)
    poly.volume = v_test
    # Recalculate volume from simplices
    calculated_volume = poly._calculate_signed_volume()
    assert np.isclose(calculated_volume, v_test)


@named_damasceno_shapes_mark
def test_surface_area_damasceno_shapes(shape):
    if shape["name"] in ("RESERVED", "Sphere"):
        return
    vertices = shape["vertices"]
    poly = ConvexPolyhedron(vertices)
    hull = ConvexHull(vertices)
    assert np.isclose(poly.surface_area, hull.area)


@pytest.mark.skipif(
    is_not_ci_circleci(), reason="Test is too slow to run during rapid development"
)
@settings(deadline=1000)
@named_damasceno_shapes_mark
@given(a_test=floats(0, 10))
def test_set_surface_area_damasceno_shapes(shape, a_test):
    if shape["name"] in ("RESERVED", "Sphere"):
        return
    if a_test == 0:
        return
    vertices = shape["vertices"]
    poly = ConvexPolyhedron(vertices)
    poly.surface_area = a_test
    calculated_area = poly._calculate_surface_area()
    assert np.isclose(calculated_area, a_test)


@named_solids_mark
def test_volume_shapes(poly):
    vertices = poly.vertices
    hull = ConvexHull(vertices)
    assert np.isclose(poly.volume, hull.volume)


@named_solids_mark
def test_surface_area_shapes(poly):
    vertices = poly.vertices
    hull = ConvexHull(vertices)
    assert np.isclose(poly.surface_area, hull.area)


# This test is slow at high precisions. Run fast locally, and test in detail on CircleCI
@pytest.mark.parametrize(
    "atol",
    [1e-1 if is_not_ci_circleci() else 5e-2],
)
@named_damasceno_shapes_mark
def test_moment_inertia_damasceno_shapes(shape, atol):
    # These shapes pass the test for a sufficiently high number of samples, but
    # the number is too high to be worth running them regularly.
    bad_shapes = [
        "Augmented Truncated Dodecahedron",
        "Deltoidal Hexecontahedron",
        "Disdyakis Triacontahedron",
        "Metabiaugmented Truncated Dodecahedron",
        "Parabiaugmented Truncated Dodecahedron",
        "Paragyrate Diminished Rhombicosidodecahedron",
        "Pentagonal Hexecontahedron",
        "Rhombic Enneacontahedron",
        "Square Cupola",
        "Triaugmented Truncated Dodecahedron",
        "Truncated Dodecahedron",
        "Truncated Icosahedron",
        "Truncated Icosidodecahedron",
    ]
    if shape["name"] in ["RESERVED", "Sphere"] + bad_shapes:
        return

    np.random.seed(0)
    poly = ConvexPolyhedron(shape["vertices"])
    coxeter_result = poly.inertia_tensor
    volume = poly.volume
    num_samples = 1000
    accept = False
    # Loop over different sampling rates to minimize the test runtime.
    while num_samples < 1e8:
        try:
            mc_result = compute_inertia_mc(poly.vertices, volume, num_samples)
            assert np.allclose(coxeter_result, mc_result, atol=atol)
            accept = True
            break
        except AssertionError:
            num_samples *= 10
            continue
    if not accept:
        raise AssertionError(
            "The test failed for shape {}.\nMC Result: "
            "\n{}\ncoxeter result: \n{}".format(
                shape["name"], mc_result, coxeter_result
            )
        )


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_iq(cube):
    assert cube.iq == approx(36 * np.pi * cube.volume**2 / cube.surface_area**3)


def test_dihedrals():
    known_shapes = {
        "Tetrahedron": np.arccos(1 / 3),
        "Cube": np.pi / 2,
        "Octahedron": np.pi - np.arccos(1 / 3),
        "Dodecahedron": np.pi - np.arctan(2),
        "Icosahedron": np.pi - np.arccos(np.sqrt(5) / 3),
    }
    for name, dihedral in known_shapes.items():
        poly = PlatonicFamily.get_shape(name)
        # The dodecahedron needs a more expansive merge to get all the
        # faces joined.
        if name == "Dodecahedron":
            poly.merge_faces(rtol=1)
        for i in range(poly.num_faces):
            for j in poly.neighbors[i]:
                assert np.isclose(poly.get_dihedral(i, j), dihedral)


def test___repr__():
    # Platonic shapes have all congruent faces, so __repr__ should never fail
    test_platonic_shapes = [
        "Cube",
        "Dodecahedron",
        "Icosahedron",
        "Octahedron",
        "Tetrahedron",
    ]
    for name in test_platonic_shapes:
        poly = PlatonicFamily.get_shape(name)
        # repr() does not return faces for ConvexPolyhedron
        poly = Polyhedron(poly.vertices, poly.faces)
        # Try to run repr() for each shape
        repr(poly)
    cuboctahedron_vertices = [
        [-1.0, 0.0, 0.0],
        [-0.5, -0.5, -0.7071067811865475],
        [-0.5, -0.5, 0.7071067811865475],
        [-0.5, 0.5, -0.7071067811865475],
        [-0.5, 0.5, 0.7071067811865475],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, -0.5, -0.7071067811865475],
        [0.5, -0.5, 0.7071067811865475],
        [0.5, 0.5, -0.7071067811865475],
        [0.5, 0.5, 0.7071067811865475],
        [1.0, 0.0, 0.0],
    ]
    icosidodecahedron_vertices = [
        [0.0, -1.618033988749895, 0.0],
        [0.0, 1.618033988749895, 0.0],
        [0.2628655560595668, -0.8090169943749475, -1.3763819204711736],
        [0.2628655560595668, 0.8090169943749475, -1.3763819204711736],
        [0.42532540417601994, -1.3090169943749475, 0.85065080835204],
        [0.42532540417601994, 1.3090169943749475, 0.85065080835204],
        [0.6881909602355868, -0.5, 1.3763819204711736],
        [0.6881909602355868, 0.5, 1.3763819204711736],
        [1.1135163644116066, -0.8090169943749475, -0.85065080835204],
        [1.1135163644116066, 0.8090169943749475, -0.85065080835204],
        [-1.3763819204711736, 0.0, -0.85065080835204],
        [-0.6881909602355868, -0.5, -1.3763819204711736],
        [-0.6881909602355868, 0.5, -1.3763819204711736],
        [1.3763819204711736, 0.0, 0.85065080835204],
        [0.9510565162951535, -1.3090169943749475, 0.0],
        [0.9510565162951535, 1.3090169943749475, 0.0],
        [0.85065080835204, 0.0, -1.3763819204711736],
        [-0.9510565162951535, -1.3090169943749475, 0.0],
        [-0.9510565162951535, 1.3090169943749475, 0.0],
        [-1.5388417685876268, -0.5, 0.0],
        [-1.5388417685876268, 0.5, 0.0],
        [1.5388417685876268, -0.5, 0.0],
        [1.5388417685876268, 0.5, 0.0],
        [-0.85065080835204, 0.0, 1.3763819204711736],
        [-1.1135163644116066, -0.8090169943749475, 0.85065080835204],
        [-1.1135163644116066, 0.8090169943749475, 0.85065080835204],
        [-0.42532540417602, -1.3090169943749475, -0.85065080835204],
        [-0.42532540417602, 1.3090169943749475, -0.85065080835204],
        [-0.2628655560595668, -0.8090169943749475, 1.3763819204711736],
        [-0.2628655560595668, 0.8090169943749475, 1.3763819204711736],
    ]
    # Now, test repr() for a few irregular polyhedra
    cuboctahedron = ConvexPolyhedron(cuboctahedron_vertices)
    cuboctahedron = Polyhedron(cuboctahedron.vertices, cuboctahedron.faces)
    repr(cuboctahedron)
    icosidodecahedron = ConvexPolyhedron(icosidodecahedron_vertices)
    icosidodecahedron = Polyhedron(icosidodecahedron.vertices, icosidodecahedron.faces)
    repr(icosidodecahedron)


def test_edges():
    # The shapes in the PlatonicFamily are normalized to unit volume
    known_shapes = {
        "Tetrahedron": np.sqrt(2) * np.cbrt(3),
        "Cube": 1,
        "Octahedron": np.power(2, 5 / 6) * np.cbrt(3 / 8),
        "Dodecahedron": np.power(2, 2 / 3) * np.cbrt(1 / (15 + np.sqrt(245))),
        "Icosahedron": np.cbrt(9 / 5 - 3 / 5 * np.sqrt(5)),
    }
    number_of_edges = {
        "Tetrahedron": 6,
        "Cube": 12,
        "Octahedron": 12,
        "Dodecahedron": 30,
        "Icosahedron": 30,
    }

    for name, edgelength in known_shapes.items():
        poly = PlatonicFamily.get_shape(name)
        # Test that the correct number of edges has been found
        assert number_of_edges[name] == len(poly.edges)
        assert number_of_edges[name] == len(poly.edge_vectors)

        # Test edge_vectors property
        for edge in poly.edge_vectors:
            assert np.isclose(np.linalg.norm(edge), edgelength)

        # Test edges property
        edges = np.asarray(
            [
                *poly.edges,
            ]
        )
        vertices = poly.vertices
        veclens = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
        assert np.allclose(veclens, edgelength)


def test_curvature():
    """Regression test against values computed with older method."""
    # The shapes in the PlatonicFamily are normalized to unit volume.
    known_shapes = {
        "Cube": 0.75,
        "Dodecahedron": 0.6703242780091758,
        "Icosahedron": 0.6715997848012972,
        "Octahedron": 0.75518565565632,
        "Tetrahedron": 0.9303430847680867,
    }

    for name, curvature in known_shapes.items():
        poly = PlatonicFamily.get_shape(name)
        # Normalize volume to ensure known_shape data is accurate
        poly.volume = 1
        assert np.isclose(poly.mean_curvature, curvature)


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


# Circumspheres cannot be generated for some Johnson and Catalan solids
@combine_marks(named_platonic_mark, named_archimedean_mark)
def test_circumsphere(poly):
    circumsphere = poly.circumsphere

    # Ensure polyhedron is centered, then compute distances.
    poly.center = [0, 0, 0]
    r2 = np.sum(poly.vertices**2, axis=1)

    assert np.allclose(r2, circumsphere.radius**2)


@combine_marks(named_platonic_mark, named_archimedean_mark)
def test_circumsphere_radius(poly):
    # Ensure polyhedron is centered, then compute distances.
    poly.center = [0, 0, 0]
    r2 = np.sum(poly.vertices**2, axis=1)

    assert np.allclose(r2, poly.circumsphere_radius**2)
    poly.circumsphere_radius *= 2
    assert np.allclose(r2 * 4, poly.circumsphere_radius**2)


def test_minimal_centered_bounding_sphere():
    """Validate circumsphere by testing the polyhedron.

    This checks that all points outside this circumsphere are also outside the
    polyhedron. Note that this is a necessary but not sufficient condition for
    correctness.
    """
    # Building convex polyhedra is the slowest part of this test, so rather
    # than testing all the shapes from this particular dataset every time we
    # instead test a random subset each time the test runs. To further speed
    # the tests, we build all convex polyhedra ahead of time. Each set of
    # random points is tested against a different random polyhedron.
    import random

    family = DOI_SHAPE_REPOSITORIES["10.1126/science.1220869"][0]
    shapes = [
        ConvexPolyhedron(s["vertices"])
        for s in random.sample(
            [s for s in family.data.values() if len(s["vertices"])],
            len(family.data) // 5,
        )
    ]

    # Use a nested function to avoid warnings from hypothesis. While the shape
    # does get modified inside the testfun, it's simply being recentered each
    # time, which is not destructive since it can be overwritten in subsequent
    # calls.
    # See https://github.com/HypothesisWorks/hypothesis/issues/377
    @settings(deadline=2000)
    @given(
        center=arrays(
            np.float64, (3,), elements=floats(-10, 10, width=64), unique=True
        ),
        points=arrays(
            np.float64, (50, 3), elements=floats(-1, 1, width=64), unique=True
        ),
        shape_index=integers(0, len(shapes) - 1),
    )
    def testfun(center, points, shape_index):
        poly = shapes[shape_index]
        poly.center = center

        sphere = poly.minimal_centered_bounding_sphere
        scaled_points = points * sphere.radius + sphere.center
        points_outside = np.logical_not(sphere.is_inside(scaled_points))

        # Verify that all points outside the circumsphere are also outside the
        # polyhedron.
        assert not np.any(np.logical_and(points_outside, poly.is_inside(scaled_points)))

        with pytest.deprecated_call():
            assert sphere_isclose(sphere, poly.circumsphere_from_center)

    testfun()


# Shapes where |aspect ratio - 1| >> 0 cannot pass this test: this includes many
# Johnson solids. Shapes with large numbers of vertices also tend to fail
@combine_marks(named_platonic_mark, named_archimedean_mark)
def test_bounding_sphere(poly):
    # Ensure polyhedron is centered, then compute distances.
    poly.center = [0, 0, 0]
    r2 = np.sum(poly.vertices**2, axis=1)

    bounding_sphere = poly.minimal_bounding_sphere
    assert np.allclose(r2, bounding_sphere.radius**2, rtol=1e-5)

    with pytest.deprecated_call():
        assert sphere_isclose(bounding_sphere, poly.bounding_sphere)


def test_inside_boundaries(convex_cube):
    assert np.all(convex_cube.is_inside(convex_cube.vertices))
    convex_cube.center = [0, 0, 0]
    assert np.all(convex_cube.is_inside(convex_cube.vertices * 0.99))
    assert not np.any(convex_cube.is_inside(convex_cube.vertices * 1.01))


def test_inside(convex_cube):
    # Use a nested function to avoid warnings from hypothesis. In this case, it
    # is safe to reuse the convex cube.
    # See https://github.com/HypothesisWorks/hypothesis/issues/377
    @given(arrays(np.float64, (100, 3), elements=floats(-10, 10, width=64)))
    def testfun(test_points):
        expected = np.all(np.logical_and(test_points >= 0, test_points <= 1), axis=1)
        actual = convex_cube.is_inside(test_points)
        np.testing.assert_allclose(expected, actual)

    testfun()


@settings(max_examples=10)
@given(
    EllipsoidSurfaceStrategy,
    arrays(np.float64, (100, 3), elements=floats(0, 1, width=64)),
)
def test_maximal_centered_bounded_sphere_convex_hulls(points, test_points):
    hull = ConvexHull(points)
    poly = ConvexPolyhedron(points[hull.vertices])
    try:
        insphere = poly.maximal_centered_bounded_sphere
    except ValueError as e:
        # Ignore cases where triangulation fails, we're not interested in
        # trying to get polytri to work for nearly degenerate cases.
        if str(e) == "Triangulation failed":
            assume(False)
    assert poly.is_inside(insphere.center)

    test_points *= insphere.radius * 3
    points_in_sphere = insphere.is_inside(test_points)
    points_in_poly = poly.is_inside(test_points)
    assert np.all(points_in_sphere <= points_in_poly)
    assert insphere.volume < poly.volume

    with pytest.deprecated_call():
        assert sphere_isclose(insphere, poly.insphere_from_center)


# Platonic and Catalan shapes have a centered insphere, but Archimedean
# and Johnson solids do not.
@named_platonic_mark
def test_insphere_platonic(poly):
    # The insphere should be centered for platonic solids.
    poly_insphere = poly.insphere
    assert sphere_isclose(
        poly_insphere, poly.maximal_centered_bounded_sphere, atol=1e-5
    )

    # The insphere of a platonic solid should be rotation invariant.
    @settings(deadline=500)
    @given(Random3DRotationStrategy)
    def check_rotation_invariance(quat):
        rotated_poly = ConvexPolyhedron(rowan.rotate(quat, poly.vertices))
        assert sphere_isclose(poly_insphere, rotated_poly.insphere, atol=1e-5)

    check_rotation_invariance()


@named_catalan_mark
def test_insphere_catalan(poly):
    # The insphere should be centered for catalan solids.
    poly_insphere = poly.insphere
    assert sphere_isclose(
        poly_insphere, poly.maximal_centered_bounded_sphere, atol=1e-5
    )

    # The insphere of a catalan solid should be rotation invariant.
    @settings(deadline=5000)
    @given(Random3DRotationStrategy)
    def check_rotation_invariance(quat):
        rotated_poly = ConvexPolyhedron(rowan.rotate(quat, poly.vertices))
        assert sphere_isclose(poly_insphere, rotated_poly.insphere, atol=1e-5)

    check_rotation_invariance()


def test_rotate_inertia(tetrahedron):
    # Use the input as noise rather than the base points to avoid precision and
    # degenerate cases provided by hypothesis.
    tet = PlatonicFamily.get_shape("Tetrahedron")
    tet.volume = 1

    @given(
        arrays(np.float64, (4, 3), elements=floats(-10, 10, width=64), unique=True),
        Random3DRotationStrategy,
    )
    def testfun(points, rotation):
        vertices = tet.vertices + points
        shape = ConvexPolyhedron(vertices)
        rotated_shape = ConvexPolyhedron(rowan.rotate(rotation, vertices))

        mat = rowan.to_matrix(rotation)
        rotated_inertia = rotate_order2_tensor(mat, shape.inertia_tensor)

        assert np.allclose(rotated_inertia, rotated_shape.inertia_tensor)

    testfun()


# Use a small range of translations to ensure that the Delaunay triangulation
# used by the MC calculation will not break.
def test_translate_inertia(convex_cube):
    # Choose a volume > 1 to test the volume scaling, but don't choose one
    # that's too large because the uncentered polyhedral calculation has
    # massive error without fixing that.
    convex_cube.volume = 2
    convex_cube.center = (0, 0, 0)
    convex_cube_inertia_tensor = convex_cube.inertia_tensor
    convex_cube_volume = convex_cube.volume

    @settings(deadline=400)
    @given(arrays(np.float64, (3,), elements=floats(-0.2, 0.2, width=64), unique=True))
    def testfun(translation):
        translated_convex_cube = ConvexPolyhedron(convex_cube.vertices + translation)

        translated_inertia = translate_inertia_tensor(
            translation, convex_cube_inertia_tensor, convex_cube_volume
        )
        mc_tensor = compute_inertia_mc(
            translated_convex_cube.vertices, convex_cube_volume, 1e4
        )

        uncentered_inertia_tensor = translated_convex_cube._compute_inertia_tensor(
            False
        )
        assert np.allclose(
            translated_inertia,
            uncentered_inertia_tensor,
            atol=2e-1,
            rtol=2e-1,
        )
        assert np.allclose(mc_tensor, uncentered_inertia_tensor, atol=2e-1, rtol=2e-1)

        assert np.allclose(mc_tensor, translated_inertia, atol=1e-2, rtol=1e-2)
        assert np.allclose(
            mc_tensor, translated_convex_cube.inertia_tensor, atol=1e-2, rtol=1e-2
        )

    testfun()


@settings(deadline=None)
@given(EllipsoidSurfaceStrategy)
def test_diagonalize_inertia(points):
    """Test that we can orient a polyhedron along its principal axes."""
    # Some of the points might be in the interior of the hull, but the
    # ConvexPolyhedron is constructed from the hull anyway so that's fine.
    poly = ConvexPolyhedron(points)

    try:
        poly.diagonalize_inertia()
        it = poly.inertia_tensor
        assert np.allclose(np.diag(np.diag(it)), it)
    except ValueError:
        # Triangulation can fail, this is a limitation of polytri and not something we
        # can address without implementing a more robust algorithm.
        return


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_form_factor(cube):
    """Validate the form factor of a polyhedron.

    At the moment this is primarily a regression test, and should be expanded for more
    rigorous validation.
    """
    cube.center = (0, 0, 0)
    cube.volume = 8

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
        dtype=float,
    )
    np.testing.assert_allclose(
        cube.compute_form_factor_amplitude(ks),
        [
            8.0,
            6.73176788,
            6.73176788,
            3.63718971,
            6.73176788,
            6.73176788,
            0.14397014,
            0.1169148,
        ],
        atol=1e-7,
    )

    # Test a translational shift.
    center = [1, 1, 1]
    cube.center = center
    np.testing.assert_allclose(
        cube.compute_form_factor_amplitude(ks),
        [
            8.0 + 0.0j,
            3.63718971 - 5.66458735j,
            3.63718971 + 5.66458735j,
            -1.51360499 - 3.30728724j,
            3.63718971 - 5.66458735j,
            3.63718971 - 5.66458735j,
            0.13823585 + 0.04022749j,
            -0.11671542 - 0.0068248j,
        ],
        atol=1e-7,
    )


@named_solids_mark
@pytest.mark.xfail(
    reason=(
        "Numerical precision problems with miniball. "
        "See https://github.com/glotzerlab/coxeter/issues/179"
    )
)
def test_get_set_minimal_bounding_sphere_radius(poly):
    _test_get_set_minimal_bounding_sphere_radius(poly)


@named_solids_mark
def test_get_set_minimal_centered_bounding_sphere_radius(poly):
    _test_get_set_minimal_bounding_sphere_radius(poly, True)


@pytest.mark.parametrize(
    "cube", ["convex_cube", "oriented_cube", "unoriented_cube"], indirect=True
)
def test_is_inside(cube):
    assert cube.is_inside(cube.center)

    @given(floats(0, 1), floats(0, 1), floats(0, 1))
    def testfun(x, y, z):
        assert cube.is_inside([[x, y, z]])

    testfun()


def test_repr_nonconvex(oriented_cube):
    assert str(oriented_cube), str(eval(repr(oriented_cube)))


def test_repr_convex(convex_cube):
    assert str(convex_cube), str(eval(repr(convex_cube)))


# Test fast locally, and in more depth on CircleCI
@pytest.mark.parametrize(
    "atol",
    [1e-2 if is_not_ci_circleci() else 5e-3],
)
@named_damasceno_shapes_mark
def test_center(shape, atol):
    poly = ConvexPolyhedron(shape["vertices"])
    coxeter_result = poly.center
    num_samples = 5000
    accept = False
    while num_samples < 5e7:
        try:
            mc_result = compute_centroid_mc(shape["vertices"], num_samples)
            assert np.allclose(coxeter_result, mc_result, atol=atol)
            accept = True
            break
        except AssertionError:
            num_samples *= 10
            continue
    if not accept:
        raise AssertionError(
            "The test failed for shape {}.\nMC Result: "
            "\n{}\ncoxeter result: \n{}".format(
                shape["name"], mc_result, coxeter_result
            )
        )


@named_solids_mark
@given(arrays(np.float64, (3,), elements=floats(-10, 10, width=64), unique=True))
def test_set_centroid(poly, centroid_vector):
    poly.centroid = centroid_vector
    coxeter_result = poly.centroid
    assert np.allclose(coxeter_result, centroid_vector, atol=1e-12)
    poly.centroid = [0, 0, 0]
    assert np.allclose(poly.centroid, [0, 0, 0], atol=1e-12)


@named_platonic_mark
def test_face_centroids(poly):
    # For platonic solids, the centroid of a face is equal to the mean of its vertices
    poly.centroid = [0, 0, 0]
    coxeter_result = poly.face_centroids
    for i, face in enumerate(poly.faces):
        face_vertices = poly.vertices[face]
        vertex_mean = np.mean(face_vertices, axis=0)
        assert np.allclose(vertex_mean, coxeter_result[i])


@given(EllipsoidSurfaceStrategy)
def test_find_simplex_equations(points):
    hull = ConvexHull(points)
    poly = ConvexPolyhedron(points[hull.vertices])
    # Check simplex equations are stored properly
    assert np.allclose(hull.equations, poly._simplex_equations)

    # Now recalculate and check answers are still correct
    poly._find_simplex_equations()
    assert np.allclose(hull.equations, poly._simplex_equations)


@named_solids_mark
def test_find_equations_and_normals(poly):
    ppoly = Polyhedron(poly.vertices, poly.faces)
    # Check face equations are stored properly
    assert np.allclose(poly.equations, ppoly._equations)
    assert np.allclose(poly.normals, ppoly.normals)

    # Now recalculate and check answers are still correct
    poly._find_equations()
    ppoly._find_equations()
    assert np.allclose(poly.equations, ppoly._equations)
    assert np.allclose(poly.normals, ppoly.normals)
