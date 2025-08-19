# Copyright (c) 2015-2025 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from coxeter.families import (
    DOI_SHAPE_REPOSITORIES,
    ArchimedeanFamily,
    CanonicalTrapezohedronFamily,
    CatalanFamily,
    Family323Plus,
    Family423,
    Family523,
    JohnsonFamily,
    PlatonicFamily,
    PrismAntiprismFamily,
    PyramidDipyramidFamily,
    RegularNGonFamily,
    TabulatedGSDShapeFamily,
    TetragonalDisphenoidFamily,
    TruncatedTetrahedronFamily,
    UniformAntiprismFamily,
    UniformDipyramidFamily,
    UniformPrismFamily,
    UniformPyramidFamily,
)
from coxeter.shapes import ConvexPolyhedron

ATOL = 1e-15
MIN_REALISTIC_PRECISION = 2e-6
MIN_DECIMALS = 6
MAX_N_POLY = 102
TEST_EXAMPLES = 32

ScienceFamily = DOI_SHAPE_REPOSITORIES["10.1126/science.1220869"][0]


def _test_parameters_outside_precision(params_list):
    """Certain input values for plane shape families can result in untriangulable
    shapes due to numeric imprecision. Filtering input values can avoid these
    unlikely cases. Note that integer inputs or inputs exactly on a boundary will
    pass, but points very close to these will not."""

    def is_close_to_shape_space_boundary(param):
        nearest_multiple = round(param / 0.25) * 0.25
        return abs(param - nearest_multiple) <= MIN_REALISTIC_PRECISION

    return any(is_close_to_shape_space_boundary(param) for param in params_list)


@given(n=integers(3, MAX_N_POLY))
@settings(max_examples=TEST_EXAMPLES)
def test_regular_ngon(n):
    poly = RegularNGonFamily.get_shape(n)
    assert len(poly.vertices) == n
    # All side lengths should be the same.
    assert (
        len(
            np.unique(
                np.linalg.norm(
                    poly.vertices - np.roll(poly.vertices, shift=-1, axis=0), axis=1
                ).round(4)
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    "family",
    [PlatonicFamily, ArchimedeanFamily, CatalanFamily, JohnsonFamily, ScienceFamily],
    ids=[
        "PlatonicFamily",
        "ArchimedeanFamily",
        "CatalanFamily",
        "JohnsonFamily",
        "ScienceFamily",
    ],
)
def test_named_family(family):
    def test_type(family):
        assert isinstance(family, TabulatedGSDShapeFamily)

    def test_iteration(family):
        for name, shape in family:
            assert isinstance(shape, ConvexPolyhedron)
            np.testing.assert_allclose(family.get_shape(name).vertices, shape.vertices)

    def test_names(family):
        for name in family.names:
            family.get_shape(name)

    test_type(family)
    test_iteration(family)
    test_names(family)


REFERENCE_MAPPING = {
    "P": PlatonicFamily,
    "A": ArchimedeanFamily,
    "C": CatalanFamily,
    "J": JohnsonFamily,
    "O": None,
}


@pytest.mark.parametrize("id, shape", ScienceFamily)
def test_science_family_properties(id, shape):
    reference = REFERENCE_MAPPING[id[0]]
    if reference is None:
        return

    # Skip specific shapes as in the original test
    if id in ("J01", "J02", "J12", "J13"):
        reference_shape = reference.get_shape(
            ScienceFamily.data[id]["alternative_name"]
        )
    else:
        reference_shape = reference.get_shape(ScienceFamily.data[id]["name"])

    np.testing.assert_allclose([shape.volume, reference_shape.volume], 1.0, rtol=1e-15)
    np.testing.assert_allclose(
        [shape.centroid, reference_shape.centroid], 0, atol=1e-15
    )
    np.testing.assert_allclose(
        reference_shape.vertices, shape.vertices, err_msg=ScienceFamily.data[id]["name"]
    )


@pytest.mark.parametrize("id, shape", ScienceFamily)
def test_science_family_faces(id, shape):
    # Test specific shapes with hardcoded face counts
    name = ScienceFamily.data[id]["name"]
    if name == "Squashed Dodecahedron":
        assert shape.num_faces == 12
    elif name == "Rhombic Icosahedron":
        assert shape.num_faces == 20
    elif name == "Rhombic Enneacontahedron":
        assert shape.num_faces == 90
    elif name == "Obtuse Golden Rhombohedron":
        assert shape.num_faces == 6
    elif name == "Acute Golden Rhombohedron":
        assert shape.num_faces == 6
    elif name == "Dürer's solid":
        assert shape.num_faces == 8


def test_shape323():
    family = Family323Plus
    # Octahedron (6, 8)
    assert len(family.get_shape(1.0, 1.0).vertices) == 6
    assert len(family.get_shape(1.0, 1.0).faces) == 8
    # Tetrahedron (4, 4)
    assert len(family.get_shape(1.0, 3.0).vertices) == 4
    assert len(family.get_shape(1.0, 3.0).faces) == 4
    # Tetrahedron (4, 4)
    assert len(family.get_shape(3.0, 1.0).vertices) == 4
    assert len(family.get_shape(3.0, 1.0).faces) == 4
    # Cube (8, 6)
    assert len(family.get_shape(3.0, 3.0).vertices) == 8
    assert len(family.get_shape(3.0, 3.0).faces) == 6


@given(a=floats(1, 3), c=floats(1, 3))
def test_shape323_intermediates(a, c):
    if _test_parameters_outside_precision([a, c]):
        return
    Family323Plus.get_shape(a, c)


def test_shape423():
    family = Family423
    # Cuboctahedron (12, 14)
    assert len(family.get_shape(1.0, 2.0).vertices) == 12
    assert len(family.get_shape(1.0, 2.0).faces) == 14
    # Octahedron (6, 8)
    assert len(family.get_shape(2.0, 2.0).vertices) == 6
    assert len(family.get_shape(2.0, 2.0).faces) == 8
    # Cube (8, 6)
    assert len(family.get_shape(1.0, 3.0).vertices) == 8
    assert len(family.get_shape(1.0, 3.0).faces) == 6
    # Rhombic Dodecahedron (14, 12)
    assert len(family.get_shape(2.0, 3.0).vertices) == 14
    assert len(family.get_shape(2.0, 3.0).faces) == 12


@given(a=floats(1, 2), c=floats(2, 3))
def test_shape423_intermediates(a, c):
    if _test_parameters_outside_precision([a, c]):
        return
    Family423.get_shape(a, c)


def test_shape523():
    family = Family523
    s = family.s

    # s is the inverse golden ratio. Check we have the correct value
    assert np.isclose(s, (np.sqrt(5) - 1) / 2)

    # Icosidodecahedron (30, 32)
    assert len(family.get_shape(1.0, family.S**2).vertices) == 30
    assert len(family.get_shape(1.0, family.S**2).faces) == 32
    # Icosahedron (12, 20)
    assert len(family.get_shape(1 * s * np.sqrt(5), family.S**2).vertices) == 12
    assert len(family.get_shape(1 * s * np.sqrt(5), family.S**2).faces) == 20
    # Dodecahedron (20, 12)
    assert len(family.get_shape(1.0, 3.0).vertices) == 20
    assert len(family.get_shape(1.0, 3.0).faces) == 12
    # Rhombic Triacontahedron (32, 30)
    assert len(family.get_shape(1 * s * np.sqrt(5), 3.0).vertices) == 32
    assert len(family.get_shape(1 * s * np.sqrt(5), 3.0).faces) == 30


@given(a=floats(1, Family523.s * np.sqrt(5)), c=floats(Family523.S**2, 3))
def test_shape523_intermediates(a, c):
    if (
        _test_parameters_outside_precision([a, c])
        or c % Family523.S**2 < MIN_REALISTIC_PRECISION
        or c % Family523.S**2 < (1 - MIN_REALISTIC_PRECISION)
    ):
        return
    Family523.get_shape(a, c)


def test_truncated_tetrahedron():
    family = TruncatedTetrahedronFamily
    # Test the endpoints (tetrahedron or octahedron).
    # Tetrahedron (4, 4)
    tet = family.get_shape(0.0)
    assert len(tet.vertices) == 4
    assert len(tet.faces) == 4

    # Octahedron (6, 8)
    tet = family.get_shape(1.0)
    assert len(tet.vertices) == 6
    assert len(tet.faces) == 8


@given(t=floats(0, 1))
def test_truncated_tetrahedron_intermediates(t):
    if _test_parameters_outside_precision([t]) or np.abs(np.round(t, 15) - t) < 2e-16:
        return
    TruncatedTetrahedronFamily.get_shape(t)


@given(n=integers(3, MAX_N_POLY))
@settings(max_examples=TEST_EXAMPLES)
def test_uniform_prisms(n):
    vertices = UniformPrismFamily.make_vertices(n=n)
    shape = UniformPrismFamily.get_shape(n=n)

    np.testing.assert_allclose(shape.centroid, 0.0, atol=ATOL)
    np.testing.assert_allclose(shape.volume, 1.0, atol=ATOL)
    np.testing.assert_allclose(shape.edge_lengths, shape.edge_lengths.mean(), atol=ATOL)
    np.testing.assert_allclose(vertices, shape.vertices)


@given(n=integers(3, MAX_N_POLY))
@settings(max_examples=TEST_EXAMPLES)
def test_uniform_antiprisms(n):
    vertices = UniformAntiprismFamily.make_vertices(n=n)
    shape = UniformAntiprismFamily.get_shape(n=n)

    np.testing.assert_allclose(shape.centroid, 0.0, atol=ATOL)
    np.testing.assert_allclose(shape.volume, 1.0, atol=ATOL)
    np.testing.assert_allclose(shape.edge_lengths, shape.edge_lengths.mean(), atol=ATOL)
    np.testing.assert_allclose(vertices, shape.vertices)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_uniform_pyramids(n):
    vertices = UniformPyramidFamily.make_vertices(n=n)
    shape = UniformPyramidFamily.get_shape(n=n)

    np.testing.assert_allclose(shape.centroid, 0.0, atol=ATOL)
    np.testing.assert_allclose(shape.volume, 1.0, atol=ATOL)
    np.testing.assert_allclose(shape.edge_lengths, shape.edge_lengths.mean(), atol=ATOL)
    np.testing.assert_allclose(vertices, shape.vertices)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_uniform_dipyramids(n):
    vertices = UniformDipyramidFamily.make_vertices(n=n)
    shape = UniformDipyramidFamily.get_shape(n=n)

    np.testing.assert_allclose(shape.centroid, 0.0, atol=ATOL)
    np.testing.assert_allclose(shape.volume, 1.0, atol=ATOL)
    np.testing.assert_allclose(shape.edge_lengths, shape.edge_lengths.mean(), atol=ATOL)
    np.testing.assert_allclose(vertices, shape.vertices)


def generate_prism_antiprism_params():
    # Map from numerical n to the string name
    number_to_name = {
        3: "Triangular",
        4: "Square",
        5: "Pentagonal",
        6: "Hexagonal",
        7: "Heptagonal",
        8: "Octagonal",
        9: "Nonagonal",
        10: "Decagonal",
    }

    shape_map = {name: shape for name, shape in PrismAntiprismFamily}

    for n in range(3, 11):
        name_prefix = number_to_name.get(n)
        if not name_prefix:
            continue

        prism_name = f"{name_prefix} Prism"
        prism_shape = shape_map.get(prism_name)
        if prism_shape:
            yield pytest.param(prism_shape, n, "prism", id=prism_name)

        antiprism_name = f"{name_prefix} Antiprism"
        antiprism_shape = shape_map.get(antiprism_name)
        if antiprism_shape:
            yield pytest.param(antiprism_shape, n, "antiprism", id=antiprism_name)


@pytest.mark.parametrize("shape, n, shape_type", generate_prism_antiprism_params())
def test_new_prism_antiprism(shape, n, shape_type):
    if shape_type == "antiprism":
        comparative_shape = UniformAntiprismFamily.get_shape(n)
        n_edges = 4 * n
        n_faces = 2 + 2 * n
    else:
        comparative_shape = UniformPrismFamily.get_shape(n)
        n_edges = 3 * n
        n_faces = 2 + n

    assert shape.num_edges == n_edges
    assert shape.num_faces == n_faces
    np.testing.assert_allclose(shape.volume, comparative_shape.volume, atol=ATOL)
    np.testing.assert_allclose(
        shape.edge_lengths, comparative_shape.edge_lengths, atol=ATOL
    )


def test_new_pyramid_dipyramid():
    with pytest.warns(DeprecationWarning, match="deprecated in favor of"):
        for i, nameshape in enumerate(PyramidDipyramidFamily):
            name, shape = nameshape

            if "Di" in name:
                n = i + 3  # count + min_n
                comparative_shape = UniformDipyramidFamily.get_shape(n)
            else:
                n = (i + 3) - 3  # count + min_n + n_pyramid
                comparative_shape = UniformPyramidFamily.get_shape(n)

            np.testing.assert_allclose(comparative_shape.centroid, 0.0, atol=ATOL)
            np.testing.assert_allclose(
                shape.volume, comparative_shape.volume, atol=ATOL
            )
            np.testing.assert_allclose(
                shape.edge_lengths, comparative_shape.edge_lengths, atol=ATOL
            )


@given(
    integers(3, MAX_N_POLY),
)
def test_trapezohedra(n):
    vertices = CanonicalTrapezohedronFamily.make_vertices(n)
    poly = CanonicalTrapezohedronFamily.get_shape(n)
    np.testing.assert_array_equal(vertices, poly.vertices)

    assert np.isclose(poly.volume, 1.0)
    assert len(np.unique(poly.edge_lengths.round(MIN_DECIMALS))) == 2 or np.allclose(
        poly.edge_lengths, 1.0
    )
    assert all([len(face) == 4 for face in poly.faces])  # All faces are kites


@given(
    floats(MIN_REALISTIC_PRECISION, 10_000_000),
    floats(MIN_REALISTIC_PRECISION, 10_000_000),
    floats(MIN_REALISTIC_PRECISION, 10_000_000),
)
def test_tetragonal_trapezohedra(a, b, c):
    vertices = TetragonalDisphenoidFamily.make_vertices(a, b, c)
    poly = TetragonalDisphenoidFamily.get_shape(a, b, c)
    np.testing.assert_array_equal(vertices, poly.vertices)

    assert np.isclose(poly.volume, 1.0)
    assert len(np.unique(poly.edge_lengths.round(MIN_DECIMALS))) in (1, 2, 3)
    assert all([len(face) == 3 for face in poly.faces])  # All faces are triangles
