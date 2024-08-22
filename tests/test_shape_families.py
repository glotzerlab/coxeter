# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from coxeter.families import (
    DOI_SHAPE_REPOSITORIES,
    ArchimedeanFamily,
    CatalanFamily,
    Family323Plus,
    Family423,
    Family523,
    JohnsonFamily,
    PlatonicFamily,
    RegularNGonFamily,
    TabulatedGSDShapeFamily,
    TruncatedTetrahedronFamily,
)
from coxeter.shapes import ConvexPolyhedron

MIN_REALISTIC_PRECISION = 2e-6


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


@pytest.mark.parametrize("n", range(3, 100))
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


def test_science_family():
    reference_mapping = {
        "P": PlatonicFamily,
        "A": ArchimedeanFamily,
        "C": CatalanFamily,
        "J": JohnsonFamily,
        "O": None,
    }
    for name, shape in ScienceFamily:
        reference = reference_mapping[name[0]]
        if reference is not None:
            try:
                np.testing.assert_allclose(
                    reference.get_shape(ScienceFamily.data[name]["name"]).vertices,
                    shape.vertices,
                )
            except KeyError as e:
                if name[0] != "J":
                    raise KeyError from e


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
