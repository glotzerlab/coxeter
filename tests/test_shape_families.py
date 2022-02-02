# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
import numpy as np
import pytest

from coxeter.families import (
    DOI_SHAPE_REPOSITORIES,
    Family323Plus,
    Family423,
    Family523,
    RegularNGonFamily,
    TruncatedTetrahedronFamily,
)


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


def test_shape_repos():
    family = DOI_SHAPE_REPOSITORIES["10.1126/science.1220869"][0]
    for key in family.data:
        if family.data[key]["name"] == "Cube":
            break
    else:
        raise AssertionError("Could not find a cube in the dataset.")

    cube = family.get_shape(key)
    assert len(cube.vertices) == 8
    assert len(cube.faces) == 6


def test_shape323():
    family = Family323Plus
    # Octahedron (6)
    assert len(family.get_shape(1, 1).vertices) == 6
    assert len(family.get_shape(1, 1).faces) == 8
    # Tetrahedron (4)
    assert len(family.get_shape(1, 3).vertices) == 4
    assert len(family.get_shape(1, 3).faces) == 4
    # Tetrahedron (4)
    assert len(family.get_shape(3, 1).vertices) == 4
    assert len(family.get_shape(3, 1).faces) == 4
    # Cube (8)
    assert len(family.get_shape(3, 3).vertices) == 8
    assert len(family.get_shape(3, 3).faces) == 6


def test_shape423():
    family = Family423
    # Cuboctahedron (12)
    assert len(family.get_shape(1, 2).vertices) == 12
    assert len(family.get_shape(1, 2).faces) == 14
    # Octahedron (6)
    assert len(family.get_shape(2, 2).vertices) == 6
    assert len(family.get_shape(2, 2).faces) == 8
    # Cube (8)
    assert len(family.get_shape(1, 3).vertices) == 8
    assert len(family.get_shape(1, 3).faces) == 6
    # Rhombic Dodecahedron (14)
    assert len(family.get_shape(2, 3).vertices) == 14
    assert len(family.get_shape(2, 3).faces) == 12


def test_shape523():
    family = Family523
    s = family.s
    # Icosidodecahedron
    assert len(family.get_shape(1, family.S**2).vertices) == 30
    assert len(family.get_shape(1, family.S**2).faces) == 32
    # Icosahedron
    assert len(family.get_shape(1 * s * np.sqrt(5), family.S**2).vertices) == 12
    assert len(family.get_shape(1 * s * np.sqrt(5), family.S**2).faces) == 20
    # Dodecahedron
    assert len(family.get_shape(1, 3).vertices) == 20
    assert len(family.get_shape(1, 3).faces) == 12
    # Rhombic Triacontahedron
    assert len(family.get_shape(1 * s * np.sqrt(5), 3).vertices) == 32
    assert len(family.get_shape(1 * s * np.sqrt(5), 3).faces) == 30


def test_truncated_tetrahedron():
    family = TruncatedTetrahedronFamily
    # Test the endpoints (tetrahedron or octahedron).
    tet = family.get_shape(0)
    assert len(tet.vertices) == 4
    assert len(tet.faces) == 4

    tet = family.get_shape(1)
    assert len(tet.vertices) == 6
    assert len(tet.faces) == 8
