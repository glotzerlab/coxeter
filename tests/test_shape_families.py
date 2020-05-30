import numpy as np
import pytest

from coxeter.shape_families import (Family323Plus, Family423, Family523,
                                    RegularNGonFamily,
                                    TruncatedTetrahedronFamily,
                                    family_from_doi)


@pytest.mark.parametrize('n', range(3, 100))
def test_regular_ngon(n):
    family = RegularNGonFamily()
    poly = family(n)
    assert len(poly.vertices) == n
    # All side lengths should be the same.
    assert len(np.unique(np.linalg.norm(
        poly.vertices - np.roll(poly.vertices, shift=-1, axis=0),
        axis=1).round(4))) == 1


def test_shape_repos():
    family = family_from_doi('10.1126/science.1220869')[0]
    for key, shape_data in family.data.items():
        if shape_data['name'] == 'Cube':
            break
    else:
        assert False, "Could not find a cube in the dataset."

    cube = family(key)
    assert len(cube.vertices) == 8
    assert len(cube.faces) == 6


def test_shape323():
    family = Family323Plus()
    # Octahedron (6)
    assert len(family(1, 1).vertices) == 6
    assert len(family(1, 1).faces) == 8
    # Tetrahedron (4)
    assert len(family(1, 3).vertices) == 4
    assert len(family(1, 3).faces) == 4
    # Tetrahedron (4)
    assert len(family(3, 1).vertices) == 4
    assert len(family(3, 1).faces) == 4
    # Cube (8)
    assert len(family(3, 3).vertices) == 8
    assert len(family(3, 3).faces) == 6


def test_shape423():
    family = Family423()
    # Cuboctahedron (12)
    assert len(family(1, 2).vertices) == 12
    assert len(family(1, 2).faces) == 14
    # Octahedron (6)
    assert len(family(2, 2).vertices) == 6
    assert len(family(2, 2).faces) == 8
    # Cube (8)
    assert len(family(1, 3).vertices) == 8
    assert len(family(1, 3).faces) == 6
    # Rhombic Dodecahedron (14)
    assert len(family(2, 3).vertices) == 14
    assert len(family(2, 3).faces) == 12


def test_shape523():
    family = Family523()
    s = family.s
    # Icosidodecahedron
    assert len(family(1, family.S**2).vertices) == 30
    assert len(family(1, family.S**2).faces) == 32
    # Icosahedron
    assert len(family(1 * s * np.sqrt(5), family.S**2).vertices) == 12
    assert len(family(1 * s * np.sqrt(5), family.S**2).faces) == 20
    # Dodecahedron
    assert len(family(1, 3).vertices) == 20
    assert len(family(1, 3).faces) == 12
    # Rhombic Triacontahedron
    assert len(family(1 * s * np.sqrt(5), 3).vertices) == 32
    assert len(family(1 * s * np.sqrt(5), 3).faces) == 30


def test_truncated_tetrahedron():
    family = TruncatedTetrahedronFamily()
    # Test the endpoints (tetrahedron or octahedron).
    tet = family(0)
    assert len(tet.vertices) == 4
    assert len(tet.faces) == 4

    tet = family(1)
    assert len(tet.vertices) == 6
    assert len(tet.faces) == 8
