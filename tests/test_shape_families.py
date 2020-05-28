from coxeter import shape_families
from coxeter.shape_families.shape_family import Family332, Family432, Family532
import numpy as np


def test_shape_repos():
    repo = shape_families.shape_repositories['10.1126/science.1220869']
    print(repo)
    print(repo('1'))


def test_shape332():
    family = Family332()
    # Octahedron (6)
    assert len(family(1, 1, 1).vertices) == 6
    assert len(family(1, 1, 1).faces) == 8
    # Tetrahedron (4)
    assert len(family(1, 1, 3).vertices) == 4
    assert len(family(1, 1, 3).faces) == 4
    # Tetrahedron (4)
    assert len(family(3, 1, 1).vertices) == 4
    assert len(family(3, 1, 1).faces) == 4
    # Cube (8)
    assert len(family(3, 1, 3).vertices) == 8
    assert len(family(3, 1, 3).faces) == 6


def test_shape432():
    family = Family432()
    # Cuboctahedron (12)
    assert len(family(1, 2, 2).vertices) == 12
    assert len(family(1, 2, 2).faces) == 14
    # Octahedron (6)
    assert len(family(2, 2, 2).vertices) == 6
    assert len(family(2, 2, 2).faces) == 8
    # Cube (8)
    assert len(family(1, 2, 3).vertices) == 8
    assert len(family(1, 2, 3).faces) == 6
    # Rhombic Dodecahedron (14)
    assert len(family(2, 2, 3).vertices) == 14
    assert len(family(2, 2, 3).faces) == 12


def test_shape532():
    family = Family532()
    s = ((5**0.5)-1.0)/2.
    S = ((5**0.5)+1.0)/2.
    # Icosidodecahedron
    assert len(family(1, 2, S**2).vertices) == 30
    assert len(family(1, 2, S**2).faces) == 32
    # Icosahedron
    assert len(family(1*s*np.sqrt(5), 2, S**2).vertices) == 12
    assert len(family(1*s*np.sqrt(5), 2, S**2).faces) == 20
    # Dodecahedron
    assert len(family(1, 2, 3).vertices) == 20
    assert len(family(1, 2, 3).faces) == 12
    # Rhombic Tricontahedron
    assert len(family(1*s*np.sqrt(5), 2, 3).vertices) == 32
    assert len(family(1*s*np.sqrt(5), 2, 3).faces) == 30


if __name__ == "__main__":
    test_shape332()
    test_shape432()
    test_shape532()
