from coxeter import shape_families
from coxeter.shape_families.shape_family import Family332, Family432, Family532
import numpy as np


def test_shape_repos():
    repo = shape_families.shape_repositories['10.1126/science.1220869']
    print(repo)
    print(repo('1'))


def test_shape332():
    family = Family332()
    print(family(1, 1, 1))


def test_shape432():
    family = Family432()
    # Cuboctahedron (12)
    print(len(family(1, 2, 2).vertices))
    # Octahedron (6)
    print(len(family(2, 2, 2).vertices))
    # Cube (8)
    print(len(family(1, 2, 3).vertices))
    # Rhombic Dodecahedron (14)
    print(len(family(2, 2, 3).vertices))


def test_shape532():
    family = Family532()
    s = ((5**0.5)-1.0)/2.
    S = ((5**0.5)+1.0)/2.
    # Icosidodecahedron
    print(len(family(1, 2, S**2).vertices))
    # Icosahedron
    print(len(family(1*s*np.sqrt(5), 2, S**2).vertices))
    # Dodecahedron
    print(len(family(1, 2, 3).vertices))
    # Rhombic Tricontahedron
    print(len(family(1*s*np.sqrt(5), 2, 3).vertices))

    print(family(1, 2, S**2).vertices)
    print(family(0.1, 2, S**2).vertices)


if __name__ == "__main__":
    test_shape432()
