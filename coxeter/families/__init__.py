# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Provide tools for generating shapes.

Shape families are coxeter's way of providing well-defined methods for
generating classes of shapes according to some set of rules. The
:class:`~.TabulatedGSDShapeFamily` group of subclasses enable the generation of shape
families according to some tabulated set of data, while other families are defined by
some set of parameters that are used to construct a shape analytically. Discrete
families -- those that contain a fixed number of individual shapes -- may be iterated
over.


Example:

    >>> from coxeter.families import PlatonicFamily
    >>> for name, shape in PlatonicFamily:
    >>>    print(name, shape.num_faces)
    ... (Cube, 6)
    ... (Dodecahedron, 12)
    ... (Icosahedron, 20)
    ... (Octahedron, 8)
    ... (Tetrahedron, 4)

    >>> cube = PlatonicFamily.get_shape("Cube")
    >>> cube.num_faces
    ... 6


For continuous families of shapes, one must instead provide the required parameters.


Example:

    >>> from coxeter.families import Family423
    >>> cube = Family423.get_shape(a=1.0, c=3.0) # These values yield a Platonic cube
    >>> cube.num_faces
    ... 6

The `DOI_SHAPE_REPOSITORIES` variable provides convenient access to the shape family or
families associated with different scientific publications. This dataset is useful for
reproducing the exact set of shapes from publications.

For convenience, shapes included in the paper *10.1126/science.1220869* may be accessed
as a single family. These are indexed in a three character format consistent with the
supplementary information of that publication.


Example:

    >>> from coxeter.families import DOI_SHAPE_REPOSITORIES
    >>> DOI_SHAPE_REPOSITORIES["10.1103/PhysRevX.4.011024"]
    ... [Family323Plus, Family423, Family523]

    >>> science_family = DOI_SHAPE_REPOSITORIES["10.1126/science.1220869"][0]
    >>> for code, shape in science_family:
    >>>     print(code, shape.num_vertices)
    ... (P01, 4)  # Tetrahedron
    ... (P02, 8)  # Octahedron
    ... (P03, 6)  # Cube
    ... (P04, 20) # Icosahedron
    ... (P05, 12) # Dodecahedron
    ... (A01, 14) # Cuboctahedron
    ... ...

"""

from .common import (
    ArchimedeanFamily,
    CatalanFamily,
    JohnsonFamily,
    PlatonicFamily,
    PrismAntiprismFamily,
    PyramidDipyramidFamily,
    RegularNGonFamily,
    UniformAntiprismFamily,
    UniformDipyramidFamily,
    UniformPrismFamily,
    UniformPyramidFamily,
)
from .doi_data_repositories import _doi_shape_collection_factory, _KeyedDefaultDict
from .plane_shape_families import (
    Family323Plus,
    Family423,
    Family523,
    TruncatedTetrahedronFamily,
)
from .shape_family import ShapeFamily
from .tabulated_shape_family import TabulatedGSDShapeFamily

# Note for devs: we want this object to be documented in the public API. The Sphinx
# method for documenting a module-level constant is placing the docstring directly below
# it (as done below). However, if the variable is defined in a different module and then
# imported here, even putting it in __all__ doesn't lead to proper detection of the
# docstring by autodoc (you can get the variable documented, but it doesn't have the new
# docstring, just the docstring of its underlying type). As a result, we have to make it
# here.
DOI_SHAPE_REPOSITORIES = _KeyedDefaultDict(_doi_shape_collection_factory)
"""A mapping of DOIs to a list of shape families.

Each known DOI is associated with a list of shape families that can be used to generate
the shapes from those papers. Currently supported DOIs are:

* 10.1126/science.1220869: :cite:`Damasceno2012`
* 10.1103/PhysRevX.4.011024: :cite:`Chen2014`
* 10.1021/nn204012y: :cite:`Damasceno2012`
"""


__all__ = [
    "DOI_SHAPE_REPOSITORIES",
    "Family323Plus",
    "Family423",
    "Family523",
    "PlatonicFamily",
    "UniformAntiprismFamily",
    "UniformDipyramidFamily",
    "UniformPrismFamily",
    "UniformPyramidFamily",
    "ArchimedeanFamily",
    "CatalanFamily",
    "JohnsonFamily",
    "PrismAntiprismFamily",
    "PyramidDipyramidFamily",
    "RegularNGonFamily",
    "ShapeFamily",
    "TabulatedGSDShapeFamily",
    "TruncatedTetrahedronFamily",
]
