# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Provide tools for generating shapes.

Shape families are coxeter's way of providing well-defined methods for
generating classes of shapes according to some set of rules. The basic
interface is defined by the :class:`~.ShapeFamily`, which is a functor that is
called to generate a shape. The :class:`~.TabulatedShapeFamily` group of
subclasses enable the generation of shape families according to some
tabulated set of data, while other families are defined by some set of
(discrete or continuous) parameters that are used to construct a shape
analytically.

The `DOI_SHAPE_REPOSITORIES` variable provides convenient access to the shape families
associated with different scientific publications. This dataset is useful for
reproducing the exact set of shapes from publications.
"""

from .common import PlatonicFamily, RegularNGonFamily
from .doi_data_repositories import _doi_shape_collection_factory, _KeyedDefaultDict
from .plane_shape_families import (
    Family323Plus,
    Family423,
    Family523,
    TruncatedTetrahedronFamily,
)
from .shape_family import ShapeFamily
from .tabulated_shape_family import TabulatedGSDShapeFamily, TabulatedShapeFamily

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
    "RegularNGonFamily",
    "ShapeFamily",
    "TabulatedShapeFamily",
    "TabulatedGSDShapeFamily",
    "TruncatedTetrahedronFamily",
]
