"""Provide tools for generating shapes.

Shape families are coxeter's way of providing well-defined methods for
generating classes of shapes according to some set of rules. The basic
interface is defined by the :class:`~.ShapeFamily`, which is a functor that is
called to generate a shape. The :class:`~.TabulatedShapeFamily` group of
subclasses enable the generation of shape families according to some
tabulated set of data, while other families are defined by some set of
(discrete or continuous) parameters that are used to construct a shape
analytically.
"""

from .common_families import PlatonicFamily, RegularNGonFamily
from .doi_data_repositories import family_from_doi
from .plane_shape_families import (
    Family323Plus,
    Family423,
    Family523,
    TruncatedTetrahedronFamily,
)
from .tabulated_shape_family import TabulatedGSDShapeFamily, TabulatedShapeFamily

__all__ = [
    "TabulatedShapeFamily",
    "TabulatedGSDShapeFamily",
    "Family323Plus",
    "Family423",
    "Family523",
    "TruncatedTetrahedronFamily",
    "family_from_doi",
    "RegularNGonFamily",
    "PlatonicFamily",
]
