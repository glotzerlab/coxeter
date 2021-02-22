# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Generate shapes from published works.

The goal of this module is to produce shapes that were used in scientific
papers. Some of these papers use some tabulated set of shapes, while others
use some analytically defined class of shapes. The
:class:`~coxeter.families.ShapeFamily` is sufficiently flexible to handle
both. The primary API offered by this module is the DOI_SHAPE_REPOSITORIES dictionary,
which maps DOIs to a list of associated shape families.
"""

import os
from collections import defaultdict

from .plane_shape_families import (
    Family323Plus,
    Family423,
    Family523,
    TruncatedTetrahedronFamily,
)
from .tabulated_shape_family import TabulatedGSDShapeFamily

_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
# Set of DOIs for which data is stored within the _DATA_FOLDER.
_DOI_TO_FILE = {
    "10.1126/science.1220869": ["science1220869.json"],
}

# Set of DOIs that are associated with a specific ShapeFamily subclass.
_DOI_TO_FAMILY = {
    "10.1103/PhysRevX.4.011024": [Family323Plus, Family423, Family523],
    "10.1021/nn204012y": [TruncatedTetrahedronFamily],
}


def _doi_shape_collection_factory(doi):
    """Produce the default shape family for a given key.

    This function is the factory used in a custom defaultdict for generating
    :class:`~coxeter.families.ShapeFamily` instances based on a given
    key when that key has not yet been seen. The purpose of using this factory
    is to delay the loading of files until they are requested. Without it, all
    files would be loaded when coxeter is imported, which introduces noticeable
    and unnecessary lag.
    """
    families = []
    if doi in _DOI_TO_FILE:
        for fn in _DOI_TO_FILE[doi]:
            families.append(
                TabulatedGSDShapeFamily.from_json_file(os.path.join(_DATA_FOLDER, fn))
            )
    if doi in _DOI_TO_FAMILY:
        for family_type in _DOI_TO_FAMILY[doi]:
            families.append(family_type())
    if not families:
        raise KeyError(
            "Provided DOI is not associated with any known data or shape families."
        )
    return families


class _KeyedDefaultDict(defaultdict):
    """A defaultdict that passes the key to the default_factory.

    This class is used so that data files are read the first time data is
    requested for shapes corresponding to a given key.
    """

    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret
