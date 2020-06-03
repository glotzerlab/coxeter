"""Tools for generating data from internally stored data sources.

The goal of this module is to produce shapes that were used in scientific
papers. Some of these papers use some tabulated set of shapes, while others
use some analytically defined class of shapes. The
:class:`~coxeter.shape_families.ShapeFamily` is sufficiently flexible to handle
both, so this module provides utilities that generate shape families when
given a particular DOI.
"""

import json
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


def _doi_shape_collection_factory(doi):
    """Produce the default shape family for a given key.

    This function is the factory used in a defaultdict for generating
    :class:`~coxeter.shape_families.ShapeFamily` instances based on a given
    key when that key has not yet been seen. The purpose of using this factory
    is to delay the loading of files until they are requested. Without it, all
    files would be loaded when coxeter is imported, which introduces noticeable
    and unnecessary lag.
    """
    # Set of DOIs for which data is stored within the data/ directory.
    doi_to_file = {
        "10.1126/science.1220869": ["science1220869.json"],
    }

    # Set of DOIs that are associated with a specific ShapeFamily subclass.
    doi_to_family = {
        "10.1103/PhysRevX.4.011024": [Family323Plus, Family423, Family523],
        "10.1021/nn204012y": [TruncatedTetrahedronFamily],
    }

    families = []
    if doi in doi_to_file:
        for fn in doi_to_file[doi]:
            with open(os.path.join(_DATA_FOLDER, fn)) as f:
                families.append(TabulatedGSDShapeFamily(json.load(f)))
    elif doi in doi_to_family:
        for family_type in doi_to_family[doi]:
            families.append(family_type())
    else:
        raise KeyError(
            "Provided DOI is not associated with any known data or " "shape families."
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


_DOI_SHAPE_REPOSITORIES = _KeyedDefaultDict(_doi_shape_collection_factory)


def family_from_doi(doi):
    """Acquire a list of shape families.

    This function produces :class:`~coxeter.shape_families.ShapeFamily`
    instances corresponding to sets of shapes that were used in the paper with
    the given DOI.

    Args:
        doi (str):
            The DOI of a paper whose shape data to find.

    Returns:
        list[:class:`~coxeter.shape_families.ShapeFamily`]:
            A list of shape families used in the paper.
    """
    try:
        return _DOI_SHAPE_REPOSITORIES[doi]
    except KeyError:
        raise ValueError(
            "coxeter does not contain any data corresponding to " "the requested DOI."
        )
