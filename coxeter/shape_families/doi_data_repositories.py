"""This module provides tools for generating data from internally stored data
sources. These data sources are stored in the JSON format that can be parsed
by the :class:`~coxeter.shape_families.TabulatedShapeFamily`."""

from .tabulated_shape_family import TabulatedGSDShapeFamily
from .plane_shape_families import Family323Plus, Family423, Family523
from collections import defaultdict
import json
import os

_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')


def _shape_collection_factory(key):
    """Factory function used in a defaultdict for generating
    :class:`~coxeter.shape_families.ShapeFamily` instances based on a given
    key."""

    # Set of keys for which data is stored within the data/ directory. Keys may
    # either be DOIs or any other identifier for a particular dataset.
    key_to_file = {
        '10.1126/science.1220869': ['science1220869.json'],
    }

    # Set of keys that are associated with a specific ShapeFamily subclass.
    key_to_family = {
        '10.1103/PhysRevX.4.011024': [Family323Plus, Family423, Family523]
    }

    families = []
    if key in key_to_file:
        for fn in key_to_file[key]:
            with open(os.path.join(_DATA_FOLDER, fn)) as f:
                families.append(TabulatedGSDShapeFamily(json.load(f)))
    elif key in key_to_family:
        for family_type in key_to_family[key]:
            families.append(family_type())
    else:
        raise KeyError("Provided key is not associated with any known data or "
                       "shape families.")
    return families


class _shape_repo_dict(defaultdict):
    """A defaultdict that passes the key to the default_factory.

    This class is used so that data files are read the first time data is
    requested for shapes corresponding to a given DOI."""
    def __init__(self):
        self.default_factory = _shape_collection_factory

    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


_DOI_SHAPE_REPOSITORIES = _shape_repo_dict()


def family_from_doi(doi):
    """Acquire a list of :class:`~coxeter.shape_families.ShapeFamily` instances
    that were used in the paper with the given DOI.

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
        raise ValueError("coxeter does not contain any data corresponding to "
                         "the requested DOI.")
