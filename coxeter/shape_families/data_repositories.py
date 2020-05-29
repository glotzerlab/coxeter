"""This module provides tools for generating data from internally stored data
sources. These data sources are stored in the JSON format that can be parsed
by the :class:`~coxeter.shape_families.TabulatedShapeFamily`."""

from .tabulated_shape_family import TabulatedGSDShapeFamily
from collections import defaultdict
import json
import os

# Set of keys for which data is stored within the data/ directory. Keys may
# either be DOIs or any other identifier for a particular dataset.
key_to_file = {
    '10.1126/science.1220869': ['science1220869.json'],
    'platonic_solids': ['platonic.json']
}


def _shape_collection_factory(doi):
    """Factory function used in a defaultdict for generating shapes from a file
    with a DOI."""
    families = []
    try:
        files = key_to_file[doi]
    except KeyError as e:
        raise KeyError("Provided DOI is not known to coxeter.") from e

    for fn in files:
        with open(os.path.join(os.path.dirname(__file__), 'data', fn)) as f:
            families.append(TabulatedGSDShapeFamily(json.load(f)))
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


def get_by_doi(doi):
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


def get_family(family_name):
    """Acquire a :class:`~coxeter.shape_families.ShapeFamily`.

    Args:
        family_name (str):
            The identifier for the desired shape family.

    Returns:
        :class:`~coxeter.shape_families.ShapeFamily`:
            A list of shape families used in the paper.
    """
    try:
        # Always return a single family here.
        return _DOI_SHAPE_REPOSITORIES[family_name][0]
    except KeyError:
        raise ValueError("coxeter does not contain any data corresponding to "
                         "the requested family name.")
