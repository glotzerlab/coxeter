"""This module provides tools for generating data from internally stored data
sources."""

from .tabulated_shape_family import TabulatedGSDShapeFamily
from collections import defaultdict
import json
import os

# Set of DOIs for which data is stored within the data/ directory.
known_files = {
    '10.1126/science.1220869': 'science1220869.json'
}


def shape_collection_factory(doi):
    """Factory function used in a defaultdict for generating shapes from a file
    with a DOI."""
    with open(os.path.join(os.path.dirname(__file__),
                           'data', known_files[doi])) as f:
        data = json.load(f)
    return TabulatedGSDShapeFamily(data)


class shape_repo_dict(defaultdict):
    """A defaultdict that passes the key to the default_factory."""
    def __init__(self):
        self.default_factory = shape_collection_factory

    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


shape_repositories = shape_repo_dict()
