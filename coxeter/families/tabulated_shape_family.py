# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Define tabulated shape families.

Tabulated shape families are defined by a dictionary of data that can be
converted into a shape by some well-defined schema. One example is the `GSD
shape schema <gsd:shapes>`, which defines how to translates dictionaries with
the appropriate formatting into a shape. These shape families may be
constructed from a JSON file that can be read into a dictionary with the
appropriate formatting.
"""

import json

from ..shape_getters import from_gsd_type_shapes
from .shape_family import ShapeFamily


class TabulatedGSDShapeFamily(ShapeFamily):
    """A tabulated shape family defined by a GSD shape schema.

    The values of the dictionary used to construct this class must adhere to
    the :ref:`GSD shape spec <gsd:shapes>`. Each mapping may contain additional
    data, which is ignored when the class is called to actually produce
    :class:`~coxeter.shapes.Shape` objects.

    Args:
        filename_or_dict (str or Mapping):
            A dictionary containing valid shape definitions or a JSON file that
            can be read into such a dictionary.
    """

    def __init__(self, data):
        self._data = data
        self._shape_names = [*data.keys()]
        self._shape_specs = [*data.values()]

    @property
    def data(self):
        """Raw JSON data for the class. Should not be used by users."""
        return self._data

    @property
    def names(self):
        """A list of names for the shapes in the family, in alphabetical order."""
        return self._shape_names

    @classmethod
    def from_json_file(cls, filename, classname=None, docstring=None):
        r"""Generate a subclass for a dataset from a JSON file.

        This method simply loads the JSON file into a dictionary and calls
        :meth:`~.from_mapping`, see that docstring for more information.

        Args:
            filename (str):
                A JSON file containing valid shape definitions.
            \*args:
                Passed on to :meth:`~.from_mapping`.
            \*\*kwargs:
                Passed on to :meth:`~.from_mapping`.

        Returns
        -------
            A subclass of this one associated with the the provided data.
        """
        with open(filename) as f:
            NewTabulatedShapeFamily = cls(data=json.load(f))  # noqa:  N806
        if classname is not None:
            NewTabulatedShapeFamily.__name__ = classname
        if docstring is not None:
            NewTabulatedShapeFamily.__doc__ = docstring
        return NewTabulatedShapeFamily

    def get_shape(self, name):
        """Use the class's data to produce a shape for the given name.

        Args:
            name (str):
                The key of the desired shape in the data dict.

        Returns
        -------
            :class:`~coxeter.shapes.Shape`: The requested shape.
        """
        return from_gsd_type_shapes(self.data[name])

    def __iter__(self):
        """Return an iterator that yields key-value pairs as tuples.

        Yields
        ------
            Iterator[Tuple[str, any]]: An iterator of key-value pairs.
        """
        for key in self.names:
            yield (key, self.get_shape(key))
