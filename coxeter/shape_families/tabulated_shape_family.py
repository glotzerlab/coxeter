"""Define tabulated shape families.

Tabulated shape families are defined by a dictionary of data that can be
converted into a shape by some well-defined schema. One example is the
`GSD shape schema <shapes>`, which defines how to translates dictionaries with
the appropriate formatting into a shape. These shape families may be
constructed from a JSON file that can be read into a dictionary with the
appropriate formatting.
"""

import copy
import json

from ..shape_getters import from_gsd_type_shapes
from .shape_family import _ShapeFamily


class TabulatedShapeFamily(_ShapeFamily):
    """A shape family corresponding to a tabulated set of shapes.

    Data can either be read from a file or provided in the form of a
    dictionary. If a filename is provided, it must be a JSON file that can be
    parsed into an appropriately formatted dictionary, namely a set of
    key-value pairs such that the call operator of this class can generate
    a :class:`~coxeter.shape_classes.Shape` from the dictionary. The raw parsed
    JSON is accessible via the :attr:`~.data` attribute. Subclasses of this
    class implement the call operator to define exactly how the dictionary
    values are converted to a shape definition.

    Args:
        filename_or_dict (str or Mapping):
            A dictionary containing valid shape definitions or a JSON file that
            can be read into such a dictionary.
    """

    def __init__(self, filename_or_dict):
        if type(filename_or_dict) is str:
            with open(filename_or_dict) as f:
                self._data = json.load(f)
        else:
            # Make a full copy to avoid modifying an input dictionary.
            self._data = copy.deepcopy(filename_or_dict)

    @property
    def data(self):
        """dict[dict]: Get the JSON data underlying the file."""
        return self._data

    def get_params(self, name):
        """Get the full dictionary of data stored for a given file.

        Returns:
            dict: The dictionary of data for a given key in the :attr:`~.data`.
        """  # noqa: D401, E501
        return self._data[name]


class TabulatedGSDShapeFamily(TabulatedShapeFamily):
    """A tabulated shape family defined by a GSD shape schema.

    The values of the dictionary used to construct this class must adhere to
    the :ref:`GSD shape spec <shapes>`. Each mapping may contain additional
    data, which is ignored when the class is called to actually produce
    :class:`~coxeter.shape_classes.Shape` objects.

    Args:
        filename_or_dict (str or Mapping):
            A dictionary containing valid shape definitions or a JSON file that
            can be read into such a dictionary.
    """

    def __call__(self, name):
        """Use the class's data to produce a shape for the given name.

        Args:
            name (str):
                The key of the desired shape in the data dict.

        Returns:
            :class:`~coxeter.shape_classes.Shape`: The requested shape.
        """
        return from_gsd_type_shapes(self.get_params(name))
