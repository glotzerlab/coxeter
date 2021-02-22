# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
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
from .shape_family import ShapeFamily


class TabulatedShapeFamily(ShapeFamily):
    """A shape family corresponding to a tabulated set of shapes.

    Data can either be read from a file or provided in the form of a
    dictionary. If a filename is provided, it must be a JSON file that can be
    parsed into an appropriately formatted dictionary, namely a set of
    key-value pairs such that the call operator of this class can generate
    a :class:`~coxeter.shapes.Shape` from the dictionary. The raw parsed
    JSON is accessible via the :attr:`~.data` attribute. Subclasses of this
    class implement the call operator to define exactly how the dictionary
    values are converted to a shape definition.
    """

    @classmethod
    def from_mapping(cls, mapping, classname=None, docstring=None):
        """Generate a subclass for a dataset from a mapping.

        Notably, this method is a _class_ factory: rather than generating a new
        instance, this method actually generates a new subclass. This design is
        consistent with the usage :class:`~.ShapeFamily` subclasses by direct
        interaction with the class (without instantiation).

        Args:
            mapping (Mapping):
                A dict-like object containing valid shape definitions.
            classname (str, optional):
                The name of the new class to use if provided (Default value: None).
            docstring (str, optional):
                The docstring to apply to the class.

        Returns:
            A subclass of this one associated with the the provided data.
        """

        class NewTabulatedShapeFamily(cls):
            # Make a full copy to avoid modifying an input dictionary.
            data = copy.deepcopy(mapping)

        # TODO: Consider dynamically setting attributes like __name__.

        if classname is not None:
            NewTabulatedShapeFamily.__name__ = classname
        if docstring is not None:
            NewTabulatedShapeFamily.__doc__ = docstring
        return NewTabulatedShapeFamily

    @classmethod
    def from_json_file(cls, filename, *args, **kwargs):
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

        Returns:
            A subclass of this one associated with the the provided data.
        """
        with open(filename) as f:
            return cls.from_mapping(json.load(f), *args, **kwargs)


class TabulatedGSDShapeFamily(TabulatedShapeFamily):
    """A tabulated shape family defined by a GSD shape schema.

    The values of the dictionary used to construct this class must adhere to
    the :ref:`GSD shape spec <shapes>`. Each mapping may contain additional
    data, which is ignored when the class is called to actually produce
    :class:`~coxeter.shapes.Shape` objects.

    Args:
        filename_or_dict (str or Mapping):
            A dictionary containing valid shape definitions or a JSON file that
            can be read into such a dictionary.
    """

    @classmethod
    def get_shape(cls, name):
        """Use the class's data to produce a shape for the given name.

        Args:
            name (str):
                The key of the desired shape in the data dict.

        Returns:
            :class:`~coxeter.shapes.Shape`: The requested shape.
        """
        return from_gsd_type_shapes(cls.data[name])
