import json
import copy
from ..shape_getters import from_gsd_type_shapes
from .shape_family import ShapeFamily


class TabulatedShapeFamily(ShapeFamily):
    """A shape family corresponding to a tabulated set of shapes.

    The shapes must be stored in a JSON file. All available data is accessible
    via the :attr:`~.data` attribute.

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
        """Get the JSON data underlying the file."""
        return self._data

    def get_params(self, name):
        """Get the full dictionary of data stored for a given file."""
        return self._data[name]


class TabulatedGSDShapeFamily(TabulatedShapeFamily):
    """A tabulated shape family defined by a GSD shape schema.

    The shapes must be stored in a JSON file that provides a dictionary of
    name->params mappings, where the params must adhere to the
    `GSD shape spec <https://gsd.readthedocs.io/en/stable/shapes.html>`_. Each
    mapping may contain additional data, which is ignored when the class is
    called to actually produce :class:`~coxeter.shape_classes.Shape` objects.

    Args:
        filename_or_dict (str or Mapping):
            A dictionary containing valid shape definitions or a JSON file that
            can be read into such a dictionary.
    """
    def __call__(self, name):
        return from_gsd_type_shapes(self.get_params(name))
