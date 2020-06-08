"""Define base classes for shapes.

This module defines the core API for shapes. The only real commonality is a
location in three dimensional space. However, creating an abstract parent
class is valuable here because it serves as part of the expected API in other
parts of the code base that either require or return shapes.
"""

from abc import ABC, abstractmethod


class Shape(ABC):
    """An abstract representation of a shape in N dimensions."""

    @property
    @abstractmethod
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        pass

    @center.setter
    @abstractmethod
    def center(self, value):
        pass

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {}


class Shape2D(Shape):
    """An abstract representation of a shape in 2 dimensions."""

    @property
    @abstractmethod
    def area(self):
        """float: Get or set the area of the shape."""
        pass

    @area.setter
    @abstractmethod
    def area(self, value):
        pass


class Shape3D(Shape):
    """An abstract representation of a shape in 3 dimensions."""

    @property
    @abstractmethod
    def volume(self):
        """float: Get or set the volume of the shape."""
        pass

    @volume.setter
    @abstractmethod
    def volume(self, value):
        pass

    @property
    @abstractmethod
    def surface_area(self):
        """float: Get or set the surface area of the shape."""
        pass

    @surface_area.setter
    @abstractmethod
    def surface_area(self, value):
        pass

    @abstractmethod
    def inertia_tensor(self):
        """:math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor."""
        pass
