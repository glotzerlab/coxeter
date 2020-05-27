from abc import ABC, abstractmethod


class Shape(ABC):
    """An abstract representation of a shape in N dimensions."""

    @property
    @abstractmethod
    def center(self):
        """float: Get or set the centroid of the shape."""
        pass

    @center.setter
    @abstractmethod
    def center(self, value):
        pass

    @property
    def gsd_shape_spec(self):
        """dict: A complete description of this shape corresponding to the
        shape specification in the GSD file format as described
        `here <https://gsd.readthedocs.io/en/stable/shapes.html>`_."""
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
        """float: Get or set the centroid of the shape."""
        pass

    @volume.setter
    @abstractmethod
    def volume(self, value):
        pass

    @property
    @abstractmethod
    def surface_area(self):
        """float: Get or set the centroid of the shape."""
        pass

    @surface_area.setter
    @abstractmethod
    def surface_area(self, value):
        pass
