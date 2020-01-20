import numpy as np


class Circle(object):
    def __init__(self, radius):
        """A circle with the given radius.

        Args:
            radius (float):
                Radius of the circle.
        """
        self._radius = radius

    @property
    def radius(self):
        """float: Radius of the circle."""
        return self._radius

    @property
    def area(self):
        """float: The area."""
        return np.pi * self.radius**2
