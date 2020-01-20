import numpy as np


class Sphere(object):
    def __init__(self, radius):
        """A sphere with the given radius.

        Args:
            radius (float):
                Radius of the sphere.
        """
        self._radius = radius

    @property
    def radius(self):
        """float: Radius of the sphere."""
        return self._radius

    @property
    def volume(self):
        """float: The volume."""
        return (4/3) * np.pi * self.radius**3

    @property
    def surface_area(self):
        """float: The surface area."""
        return 4 * np.pi * self.radius**2
