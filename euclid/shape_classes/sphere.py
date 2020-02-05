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

    @property
    def inertia_tensor(self):
        """float: Get the inertia tensor. Assumes constant density of 1."""
        V = self.volume
        Ixx = V * 2/5 * self.radius**2
        return np.diag([Ixx, Ixx, Ixx])

    @property
    def iq(self):
        """float: The isoperimetric quotient. This is 1 by definition for
        spheres."""
        return 1
