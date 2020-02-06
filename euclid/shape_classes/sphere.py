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

    @radius.setter
    def radius(self, radius):
        self._radius = radius

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

    def is_inside(self, points):
        """Determine whether a set of points are contained in this sphere.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                sphere.
        """
        points = np.atleast_2d(points)
        return np.linalg.norm(points, axis=-1) <= self.radius
