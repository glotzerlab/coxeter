import numpy as np
from scipy.special import ellipkinc, ellipeinc
from .utils import translate_inertia_tensor
from.base_classes import Shape3D


class Ellipsoid(Shape3D):
    def __init__(self, a, b, c, center=(0, 0, 0)):
        """An ellipsoid with principal axes a, b, and c.

        Args:
            a (float):
                Principal axis a of the ellipsoid (radius in the x direction).
            b (float):
                Principal axis b of the ellipsoid (radius in the y direction).
            c (float):
                Principal axis c of the ellipsoid (radius in the z direction).
            center (Sequence[float]):
                The coordinates of the center of the circle (Default
                value: (0, 0, 0)).
        """
        self._a = a
        self._b = b
        self._c = c
        self._center = np.asarray(center)

    @property
    def gsd_shape_spec(self):
        """dict: A complete description of this shape corresponding to the
        shape specification in the GSD file format as described
        `here <https://gsd.readthedocs.io/en/stable/shapes.html>`_."""
        return {'type': 'Ellipsoid', 'a': self._a, 'b': self._b, 'c': self._c}

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = np.asarray(value)

    @property
    def a(self):
        """float: Length of principal axis a (radius in the x direction)."""
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def b(self):
        """float: Length of principal axis b (radius in the y direction)."""
        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    @property
    def c(self):
        """float: Length of principal axis c (radius in the z direction)."""
        return self._c

    @c.setter
    def c(self, c):
        self._c = c

    @property
    def volume(self):
        """float: The volume."""
        return (4/3) * np.pi * self.a * self.b * self.c

    @property
    def surface_area(self):
        """float: The surface area."""
        # Implemented from this example:
        # https://www.johndcook.com/blog/2014/07/06/ellipsoid-surface-area/
        # It requires that a >= b >= c, so we sort the principal axes:
        c, b, a = sorted([self.a, self.b, self.c])
        if a > c:
            phi = np.arccos(c/a)
            m = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
            elliptic_part = ellipeinc(phi, m) * np.sin(phi)**2
            elliptic_part += ellipkinc(phi, m) * np.cos(phi)**2
            elliptic_part /= np.sin(phi)
        else:
            elliptic_part = 1

        result = 2 * np.pi * (c**2 + a * b * elliptic_part)
        return result

    @property
    def inertia_tensor(self):
        """float: Get the inertia tensor. Assumes constant density of 1."""
        V = self.volume
        Ixx = V/5 * (self.b**2 + self.c**2)
        Iyy = V/5 * (self.a**2 + self.c**2)
        Izz = V/5 * (self.a**2 + self.b**2)
        inertia_tensor = np.diag([Ixx, Iyy, Izz])
        return translate_inertia_tensor(
            self.center, inertia_tensor, self.volume)

    @property
    def iq(self):
        """float: The isoperimetric quotient."""
        V = self.volume
        S = self.surface_area
        return np.pi * 36 * V**2 / (S**3)

    def is_inside(self, points):
        """Determine whether a set of points are contained in this ellipsoid.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                ellipsoid.
        """
        points = np.atleast_2d(points)
        scale = np.array([self.a, self.b, self.c])
        return np.linalg.norm(points / scale, axis=-1) <= 1
