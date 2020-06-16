"""Defines an ellipsoid."""

import numpy as np
from scipy.special import ellipeinc, ellipkinc

from .base_classes import Shape3D
from .utils import translate_inertia_tensor


class Ellipsoid(Shape3D):
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

    def __init__(self, a, b, c, center=(0, 0, 0)):
        self._a = a
        self._b = b
        self._c = c
        self._center = np.asarray(center)

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {"type": "Ellipsoid", "a": self._a, "b": self._b, "c": self._c}

    @property
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self._center

    @center.setter
    def center(self, value):
        self._center = np.asarray(value)

    @property
    def a(self):
        """float: Get or set the length of principal axis a (the x radius)."""  # noqa: D402, E501
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def b(self):
        """float: Get or set the length of principal axis b (the y radius)."""  # noqa: D402, E501
        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    @property
    def c(self):
        """float: Get or set the length of principal axis c (the z radius)."""  # noqa: D402, E501
        return self._c

    @c.setter
    def c(self, c):
        self._c = c

    @property
    def volume(self):
        """float: Get the volume."""
        return (4 / 3) * np.pi * self.a * self.b * self.c

    @property
    def surface_area(self):
        """float: Get the surface area."""
        # Implemented from this example:
        # https://www.johndcook.com/blog/2014/07/06/ellipsoid-surface-area/
        # It requires that a >= b >= c, so we sort the principal axes:
        c, b, a = sorted([self.a, self.b, self.c])
        if a > c:
            phi = np.arccos(c / a)
            m = (a ** 2 * (b ** 2 - c ** 2)) / (b ** 2 * (a ** 2 - c ** 2))
            elliptic_part = ellipeinc(phi, m) * np.sin(phi) ** 2
            elliptic_part += ellipkinc(phi, m) * np.cos(phi) ** 2
            elliptic_part /= np.sin(phi)
        else:
            elliptic_part = 1

        result = 2 * np.pi * (c ** 2 + a * b * elliptic_part)
        return result

    @property
    def inertia_tensor(self):
        """float: Get the inertia tensor.

        Assumes a constant density of 1.
        """
        vol = self.volume
        i_xx = vol / 5 * (self.b ** 2 + self.c ** 2)
        i_yy = vol / 5 * (self.a ** 2 + self.c ** 2)
        i_zz = vol / 5 * (self.a ** 2 + self.b ** 2)
        inertia_tensor = np.diag([i_xx, i_yy, i_zz])
        return translate_inertia_tensor(self.center, inertia_tensor, vol)

    @property
    def iq(self):
        """float: Get the isoperimetric quotient."""
        return np.pi * 36 * self.volume ** 2 / (self.surface_area ** 3)

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
        points = np.atleast_2d(points) - self.center
        scale = np.array([self.a, self.b, self.c])
        return np.linalg.norm(points / scale, axis=-1) <= 1
