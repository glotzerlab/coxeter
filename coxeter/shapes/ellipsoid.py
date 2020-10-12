"""Defines an ellipsoid."""

import numpy as np
from scipy.special import ellipeinc, ellipkinc

from .base_classes import Shape3D
from .utils import translate_inertia_tensor


class Ellipsoid(Shape3D):
    """An ellipsoid with principal axes a, b, and c.

    Args:
        a (float):
            Principal axis a of the ellipsoid (radius in the :math:`x`
            direction).
        b (float):
            Principal axis b of the ellipsoid (radius in the :math:`y`
            direction).
        c (float):
            Principal axis c of the ellipsoid (radius in the :math:`z`
            direction).
        center (Sequence[float]):
            The coordinates of the center of the ellipsoid (Default
            value: (0, 0, 0)).

    Example:
        >>> ellipsoid = coxeter.shapes.Ellipsoid(1.0, 3.0, 2.0)
        >>> ellipsoid.a
        1.0
        >>> ellipsoid.b
        3.0
        >>> ellipsoid.c
        2.0
        >>> ellipsoid.center
        array([0, 0, 0])
        >>> ellipsoid.gsd_shape_spec
        {'type': 'Ellipsoid', 'a': 1.0, 'b': 3.0, 'c': 2.0}
        >>> ellipsoid.inertia_tensor
        array([[65.34512...,  0.        ,  0.        ],
               [ 0.        , 25.13274...,  0.        ],
               [ 0.        ,  0.        , 50.26548...]])
        >>> ellipsoid.iq
        0.61161...
        >>> ellipsoid.surface_area
        48.88214...
        >>> ellipsoid.volume
        25.13274...

    """

    def __init__(self, a, b, c, center=(0, 0, 0)):
        self.a = a
        self.b = b
        self.c = c
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
        """float: Get or set the length of principal axis a (the :math:`x` radius)."""  # noqa: D402, E501
        return self._a

    @a.setter
    def a(self, a):
        if a <= 0:
            raise ValueError("a must be greater than zero.")
        self._a = a

    @property
    def b(self):
        """float: Get or set the length of principal axis b (the :math:`y` radius)."""  # noqa: D402, E501
        return self._b

    @b.setter
    def b(self, b):
        if b <= 0:
            raise ValueError("b must be greater than zero.")
        self._b = b

    @property
    def c(self):
        """float: Get or set the length of principal axis c (the :math:`z` radius)."""  # noqa: D402, E501
        return self._c

    @c.setter
    def c(self, c):
        if c <= 0:
            raise ValueError("c must be greater than zero.")
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

        Example:
            >>> ellipsoid = coxeter.shapes.Ellipsoid(1.0, 2.0, 3.0)
            >>> ellipsoid.is_inside([[0, 0, 0], [100, 1, 1]])
            array([ True, False])

        """
        points = np.atleast_2d(points) - self.center
        scale = np.array([self.a, self.b, self.c])
        return np.linalg.norm(points / scale, axis=-1) <= 1
