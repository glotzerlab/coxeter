"""Defines an ellipse."""

import numpy as np
from scipy.special import ellipe

from .base_classes import Shape2D


class Ellipse(Shape2D):
    """An ellipse with principal axes a and b.

    Args:
        a (float):
            Principal axis a of the ellipse (radius in the :math:`x`
            direction).
        b (float):
            Principal axis b of the ellipse (radius in the :math:`y`
            direction).
        center (Sequence[float]):
            The coordinates of the center of the ellipse (Default
            value: (0, 0, 0)).

    Example:
        >>> ellipse = coxeter.shapes.Ellipse(1.0, 2.0)
        >>> ellipse.a
        1.0
        >>> ellipse.b
        2.0
        >>> ellipse.area
        6.28318...
        >>> ellipse.center
        array([0, 0, 0])
        >>> ellipse.circumference
        9.68844...
        >>> ellipse.eccentricity
        0.86602...
        >>> ellipse.gsd_shape_spec
        {'type': 'Ellipsoid', 'a': 1.0, 'b': 2.0}
        >>> ellipse.iq
        0.84116...
        >>> ellipse.perimeter
        9.68844...
        >>> ellipse.polar_moment_inertia
        7.85398...

    """

    def __init__(self, a, b, center=(0, 0, 0)):
        if a <= 0:
            raise ValueError("a must be greater than zero.")
        if b <= 0:
            raise ValueError("b must be greater than zero.")
        self._a = a
        self._b = b
        self._center = np.asarray(center)

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {"type": "Ellipsoid", "a": self._a, "b": self._b}

    @property
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self._center

    @center.setter
    def center(self, value):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        self._center = np.asarray(value)

    @property
    def a(self):
        """float: Length of principal axis a (radius in the :math:`x` direction)."""  # noqa: D402, E501
        return self._a

    @a.setter
    def a(self, a):
        if a <= 0:
            raise ValueError("a must be greater than zero.")
        self._a = a

    @property
    def b(self):
        """float: Length of principal axis b (radius in the :math:`y` direction)."""  # noqa: D402, E501
        return self._b

    @b.setter
    def b(self, b):
        if b <= 0:
            raise ValueError("a must be greater than zero.")
        self._b = b

    @property
    def area(self):
        """float: The area."""
        return np.pi * self.a * self.b

    @property
    def eccentricity(self):
        """float: The eccentricity."""
        # Requires that a >= b, so we sort the principal axes:
        b, a = sorted([self.a, self.b])
        e = np.sqrt(1 - b ** 2 / a ** 2)
        return e

    @property
    def perimeter(self):
        """float: The perimeter."""
        # Implemented from this example:
        # https://scipython.com/book/chapter-8-scipy/examples/the-circumference-of-an-ellipse/
        # It requires that a >= b, so we sort the principal axes:
        b, a = sorted([self.a, self.b])
        result = 4 * a * ellipe(self.eccentricity ** 2)
        return result

    @property
    def circumference(self):
        """float: Alias for :meth:`~.Ellipse.perimeter`."""
        return self.perimeter

    @property
    def planar_moments_inertia(self):
        r"""Get the planar moments of inertia.

        Moments are computed with respect to the :math:`x` and :math:`y`
        axes. In addition to the two planar moments, this property also
        provides the product of inertia.

        The `planar moments <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        and the
        `product <https://en.wikipedia.org/wiki/Second_moment_of_area#Product_moment_of_area>`__
        of inertia are defined by the formulas:

        .. math::
            \begin{align}
                I_x &= {\int \int}_A y^2 dA = \frac{\pi}{4} a b^3 = \frac{Ab^2}{4} \\
                I_y &= {\int \int}_A x^2 dA = \frac{\pi}{4} a^3 b = \frac{Aa^2}{4} \\
                I_{xy} &= {\int \int}_A xy dA = 0 \\
            \end{align}

        These formulas are given
        `here <https://en.wikipedia.org/wiki/List_of_second_moments_of_area>`__. Note
        that the product moment is zero by symmetry.
        """  # noqa: E501
        area = self.area
        i_x = area / 4 * self.b ** 2
        i_y = area / 4 * self.a ** 2
        i_xy = 0

        # Apply parallel axis theorem from the center
        i_x += area * self.center[0] ** 2
        i_y += area * self.center[1] ** 2
        i_xy += area * self.center[0] * self.center[1]
        return i_x, i_y, i_xy

    @property
    def iq(self):
        """float: The isoperimetric quotient."""
        return np.min([4 * np.pi * self.area / (self.perimeter ** 2), 1])
