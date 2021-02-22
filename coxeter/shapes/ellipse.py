# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines an ellipse."""

import numpy as np
from scipy.special import ellipe

from . import Circle, Shape2D


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
            The coordinates of the centroid of the ellipse (Default
            value: (0, 0, 0)).

    Example:
        >>> ellipse = coxeter.shapes.Ellipse(1.0, 2.0)
        >>> ellipse.a
        1.0
        >>> ellipse.b
        2.0
        >>> ellipse.area
        6.28318...
        >>> ellipse.centroid
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
        self.a = a
        self.b = b
        self.centroid = center

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {"type": "Ellipsoid", "a": self.a, "b": self.b}

    @property
    def centroid(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self._centroid

    @centroid.setter
    def centroid(self, value):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        self._centroid = np.asarray(value)

    @property
    def a(self):
        """float: Length of principal axis a (radius in the :math:`x` direction)."""  # noqa: D402, E501
        return self._a

    @a.setter
    def a(self, value):
        if value > 0:
            self._a = value
        else:
            raise ValueError("a must be greater than zero.")

    @property
    def b(self):
        """float: Length of principal axis b (radius in the :math:`y` direction)."""  # noqa: D402, E501
        return self._b

    @b.setter
    def b(self, value):
        if value > 0:
            self._b = value
        else:
            raise ValueError("b must be greater than zero.")

    def _rescale(self, scale):
        """Multiply length scale.

        Args:
            scale (float):
                Scale factor.
        """
        self.a *= scale
        self.b *= scale

    @property
    def area(self):
        """float: Get or set the area."""
        return np.pi * self.a * self.b

    @area.setter
    def area(self, value):
        if value > 0:
            scale = np.sqrt(value / self.area)
            self._rescale(scale)
        else:
            raise ValueError("Area must be greater than zero.")

    @property
    def eccentricity(self):
        r"""float: The eccentricity.

        An ellipse's eccentricity is defined as :math:`e = \sqrt{1 -
        \frac{b^2}{a^2}}` where :math:`b` is the length of the smaller
        semi-axis and :math:`a` is the length of the larger semi-axis.
        """
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

    @perimeter.setter
    def perimeter(self, value):
        if value > 0:
            scale = value / self.perimeter
            self._rescale(scale)
        else:
            raise ValueError("Perimeter must be greater than zero.")

    @property
    def circumference(self):
        """float: Alias for `Ellipse.perimeter`."""
        return self.perimeter

    @circumference.setter
    def circumference(self, value):
        self.perimeter = value

    @property
    def planar_moments_inertia(self):
        r"""list[float, float, float]: Get the planar and product moments of inertia.

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

        # Apply parallel axis theorem from the centroid
        i_x += area * self.centroid[0] ** 2
        i_y += area * self.centroid[1] ** 2
        i_xy += area * self.centroid[0] * self.centroid[1]
        return i_x, i_y, i_xy

    def distance_to_surface(self, angles):  # noqa: D102
        return np.sqrt(
            (self.a * self.a + self.b * self.b)
            / (
                1
                + (self.a * self.a)
                / (self.b * self.b)
                * np.sin(angles)
                * np.sin(angles)
                + (self.b * self.b)
                / (self.a * self.a)
                * np.cos(angles)
                * np.cos(angles)
            )
        )

    @property
    def iq(self):
        """float: The isoperimetric quotient."""
        return np.min([4 * np.pi * self.area / (self.perimeter ** 2), 1])

    @property
    def minimal_centered_bounding_circle(self):
        """:class:`~.Circle`: Get the smallest bounding concentric circle."""
        return Circle(max(self.a, self.b), self.centroid)

    @property
    def minimal_bounding_circle(self):
        """:class:`~.Circle`: Get the smallest bounding circle."""
        return Circle(max(self.a, self.b), self.centroid)

    @property
    def maximal_centered_bounded_circle(self):
        """:class:`~.Circle`: Get the largest bounded concentric circle."""
        return Circle(min(self.a, self.b), self.centroid)

    @property
    def maximal_bounded_circle(self):
        """:class:`~.Circle`: Get the largest bounded circle."""
        return Circle(min(self.a, self.b), self.centroid)

    def is_inside(self, points):
        """Determine whether a set of points are contained in this ellipse.

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
            >>> ellipse = coxeter.shapes.Ellipse(1.0, 2.0)
            >>> ellipse.is_inside([[0, 0, 0], [100, 1, 1]])
            array([ True, False])

        """
        points = np.atleast_2d(points) - self.centroid
        scale = np.array([self.a, self.b, np.inf])
        return np.logical_and(
            np.all(points / scale <= 1, axis=-1),
            # At present ellipsoids are not orientable, so the z position must
            # match exactly.
            np.isclose(points[:, 2], 0),
        )

    def __repr__(self):
        return (
            f"coxeter.shapes.Ellipse(a={self.a}, b={self.b}, "
            f"center={self.centroid.tolist()})"
        )
