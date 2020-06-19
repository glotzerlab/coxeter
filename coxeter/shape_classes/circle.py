"""Defines a circle."""

import numpy as np

from .base_classes import Shape2D


class Circle(Shape2D):
    """A circle with the given radius.

    Args:
        radius (float):
            Radius of the circle.
        center (Sequence[float]):
            The coordinates of the center of the circle (Default
            value: (0, 0, 0)).
    """

    def __init__(self, radius, center=(0, 0, 0)):
        self._radius = radius
        self._center = np.asarray(center)

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {"type": "Sphere", "diameter": 2 * self._radius}

    @property
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self._center

    @center.setter
    def center(self, value):
        self._center = np.asarray(value)

    @property
    def radius(self):
        """float: Get the radius of the circle."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def area(self):
        """float: Get the area of the circle."""
        return np.pi * self.radius ** 2

    @area.setter
    def area(self, value):
        self._radius = np.sqrt(value / np.pi)

    @property
    def eccentricity(self):
        """float: Get the eccentricity of the circle.

        This is 0 by definition for circles.
        """
        return 0

    @property
    def perimeter(self):
        """float: Get the perimeter of the circle."""
        return 2 * np.pi * self.radius

    @property
    def circumference(self):
        """float: Get the circumference, alias for :meth:`~.Circle.perimeter`."""
        return self.perimeter

    @property
    def planar_moments_inertia(self):
        r"""Get the planar moments of inertia.

        Moments are computed with respect to the x and y axis. In addition to
        the two planar moments, this property also provides the product of
        inertia.

        The `planar moments <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        and the
        `product <https://en.wikipedia.org/wiki/Second_moment_of_area#Product_moment_of_area>`__
        of inertia are defined by the formulas:

        .. math::
            \begin{align}
                I_x &= {\int \int}_A y^2 dA = \frac{\pi}{4} r^4 = \frac{Ar^2}{4} \\
                I_y &= {\int \int}_A x^2 dA = \frac{\pi}{4} r^4 = \frac{Ar^2}{4}\\
                I_{xy} &= {\int \int}_A xy dA = 0 \\
            \end{align}

        These formulas are given
        `here <https://en.wikipedia.org/wiki/List_of_second_moments_of_area>`__. Note
        that the product moment is zero by symmetry.
        """  # noqa: E501
        area = self.area
        i_x = i_y = area / 4 * self.radius ** 2
        i_xy = 0

        # Apply parallel axis theorem from the center
        i_x += area * self.center[0] ** 2
        i_y += area * self.center[1] ** 2
        i_xy += area * self.center[0] * self.center[1]
        return i_x, i_y, i_xy

    @property
    def polar_moment_inertia(self):
        """Get the polar moment of inertia.

        The `polar moment of inertia <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        is always calculated about an axis perpendicular to the circle (i.e. the
        normal vector) placed at the centroid of the circle.

        The polar moment is computed as the sum of the two planar moments of
        inertia.
        """  # noqa: E501
        return np.sum(self.planar_moments_inertia[:2])

    @property
    def iq(self):
        """float: The isoperimetric quotient.

        This is 1 by definition for circles.
        """
        return 1
