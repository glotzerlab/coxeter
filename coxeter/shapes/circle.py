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

    Example:
        >>> circle = coxeter.shapes.circle.Circle(radius=1.0, center=(1, 1, 1))
        >>> import numpy as np
        >>> assert np.isclose(circle.area, np.pi)
        >>> circle.center
        array([1, 1, 1])
        >>> assert np.isclose(circle.circumference, 2 * np.pi)
        >>> circle.eccentricity
        0
        >>> circle.gsd_shape_spec
        {'type': 'Sphere', 'diameter': 2.0}
        >>> circle.iq
        1
        >>> assert np.isclose(circle.perimeter, 2 * np.pi)
        >>> assert np.allclose(
        ...   circle.planar_moments_inertia,
        ...   (5. / 4. * np.pi, 5. / 4. * np.pi, np.pi))
        >>> assert np.isclose(circle.polar_moment_inertia, 5. / 2. * np.pi)
        >>> circle.radius
        1.0

    """

    def __init__(self, radius, center=(0, 0, 0)):
        self.radius = radius
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
    def radius(self, r):
        if r > 0:
            self._radius = r
        else:
            raise ValueError("Radius must be greater than zero.")

    @property
    def area(self):
        """float: Get the area of the circle."""
        return np.pi * self.radius ** 2

    @area.setter
    def area(self, value):
        if value > 0:
            self.radius = np.sqrt(value / np.pi)
        else:
            raise ValueError("Area must be greater than zero.")

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

    @perimeter.setter
    def perimeter(self, value):
        if value > 0:
            self.radius = value / (2 * np.pi)
        else:
            raise ValueError("Perimeter must be greater than zero.")

    @property
    def circumference(self):
        """float: Get the circumference, alias for :meth:`~.Circle.perimeter`."""
        return self.perimeter

    @circumference.setter
    def circumference(self, value):
        self.perimeter = value

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
                I_x &= {\int \int}_A y^2 dA = \frac{\pi}{4} r^4 = \frac{Ar^2}{4} \\
                I_y &= {\int \int}_A x^2 dA = \frac{\pi}{4} r^4 = \frac{Ar^2}{4} \\
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
    def iq(self):
        """float: The isoperimetric quotient.

        This is 1 by definition for circles.
        """
        return 1
