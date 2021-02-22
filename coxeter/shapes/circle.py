# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines a circle."""

import numpy as np

from .base_classes import Shape2D


class Circle(Shape2D):
    """A circle with the given radius.

    Args:
        radius (float):
            Radius of the circle.
        center (Sequence[float]):
            The coordinates of the centroid of the circle (Default
            value: (0, 0, 0)).

    Example:
        >>> circle = coxeter.shapes.circle.Circle(radius=1.0, center=(1, 1, 1))
        >>> import numpy as np
        >>> assert np.isclose(circle.area, np.pi)
        >>> circle.centroid
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
        self.centroid = center

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {"type": "Sphere", "diameter": 2 * self.radius}

    @property
    def centroid(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self._centroid

    @centroid.setter
    def centroid(self, value):
        self._centroid = np.asarray(value)

    @property
    def radius(self):
        """float: Get the radius of the circle."""
        return self._radius

    @radius.setter
    def radius(self, value):
        if value > 0:
            self._radius = value
        else:
            raise ValueError("Radius must be greater than zero.")

    def _rescale(self, scale):
        """Multiply length scale.

        Args:
            scale (float):
                Scale factor.
        """
        self.radius *= scale

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
        """float: Get the circumference, alias for `Circle.perimeter`."""
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

        # Apply parallel axis theorem from the centroid
        i_x += area * self.centroid[0] ** 2
        i_y += area * self.centroid[1] ** 2
        i_xy += area * self.centroid[0] * self.centroid[1]
        return i_x, i_y, i_xy

    def is_inside(self, points):
        """Determine whether a set of points are contained in this circle.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                circle.

        Example:
            >>> circle = coxeter.shapes.Circle(1.0)
            >>> circle.is_inside([[0, 0, 0], [20, 20, 20]])
            array([ True, False])

        """
        points = np.atleast_2d(points) - self.centroid
        return np.logical_and(
            np.linalg.norm(points, axis=-1) <= self.radius,
            # At present circles are not orientable, so the z position must
            # match exactly.
            np.isclose(points[:, 2], 0),
        )

    @property
    def iq(self):
        """float: The isoperimetric quotient.

        This is 1 by definition for circles.
        """
        return 1

    def distance_to_surface(self, angles):  # noqa: D102
        return np.ones_like(angles) * self.radius

    @property
    def minimal_bounding_circle(self):
        """:class:`~.Circle`: Get the smallest bounding circle."""
        return Circle(self.radius, self.centroid)

    @property
    def minimal_centered_bounding_circle(self):
        """:class:`~.Circle`: Get the smallest bounding concentric circle."""
        return Circle(self.radius, self.centroid)

    @property
    def maximal_bounding_circle(self):
        """:class:`~.Circle`: Get the largest bounded circle."""
        return Circle(self.radius, self.centroid)

    @property
    def maximal_centered_bounded_circle(self):
        """:class:`~.Circle`: Get the largest bounded concentric circle."""
        return Circle(self.radius, self.centroid)

    def __repr__(self):
        return (
            f"coxeter.shapes.Circle(radius={self.radius}, "
            f"center={self.centroid.tolist()})"
        )
