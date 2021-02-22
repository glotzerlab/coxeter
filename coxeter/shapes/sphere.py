# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines an circle."""

import numpy as np

from .base_classes import Shape3D
from .utils import translate_inertia_tensor


class Sphere(Shape3D):
    """A sphere with the given radius.

    Args:
        radius (float):
            Radius of the sphere.
        center (Sequence[float]):
            The coordinates of the centroid of the sphere (Default
            value: (0, 0, 0)).

    Example:
        >>> sphere = coxeter.shapes.Sphere(1.0)
        >>> assert np.isclose(sphere.radius, 1.0)
        >>> assert np.allclose(sphere.centroid, [0., 0., 0.])
        >>> sphere.gsd_shape_spec
        {'type': 'Sphere', 'diameter': 2.0}
        >>> assert np.allclose(
        ...   np.diag(sphere.inertia_tensor),
        ...   8. / 15. * np.pi)
        >>> sphere.iq
        1
        >>> sphere.surface_area
        12.56637...
        >>> sphere.volume
        4.18879...

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
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        self._centroid = np.asarray(value)

    @property
    def radius(self):
        """float: Get or set the radius of the sphere."""
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
    def volume(self):
        """float: Get the volume of the sphere."""
        return (4 / 3) * np.pi * self.radius ** 3

    @volume.setter
    def volume(self, value):
        if value > 0:
            self.radius = (3 * value / (4 * np.pi)) ** (1 / 3)
        else:
            raise ValueError("Volume must be greater than zero.")

    @property
    def surface_area(self):
        """float: Get the surface area."""
        return 4 * np.pi * self.radius ** 2

    @surface_area.setter
    def surface_area(self, value):
        if value > 0:
            self.radius = np.sqrt(value / (4 * np.pi))
        else:
            raise ValueError("Surface area must be greater than zero.")

    @property
    def inertia_tensor(self):
        """:math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor.

        Assumes a constant density of 1.
        """
        vol = self.volume
        i_xx = vol * 2 / 5 * self.radius ** 2
        inertia_tensor = np.diag([i_xx, i_xx, i_xx])
        return translate_inertia_tensor(self.centroid, inertia_tensor, vol)

    @property
    def iq(self):
        """float: The isoperimetric quotient.

        This is 1 by definition for spheres.
        """
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

        Example:
            >>> sphere = coxeter.shapes.Sphere(1.0)
            >>> sphere.is_inside([[0, 0, 0], [20, 20, 20]])
            array([ True, False])

        """
        points = np.atleast_2d(points) - self.centroid
        return np.linalg.norm(points, axis=-1) <= self.radius

    def compute_form_factor_amplitude(self, q, density=1.0):  # noqa: D102
        # Use the parent docstring.

        # The formula for a the form factor of a sphere may be found here:
        # http://gisaxs.com/index.php/Form_Factor:Sphere
        # (among other sources).
        q = np.atleast_2d(q)
        form_factor = np.empty(q.shape[0], dtype=np.complex128)
        q_sqs = np.sum(q * q, axis=-1)
        zero_q = np.isclose(q_sqs, 0)
        form_factor[zero_q] = self.volume
        # Two notes are in order for the formula below:
        #   - np.sinc(x) gives sin(pi*x)/(pi*x)
        #   - The expression below is the familiar expression for the form factor of a
        #     sphere, but it must be shifted to the sphere's position.
        qr = np.sqrt(q_sqs[~zero_q]) * self.radius
        form_factor[~zero_q] = (
            4 * np.pi * self.radius * (np.sinc(qr / np.pi) - np.cos(qr))
        ) / q_sqs[~zero_q]

        # Shift the form factor to the particle's position and scale by density.
        form_factor *= density * np.exp(-1j * np.dot(q, self.centroid))
        return form_factor

    @property
    def minimal_centered_bounding_sphere(self):
        """:class:`~.Sphere`: Get the smallest bounding concentric sphere."""
        return Sphere(self.radius, self.centroid)

    @property
    def minimal_bounding_sphere(self):
        """:class:`~.Sphere`: Get the smallest bounding sphere."""
        return Sphere(self.radius, self.centroid)

    @property
    def maximal_centered_bounded_sphere(self):
        """:class:`~.Sphere`: Get the largest bounded concentric sphere."""
        return Sphere(self.radius, self.centroid)

    @property
    def maximal_bounded_sphere(self):
        """:class:`~.Sphere`: Get the largest bounded sphere."""
        return Sphere(self.radius, self.centroid)

    def __repr__(self):
        return (
            f"coxeter.shapes.Sphere(radius={self.radius}, "
            f"center={self.centroid.tolist()})"
        )
