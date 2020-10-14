"""Define base classes for shapes.

This module defines the core API for shapes. The only real commonality is a
location in three dimensional space. However, creating an abstract parent
class is valuable here because it serves as part of the expected API in other
parts of the code base that either require or return shapes.
"""

from abc import ABC, abstractmethod

import numpy as np


class Shape(ABC):
    """An abstract representation of a shape in N dimensions."""

    @property
    @abstractmethod
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        pass

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        raise NotImplementedError

    def is_inside(self, points):
        """Determine whether points are contained in this shape.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the shape.
        """
        raise NotImplementedError

    def compute_form_factor_amplitude(self, q):
        r"""Calculate the form factor intensity.

        In solid state physics,
        `scattering theory <https://en.wikipedia.org/wiki/Scattering_theory>`__ is
        concerned with understanding the ways in which radiation is scattered from a
        sample. For a single point particle at position P, the the amplitude of a
        scattered wave observed at position Q is the product of the incoming wave
        amplitude (which follows the standard equation for a traveling wave) and the
        scattering density at P. For a crystal composed of many point particles, the
        intensity of the resulting superposition of waves can be identified as the
        Fourier transform of the total scattering density. When the particles are not
        point particles, the scattering density of the particles in their local
        coordinate systems are no longer identical. Conveniently, this component is
        separable in the Fourier transform of the total density; as a result, the
        scattering scattering intensity can be decomposed into two terms, the Fourier
        transform of the distribution of scatterers and the Fourier transform of each
        scatterer in its local coordinate system. The first term is known as the
        `static structure factor <https://en.wikipedia.org/wiki/Structure_factor>`__
        :math:`S(\vec{q})` and describes the spatial distribution of scatterers, while
        the second term is called the
        `form factor <http://gisaxs.com/index.php/Form_Factor>`__
        :math:`f(\vec{q})` and describes the local scattering profile.

        While the form factor (the scattering intensity) can be measured from
        diffraction experiments, the Fourier transform of a single particle cannot.
        However, it can be computed theoretically for a known scattering volume and can
        be inserted directly into the expression for the total scattering intensity.
        This local profile directly describes the wave emitted from a single scatterer
        (in reciprocal space) and is known as the form factor amplitude. This function
        computes the form factor amplitude for a given wavevector :math:`q`.
        """
        raise NotImplementedError(
            "The form factor calculation is not implemented for this shape."
        )

    def plot(self):
        """Plot the shape."""
        raise NotImplementedError("Plotting is not implemented for this shape.")


class Shape2D(Shape):
    """An abstract representation of a shape in 2 dimensions."""

    @property
    @abstractmethod
    def area(self):
        """float: Get or set the area of the shape."""
        pass

    @property
    @abstractmethod
    def perimeter(self):
        """float: Get the perimeter of the shape."""
        pass

    @property
    def planar_moments_inertia(self):
        r"""list[float, float, float]: Get the planar and product moments of inertia.

        Moments are computed with respect to the :math:`x` and :math:`y`
        axes. In addition to the two planar moments, this property also
        provides the product of inertia.

        The `planar moments of inertia <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        and the
        `product of inertia <https://en.wikipedia.org/wiki/Second_moment_of_area#Product_moment_of_area>`__
        define the in-plane area distribution.
        """  # noqa: E501
        raise NotImplementedError

    @property
    def polar_moment_inertia(self):
        """float: Get the polar moment of inertia.

        The `polar moment of inertia <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        is always calculated about an axis perpendicular to the shape (i.e. the
        normal vector).

        The polar moment is computed as the sum of the two planar moments of inertia.
        """  # noqa: E501
        return np.sum(self.planar_moments_inertia[:2])

    @property
    def iq(self):
        r"""float: The isoperimetric quotient.

        The `isoperimetric quotient
        <https://mathworld.wolfram.com/IsoperimetricQuotient.html>`__ is the ratio of
        the area of a shape to the area of a circle with the same perimeter. Given a
        shape of area :math:`A` and perimeter :math:`p`, the circle with the same
        perimeter has radius :math:`r_p = \frac{p}{2\pi}` and therefore has an area
        :math:`A_{circle} = \pi r_p^2 = \frac{p^2}{4\pi}`. Therefore, we have that:

        .. math::
            \begin{align}
                IQ &= \frac{A}{A_{circle}} \\
                   &= \frac{4\pi A}{p^2}
            \end{align}
        """  # noqa: E501
        return 4 * np.pi * self.area / (self.perimeter ** 2)


class Shape3D(Shape):
    """An abstract representation of a shape in 3 dimensions."""

    @property
    @abstractmethod
    def volume(self):
        """float: Get or set the volume of the shape."""
        pass

    @property
    @abstractmethod
    def surface_area(self):
        """float: Get or set the surface area of the shape."""
        pass

    def inertia_tensor(self):
        """:math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor."""
        raise NotImplementedError

    @property
    def iq(self):
        r"""float: The isoperimetric quotient.

        The `isoperimetric quotient
        <https://mathworld.wolfram.com/IsoperimetricQuotient.html>`__ is the ratio of
        the volume of a shape to the volume of a sphere with the same perimeter. Given a
        shape of volume :math:`A` and surface :math:`S`, the sphere with the same
        surface has radius :math:`r_S = \sqrt{\frac{S}{4\pi}}` and therefore has volume
        :math:`V_{sphere} = \frac{4}{3} \pi r_S^3 = \frac{S^{3/2}}{\sqrt{4\pi}}`.
        Taking the ratio of volumes gives:

        .. math::
            \begin{equation}
                \frac{V}{V_{sphere}} = \frac{6\sqrt{\pi} V}{S^{3/2}}
            \end{equation}

        To avoid inconvenient fractional exponents, the isoperimetric quotient is
        conventionally defined as the square of this quantity:

        .. math::
            \begin{align}
                IQ &= \left(\frac{V}{V_{sphere}}\right)^2 \\
                   &= \frac{36\pi V^2}{S^3}
            \end{align}
        """  # noqa: E501
        return np.pi * 36 * self.volume ** 2 / (self.surface_area ** 3)
