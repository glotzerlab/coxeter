"""Define base classes for shapes.

This module defines the core API for shapes. The only real commonality is a
location in three dimensional space. However, creating an abstract parent
class is valuable here because it serves as part of the expected API in other
parts of the code base that either require or return shapes.
"""

from abc import ABC, abstractmethod


class Shape(ABC):
    """An abstract representation of a shape in N dimensions."""

    @property
    @abstractmethod
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        pass

    @center.setter
    @abstractmethod
    def center(self, value):
        pass

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {}

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
        `form factor <https://en.wikipedia.org/wiki/Atomic_form_factor>`__
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


class Shape2D(Shape):
    """An abstract representation of a shape in 2 dimensions."""

    @property
    @abstractmethod
    def area(self):
        """float: Get or set the area of the shape."""
        pass

    @area.setter
    @abstractmethod
    def area(self, value):
        pass


class Shape3D(Shape):
    """An abstract representation of a shape in 3 dimensions."""

    @property
    @abstractmethod
    def volume(self):
        """float: Get or set the volume of the shape."""
        pass

    @volume.setter
    @abstractmethod
    def volume(self, value):
        pass

    @property
    @abstractmethod
    def surface_area(self):
        """float: Get or set the surface area of the shape."""
        pass

    @surface_area.setter
    @abstractmethod
    def surface_area(self, value):
        pass

    @abstractmethod
    def inertia_tensor(self):
        """:math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor."""
        pass
