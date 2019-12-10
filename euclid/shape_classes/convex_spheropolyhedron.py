import numpy as np
from .convex_polyhedron import ConvexPolyhedron


class ConvexSpheropolyhedron(object):
    def __init__(self, vertices, radius):
        """A convex spheropolyhedron.

        A convex spheropolyhedron is defined as a convex polyhedron plus a
        rounding radius.

        Args:
            vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
                The vertices of the underlying polyhedron.
            radius (float):
                The rounding radius of the spheropolyhedron.
        """
        self._polyhedron = ConvexPolyhedron(vertices)
        self._radius = radius

    @property
    def volume(self):
        """float: The volume."""
        return self._polyhedron.volume + (4/3)*np.pi*self._radius**3

    @property
    def radius(self):
        """float: The rounding radius."""
        return self._radius
