import numpy as np
from .convex_polyhedron import ConvexPolyhedron


class ConvexSpheropolyhedron(object):
    def __init__(self, vertices, radius):
        """A convex spheropolyhedron.

        A convex spheropolyhedron is defined as a convex polyhedron plus a
        rounding radius. All properties of the underlying polyhedron (the
        vertices, the facets and their neighbors, etc) can be accessed directly
        through :attr:`.polyhedron`.

        Args:
            vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
                The vertices of the underlying polyhedron.
            radius (float):
                The rounding radius of the spheropolyhedron.
        """
        self._polyhedron = ConvexPolyhedron(vertices)
        self._radius = radius

    @property
    def polyhedron(self):
        """:class:`~euclid.shape_classes.convex_polyhedron.ConvexPolyhedron`:
        The underlying polyhedron."""
        return self._polyhedron

    @property
    def volume(self):
        """float: The volume."""
        # WRONG:Need to add the cylinder volumes too.
        return self.polyhedron.volume + (4/3)*np.pi*self._radius**3

    @property
    def radius(self):
        """float: The rounding radius."""
        return self._radius

    @property
    def surface_area(self):
        """float: The surface area."""
        pass
