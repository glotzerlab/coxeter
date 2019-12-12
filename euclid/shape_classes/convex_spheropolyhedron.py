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
        V_poly = self.polyhedron.volume
        V_sphere = (4/3)*np.pi*self._radius**3
        V_cyl = 0

        # For every pair of faces, find the dihedral angle, divide by 2*pi to
        # get the fraction of a cylinder it includes, then multiply by the edge
        # length to get the cylinder contribution.
        for i, j, edge in self.polyhedron._get_facet_intersections():
            phi = self.polyhedron.get_dihedral(i, j)
            edge_length = np.linalg.norm(self.polyhedron.vertices[edge[0]] -
                                         self.polyhedron.vertices[edge[1]])
            V_cyl += (np.pi*self.radius**2)*(phi/(2*np.pi))*edge_length

        return V_poly + V_sphere + V_cyl

    @property
    def radius(self):
        """float: The rounding radius."""
        return self._radius

    @property
    def surface_area(self):
        """float: The surface area."""
        A_poly = self.polyhedron.surface_area
        A_sphere = 4*np.pi*self._radius**2
        A_cyl = 0

        # For every pair of faces, find the dihedral angle, divide by 2*pi to
        # get the fraction of a cylinder it includes, then multiply by the edge
        # length to get the cylinder contribution.
        for i, j, edge in self._get_facet_intersections():
            phi = self.polyhedron.get_dihedral(i, j)
            edge_vector = self.vertices[edge[0]] - self.vertices[edge[1]]
            edge_length = np.linalg.norm(edge_vector)
            A_cyl += (2*np.pi*self.radius)*(phi/2*np.pi)*edge_length

        return A_poly + A_sphere + A_cyl
