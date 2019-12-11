from scipy.spatial import ConvexHull
import numpy as np
from .polyhedron import Polyhedron


class ConvexPolyhedron(Polyhedron):
    def __init__(self, vertices):
        """A convex polyhedron.

        A convex polyhedron is defined as the convex hull of its vertices. The
        class is a simple extension of :class:`~.Polyhedron` that builds the
        facets from the simplices of the convex hull. This class also includes
        various additional properties that can be used to characterize the
        geometric features of the polyhedron.

        Args:
            vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
                The vertices of the polyhedron.
        """
        hull = ConvexHull(vertices)
        super(ConvexPolyhedron, self).__init__(vertices, hull.simplices)
        self.merge_facets()

    @property
    def mean_curvature(self):
        R"""float: The integrated, normalized mean curvature
        :math:`R = \sum_i (1/2) L_i (\pi - \phi_i) / (4 \pi)` with edge lengths
        :math:`L_i` and dihedral angles :math:`\phi_i` (see :cite:`Irrgang2017`
        for more information).
        """
        R = 0
        for i, j, edge in self._get_facet_intersections():
            # Don't double count neighbors.
            if j < i:
                continue
            phi = self.get_dihedral(i, j)
            edge_vert = self.vertices[edge[0]] - self.vertices[edge[1]]
            length = np.linalg.norm(edge_vert)
            R += length * (np.pi - phi)
        return R / (8 * np.pi)

    @property
    def tau(self):
        R"""float: The parameter :math:`tau = \frac{S}{4\pi R^2}` defined in
        :cite:`Naumann19841` that is closely related to the Pitzer acentric
        factor. This quantity appears relevant to the third and fourth virial
        coefficient for hard polyhedron fluids.
        """
        R = self.mean_curvature
        return 4*np.pi*R*R/self.surface_area

    @property
    def asphericity(self):
        """float: The asphericity as defined in :cite:`Irrgang2017`."""
        return self.mean_curvature*self.surface_area/(3*self.volume)
