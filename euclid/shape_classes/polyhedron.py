from scipy.spatial import ConvexHull
import numpy as np
from .polygon import Polygon
from scipy.sparse.csgraph import connected_components


def _facet_to_edges(facet, reverse=False):
    """Convert a facet (a sequence of vertices) into a sequence of edges
    (tuples)."""
    shift = 1 if reverse else -1
    return list(zip(*np.stack((facet, np.roll(facet, shift)))))


class Polyhedron(object):
    def __init__(self, vertices, facets):
        """A general polyhedron.

        If only vertices are passed in, the result is a convex polyhedron
        defined by these vertices. If facets are provided, the resulting
        polyhedron may be nonconvex.

        The polyhedron is assumed to be of unit mass and constant density.

        """
        self._vertices = np.array(vertices, dtype=np.float64)
        self._facets = [facet for facet in facets]
        self.sort_facets()

    def _find_equations(self):
        """Find equations of facets."""
        self._equations = np.empty((len(self.facets), 4))
        for i, facet in enumerate(self.facets):
            # The direction of the normal is selected such that vertices that
            # are already ordered counterclockwise will point outward.
            normal = np.cross(
                self.vertices[facet[2]] - self.vertices[facet[1]],
                self.vertices[facet[0]] - self.vertices[facet[1]])
            normal /= np.linalg.norm(normal)
            self._equations[i, :3] = normal
            self._equations[i, 3] = normal.dot(self.vertices[facet[0]])

    def _find_neighbors(self):
        """Find neighbors of facets (assumes facets are ordered)."""
        # First enumerate all edges of each neighbor. We include both
        # directions of the edges for comparison.
        facet_edges = [set(_facet_to_edges(f) +
                           _facet_to_edges(f, True)) for f in self.facets]

        # Find any facets that share neighbors.
        self._neighbors = [[] for _ in range(self.num_facets)]
        for i in range(self.num_facets):
            for j in range(i+1, self.num_facets):
                if len(facet_edges[i].intersection(facet_edges[j])) > 0:
                    self._neighbors[i].append(j)
                    self._neighbors[j].append(i)
            self._neighbors[i] = np.array(self._neighbors[i])

    def merge_facets(self, atol=1e-8, rtol=1e-5):
        """Merge facets of a polyhedron.

        For convex polyhedra, facets will automatically be merged to an
        appropriate degree.  However, the merging of facets must be based on a
        tolerance (we may need to provide two such parameters depending on how
        we perform the merge), so we need to expose this method to allow the
        user to redo the merge with a different tolerance."""
        # Construct a graph where connectivity indicates merging, then identify
        # connected components to merge.
        merge_graph = np.zeros((self.num_facets, self.num_facets))
        for i in range(self.num_facets):
            for j in self._neighbors[i]:
                eq1, eq2 = self._equations[[i, j]]
                if np.allclose(eq1, eq2, atol=atol, rtol=rtol) or \
                        np.allclose(eq1, -eq2, atol=atol, rtol=rtol):
                    merge_graph[i, j] = 1

        _, labels = connected_components(merge_graph, directed=False,
                                         return_labels=True)
        new_facets = [set() for _ in range(len(np.unique(labels)))]
        for i, facet in enumerate(self.facets):
            new_facets[labels[i]].update(facet)

        self._facets = [np.asarray(list(f)) for f in new_facets]
        self.sort_facets()

    @property
    def neighbors(self):
        """The neighbors of each facet. Facets are defined to be neighbors if
        they share an edge."""
        return self._neighbors

    @property
    def normals(self):
        """The normal vectors to each facet."""
        return self._equations[:, :3]

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def num_facets(self):
        return len(self.facets)

    def sort_facets(self):
        """Ensure that all facets are ordered such that the normals are
        counterclockwise and point outwards.

        This algorithm proceeds in four steps. First, it ensures that each
        facet is ordered in either clockwise or counterclockwise order such
        that edges can be found from the sequence of the vertices in each
        facet. Next, it calls the neighbor finding routine to establish with
        facets are neighbors. Then, it performs a breadth-first search,
        reorienting facets to match the orientation of the first facet.
        Finally, it computes the signed volume to determine whether or not all
        the normals need to be flipped.
        """
        # We first ensure that facet vertices are sequentially ordered by
        # constructing a Polygon and updating the facet (in place), which
        # enables finding neighbors.
        for facet in self.facets:
            facet[:] = np.asarray([
                np.where(np.all(self.vertices == vertex, axis=1))[0][0]
                for vertex in Polygon(self.vertices[facet],
                                      planar_tolerance=1e-4).vertices
            ])
        self._find_neighbors()

        # The initial facet sets the order of the others.
        visited_facets = []
        remaining_facets = [0]
        while len(remaining_facets):
            current_facet = remaining_facets[-1]
            visited_facets.append(current_facet)
            remaining_facets.pop()

            # Search for common edges between pairs of facets, then check the
            # ordering of the edge to determine relative facet orientation.
            current_edges = _facet_to_edges(self.facets[current_facet])
            for neighbor in self._neighbors[current_facet]:
                if neighbor in visited_facets:
                    continue
                remaining_facets.append(neighbor)

                # Two facets can only share a single edge (otherwise they would
                # be coplanar), so we can break as soon as we find the
                # neighbor. Flip the neighbor if the edges are identical.
                for edge in _facet_to_edges(self.facets[neighbor]):
                    if edge in current_edges:
                        self._facets[neighbor] = self._facets[neighbor][::-1]
                        break
                    elif edge[::-1] in current_edges:
                        break
                visited_facets.append(neighbor)

        # Now compute the signed area and flip all the orderings if the area is
        # negative.
        self._find_equations()
        if self.volume < 0:
            for i in range(len(self.facets)):
                self._facets[i] = self._facets[i][::-1]
                self._equations[i] *= -1

    @property
    def vertices(self):
        """Get the vertices of the polyhedron."""
        return self._vertices

    @property
    def facets(self):
        """Get the polyhedron's facets."""
        return self._facets

    @property
    def volume(self):
        """Get or set the polyhedron's volume (setting rescales vertices)."""
        ds = self._equations[:, 3]
        return np.sum(ds*self.get_facet_area())/3

    @volume.setter
    def volume(self, new_volume):
        scale_factor = (new_volume/self.volume)**(1/3)
        self._vertices *= scale_factor
        self._equations[:, 3] *= scale_factor

    def get_facet_area(self, facets=None):
        """Get the total surface area of a set of facets.

        Args:
            facets (int, sequence, or None):
                The index of a facet or a set of facet indices for which to
                find the area. If None, finds the area of all facets (Default
                value: None).

        Returns:
            list: The area of each facet.
        """
        if facets is None:
            facets = range(len(self.facets))

        areas = np.empty(len(facets))
        for i, facet_index in enumerate(facets):
            facet = self.facets[facet_index]
            poly = Polygon(self.vertices[facet], planar_tolerance=1e-4)
            areas[i] = poly.area

        return areas

    @property
    def surface_area(self):
        """The surface area."""
        return np.sum(self.get_facet_area())

    def triangulation(self):
        """Generate a triangulation of the surface of the polyhedron.

        This algorithm constructs Polygons from each of the facets and then
        triangulates each of these to provide a total triangulation.
        """
        for facet in self.facets:
            poly = Polygon(self.vertices[facet], planar_tolerance=1e-4)
            yield from poly.triangulation()

    @property
    def inertia_tensor(self):
        """The inertia tensor computed about the center of mass.

        Computed using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """
        simplices = np.array(list(self.triangulation())) - self.center

        volumes = np.abs(np.linalg.det(simplices)/6)

        def triangle_integrate(f):
            R"""Compute integrals of the form :math:`\int\int\int f(x, y, z)
            dx dy dz` over a set of triangles. Omits a factor of v/20."""
            fv1 = f(simplices[:, 0, :])
            fv2 = f(simplices[:, 1, :])
            fv3 = f(simplices[:, 2, :])
            fvsum = (f(simplices[:, 0, :] +
                       simplices[:, 1, :] +
                       simplices[:, 2, :]))
            return np.sum((volumes/20)*(fv1 + fv2 + fv3 + fvsum))

        Ixx = triangle_integrate(lambda t: t[:, 1]**2 + t[:, 2]**2)
        Ixy = triangle_integrate(lambda t: -t[:, 0]*t[:, 1])
        Ixz = triangle_integrate(lambda t: -t[:, 0]*t[:, 2])
        Iyy = triangle_integrate(lambda t: t[:, 0]**2 + t[:, 2]**2)
        Iyz = triangle_integrate(lambda t: -t[:, 1]*t[:, 2])
        Izz = triangle_integrate(lambda t: t[:, 0]**2 + t[:, 1]**2)

        return np.array([[Ixx, Ixy, Ixz],
                         [Ixy,   Iyy, Iyz],
                         [Ixz,   Iyz,   Izz]])

    @property
    def center(self):
        """Get or set the polyhedron's centroid (setting rescales vertices)."""
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self._vertices += (np.asarray(value) - self.center)
        self._find_equations()

    @property
    def circumsphere_radius(self):
        """Get or set the polyhedron's circumsphere radius (setting rescales
        vertices)."""
        return np.linalg.norm(self.vertices, axis=1).max()

    @circumsphere_radius.setter
    def circumsphere_radius(self, new_radius):
        scale_factor = new_radius/self.circumsphere_radius
        self._vertices *= scale_factor
        self._equations[:, 3] *= scale_factor

    @property
    def iq(self):
        """The isoperimetric quotient."""
        V = self.volume
        S = self.surface_area
        return np.pi * 36 * V * V / (S * S * S)

    def get_dihedral(self, a, b):
        """Get the dihedral angle between a pair of facets.

        The dihedral is computed from the dot product of the facet normals.

        Args:
            a (int):
                The index of the first facet.
            b (int):
                The index of the secondfacet.

        Returns:
            float: The dihedral angle in radians.
        """
        if b not in self.neighbors[a]:
            raise ValueError("The two facets are not neighbors.")
        n1, n2 = self._equations[[a, b], :3]
        return np.arccos(np.dot(-n1, n2))


class ConvexPolyhedron(Polyhedron):
    def __init__(self, vertices, facets=None):
        """A general polyhedron.

        If only vertices are passed in, the result is a convex polyhedron
        defined by these vertices. If facets are provided, the resulting
        polyhedron may be nonconvex.

        The polyhedron is assumed to be of unit mass and constant density.

        """
        hull = ConvexHull(vertices)
        super(ConvexPolyhedron, self).__init__(vertices, hull.simplices)
        self.merge_facets()

    @property
    def mean_curvature(self):
        R"""The mean curvature of the polyhedron.

        We follow the convention defined in :cite:`Irrgang2017`.  Mean
        curvature R for a polyhedron is determined from the edge lengths
        :math:`L_i` and dihedral angles :math:`\phi_i` and is given by
        :math:`\sum_i (1/2) L_i (\pi - \phi_i) / (4 \pi)`.
        """
        R = 0
        for i in range(self.num_facets):
            for j in self.neighbors[i]:
                # Don't double count neighbors.
                if j < i:
                    continue
                phi = self.get_dihedral(i, j)
                # Include both directions for one facet to get a unique edge.
                f1 = set(_facet_to_edges(self.facets[i]))
                f2 = set(_facet_to_edges(self.facets[j]) +
                         _facet_to_edges(self.facets[j], reverse=True))
                edge = list(f1.intersection(f2))
                assert len(edge) == 1
                edge = edge[0]
                edge_vert = self.vertices[edge[0]] - self.vertices[edge[1]]
                length = np.linalg.norm(edge_vert)
                R += length * (np.pi - phi)
        return R / (8 * np.pi)

    @property
    def tau(self):
        R"""The :math:`tau` parameter of aspheriity.

        The quantity :math:`tau = \frac{S}{4\pi R^2}` is defined in
        :cite:`Naumann19841` that is closely related to the Pitzer acentric
        factor. This quantity appears relevant to the third and fourth virial
        coefficient for hard polyhedron fluids.
        """
        R = self.mean_curvature
        return 4*np.pi*R*R/self.surface_area

    @property
    def asphericity(self):
        """The asphericity as defined in :cite:`Irrgang2017`."""
        return self.mean_curvature*self.surface_area/(3*self.volume)

    # WARNING: The insphere radius calculation provided here is only valid for
    # regular polyhedra. We should be careful indicating what we're providing
    # here.
    @property
    def insphere_radius(self):
        """Get or set the polyhedron's insphere radius (setting rescales
        vertices)."""
        return np.abs(self._equations[:, 3]).max()

    @insphere_radius.setter
    def insphere_radius(self, new_radius):
        scale_factor = new_radius/self.insphere_radius
        self._vertices *= scale_factor
        self._equations[:, 3] *= scale_factor
