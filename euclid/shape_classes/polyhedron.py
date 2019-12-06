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
    def __init__(self, vertices, facets=None, normals=None):
        """A general polyhedron.

        If only vertices are passed in, the result is a convex polyhedron
        defined by these vertices. If facets are provided, the resulting
        polyhedron may be nonconvex.

        The polyhedron is assumed to be of unit mass and constant density.

        """
        self._vertices = np.array(vertices, dtype=np.float64)
        if facets is None:
            hull = ConvexHull(vertices)
            self._facets = [facet for facet in hull.simplices]
        else:
            # TODO: Add some sanity checks here.
            self._facets = facets

        if normals is not None:
            self._equations = np.empty((len(self.facets), 4))
            for i, (facet, normal) in enumerate(zip(self.facets, normals)):
                self._equations[i, :3] = normal
                # Arbitrarily choose to use the first vertex of each facet.
                self._equations[i, 3] = normal.dot(self.vertices[facet[0]])
        else:
            self._find_equations()

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
            self._neighbors[i] = np.array(self._neighbors[i])

    def merge_facets(self, tolerance=1e-6):
        """Merge facets of a polyhedron.

        For convex polyhedra, facets will automatically be merged to an
        appropriate degree.  However, the merging of facets must be based on a
        tolerance (we may need to provide two such parameters depending on how
        we perform the merge), so we need to expose this method to allow the
        user to redo the merge with a different tolerance."""
        self._find_neighbors()

        # Construct a graph where connectivity indicates merging, then identify
        # connected components to merge.
        merge_graph = np.zeros((self.num_facets, self.num_facets))
        for i in range(self.num_facets):
            for j in self._neighbors[i]:
                if np.allclose(self._equations[i], self._equations[j]) or \
                        np.allclose(self._equations[i], -self._equations[j]):
                    merge_graph[i, j] = 1

        _, labels = connected_components(merge_graph, directed=False,
                                         return_labels=True)
        new_facets = [set() for _ in range(len(np.unique(labels)))]
        for i, facet in enumerate(self.facets):
            new_facets[labels[i]].update(facet)

        self._facets = [np.asarray(list(f)) for f in new_facets]
        self._find_equations()

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
                for vertex in Polygon(self.vertices[facet]).vertices
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
    def volume(self, value):
        pass

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
            poly = Polygon(self.vertices[facet])
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
            poly = Polygon(self.vertices[facet])
            yield from poly.triangulation()

    @property
    def inertia_tensor(self):
        """The inertia tensor computed about the center of mass.

        Computed using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """
        simplices = np.array(list(self.triangulation())) - self.center
        volumes = np.abs(np.linalg.det(simplices)/6)

        def compute(f):
            R"""Integrate functions of the form :math:`\int\int\int f(x, y, z)
            dx dy dz` over a set of triangles. Omits a factor of v/20."""
            fv1 = f(simplices[:, 0, :])
            fv2 = f(simplices[:, 1, :])
            fv3 = f(simplices[:, 2, :])
            fvsum = (f(simplices[:, 0, :] +
                       simplices[:, 1, :] +
                       simplices[:, 2, :]))
            return np.sum((volumes/20)*(fv1 + fv2 + fv3 + fvsum))

        Ixx = compute(lambda t: t[:, 1]**2 + t[:, 2]**2)
        Ixy = compute(lambda t: -t[:, 0]*t[:, 1])
        Ixz = compute(lambda t: -t[:, 0]*t[:, 2])
        Iyy = compute(lambda t: t[:, 0]**2 + t[:, 2]**2)
        Iyz = compute(lambda t: -t[:, 1]*t[:, 2])
        Izz = compute(lambda t: t[:, 0]**2 + t[:, 1]**2)

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
    def insphere_radius(self):
        """Get or set the polyhedron's insphere radius (setting rescales
        vertices)."""
        pass

    @insphere_radius.setter
    def insphere_radius(self, value):
        pass

    @property
    def circumsphere_radius(self):
        """Get or set the polyhedron's circumsphere radius (setting rescales
        vertices)."""
        pass

    @circumsphere_radius.setter
    def circumsphere_radius(self, value):
        pass

    @property
    def asphericity(self):
        """The asphericity."""
        pass

    @property
    def iq(self):
        """The isoperimetric quotient."""
        pass

    @property
    def tau(self):
        """The sphericity measure defined in
        https://www.sciencedirect.com/science/article/pii/0378381284800199."""
        pass

    @property
    def facet_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        facets."""
        pass

    @property
    def vertex_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        vertex."""
        pass
