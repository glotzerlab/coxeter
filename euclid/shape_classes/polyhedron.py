from scipy.spatial import ConvexHull
import numpy as np
from .polygon import Polygon


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
        """Find equations of facets.

        This function makes no guarantees about the direction of the normal it
        chooses."""
        self._equations = np.empty((len(self.facets), 4))
        for i, facet in enumerate(self.facets):
            v0 = self.vertices[facet[0]]
            v1 = self.vertices[facet[1]]
            v2 = self.vertices[facet[2]]
            normal = np.cross(v2 - v1, v0 - v1)
            self._equations[i, :3] = normal / np.linalg.norm(normal)
            self._equations[i, 3] = normal.dot(self.vertices[facet[0]])

    def _find_neighbors(self):
        """Find neighbors of facets."""
        # First enumerate all edges of each neighbor. We include both
        # directions of the edges for comparison.
        facet_edges = []
        for facet in self.facets:
            forward_edges = set(zip(*np.stack((facet, np.roll(facet, 1)))))
            reverse_edges = set(zip(*np.stack((facet, np.roll(facet, -1)))))
            facet_edges.append(forward_edges.union(reverse_edges))

        # Find any facets that share neighbors.
        num_facets = len(facet_edges)
        self._connectivity_graph = np.zeros((num_facets, num_facets))
        for i in range(num_facets):
            for j in range(i+1, num_facets):
                if len(facet_edges[i].intersection(facet_edges[j])) > 0:
                    self._connectivity_graph[i, j] = 1
                    # For symmetry, can be removed when no longer useful for
                    # debugging.
                    # self._connectivity_graph[j, i] = 1

    def merge_facets(self, tolerance=1e-6):
        """Merge facets of a polyhedron.

        For convex polyhedra, facets will automatically be merged to an
        appropriate degree.  However, the merging of facets must be based on a
        tolerance (we may need to provide two such parameters depending on how
        we perform the merge), so we need to expose this method to allow the
        user to redo the merge with a different tolerance."""
        self._find_equations()
        self._find_neighbors()

        # Test if these are coplanar.
        num_facets = len(self.facets)
        merge_graph = np.zeros((num_facets, num_facets))
        for i, j in zip(*np.where(self._connectivity_graph)):
            if np.allclose(self._equations[i, :], self._equations[j, :]) or \
                    np.allclose(self._equations[i, :], -self._equations[j, :]):
                merge_graph[i, j] = 1

        # Merge selected facets.
        new_facets = []
        remaining_facets = np.arange(merge_graph.shape[0]).tolist()
        for i in remaining_facets:
            cur_set = set(self.facets[i])
            other_facets = np.where(merge_graph[i])[0]
            for j in other_facets:
                cur_set = cur_set.union(self.facets[j])
                remaining_facets.remove(j)
            new_facets.append(np.array(list(cur_set)))

        self._facets = new_facets
        self._find_equations()

    @property
    def normals(self):
        """The normal vectors to each facet."""
        return self._equations[:, :3]

    def sort_facets(self):
        """Ensure that all facets are ordered such that the normals are
        counterclockwise and point outwards."""

        # Finding neighbors does not require that facets be ordered the same
        # way, but they do need to be ordered such that edges can be identified
        # by consecutive vertices in the facet list. To generate the ordering,
        # we simply construct a Polygon from the vertices in each facet.
        for i in range(len(self.facets)):
            poly = Polygon(self.vertices[self.facets[i]])
            # Add the vertices in order.
            new_facet = []
            for vertex in poly.vertices:
                vertex_id = np.where(np.all(self.vertices == vertex, axis=1))[0][0]
                new_facet.append(vertex_id)
            self._facets[i] = np.asarray(new_facet)

        # Need to know which facets are neighbors for this to work.
        self._find_neighbors()

        visited_facets = []
        current_facet_index = 0
        remaining_facets = [current_facet_index]
        # Use a while loop because we don't want to loop over an object whose
        # size is changing (as we remove facets that still need to be
        # reordered. The initial facet sets the orientation of all the others.
        while len(visited_facets) < len(self.facets):
            visited_facets.append(current_facet_index)
            remaining_facets.remove(current_facet_index)

            # Get all edges of the current facet
            current_facet = self.facets[current_facet_index]
            current_edges = [(current_facet[i], current_facet[(i+1) % len(current_facet)])
                             for i in range(len(current_facet))]

            # Get all neighbors, then check each one to see if needs
            # reordering (unless it has already been checked).
            current_neighbor_indices = np.where(
                self._connectivity_graph[current_facet_index])[0]
            for neighbor in current_neighbor_indices:
                # Any neighbor that has not itself been reordered should be
                # added to the queue of possible reordered vertices.
                if neighbor in visited_facets:
                    continue
                else:
                    remaining_facets.append(neighbor)
                neighbor_facet = self.facets[neighbor]
                neighbor_facet = np.concatenate((neighbor_facet, neighbor_facet[[0]]))

                # Two facets can only share a single edge (otherwise they would
                # be coplanar), so we can break as soon as we find the
                # neighbor.
                for i in range(len(neighbor_facet)-1):
                    edge = (neighbor_facet[i], neighbor_facet[i+1])
                    if edge in current_edges:
                        # This requires a flip
                        self._facets[neighbor] = self._facets[neighbor][::-1]
                        break
                    elif edge[::-1] in current_edges:
                        # This is the desired orientation
                        break
                visited_facets.append(neighbor)

            if len(remaining_facets):
                current_facet_index = remaining_facets[0]

        # Now compute the signed area and flip all the orderings if the area is
        # negative.
        self._find_equations()
        if self.volume < 0:
            for i in range(len(self.facets)):
                self._facets[i] = self._facets[i][::-1]
        self._find_equations()

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
        # Arbitrary choice, use the first vertex in the face to compute the
        # distance to the plane.
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

    @property
    def inertia_tensor(self):
        """The inertia tensor.

        Computed using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """
        try:
            return self._inertia_tensor
        except AttributeError:
            centered_vertices = self.vertices - self.center
            simplices = centered_vertices[self.faces]

            volumes = np.abs(np.linalg.det(simplices)/6)

            fxx = lambda triangles: triangles[:, 1]**2 + triangles[:, 2]**2 # noqa
            fxy = lambda triangles: -triangles[:, 0]*triangles[:, 1] # noqa
            fxz = lambda triangles: -triangles[:, 0]*triangles[:, 2] # noqa
            fyy = lambda triangles: triangles[:, 0]**2 + triangles[:, 2]**2 # noqa
            fyz = lambda triangles: -triangles[:, 1]*triangles[:, 2] # noqa
            fzz = lambda triangles: triangles[:, 0]**2 + triangles[:, 1]**2 # noqa

            def compute(f):
                return f(simplices[:, 0, :]) + f(simplices[:, 1, :]) + \
                    f(simplices[:, 2, :]) + f(simplices[:, 0, :] +
                                              simplices[:, 1, :] +
                                              simplices[:, 2, :])

            Ixx = (compute(fxx)*volumes/20).sum()
            Ixy = (compute(fxy)*volumes/20).sum()
            Ixz = (compute(fxz)*volumes/20).sum()
            Iyy = (compute(fyy)*volumes/20).sum()
            Iyz = (compute(fyz)*volumes/20).sum()
            Izz = (compute(fzz)*volumes/20).sum()

            self._inertia_tensor = np.array([[Ixx, Ixy, Ixz],
                                            [Ixy,   Iyy, Iyz],
                                            [Ixz,   Iyz,   Izz]])

            return self._inertia_tensor

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
