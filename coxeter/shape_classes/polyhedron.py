import numpy as np
from .polygon import Polygon, _is_simple
from .convex_polygon import ConvexPolygon
from .convex_polygon import _is_convex
from scipy.sparse.csgraph import connected_components
import rowan

try:
    import miniball
    MINIBALL = True
except ImportError:
    MINIBALL = False


def _facet_to_edges(facet, reverse=False):
    """Convert a facet (a sequence of vertices) into a sequence of edges
    (tuples).

    Args:
        facet (array-like):
            A facet composed of vertex indices.
        reverse (bool):
            Whether to return the edges in reverse.
    """
    shift = 1 if reverse else -1
    return list(zip(*np.stack((facet, np.roll(facet, shift)))))


class Polyhedron(object):
    def __init__(self, vertices, facets, facets_are_convex=None):
        """A three-dimensional polytope.

        A polyhedron is defined by a set of vertices and a set of facets
        composed of the vertices. On construction, the facets are reordered
        counterclockwise with respect to an outward normal. The polyhedron
        provides various standard geometric calculations, such as volume and
        surface area. Most features of the polyhedron can be accessed via
        properties, including the plane equations defining the facets and the
        neighbors of each facet.

        .. note::

            For the purposes of calculations like moments of inertia, the
            polyhedron is assumed to be of constant, unit density.

        Args:
            vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
                The vertices of the polyhedron.
            facets (list(list)):
                The facets of the polyhedron.
            facets_are_convex (bool):
                Whether or not the facets of the polyhedron are all convex.
                This is used to determine whether certain operations like
                coplanar facet merging are allowed (Default value: False).
        """
        self._vertices = np.array(vertices, dtype=np.float64)
        self._facets = [facet for facet in facets]
        if facets_are_convex is None:
            facets_are_convex = all(len(facet) == 3 for facet in facets)
        self._facets_are_convex = facets_are_convex
        self._find_equations()
        self._find_neighbors()

    def _find_equations(self):
        """Find the plane equations of the polyhedron facets."""
        self._equations = np.empty((len(self.facets), 4))
        for i, facet in enumerate(self.facets):
            # The direction of the normal is selected such that vertices that
            # are already ordered counterclockwise will point outward.
            normal = np.cross(
                self.vertices[facet[2]] - self.vertices[facet[1]],
                self.vertices[facet[0]] - self.vertices[facet[1]])
            normal /= np.linalg.norm(normal)
            self._equations[i, :3] = normal
            # Sign conventions chosen to match scipy.spatial.ConvexHull
            # We use ax + by + cz + d = 0 (not ax + by + cz = d)
            self._equations[i, 3] = -normal.dot(self.vertices[facet[0]])

    def _find_neighbors(self):
        """Find neighbors of facets."""
        self._neighbors = [[] for _ in range(self.num_facets)]
        for i, j, _ in self._get_facet_intersections():
            self._neighbors[i].append(j)
            self._neighbors[j].append(i)
        self._neighbors = [np.array(neigh) for neigh in self._neighbors]

    def _get_facet_intersections(self):
        """A generator that yields tuples of the form (facet, neighbor,
        (vertex1, vertex2)) indicating neighboring facets and their common
        edge."""
        # First enumerate all edges of each neighbor. We include both
        # directions of the edges for comparison.
        facet_edges = [set(_facet_to_edges(f) +
                           _facet_to_edges(f, True)) for f in self.facets]

        for i in range(self.num_facets):
            for j in range(i+1, self.num_facets):
                common_edges = facet_edges[i].intersection(facet_edges[j])
                if len(common_edges) > 0:
                    # Can never have multiple intersections, but we should have
                    # the same edge show up twice (forward and reverse).
                    assert len(common_edges) == 2
                    common_edge = list(common_edges)[0]
                    yield (i, j, (common_edge[0], common_edge[1]))

    def merge_facets(self, atol=1e-8, rtol=1e-5):
        """Merge coplanar facets to a given tolerance.

        Whether or not facets should be merged is determined using
        :func:`numpy.allclose` to compare the plane equations of neighboring
        facets. Connected components of mergeable facets are then merged into
        a single facet.  This method can be safely called many times with
        different tolerances, however, the operation is destructive in the
        sense that merged facets cannot be recovered. Users wishing to undo a
        merge to attempt a less expansive merge must build a new polyhedron.

        Args:
            atol (float):
                Absolute tolerance for :func:`numpy.allclose`.
            rtol (float):
                Relative tolerance for :func:`numpy.allclose`.
        """
        if not self._facets_are_convex:
            # Can only sort facets if they are guaranteed to be convex.
            raise ValueError(
                "Faces cannot be merged unless they are convex because the "
                "correct ordering of vertices in a facet cannot be determined "
                "for nonconvex faces.")

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
        """list(:class:`numpy.ndarray`): A list where the
        :math:`i^{\\text{th}}` element is an array of indices of facets that
        are neighbors of facet :math:`i`.
        """
        return self._neighbors

    @property
    def normals(self):
        """:math:`(N, 3)` :class:`numpy.ndarray`: The normal vectors to each
        facet."""
        return self._equations[:, :3]

    @property
    def num_vertices(self):
        """int: The number of vertices."""
        return self.vertices.shape[0]

    @property
    def num_facets(self):
        """int: The number of facets."""
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

        .. note::
            This method can only be called for polyhedra whose faces are all
            convex (i.e. constructed with ``facets_are_convex=True``).
        """
        if not self._facets_are_convex:
            # Can only sort facets if they are guaranteed to be convex.
            raise ValueError(
                "Faces cannot be sorted unless they are convex because the "
                "correct ordering of vertices in a facet cannot be determined "
                "for nonconvex faces.")

        # We first ensure that facet vertices are sequentially ordered by
        # constructing a Polygon and updating the facet (in place), which
        # enables finding neighbors.
        for facet in self.facets:
            polygon = ConvexPolygon(
                self.vertices[facet], planar_tolerance=1e-4)
            if _is_convex(polygon.vertices, polygon.normal):
                facet[:] = np.asarray([
                    np.where(np.all(self.vertices == vertex, axis=1))[0][0]
                    for vertex in polygon.vertices
                ])
            elif not _is_simple(polygon.vertices):
                raise ValueError("The vertices of each facet must be provided "
                                 "in counterclockwise order relative to the "
                                 "facet normal unless the facet is a convex "
                                 "polygon.")
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
        """:math:`(N, 3)` :class:`numpy.ndarray`: Get the vertices of the
        polyhedron."""
        return self._vertices

    @property
    def facets(self):
        """list(:class:`numpy.ndarray`): Get the polyhedron's facets."""
        return self._facets

    @property
    def volume(self):
        """float: Get or set the polyhedron's volume (setting rescales
        vertices)."""
        ds = -self._equations[:, 3]
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
            :class:`numpy.ndarray`: The area of each facet.
        """
        if facets is None:
            facets = range(len(self.facets))
        elif type(facets) is int:
            facets = [facets]

        areas = np.empty(len(facets))
        for i, facet_index in enumerate(facets):
            facet = self.facets[facet_index]
            poly = ConvexPolygon(self.vertices[facet], planar_tolerance=1e-4)
            areas[i] = poly.area

        return areas

    @property
    def surface_area(self):
        """float: Get the surface area."""
        return np.sum(self.get_facet_area())

    def _triangulation(self):
        """Generate a triangulation of the surface of the polyhedron.

        This algorithm constructs Polygons from each of the facets and then
        triangulates each of these to provide a total triangulation.
        """
        for facet in self.facets:
            poly = Polygon(self.vertices[facet], planar_tolerance=1e-4)
            yield from poly._triangulation()

    def _point_plane_distances(self, points):
        """Computes the distances from a set of points to each plane.

        Distances that are <= 0 are inside and > 0 are outside.

        Returns:
            :math:`(N_{points}, N_{planes})` :class:`numpy.ndarray`: The
            distance from each point to each plane.
        """
        points = np.atleast_2d(points)
        dots = np.inner(points, self._equations[:, :3])
        distances = dots + self._equations[:, 3]
        return distances

    @property
    def inertia_tensor(self):
        """:math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor
        computed about the center of mass (uses the algorithm described in
        :cite:`Kallay2006`).
        """
        simplices = np.array(list(self._triangulation())) - self.center

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
                         [Ixy, Iyy, Iyz],
                         [Ixz, Iyz, Izz]])

    @property
    def center(self):
        """float: Get or set the polyhedron's centroid (setting rescales
        vertices)."""
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self._vertices += (np.asarray(value) - self.center)
        self._find_equations()

    @property
    def bounding_sphere(self):
        """The bounding sphere of the polyhedron, given by a center and a
        radius."""
        if not MINIBALL:
            raise ImportError("The miniball module must be installed. It can "
                              "be installed as an extra with coxeter (e.g. "
                              "with pip install coxeter[bounding_sphere], or "
                              "directly from PyPI using pip install miniball."
                              )

        # The algorithm in miniball involves solving a linear system and
        # can therefore occasionally be somewhat unstable. Applying a
        # random rotation will usually fix the issue.
        max_attempts = 10
        attempt = 0
        current_rotation = [1, 0, 0, 0]
        vertices = self.vertices
        while attempt < max_attempts:
            attempt += 1
            try:
                center, r2 = miniball.get_bounding_ball(vertices)
                break
            except np.linalg.LinAlgError:
                current_rotation = rowan.random.rand(1)
                vertices = rowan.rotate(current_rotation, vertices)
        else:
            raise RuntimeError("Unable to solve for a bounding sphere.")

        # The center must be rotated back to undo any rotation.
        center = rowan.rotate(rowan.conjugate(current_rotation), center)

        return center, np.sqrt(r2)

    @property
    def circumsphere(self):
        """float: Get the polyhedron's circumsphere."""
        points = self.vertices[1:] - self.vertices[0]
        half_point_lengths = np.sum(points*points, axis=1)/2
        x, resids, _, _ = np.linalg.lstsq(points, half_point_lengths, None)
        if len(self.vertices) > 4 and not np.isclose(resids, 0):
            raise RuntimeError("No circumsphere for this polyhedron.")

        return x + self.vertices[0], np.linalg.norm(x)

    @property
    def iq(self):
        """float: The isoperimetric quotient.

        """
        # TODO: allow for non-spherical reference ratio (changes the prefactor)
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
                The index of the second facet.

        Returns:
            float: The dihedral angle in radians.
        """
        if b not in self.neighbors[a]:
            raise ValueError("The two facets are not neighbors.")
        n1, n2 = self._equations[[a, b], :3]
        return np.arccos(np.dot(-n1, n2))

    def plot(self, ax, plot_verts=False, label_verts=False):
        """Plot the polyhedron.

        Note that the ``ax`` argument should be a 3D axes object; passing in a
        2D axes will result in wrong behavior.

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes on which to draw the polyhedron.
            plot_verts (bool):
                If True, scatter points will be added at the vertices (Default
                value: False).
            label_verts (bool):
                If True, vertex indices will be added next to the vertices
                (Default value: False).
        """
        # TODO: Generate axis if one is not provided.
        # Determine dimensionality.
        for i, facet in enumerate(self.facets):
            verts = self.vertices[facet]
            verts = np.concatenate((verts, verts[[0]]))
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2])

        if plot_verts:
            ax.scatter(self.vertices[:, 0],
                       self.vertices[:, 1],
                       self.vertices[:, 2])
        if label_verts:
            # Typically a good shift for plotting the labels
            shift = (np.max(self.vertices[:, 2]) -
                     np.min(self.vertices[:, 2]))*0.025
            for i, vert in enumerate(self.vertices):
                ax.text(vert[0], vert[1], vert[2] + shift, '{}'.format(i),
                        fontsize=10)
