"""Defines a polyhedron."""

import numpy as np
import rowan
from scipy.sparse.csgraph import connected_components

from .base_classes import Shape3D
from .convex_polygon import ConvexPolygon, _is_convex
from .polygon import Polygon, _is_simple
from .sphere import Sphere
from .utils import translate_inertia_tensor

try:
    import miniball

    MINIBALL = True
except ImportError:
    MINIBALL = False


def _face_to_edges(face, reverse=False):
    """Convert a face into a sequence of edges (tuples).

    Args:
        face (array-like):
            A face composed of vertex indices.
        reverse (bool):
            Whether to return the edges in reverse.

    Returns:
        list[tuple[int, int]]:
            A list of edges where each is a tuple of a pair of vertices.
    """
    shift = 1 if reverse else -1
    return list(zip(*np.stack((face, np.roll(face, shift)))))


class Polyhedron(Shape3D):
    """A three-dimensional polytope.

    A polyhedron is defined by a set of vertices and a set of faces
    composed of the vertices. On construction, the faces are reordered
    counterclockwise with respect to an outward normal. The polyhedron
    provides various standard geometric calculations, such as volume and
    surface area. Most features of the polyhedron can be accessed via
    properties, including the plane equations defining the faces and the
    neighbors of each face.

    .. note::

        For the purposes of calculations like moments of inertia, the
        polyhedron is assumed to be of constant, unit density.

    Args:
        vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
            The vertices of the polyhedron.
        faces (list(list)):
            The faces of the polyhedron.
        faces_are_convex (bool, optional):
            Whether or not the faces of the polyhedron are all convex.
            This is used to determine whether certain operations like
            coplanar face merging are allowed (Default value: False).
    """

    def __init__(self, vertices, faces, faces_are_convex=None):
        self._vertices = np.array(vertices, dtype=np.float64)
        self._faces = [face for face in faces]
        if faces_are_convex is None:
            faces_are_convex = all(len(face) == 3 for face in faces)
        self._faces_are_convex = faces_are_convex
        self._find_equations()
        self._find_neighbors()

    def _find_equations(self):
        """Find the plane equations of the polyhedron faces."""
        self._equations = np.empty((len(self.faces), 4))
        for i, face in enumerate(self.faces):
            # The direction of the normal is selected such that vertices that
            # are already ordered counterclockwise will point outward.
            normal = np.cross(
                self.vertices[face[2]] - self.vertices[face[1]],
                self.vertices[face[0]] - self.vertices[face[1]],
            )
            normal /= np.linalg.norm(normal)
            self._equations[i, :3] = normal
            # Sign conventions chosen to match scipy.spatial.ConvexHull
            # We use ax + by + cz + d = 0 (not ax + by + cz = d)
            self._equations[i, 3] = -normal.dot(self.vertices[face[0]])

    def _find_neighbors(self):
        """Find neighbors of faces."""
        self._neighbors = [[] for _ in range(self.num_faces)]
        for i, j, _ in self._get_face_intersections():
            self._neighbors[i].append(j)
            self._neighbors[j].append(i)
        self._neighbors = [np.array(neigh) for neigh in self._neighbors]

    def _get_face_intersections(self):
        """Get pairs of faces and their common edges.

        This function yields a generator of tuples of the form (face, neighbor,
        (vertex1, vertex2)) indicating neighboring faces and their common
        edge.
        """
        # First enumerate all edges of each neighbor. We include both
        # directions of the edges for comparison.
        face_edges = [
            set(_face_to_edges(f) + _face_to_edges(f, True)) for f in self.faces
        ]

        for i in range(self.num_faces):
            for j in range(i + 1, self.num_faces):
                common_edges = face_edges[i].intersection(face_edges[j])
                if len(common_edges) > 0:
                    # Can never have multiple intersections, but we should have
                    # the same edge show up twice (forward and reverse).
                    assert len(common_edges) == 2
                    common_edge = list(common_edges)[0]
                    yield (i, j, (common_edge[0], common_edge[1]))

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {
            "type": "Mesh",
            "vertices": self._vertices.tolist(),
            "faces": self._faces,
        }

    def merge_faces(self, atol=1e-8, rtol=1e-5):
        """Merge coplanar faces to a given tolerance.

        Whether or not faces should be merged is determined using
        :func:`numpy.allclose` to compare the plane equations of neighboring
        faces. Connected components of mergeable faces are then merged into
        a single face.  This method can be safely called many times with
        different tolerances, however, the operation is destructive in the
        sense that merged faces cannot be recovered. Users wishing to undo a
        merge to attempt a less expansive merge must build a new polyhedron.

        Args:
            atol (float):
                Absolute tolerance for :func:`numpy.allclose`.
            rtol (float):
                Relative tolerance for :func:`numpy.allclose`.
        """
        if not self._faces_are_convex:
            # Can only sort faces if they are guaranteed to be convex.
            raise ValueError(
                "Faces cannot be merged unless they are convex because the "
                "correct ordering of vertices in a face cannot be determined "
                "for nonconvex faces."
            )

        # Construct a graph where connectivity indicates merging, then identify
        # connected components to merge.
        merge_graph = np.zeros((self.num_faces, self.num_faces))
        for i in range(self.num_faces):
            for j in self._neighbors[i]:
                eq1, eq2 = self._equations[[i, j]]
                if np.allclose(eq1, eq2, atol=atol, rtol=rtol) or np.allclose(
                    eq1, -eq2, atol=atol, rtol=rtol
                ):
                    merge_graph[i, j] = 1

        _, labels = connected_components(
            merge_graph, directed=False, return_labels=True
        )
        new_faces = [set() for _ in range(len(np.unique(labels)))]
        for i, face in enumerate(self.faces):
            new_faces[labels[i]].update(face)

        self._faces = [np.asarray(list(f)) for f in new_faces]
        self.sort_faces()

    @property
    def neighbors(self):
        r"""list(:class:`numpy.ndarray`): Get neighboring pairs of faces.

        The neighbors are provided as a list where the :math:`i^{\text{th}}`
        element is an array of indices of faces that are neighbors of face
        :math:`i`.
        """
        return self._neighbors

    @property
    def normals(self):
        """:math:`(N, 3)` :class:`numpy.ndarray`: Get the face normals."""
        return self._equations[:, :3]

    @property
    def num_vertices(self):
        """int: Get the number of vertices."""
        return self.vertices.shape[0]

    @property
    def num_faces(self):
        """int: Get the number of faces."""
        return len(self.faces)

    def sort_faces(self):  # noqa: C901
        """Sort faces of the polyhedron.

        This method ensures that all faces are ordered such that the normals
        are counterclockwise and point outwards. This algorithm proceeds in
        four steps. First, it ensures that each face is ordered in either
        clockwise or counterclockwise order such that edges can be found from
        the sequence of the vertices in each face. Next, it calls the neighbor
        finding routine to establish with faces are neighbors. Then, it
        performs a breadth-first search, reorienting faces to match the
        orientation of the first face.  Finally, it computes the signed volume
        to determine whether or not all the normals need to be flipped.

        .. note::
            This method can only be called for polyhedra whose faces are all
            convex (i.e. constructed with ``faces_are_convex=True``).
        """
        if not self._faces_are_convex:
            # Can only sort faces if they are guaranteed to be convex.
            raise ValueError(
                "Faces cannot be sorted unless they are convex because the "
                "correct ordering of vertices in a face cannot be determined "
                "for nonconvex faces."
            )

        # We first ensure that face vertices are sequentially ordered by
        # constructing a Polygon and updating the face (in place), which
        # enables finding neighbors.
        for face in self.faces:
            polygon = ConvexPolygon(self.vertices[face], planar_tolerance=1e-4)
            if _is_convex(polygon.vertices, polygon.normal):
                face[:] = np.asarray(
                    [
                        np.where(np.all(self.vertices == vertex, axis=1))[0][0]
                        for vertex in polygon.vertices
                    ]
                )
            elif not _is_simple(polygon.vertices):
                raise ValueError(
                    "The vertices of each face must be provided "
                    "in counterclockwise order relative to the "
                    "face normal unless the face is a convex "
                    "polygon."
                )
        self._find_neighbors()

        # The initial face sets the order of the others.
        visited_faces = []
        remaining_faces = [0]
        while len(remaining_faces):
            current_face = remaining_faces[-1]
            visited_faces.append(current_face)
            remaining_faces.pop()

            # Search for common edges between pairs of faces, then check the
            # ordering of the edge to determine relative face orientation.
            current_edges = _face_to_edges(self.faces[current_face])
            for neighbor in self._neighbors[current_face]:
                if neighbor in visited_faces:
                    continue
                remaining_faces.append(neighbor)

                # Two faces can only share a single edge (otherwise they would
                # be coplanar), so we can break as soon as we find the
                # neighbor. Flip the neighbor if the edges are identical.
                for edge in _face_to_edges(self.faces[neighbor]):
                    if edge in current_edges:
                        self._faces[neighbor] = self._faces[neighbor][::-1]
                        break
                    elif edge[::-1] in current_edges:
                        break
                visited_faces.append(neighbor)

        # Now compute the signed area and flip all the orderings if the area is
        # negative.
        self._find_equations()
        if self.volume < 0:
            for i in range(len(self.faces)):
                self._faces[i] = self._faces[i][::-1]
                self._equations[i] *= -1

    @property
    def vertices(self):
        """:math:`(N, 3)` :class:`numpy.ndarray`: Get the vertices of the polyhedron."""  # noqa: E501
        return self._vertices

    @property
    def faces(self):
        """list(:class:`numpy.ndarray`): Get the polyhedron's faces."""
        return self._faces

    @property
    def volume(self):
        """float: Get or set the polyhedron's volume."""
        ds = -self._equations[:, 3]
        return np.sum(ds * self.get_face_area()) / 3

    @volume.setter
    def volume(self, new_volume):
        scale_factor = (new_volume / self.volume) ** (1 / 3)
        self._vertices *= scale_factor
        self._equations[:, 3] *= scale_factor

    def get_face_area(self, faces=None):
        """Get the total surface area of a set of faces.

        Args:
            faces (int, sequence, or None):
                The index of a face or a set of face indices for which to
                find the area. If None, finds the area of all faces (Default
                value: None).

        Returns:
            :class:`numpy.ndarray`: The area of each face.
        """
        if faces is None:
            faces = range(len(self.faces))
        elif type(faces) is int:
            faces = [faces]

        areas = np.empty(len(faces))
        for i, face_index in enumerate(faces):
            face = self.faces[face_index]
            poly = ConvexPolygon(self.vertices[face], planar_tolerance=1e-4)
            areas[i] = poly.area

        return areas

    @property
    def surface_area(self):
        """float: Get the surface area."""
        return np.sum(self.get_face_area())

    def _surface_triangulation(self):
        """Generate a triangulation of the surface of the polyhedron.

        This algorithm constructs Polygons from each of the faces and then
        triangulates each of these to provide a total triangulation.
        """
        for face in self.faces:
            poly = Polygon(self.vertices[face], planar_tolerance=1e-4)
            yield from poly._triangulation()

    def _point_plane_distances(self, points):
        """Compute the distances from a set of points to each plane.

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
        """:math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor.

        The inertia tensor is computed using the algorithm described in
        :cite:`Kallay2006`.

        Note:
            For improved stability, the inertia tensor is computed about the
            center of mass and then shifted rather than directly computed in
            the global frame.
        """
        it = self._compute_inertia_tensor()
        return translate_inertia_tensor(self.center, it, self.volume)

    def _compute_inertia_tensor(self, centered=True):
        """Compute the inertia tensor.

        Internal function for computing the inertia tensor that supports both
        centered and uncentered calculations. Primarily of use for testing and
        validation purposes.
        """
        simplices = np.array(list(self._surface_triangulation()))
        if centered:
            simplices -= self.center

        volumes = np.abs(np.linalg.det(simplices) / 6)

        def triangle_integrate(f):
            r"""Integrate f over the simplices.

            This function computes integrals of the form
            :math:`\int\int\int f(x, y, z) dx dy dz` over a set of triangles.
            """
            fv1 = f(simplices[:, 0, :])
            fv2 = f(simplices[:, 1, :])
            fv3 = f(simplices[:, 2, :])
            fvsum = f(simplices[:, 0, :] + simplices[:, 1, :] + simplices[:, 2, :])
            return np.sum((volumes / 20) * (fv1 + fv2 + fv3 + fvsum))

        i_xx = triangle_integrate(lambda t: t[:, 1] ** 2 + t[:, 2] ** 2)
        i_xy = triangle_integrate(lambda t: -t[:, 0] * t[:, 1])
        i_xz = triangle_integrate(lambda t: -t[:, 0] * t[:, 2])
        i_yy = triangle_integrate(lambda t: t[:, 0] ** 2 + t[:, 2] ** 2)
        i_yz = triangle_integrate(lambda t: -t[:, 1] * t[:, 2])
        i_zz = triangle_integrate(lambda t: t[:, 0] ** 2 + t[:, 1] ** 2)

        return np.array([[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]])

    @property
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self._vertices += np.asarray(value) - self.center
        self._find_equations()

    @property
    def bounding_sphere(self):
        """:class:`~.Sphere`: Get the center and radius of the bounding sphere."""
        if not MINIBALL:
            raise ImportError(
                "The miniball module must be installed. It can "
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

        return Sphere(np.sqrt(r2), center)

    @property
    def circumsphere(self):
        """:class:`~.Sphere`: Get the polyhedron's circumsphere."""
        points = self.vertices[1:] - self.vertices[0]
        half_point_lengths = np.sum(points * points, axis=1) / 2
        x, resids, _, _ = np.linalg.lstsq(points, half_point_lengths, None)
        if len(self.vertices) > 4 and not np.isclose(resids, 0):
            raise RuntimeError("No circumsphere for this polyhedron.")

        return Sphere(np.linalg.norm(x), x + self.vertices[0])

    @property
    def iq(self):
        """float: The isoperimetric quotient."""
        # TODO: allow for non-spherical reference ratio (changes the prefactor)
        return np.pi * 36 * self.volume ** 2 / (self.surface_area ** 3)

    def get_dihedral(self, a, b):
        """Get the dihedral angle between a pair of faces.

        The dihedral is computed from the dot product of the face normals.

        Args:
            a (int):
                The index of the first face.
            b (int):
                The index of the second face.

        Returns:
            float: The dihedral angle in radians.
        """
        if b not in self.neighbors[a]:
            raise ValueError("The two faces are not neighbors.")
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
        for face in self.faces:
            verts = self.vertices[face]
            verts = np.concatenate((verts, verts[[0]]))
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2])

        if plot_verts:
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
        if label_verts:
            # Typically a good shift for plotting the labels
            shift = (np.max(self.vertices[:, 2]) - np.min(self.vertices[:, 2])) * 0.025
            for i, vert in enumerate(self.vertices):
                ax.text(vert[0], vert[1], vert[2] + shift, "{}".format(i), fontsize=10)

    def diagonalize_inertia(self):
        """Orient the shape along its principal axes.

        The principal axes of a shape are defined by the eigenvectors of the inertia
        tensor. This method computes the inertia tensor of the shape, diagonalizes it,
        and then rotates the shape by the corresponding orthogonal transformation.
        """
        principal_moments, principal_axes = np.linalg.eigh(self.inertia_tensor)
        self._vertices = np.dot(self._vertices, principal_axes)
