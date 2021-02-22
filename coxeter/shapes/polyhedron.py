# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines a polyhedron."""

import warnings

import numpy as np
import rowan
from scipy.sparse.csgraph import connected_components

from ..extern.polyhedron.polyhedron import Polyhedron as PolyInside
from ..extern.polytri import polytri
from .base_classes import Shape3D
from .convex_polygon import ConvexPolygon, _is_convex
from .polygon import Polygon, _is_simple
from .sphere import Sphere
from .utils import _generate_ax, _set_3d_axes_equal, translate_inertia_tensor

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

    Example:
        >>> cube = coxeter.shapes.ConvexPolyhedron(
        ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
        ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]])
        >>> cube = coxeter.shapes.Polyhedron(
        ...   vertices=cube.vertices, faces=cube.faces)
        >>> bounding_sphere = cube.minimal_bounding_sphere
        >>> import numpy as np
        >>> assert np.isclose(bounding_sphere.radius, np.sqrt(3))
        >>> cube.center
        array([0., 0., 0.])
        >>> cube.faces
        [array([4, 5, 1, 0], dtype=int32), array([0, 2, 6, 4], dtype=int32),
        array([6, 7, 5, 4], dtype=int32), array([0, 1, 3, 2], dtype=int32),
        array([5, 7, 3, 1], dtype=int32), array([2, 3, 7, 6], dtype=int32)]
        >>> cube.gsd_shape_spec
        {'type': 'Mesh', 'vertices': [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]], 'faces':
        [array([4, 5, 1, 0], dtype=int32), array([0, 2, 6, 4], dtype=int32),
        array([6, 7, 5, 4], dtype=int32), array([0, 1, 3, 2], dtype=int32),
        array([5, 7, 3, 1], dtype=int32), array([2, 3, 7, 6], dtype=int32)]}
        >>> assert np.allclose(
        ...   cube.inertia_tensor,
        ...   np.diag([16. / 3., 16. / 3., 16. / 3.]))
        >>> assert np.isclose(cube.iq, np.pi / 6.)
        >>> cube.neighbors
        [array([1, 2, 3, 4]), array([0, 2, 3, 5]), array([0, 1, 4, 5]),
        array([0, 1, 4, 5]), array([0, 2, 3, 5]), array([1, 2, 3, 4])]
        >>> cube.normals
        array([[ 0.,  0.,  1.],
               [ 0.,  1., -0.],
               [-1.,  0.,  0.],
               [ 1., -0.,  0.],
               [ 0., -1.,  0.],
               [ 0.,  0., -1.]])
        >>> cube.num_faces
        6
        >>> cube.num_vertices
        8
        >>> assert np.isclose(cube.surface_area, 24.0)
        >>> cube.vertices
        array([[ 1.,  1.,  1.],
               [ 1., -1.,  1.],
               [ 1.,  1., -1.],
               [ 1., -1., -1.],
               [-1.,  1.,  1.],
               [-1., -1.,  1.],
               [-1.,  1., -1.],
               [-1., -1., -1.]])
        >>> assert np.isclose(cube.volume, 8.0)

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
            "vertices": self.vertices.tolist(),
            "faces": self.faces,
        }

    def _rescale(self, scale):
        """Multiply length scale.

        Args:
            scale (float):
                Scale factor.
        """
        self._vertices *= scale
        self._equations[:, 3] *= scale

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
        """:math:`(N, 3)` :class:`numpy.ndarray`: Get the vertices of the polyhedron."""
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
    def volume(self, value):
        scale = (value / self.volume) ** (1 / 3)
        self._rescale(scale)

    def get_face_area(self, faces=None):
        """Get the total surface area of a set of faces.

        Args:
            faces (int, sequence, or None):
                The index of a face or a set of face indices for which to
                find the area. If None, finds the area of all faces (Default
                value: None).

        Returns:
            :class:`numpy.ndarray`: The area of each face.

        Example:
            >>> cube = coxeter.shapes.ConvexPolyhedron(
            ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
            ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]])
            >>> cube = coxeter.shapes.Polyhedron(
            ...   vertices=cube.vertices,faces=cube.faces)
            >>> import numpy as np
            >>> assert np.allclose(
            ...   cube.get_face_area([1, 2, 3]),
            ...   [4., 4., 4.])

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

    @surface_area.setter
    def surface_area(self, value):
        if value > 0:
            scale = np.sqrt(value / self.surface_area)
            self._rescale(scale)
        else:
            raise ValueError("Surface area must be greater than zero.")

    def _surface_triangulation(self):
        """Generate a triangulation of the surface of the polyhedron.

        This algorithm constructs Polygons from each of the faces and then
        triangulates each of these to provide a total triangulation.
        """
        for face in self.faces:
            yield from polytri.triangulate(self.vertices[face])

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
    def centroid(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape.

        The centroid is computed using the algorithm described in
        :cite:`Eberly2002`.
        """  # noqa: E501
        # We could call self.volume, but this algorithm gets it for free as
        # part of the centroid integral so we might as well use it.
        volume = 0
        center = np.zeros(3)

        # >90% of the time is spent in generating the surface triangulation, so
        # there's not much to be gained by vectorizing the loop.
        for triangle in self._surface_triangulation():
            v0, v1, v2 = triangle
            v01 = v1 - v0
            v02 = v2 - v0
            # numpy.cross is relatively slow for a single vector cross product.
            normal = [
                v01[1] * v02[2] - v01[2] * v02[1],
                v01[2] * v02[0] - v01[0] * v02[2],
                v01[0] * v02[1] - v01[1] * v02[0],
            ]

            t0 = v0 + v1
            f1 = t0 + v2
            t1 = v0 * v0
            t2 = t1 + v1 * t0
            f2 = t2 + v2 * f1

            # This could equivalently use the y or z components.
            volume += normal[0] * f1[0]
            center += normal * f2

        return np.array(center) / volume / 4

    @centroid.setter
    def centroid(self, value):
        self._vertices += np.asarray(value) - self.centroid
        self._find_equations()

    @property
    def bounding_sphere(self):
        """:class:`~.Sphere`: Get the polyhedron's bounding sphere."""
        warnings.warn(
            "The bounding_sphere property is deprecated, use "
            "minimal_bounding_sphere instead",
            DeprecationWarning,
        )

        return self.minimal_bounding_sphere

    @property
    def minimal_bounding_sphere(self):
        """:class:`~.Sphere`: Get the polyhedron's bounding sphere."""
        if not MINIBALL:
            raise ImportError(
                "The miniball module must be installed. It can "
                "be installed as an extra with coxeter (e.g. "
                'with "pip install coxeter[bounding_sphere]") or '
                'directly from PyPI using "pip install miniball".'
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
        """:class:`~.Sphere`: Get the polyhedron's circumsphere.

        A `circumsphere
        <https://en.wikipedia.org/wiki/Circumscribed_sphere>`__ must touch
        all the points of the polyhedron. A circumsphere exists if and only if
        there is a point equidistant from all the vertices. The circumsphere is
        found by finding the least squares solution of the overdetermined system
        of linear equations defined by this constraint, and the circumsphere
        only exists if the resulting solution has no residual.

        Raises:
            RuntimeError: If no circumsphere exists for this polyhedron.
        """
        # The circumsphere is defined by center C and radius r. For vertex i
        # with position r_i, dot(r_i - C, r_i - C) = r^2, which is equivalent
        # to dot(r_i, r_i) - 2 dot(C, r_i) + dot(C, C) = r^2, a system of
        # quadratic equations. If we choose r_0 as the origin, then dot(C, C) =
        # r^2 and we instead have the linear equations dot(p_i, p_i) / 2 =
        # dot(C, p_i) where p_i = r_i - r_0. This is the set of equations that
        # we solve.
        points = self.vertices[1:] - self.vertices[0]
        half_point_lengths = np.sum(points * points, axis=1) / 2
        x, resids, _, _ = np.linalg.lstsq(points, half_point_lengths, None)
        if len(self.vertices) > 4 and not np.isclose(resids, 0):
            raise RuntimeError("No circumsphere for this polyhedron.")

        return Sphere(np.linalg.norm(x), x + self.vertices[0])

    @property
    def circumsphere_radius(self):
        """float: Get the radius of the polygon's circumsphere."""
        return self.circumsphere.radius

    @circumsphere_radius.setter
    def circumsphere_radius(self, value):
        self._rescale(value / self.circumsphere_radius)

    @property
    def insphere(self):
        """:class:`~.Sphere`: Get the polyhedron's insphere.

        Note:
            The insphere of a polyhedron is defined as the sphere contained within
            the polyhedron that is tangent to all its faces. This condition
            uniquely defines the sphere, if it exists. The set of equations
            defined by this equation is solved using a least squares approach,
            with the magnitude of the residual used to determine whether or not
            the insphere exists.

        """
        # The insphere is defined by center C and radius r. For face i
        # defined by its unit normal n_i and any point in the plane (choose a
        # vertex v_i for convenience), we must have dot(C + r n_i - v_i, n_i) = 0.
        # Defining the vector Cr = (C_x, C_y, C_z, r), and the augmented
        # normals m = (n_x, n_y, n_z, 1), rearranging gives the equations that
        # we solve: dot(m_i, cr) = dot(n_i, v_i).
        first_vertices = np.array([verts[0] for verts in self.faces])
        b = np.sum(self.normals * self.vertices[first_vertices], axis=-1)
        a = np.hstack((self.normals, np.ones((self.num_faces, 1))))
        x, resids, _, _ = np.linalg.lstsq(a, b, None)
        if len(self.vertices) > 4 and not np.isclose(resids, 0):
            raise RuntimeError("No insphere for this polyhedron.")

        return Sphere(x[3], x[:3])

    @property
    def insphere_radius(self):
        """float: Get the radius of the polygon's insphere."""
        return self.insphere.radius

    @insphere_radius.setter
    def insphere_radius(self, value):
        self._rescale(value / self.insphere_radius)

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

        Example:
            >>> cube = coxeter.shapes.ConvexPolyhedron(
            ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
            ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]])
            >>> cube = coxeter.shapes.Polyhedron(
            ...   vertices=cube.vertices, faces=cube.faces)
            >>> import numpy as np
            >>> assert np.isclose(cube.get_dihedral(1, 2), np.pi / 2.)

        """
        if b not in self.neighbors[a]:
            raise ValueError("The two faces are not neighbors.")
        n1, n2 = self._equations[[a, b], :3]
        return np.arccos(np.dot(-n1, n2))

    def plot(self, ax=None, plot_verts=False, label_verts=False):
        """Plot the polyhedron.

        Note that the ``ax`` argument should be a 3D axes object; passing in
        a 2D axes object will result in wrong behavior.

        Args:
            ax (:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`):
                The axes on which to draw the polyhedron. Axes will be
                created if this is None (Default value: None).
            plot_verts (bool):
                If True, scatter points will be added at the vertices
                (Default value: False).
            label_verts (bool):
                If True, vertex indices will be added next to the vertices
                (Default value: False).
        """
        ax = _generate_ax(ax, axes3d=True)

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
        _set_3d_axes_equal(ax)

    def diagonalize_inertia(self):
        """Orient the shape along its principal axes.

        The principal axes of a shape are defined by the eigenvectors of the inertia
        tensor. This method computes the inertia tensor of the shape, diagonalizes it,
        and then rotates the shape by the corresponding orthogonal transformation.

        Example:
            >>> cube = coxeter.shapes.ConvexPolyhedron(
            ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
            ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]])
            >>> cube = coxeter.shapes.Polyhedron(
            ...   vertices=cube.vertices, faces=cube.faces)
            >>> cube.diagonalize_inertia()
            >>> cube.vertices
            array([[ 1.,  1.,  1.],
                   [ 1., -1.,  1.],
                   [ 1.,  1., -1.],
                   [ 1., -1., -1.],
                   [-1.,  1.,  1.],
                   [-1., -1.,  1.],
                   [-1.,  1., -1.],
                   [-1., -1., -1.]])

        """
        principal_moments, principal_axes = np.linalg.eigh(self.inertia_tensor)
        self._vertices = np.dot(self._vertices, principal_axes)

    def compute_form_factor_amplitude(self, q, density=1.0):  # noqa: D102
        """Calculate the form factor intensity.

        The form factor amplitude of a polyhedron is computed according to the
        derivation provided in this dissertation:
        https://deepblue.lib.umich.edu/handle/2027.42/120906.
        In brief, two applications of Stokes theorem (or to use the names more
        familiar from elementary vector calculus, the application of the divergence
        theorem followed by the classic Kelvin-Stokes theorem) are used to reduce the
        volume integral over a polyhedron into a series of line integrals around the
        boundaries of each polygonal face.

        For more generic information about form factors, see
        `Shape.compute_form_factor_amplitude`.
        """
        # If we wish to use this formula more productively in the future, it may be
        # worthwhile to compare against the method proposed here:
        # https://journals.iucr.org/j/issues/2017/05/00/fs5152/
        # That paper directly performs the Fourier integrals rather than attempting to
        # reduce their dimensionality first.
        #
        # Since the polyhedron is represented as a collection of vertices that are
        # translated when a new center (and if we implement rotation, it will probably
        # be implemented as a direct change to the vertices as well), there is no need
        # to treat translations and rotations in any special manner. However, if this
        # ever changes, the relevant changes would be to:
        #   1) Rotate all the k vectors by the _inverse_ of the orientation, i.e.
        #      k = rowan.rotate(rowan.conjugate(self.orientation), k)
        #   2) Rotate and translate the final form factors, i.e.
        #      for i, k in enumerate(q):
        #          form_factor[i] *= np.exp(-1j * np.dot(
        #              k, rowan.rotate(rowan.inverse(self.orientation), self.center)))
        form_factor = np.zeros((len(q),), dtype=np.complex128)

        # Handle zeros q vector cases up front to allow looping over faces without
        # double checking internally.
        q_sqs = np.sum(q * q, axis=-1)
        zero_q = np.isclose(q_sqs, 0)
        form_factor[zero_q] = self.volume

        for face, eqn in zip(self.faces, self._equations):
            # Calculate each face's form factor as a polygon. This implementation aims
            # at clarity over efficiency (a true form factor calculation would need to
            # efficiently loop over many shapes anyway). Note that we have to negate the
            # distance in the line below due to our equation sign convention (see
            # _find_equations).
            face_normal, d = eqn[:3], -eqn[3]
            face_polygon = Polygon(self.vertices[face], face_normal)
            face_form_factors = face_polygon.compute_form_factor_amplitude(q[~zero_q])

            # Translate the calculation into the reference frame of the polyhedron.
            qs_dot_norm = np.dot(q[~zero_q], face_normal)
            exp_qr = np.exp(-1j * qs_dot_norm * d)
            form_factor[~zero_q] += (
                qs_dot_norm * (1j * face_form_factors * exp_qr)
            ) / q_sqs[~zero_q]

        return form_factor

    def is_inside(self, points):
        """Determine whether points are contained in this polyhedron.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                polyhedron.
        """
        # polytri generates a triangulation directly from the vertices. We need
        # to map this back to index positions to feed to the polyhedron winding
        # number calculation.
        vertex_to_index = {tuple(v): i for i, v in enumerate(self.vertices)}

        triangles = [
            [vertex_to_index[tuple(v)] for v in triangle]
            for triangle in self._surface_triangulation()
        ]

        winding_number_calculator = PolyInside(triangles, self.vertices)

        def _check_inside(p):
            """Check if point is inside, including boundary points.

            The polyhedron check will will raise a ValueError for points on the
            boundary, which we want to be inside.
            """
            try:
                return winding_number_calculator.winding_number(p) != 0
            except ValueError as e:
                if str(e) in (
                    "vertex coincides with origin",
                    "vertices collinear with origin",
                    "vertices coplanar with origin",
                ):
                    return True
                else:
                    raise

        return np.array([_check_inside(p) for p in np.atleast_2d(points)])

    def __repr__(self):
        return (
            f"coxeter.shapes.Polyhedron(vertices={self.vertices.tolist()}, "
            f"faces={np.asarray(self.faces).tolist()})"
        )
