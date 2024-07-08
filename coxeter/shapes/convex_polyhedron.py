# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Defines a convex polyhedron."""

import warnings
from numbers import Number

import numpy as np
import rowan
from scipy.spatial import ConvexHull

from .polyhedron import Polyhedron
from .sphere import Sphere
from .utils import translate_inertia_tensor


class ConvexPolyhedron(Polyhedron):
    """A convex polyhedron.

    A convex polyhedron is defined as the convex hull of its vertices. The
    class is an extension of :class:`~.Polyhedron` that builds the
    faces from the simplices of the convex hull. Simplices are stored and class methods
    are optimized to make use of the triangulation, as well as  special properties of
    convex solids in three dimensions. This class also includes various additional
    properties that can be used to characterize geometric features of the polyhedron.

    Args:
        vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
            The vertices of the polyhedron.

    Example:
        >>> cube = coxeter.shapes.ConvexPolyhedron(
        ...   [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        ...    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
        >>> import numpy as np
        >>> assert np.isclose(cube.asphericity, 1.5)
        >>> bounding_sphere = cube.minimal_bounding_sphere
        >>> assert np.isclose(bounding_sphere.radius, np.sqrt(3))
        >>> cube.centroid
        array([0., 0., 0.])
        >>> circumsphere = cube.circumsphere
        >>> assert np.isclose(circumsphere.radius, np.sqrt(3))
        >>> cube.faces
        [array([0, 2, 6, 4], dtype=int32), array([0, 4, 5, 1], dtype=int32),
        array([4, 6, 7, 5], dtype=int32), array([0, 1, 3, 2], dtype=int32),
        array([2, 3, 7, 6], dtype=int32), array([1, 5, 7, 3], dtype=int32)]
        >>> cube.gsd_shape_spec
        {'type': 'ConvexPolyhedron', 'vertices': [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]}
        >>> assert np.allclose(
        ...   cube.inertia_tensor,
        ...   np.diag([16. / 3., 16. / 3., 16. / 3.]))
        >>> sphere = cube.maximal_centered_bounded_sphere
        >>> sphere.radius
        1.0
        >>> assert np.isclose(cube.iq, np.pi / 6.)
        >>> assert np.isclose(cube.mean_curvature, 1.5)
        >>> cube.neighbors
        [array([1, 2, 3, 4]), array([0, 2, 3, 5]), array([0, 1, 4, 5]),
        array([0, 1, 4, 5]), array([0, 2, 3, 5]), array([1, 2, 3, 4])]
        >>> cube.normals
        array([[-0., -0., -1.],
               [ 0., -1.,  0.],
               [ 1., -0., -0.],
               [-1., -0., -0.],
               [ 0.,  1., -0.],
               [-0., -0.,  1.]])
        >>> cube.num_faces
        6
        >>> cube.num_vertices
        8
        >>> cube.surface_area
        24.0
        >>> assert np.isclose(cube.tau, 3. / 8. * np.pi)
        >>> cube.vertices
        array([[-1., -1., -1.],
               [-1., -1.,  1.],
               [-1.,  1., -1.],
               [-1.,  1.,  1.],
               [ 1., -1., -1.],
               [ 1., -1.,  1.],
               [ 1.,  1., -1.],
               [ 1.,  1.,  1.]])
        >>> assert np.isclose(cube.volume, 8.)

    """

    def __init__(self, vertices):
        self._vertices = np.array(vertices, dtype=np.float64)
        self._ndim = self._vertices.shape[1]
        hull = ConvexHull(self._vertices)
        self._faces_are_convex = True

        if not len(hull.vertices) == len(self._vertices):
            # Identify vertices that do not contribute to the convex hull
            nonconvex_vertices = sorted(
                set(range(len(self._vertices))) - set(hull.vertices)
            )

            raise ValueError(
                "Input vertices must be a convex set. "
                + f"Vertices {nonconvex_vertices} are inside the shape (or coplanar)."
            )

        # Transfer data in from convex hull, then clean up the results.
        self._consume_hull(hull)
        self._combine_simplices()

        # Sort simplices. This method also calculates simplex equations and the centroid
        self._sort_simplices()
        self.sort_faces()

    def _consume_hull(self, hull):
        """Extract data from ConvexHull.

        Data is moved from convex hull into private variables.
        """
        assert (
            self._ndim == hull.ndim
        ), "Input points are coplanar or close to coplanar."

        self._simplices = hull.simplices[:]
        self._simplex_equations = hull.equations[:]
        self._simplex_neighbors = hull.neighbors[:]
        self._volume = hull.volume
        self._area = hull.area
        self._maximal_extents = np.array([hull.min_bound, hull.max_bound])

    def _combine_simplices(self, tol: float = 2e-15):
        """Combine simplices into faces, merging based on simplex equations.

        Coplanar faces will have identical equations (within rounding tolerance). Values
        should be about an order of magnitude greater than machine epsilon.

        Args:
            tol (float, optional):
                Floating point tolerance within which values are considered identical.
                (Default value: 2e-15).

        """
        is_coplanar = np.all(
            np.abs(self._simplex_equations[:, None] - self._simplex_equations) < tol,
            axis=2,
        )
        coplanar_indices = [[] for _ in range(len(self._simplices))]

        # Iterate over coplanar indices to build face index lists
        for face, index in zip(*is_coplanar.nonzero()):
            coplanar_indices[face].append(index)

        # Remove duplicate faces, then sort the face indices by their minimum value
        coplanar_indices = sorted(set(map(tuple, coplanar_indices)), key=lambda x: x[0])

        # Extract vertex indices from simplex indices and remove duplicates
        faces = [np.unique(self.simplices[[ind]]) for ind in coplanar_indices]
        self._faces = faces

        # Copy the simplex equation for one of the simplices on each face
        self._equations = self._simplex_equations[
            [equation_index[0] for equation_index in coplanar_indices]
        ]
        # Convert the simplex indices to numpy arrays and save
        self._coplanar_simplices = list(map(np.array, coplanar_indices))

    def _sort_simplices(self):
        """Reorder simplices counterclockwise relative to the plane they lie on.

        This does NOT change the *order* of simplices in the list.
        """
        # Get correct-quadrant angles about the simplex normal
        vertices = self._vertices[self._simplices]

        # Get the absolute angles of each vertex and fit to unit circle
        angles = np.arctan2(vertices[:, 1], vertices[:, 0])
        angles = np.mod(angles - angles[0], 2 * np.pi)

        # Calculate distances
        distances = np.linalg.norm(vertices, axis=1)

        # Create a tuple of distances and angles to use for lexicographical sorting
        vert_order = np.lexsort((distances, angles))

        # Apply orientation reordering to every simplex
        self._simplices = np.take_along_axis(
            self._simplices, np.argsort(vert_order), axis=1
        )

        # Compute N,3,2 array of simplex edges from an N,3 array of simplices
        def _tri_to_edge_tuples(simplices, shift=-3):
            # edges = np.column_stack((simplices, np.roll(simplices, shift, axis=1)))
            edges = np.roll(
                np.repeat(simplices, repeats=2, axis=1), shift=shift, axis=1
            )
            edges = edges.reshape(-1, 3, 2)
            # Convert to tuples for fast comparison
            return [[tuple(edge) for edge in triedge] for triedge in edges]

        simplex_edges = _tri_to_edge_tuples(self._simplices, shift=-1)

        # Now, reorient simplices to match the orientation of simplex 0
        visited_simplices = []
        remaining_simplices = [0]
        while len(remaining_simplices):
            current_simplex = remaining_simplices[-1]
            visited_simplices.append(current_simplex)
            remaining_simplices.pop()

            # Search for common edges between pairs of simplices, then check the
            # ordering of the edge to determine relative face orientation.
            current_edges = simplex_edges[current_simplex]
            for neighbor in self._simplex_neighbors[current_simplex]:
                if neighbor in visited_simplices:
                    continue
                remaining_simplices.append(neighbor)

                # Two faces can only share a single edge (otherwise they would
                # be coplanar), so we can break as soon as we find the
                # neighbor. Flip the neighbor if the edges are identical.
                for edge in simplex_edges[neighbor]:
                    if edge in current_edges:
                        self._simplices[neighbor] = self._simplices[neighbor][::-1]
                        simplex_edges[neighbor] = [
                            (j, i) for i, j in simplex_edges[neighbor]
                        ][::-1]
                        break
                    elif edge[::-1] in current_edges:
                        break
                visited_simplices.append(neighbor)

        # Flip if calcualted volume is negative
        if self._calculate_signed_volume() < 0:
            self._simplices = self._simplices[:, ::-1, ...]

        # Recompute simplex equations and centroid from the new triangulation.
        self._find_simplex_equations()
        self._centroid_from_triangulated_surface()

    @property
    def volume(self):
        """float: Get or set the polyhedron's volume."""
        return self._volume

    @volume.setter
    def volume(self, value: Number):
        scale_factor = np.cbrt(value / self._volume)
        self._rescale(scale_factor)

    @property
    def surface_area(self):
        """float: Get or set the surface area."""
        return self._area

    @surface_area.setter
    def surface_area(self, value: Number):
        scale_factor = np.sqrt(value / self._area)
        self._rescale(scale_factor)

    def _calculate_surface_area(self):
        new_area = self._find_triangle_array_area(self._vertices[self._simplices])
        self._area = new_area
        return self._area

    def _rescale(self, scale_factor: Number):
        """Scale polytope by changing the length of the edges.

        Args:
            scale_factor (int or float):
                Multiplier to scale edges by. Volume and surface area setters preconvert
                the scale_factor to the correct value for the desired property.
        """
        self._vertices *= scale_factor
        self._equations[:, 3] *= scale_factor
        self._simplex_equations[:, 3] *= scale_factor
        self._volume = self._volume * scale_factor**3
        self._area = self._area * scale_factor**2

        # Recalculate centroid of shape
        self._centroid_from_triangulated_surface()

    def _calculate_signed_volume(self):
        """Calculate the signed volume of the polyhedron.

        This class splits the shape into tetrahedra, then sums their contributing
        volumes. The external volume property will always be a positive value, but
        accessing the signed volume can be useful for some mathematical operations.

        Returns
        -------
            float: Signed volume of the polyhedron.
        """
        signed_volume = np.sum(np.linalg.det(self._vertices[self._simplices]) / 6)
        self._volume = abs(signed_volume)
        return signed_volume

    def get_face_area(self, face=None):
        """Get the total surface area of a set of faces.

        Args:
            faces (int, sequence, or None):
                The index of a face or a set of face indices for which to
                find the area. If None, finds the area of all faces.
                (Default value: None).

        Returns
        -------
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
        self._simplex_areas = self._find_triangle_array_area(
            self._vertices[self._simplices], sum_result=False
        )
        if face is None:
            # Return face area for every face.
            return [
                np.sum(self._simplex_areas[self._coplanar_simplices[fac]])
                for fac in range(self.num_faces)
            ]
        elif face == "total":
            # Return total surface area
            return np.sum(self._simplex_areas)
        elif hasattr(face, "__len__"):
            # Return face areas for a list of faces
            return [
                np.sum(self._simplex_areas[self._coplanar_simplices[fac]])
                for fac in face
            ]
        else:
            # Return face area of a single face
            return np.sum(self._simplex_areas[self._coplanar_simplices[face]])

    def _find_equations(self):
        """Find the plane equations of the polyhedron faces."""
        # This method only takes the first three items from each face, so it is
        # unaffected by face structure or ordering.
        point_on_face_indices = []
        for face in self.faces:
            point_on_face_indices.append(face[0:3])
        vertices = self._vertices[point_on_face_indices]

        # Calculate the directions of the normals
        v1 = vertices[:, 2] - vertices[:, 1]
        v2 = vertices[:, 0] - vertices[:, 1]
        normals = np.cross(v1, v2)

        # Normalize the normals
        norms = np.linalg.norm(normals, axis=1)
        normals /= norms[:, None]

        _equations = np.empty((len(self._faces), 4))
        _equations[:, :3] = normals
        _equations[:, 3] = -np.einsum(
            "ij,ij->i", normals, vertices[:, 0]
        )  # dot product
        self._equations = _equations

    @property
    def equations(self):
        """:math:`(N, 4)` :class:`numpy.ndarray`: Get plane equations for each face.

        Sign convention matches Scipy Convex Hull (ax + by + cz + d = 0).
        """
        return self._equations

    @property
    def normals(self):
        """:math:`(N, 3)` :class:`numpy.ndarray`: Get normal vectors for each face."""
        return self._equations[:, :3]

    def _find_simplex_equations(self):
        """Find the plane equations of the polyhedron simplices."""
        abc = self._vertices[self._simplices]
        a = abc[:, 0]
        b = abc[:, 1]
        c = abc[:, 2]
        n = np.cross((b - a), (c - a))
        n /= np.linalg.norm(n, axis=1)[:, None]

        _equations = np.empty((len(self._simplices), 4))
        _equations[:, :3] = n
        _equations[:, 3] = -np.einsum("ij,ij->i", n, a)
        self._simplex_equations = _equations

    @property
    def centroid(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the center of mass.

        The centroid is calculated using the curl theorem over the surface simplices.
        """
        return self._centroid

    @centroid.setter
    def centroid(self, value):
        assert len(value) == 3, "Centroid must be a point in 3-space."
        self._vertices += np.asarray(value) - self.centroid
        self._find_equations()
        self._find_simplex_equations()
        self._centroid_from_triangulated_surface()
        self._calculate_signed_volume()

    def _centroid_from_triangulated_surface(self):
        abc = self._vertices[self._simplices]
        a = abc[:, 0]
        b = abc[:, 1]
        c = abc[:, 2]
        n = np.cross((b - a), (c - a))
        self._centroid = (
            1
            / (48 * self._volume)
            * np.sum(n * ((a + b) ** 2 + (b + c) ** 2 + (a + c) ** 2), axis=0)
        )

    @property
    def faces(self):
        """list(:class:`numpy.ndarray`): Get the polyhedron's faces."""
        return self._faces

    def _find_face_centroids(self):
        simplex_centroids = np.mean(self._vertices[self._simplices], axis=1)  # (N,3)
        self._simplex_areas = self._find_triangle_array_area(
            self._vertices[self._simplices], sum_result=False
        )  # (N,)
        self._face_centroids = []
        for face in self._coplanar_simplices:
            self._face_centroids.append(
                np.sum(
                    simplex_centroids[face] * self._simplex_areas[face][:, None],
                    axis=0,
                )
                / np.sum(self._simplex_areas[face]),  # Rescale by area of face
            )
        self._face_centroids = np.array(self._face_centroids)

    @property
    def face_centroids(self):
        """Calculate the centroid (center of mass) of each polygonal face.

        Returns
        -------
            :math:`(N,3)` :class:`numpy.ndarray`:
                Array of centroids for each face.
        """
        self._find_face_centroids()
        return self._face_centroids

    @property
    def neighbors(self):
        r"""list(:class:`numpy.ndarray`): Get neighboring pairs of faces.

        The neighbors are provided as a list where the :math:`i^{\text{th}}`
        element is an array of indices of faces that are neighbors of face
        :math:`i`.
        """
        return self._neighbors

    def sort_faces(self):
        """Reorder faces counterclockwise relatative to the plane they lie on.

        This does NOT change the *order* of faces in the list.
        """
        # Get correct-quadrant angles about the face normal
        sorted_faces = []

        for i, face in enumerate(self._faces):
            vertices = self._vertices[face]

            # Rotate the face's points into the XY plane
            normal = self._equations[i][:3]
            rotation, _ = rowan.mapping.kabsch(
                [normal, -normal], [[0, 0, 1], [0, 0, -1]]
            )
            vertices = np.dot(vertices - np.mean(vertices, axis=0), rotation.T)

            # Get the absolute angles of each vertex and fit to unit circle
            angles = np.arctan2(vertices[:, 1], vertices[:, 0])
            angles = np.mod(angles - angles[0], 2 * np.pi)

            # Calculate distances
            distances = np.linalg.norm(vertices, axis=1)

            # Create a tuple of distances and angles to use for lexicographical sorting
            vert_order = np.lexsort((distances, angles))

            # Apply reordering to every simplex
            sorted_faces.append(face[vert_order])

        self._faces = sorted_faces
        self._find_neighbors()

    def _surface_triangulation(self):
        """Output the vertices of simplices composing the polyhedron's surface.

        Returns
        -------
            :math:`(N,3,3)` :class:`numpy.ndarray`:
                Array of vertices for simplices composing the polyhedron's surface.
        """
        return self.vertices[self._simplices]

    @property
    def simplices(self):
        """Output the vertex indices of simplices composing the polyhedron's surface.

        Returns
        -------
            :math:`(N,3)` :class:`numpy.ndarray`:
                Array of vertex indices of simplices making up the polyhedron's surface.
        """
        return self._simplices

    def _find_triangle_array_area(self, triangle_vertices, sum_result=True):
        """
        Get the areas of each triangle in an input array.

        Args:
            angles (:math:`(N, 3, 3)` :class:`numpy.ndarray`):
                Array of vertices for the triangles that will have their area computed.
            sum_result (bool):
                Whether the output should be summed.
                (Default value: True).

        Returns
        -------
            :math:`(N, )` :class:`numpy.ndarray` or float:
                Boolean array indicating which points are contained in thepolyhedron.
                If sum_result is True, a single value is returned.

        """
        v1 = triangle_vertices[:, 2] - triangle_vertices[:, 1]
        v2 = triangle_vertices[:, 0] - triangle_vertices[:, 1]
        unnormalized_normals = np.cross(v1, v2, axis=1)

        if sum_result:
            return np.sum(np.linalg.norm(unnormalized_normals, axis=1)) / 2
        else:
            return np.linalg.norm(unnormalized_normals, axis=1) / 2

    @property
    def inertia_tensor(self):
        """:math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor.

        The inertia tensor for convex shapes is computed using the algorithm described
        in :cite:`Messner1980`.

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
        centered and uncentered calculations.
        """

        def _quadrature_points(abc):
            """Compute quadrature points at which to evaluate the integral."""
            scalars = [
                [5 / 3, 5 / 3, 5 / 3],
                [[1], [1], [3]],
                [[3], [1], [1]],
                [[1], [3], [1]],
            ]

            q = np.zeros((abc.shape[0], 3, 4))

            for i in range(4):
                q[:, :, i] = np.sum(abc * scalars[i], axis=1)
            q /= 5
            return q

        # Define triangle array
        abc = self.vertices[self.simplices]
        if centered:
            abc -= self.centroid
        n = self._simplex_equations[:, :3]
        nt = n.T
        q = _quadrature_points(abc)
        q2 = q**2
        q3 = q**3

        # Define quadrature weights (from integrating a third order polynomial over a
        # triangular domain). The weights have been normalized such that their sum is 1.
        w = np.array([[-9 / 16, 25 / 48, 25 / 48, 25 / 48]]).T

        # Face simplex areas
        at = self._find_triangle_array_area(abc, sum_result=False) * 2

        # Calculate and sum on-diagonal contributions to the moment of inertia.
        def i_nn(nt, q3, w, at, sub):
            return np.einsum("ij, j, jik, kl ->", nt[sub, :], at, q3[:, sub, :], w) / 6

        # Calculate and sum off-diagonal contributions to the moment of inertia.
        def i_nm(n, q, q2, w, at, sub):
            return (
                -(
                    np.einsum(
                        "ij,ki,k,k->",
                        w,
                        q2[:, sub[0], :] * q[:, sub[1], :],
                        n[:, sub[0]],
                        at,
                    )
                    + np.einsum(
                        "ij,ki,k,k->",
                        w,
                        q[:, sub[0], :] * q2[:, sub[1], :],
                        n[:, sub[1]],
                        at,
                    )
                )
                / 8
            )

        i_xx = i_nn(nt, q3, w, at, sub=[1, 2])
        i_xy = i_nm(n, q, q2, w, at, sub=[0, 1])
        i_xz = i_nm(n, q, q2, w, at, sub=[0, 2])
        i_yy = i_nn(nt, q3, w, at, sub=[0, 2])
        i_yz = i_nm(n, q, q2, w, at, sub=[1, 2])
        i_zz = i_nn(nt, q3, w, at, sub=[0, 1])

        return np.array([[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]])

    def diagonalize_inertia(self):
        """Orient the shape along its principal axes.

        The principal axes of a shape are defined by the eigenvectors of the inertia
        tensor. This method computes the inertia tensor of the shape, diagonalizes it,
        and then rotates the shape by the corresponding orthogonal transformation.

        Example:
            >>> cube = coxeter.shapes.ConvexPolyhedron(
            ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
            ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]])
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
        _, principal_axes = np.linalg.eigh(self.inertia_tensor)
        self._vertices = np.dot(self._vertices, principal_axes)
        self._sort_simplices()

    @property
    def mean_curvature(self):
        r"""float: The integrated, normalized mean curvature.

        This quantity is calculated by the formula
        :math:`R = \sum_i (1/2) L_i (\pi - \phi_i) / (4 \pi)` with edge lengths
        :math:`L_i` and dihedral angles :math:`\phi_i` (see :cite:`Irrgang2017`
        for more information).
        """
        unnorm_r = 0
        for i, j, edge in self._get_face_intersections():
            phi = self.get_dihedral(i, j)
            edge_vector = self.vertices[edge[0]] - self.vertices[edge[1]]
            edge_length = np.linalg.norm(edge_vector)
            unnorm_r += edge_length * (np.pi - phi)
        return unnorm_r / (8 * np.pi)

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <gsd:shapes>`."""  # noqa: D401
        return {"type": "ConvexPolyhedron", "vertices": self.vertices.tolist()}

    @property
    def tau(self):
        r"""float: Get the parameter :math:`\tau = \frac{4\pi R^2}{S}`.

        This parameter is defined in :cite:`Naumann19841` and is closely
        related to the Pitzer acentric factor. This quantity appears relevant
        to the third and fourth virial coefficient for hard polyhedron fluids.
        """
        mc = self.mean_curvature
        return 4 * np.pi * mc * mc / self.surface_area

    @property
    def asphericity(self):
        """float: Get the asphericity as defined in :cite:`Irrgang2017`."""
        return self.mean_curvature * self.surface_area / (3 * self.volume)

    @property
    def num_edges(self):
        """int: Get the number of edges."""
        # Calculate number of edges from Euler Characteristic
        return self.num_vertices + self.num_faces - 2

    def is_inside(self, points):
        """Determine whether points are contained in this polyhedron.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns
        -------
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                polyhedron.
        """
        return np.all(self._point_plane_distances(points) <= 0, axis=1)

    @property
    def insphere_from_center(self):
        """:class:`~.Sphere`: Get the largest concentric inscribed sphere."""
        warnings.warn(
            "The insphere_from_center property is deprecated, use "
            "maximal_centered_bounded_sphere instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.maximal_centered_bounded_sphere

    @property
    def circumsphere_from_center(self):
        """:class:`~.Sphere`: Get the smallest circumscribed sphere centered at the centroid.

        The requirement that the sphere be centered at the centroid of the
        shape distinguishes this sphere from most typical circumsphere
        calculations.
        """  # noqa: E501
        warnings.warn(
            "The circumsphere_from_center property is deprecated, use "
            "minimal_centered_bounding_sphere instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.minimal_centered_bounding_sphere

    @property
    def minimal_centered_bounding_sphere(self):
        """:class:`~.Sphere`: Get the smallest bounding concentric sphere."""
        # The radius is determined by the furthest vertex from the center.
        return Sphere(
            np.linalg.norm(self.vertices - self.center, axis=-1).max(), self.center
        )

    @property
    def maximal_centered_bounded_sphere(self):
        """:class:`~.Sphere`: Get the largest bounded concentric sphere."""
        # The radius is determined by the furthest vertex from the center.
        center = self.center
        distances = self._point_plane_distances(center).squeeze()
        if any(distances > 0):
            raise ValueError(
                "The centroid is not contained in the shape. The "
                "insphere from center is not defined."
            )
        min_distance = -np.max(distances)
        return Sphere(min_distance, center)

    def _plato_primitive(self, backend):
        return backend.ConvexPolyhedra(
            positions=np.array([self.center]),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            colors=np.array([[0.5, 0.5, 0.5, 1]]),
            vertices=self.vertices,
        )
