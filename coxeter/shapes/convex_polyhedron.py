# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines a convex polyhedron."""

import warnings
from collections import defaultdict
from numbers import Number

import numpy as np
from scipy.spatial import ConvexHull

from .polyhedron import Polyhedron
from .sphere import Sphere


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
        fast (bool, optional):
            Creation mode for the polyhedron. fast=False (default) will perform all
            checks and precalculate most properties. fast=True  will precalculate some
            properties, but will not find face neighbors or sort face indices. These
            calculations will instead be performed when required by another method.
            (Default value: False).

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

    def __init__(self, vertices, fast=True):
        self._vertices = np.array(vertices, dtype=np.float64)
        self._ndim = self._vertices.shape[1]
        self._convex_hull = ConvexHull(self._vertices)
        self._faces_are_convex = True

        if not len(self._convex_hull.vertices) == len(self._vertices):
            warnings.warn("Not all vertices used for hull.", RuntimeWarning)

            # Remove internal points
            self._vertices = self._vertices[self._convex_hull.vertices]

        # Transfer data in from convex hull, then clean up the results.
        self._consume_hull()
        self._combine_simplices()

        # Sort simplices. This method also calculates simplex equations and the centroid
        self._sort_simplices()

        # If mode is NOT fast, perform additional initializiation steps
        if fast is False:
            self._find_coplanar_simplices()
            self.sort_faces()
        else:
            self._faces_are_sorted = False

    def _consume_hull(self):
        """Extract data from ConvexHull.

        Data is moved from convex hull into private variables. This method deletes the
        original hull in order to avoid double storage, and chec.
        """
        assert (
            self._ndim == self._convex_hull.ndim
        ), "Input points are coplanar or close to coplanar."

        self._simplices = self._convex_hull.simplices[:]
        self._simplex_equations = self._convex_hull.equations[:]
        self._simplex_neighbors = self._convex_hull.neighbors[:]
        self._volume = self._convex_hull.volume
        self._area = self._convex_hull.area
        self._maximal_extents = np.array(
            [self._convex_hull.min_bound, self._convex_hull.max_bound]
        )

        # Clean up the result.
        del self._convex_hull

    def _combine_simplices(self, rounding: int = 15):
        """Combine simplices into faces, merging based on simplex equations.

        Coplanar faces will have identical equations (within rounding tolerance). Values
        should be slightly larger than machine epsilon (e.g. rounding=15 for ~1e-15)

        Args:
            rounding (int, optional):
                Integer number of decimal places to round equations to.
                (Default value: 15).

        """
        equation_groups = defaultdict(list)

        # Iterate over all simplex equations
        for i, equation in enumerate(self._simplex_equations):
            # Convert to hashable key
            equation_key = tuple(equation.round(rounding))

            # Store vertex indices from the new simplex under the correct equation key
            equation_groups[equation_key].extend(self._simplices[i])

        # Combine elements with the same plan equation and remove duplicate indices
        ragged_faces = [
            np.fromiter(set(group), np.int32) for group in equation_groups.values()
        ]
        self._faces = ragged_faces
        self._equations = np.array(list(equation_groups.keys()))

    def _find_coplanar_simplices(self, rounding: int = 15):
        """
        Get lists of simplex indices for coplanar simplices.

        Args:
            rounding (int, optional):
                Integer number of decimal places to round equations to.
                (Default value: 15).


        """
        # Combine simplex indices
        equation_groups = defaultdict(list)

        # Iterate over all simplex equations
        for i, equation in enumerate(self._simplex_equations):
            # Convert equation into hashable tuple
            equation_key = tuple(equation.round(rounding))
            equation_groups[equation_key].append(i)
        ragged_coplanar_indices = [
            np.fromiter(set(group), np.int32) for group in equation_groups.values()
        ]

        self._coplanar_simplices = ragged_coplanar_indices

    def _sort_simplices(self):
        """Reorder simplices counterclockwise relatative to the plane they lie on.

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
        """Internal method to calculate the signed volume of the polyhedron.

        This class splits the shape into tetrahedra, then sums their contributing
        volumes. The external volume property will always be a positive value, but
        accessing the signed volume can be useful for some mathematical operations.

        Returns:
            float: Signed volume of the polyhedron.
        """
        signed_volume = np.sum(np.linalg.det(self._vertices[self._simplices]) / 6)
        self._volume = abs(signed_volume)
        return signed_volume

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
        return np.all(self._point_plane_distances(points) <= 0, axis=1)

    @property
    def insphere_from_center(self):
        """:class:`~.Sphere`: Get the largest concentric inscribed sphere."""
        warnings.warn(
            "The insphere_from_center property is deprecated, use "
            "maximal_centered_bounded_sphere instead",
            DeprecationWarning,
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
