# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines a convex polyhedron."""

import warnings

import numpy as np
from scipy.spatial import ConvexHull

from .polyhedron import Polyhedron
from .sphere import Sphere


class ConvexPolyhedron(Polyhedron):
    """A convex polyhedron.

    A convex polyhedron is defined as the convex hull of its vertices. The
    class is a simple extension of :class:`~.Polyhedron` that builds the
    faces from the simplices of the convex hull. This class also includes
    various additional properties that can be used to characterize the
    geometric features of the polyhedron.

    Args:
        vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
            The vertices of the polyhedron.

    Example:
        >>> cube = coxeter.shapes.ConvexPolyhedron(
        ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
        ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]])
        >>> import numpy as np
        >>> assert np.isclose(cube.asphericity, 1.5)
        >>> bounding_sphere = cube.minimal_bounding_sphere
        >>> assert np.isclose(bounding_sphere.radius, np.sqrt(3))
        >>> cube.center
        array([0., 0., 0.])
        >>> circumsphere = cube.circumsphere
        >>> assert np.isclose(circumsphere.radius, np.sqrt(3))
        >>> cube.faces
        [array([4, 5, 1, 0], dtype=int32), array([0, 2, 6, 4], dtype=int32),
        array([6, 7, 5, 4], dtype=int32), array([0, 1, 3, 2], dtype=int32),
        array([5, 7, 3, 1], dtype=int32), array([2, 3, 7, 6], dtype=int32)]
        >>> cube.gsd_shape_spec
        {'type': 'ConvexPolyhedron', 'vertices': [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]}
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
        array([[-0., -0.,  1.],
               [-0.,  1., -0.],
               [-1.,  0., -0.],
               [ 1., -0., -0.],
               [-0., -1.,  0.],
               [-0., -0., -1.]])
        >>> cube.num_faces
        6
        >>> cube.num_vertices
        8
        >>> cube.surface_area
        24.0
        >>> assert np.isclose(cube.tau, 3. / 8. * np.pi)
        >>> cube.vertices
        array([[ 1.,  1.,  1.],
               [ 1., -1.,  1.],
               [ 1.,  1., -1.],
               [ 1., -1., -1.],
               [-1.,  1.,  1.],
               [-1., -1.,  1.],
               [-1.,  1., -1.],
               [-1., -1., -1.]])
        >>> assert np.isclose(cube.volume, 8.)

    """

    def __init__(self, vertices):
        hull = ConvexHull(vertices)
        super(ConvexPolyhedron, self).__init__(vertices, hull.simplices, True)
        self.merge_faces()

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
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
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
