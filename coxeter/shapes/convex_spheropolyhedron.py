# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines a convex spheropolyhedron.

A convex spheropolyhedron is defined by the Minkowski sum of a convex
polyhedron and a sphere of some radius.
"""

import numpy as np

from .base_classes import Shape3D
from .convex_polyhedron import ConvexPolyhedron


class ConvexSpheropolyhedron(Shape3D):
    """A convex spheropolyhedron.

    A convex spheropolyhedron is defined as a convex polyhedron plus a
    rounding radius. All properties of the underlying polyhedron (the
    vertices, the faces and their neighbors, etc.) can be accessed directly
    through :attr:`.polyhedron`.

    Args:
        vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
            The vertices of the underlying polyhedron.
        radius (float):
            The rounding radius of the spheropolyhedron.

    Example:
        >>> spherocube = coxeter.shapes.ConvexSpheropolyhedron(
        ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
        ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]],
        ...   radius=0.5)
        >>> spherocube.gsd_shape_spec
        {'type': 'ConvexPolyhedron', 'vertices': [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]], 'rounding_radius': 0.5}
        >>> cube = spherocube.polyhedron
        >>> cube.vertices
        array([[ 1.,  1.,  1.],
               [ 1., -1.,  1.],
               [ 1.,  1., -1.],
               [ 1., -1., -1.],
               [-1.,  1.,  1.],
               [-1., -1.,  1.],
               [-1.,  1., -1.],
               [-1., -1., -1.]])
        >>> spherocube.radius
        0.5
        >>> spherocube.surface_area
        45.991...
        >>> spherocube.vertices
        array([[ 1.,  1.,  1.],
               [ 1., -1.,  1.],
               [ 1.,  1., -1.],
               [ 1., -1., -1.],
               [-1.,  1.,  1.],
               [-1., -1.,  1.],
               [-1.,  1., -1.],
               [-1., -1., -1.]])
        >>> spherocube.volume
        25.235...

    """

    def __init__(self, vertices, radius):
        self._polyhedron = ConvexPolyhedron(vertices)
        self.radius = radius

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {
            "type": "ConvexPolyhedron",
            "vertices": self.polyhedron.vertices.tolist(),
            "rounding_radius": self.radius,
        }

    @property
    def polyhedron(self):
        """:class:`~.ConvexPolyhedron`: The underlying polyhedron."""
        return self._polyhedron

    @property
    def vertices(self):
        """Get the vertices of the spheropolyhedron."""
        return self.polyhedron.vertices

    def _rescale(self, scale):
        """Multiply length scale.

        Args:
            scale (float):
                Scale factor.
        """
        self.polyhedron._rescale(scale)
        self.radius *= scale

    @property
    def volume(self):
        """float: The volume."""
        # Compute the volume as the sum of 4 terms:
        # 1) The volume of the underlying polyhedron.
        # 2) The volume of the spherical caps on the vertices, which sum up to
        #    a single sphere with the spheropolyhedron's rounding radius.
        # 3) The volume of cylindrical wedges along the edges, which are
        #    computed using a standard cylinder formula then using the dihedral
        #    angle of the face to determine what fraction of the cylinder to
        #    include.
        # 4) The volume of the extruded faces, which is the surface area of
        #    each face multiplied by the rounding radius.
        v_poly = self.polyhedron.volume
        v_sphere = (4 / 3) * np.pi * self.radius ** 3
        v_cyl = 0
        v_face = self.polyhedron.surface_area * self.radius

        # For every pair of faces, find the dihedral angle, divide by 2*pi to
        # get the fraction of a cylinder it includes, then multiply by the edge
        # length to get the cylinder contribution.
        for i, j, edge in self.polyhedron._get_face_intersections():
            phi = self.polyhedron.get_dihedral(i, j)
            edge_length = np.linalg.norm(
                self.polyhedron.vertices[edge[0]] - self.polyhedron.vertices[edge[1]]
            )
            v_cyl += (np.pi * self.radius ** 2) * (phi / (2 * np.pi)) * edge_length

        return v_poly + v_sphere + v_face + v_cyl

    @volume.setter
    def volume(self, value):
        scale = (value / self.volume) ** (1 / 3)
        self._rescale(scale)

    @property
    def radius(self):
        """float: The rounding radius."""
        return self._radius

    @radius.setter
    def radius(self, value):
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Rounding radius must be greater than or equal to zero.")

    @property
    def surface_area(self):
        """float: Get the surface area."""
        # Compute the surface area as the sum of 3 terms:
        # 1) The (now extruded) surface area of the underlying polyhedron.
        # 2) The surface are of the spherical vertex caps, which is just the
        #    surface area of a single sphere with the rounding radius.
        # 3) The surface area of cylindrical wedges along the edges, which are
        #    computed using a standard cylinder formula then using the dihedral
        #    angle of the face to determine what fraction of the cylinder to
        #    include.
        a_poly = self.polyhedron.surface_area
        a_sphere = 4 * np.pi * self.radius ** 2
        a_cyl = 0

        # For every pair of faces, find the dihedral angle, divide by 2*pi to
        # get the fraction of a cylinder it includes, then multiply by the edge
        # length to get the cylinder contribution.
        for i, j, edge in self.polyhedron._get_face_intersections():
            phi = self.polyhedron.get_dihedral(i, j)
            edge_length = np.linalg.norm(
                self.polyhedron.vertices[edge[0]] - self.polyhedron.vertices[edge[1]]
            )
            a_cyl += (2 * np.pi * self.radius) * (phi / (2 * np.pi)) * edge_length

        return a_poly + a_sphere + a_cyl

    @surface_area.setter
    def surface_area(self, value):
        if value > 0:
            scale = np.sqrt(value / self.surface_area)
            self._rescale(scale)
        else:
            raise ValueError("Surface area must be greater than zero.")

    def is_inside(self, points):
        """Determine whether points are contained in this spheropolyhedron.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                spheropolyhedron.

        Example:
            >>> sphero = coxeter.shapes.ConvexSpheropolyhedron(
            ...   [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
            ...    [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]],
            ...   radius=0.5)
            >>> sphero.is_inside([[0, 0, 0], [10, 10, 10]])
            array([ True, False])

        """
        # Determine which points are in the polyhedron and which are in the
        # bounded volume of faces extruded by the rounding radius
        points = np.atleast_2d(points)
        point_plane_distances = self.polyhedron._point_plane_distances(points)
        in_polyhedron = np.all(point_plane_distances <= 0, axis=1)

        # Exit early if all points are inside the convex polyhedron
        if np.all(in_polyhedron):
            return in_polyhedron

        # Compute extrusions of the faces
        extruded_faces = []
        for face, normal in zip(self.polyhedron.faces, self.polyhedron.normals):
            base_vertices = self.polyhedron.vertices[face]
            extruded_vertices = base_vertices + self.radius * normal
            extruded_faces.append(
                ConvexPolyhedron([*base_vertices, *extruded_vertices])
            )

        # Select the points between the inner polyhedron and extruded space
        # and then filter them using the point-face distances
        point_faces_in_polyhedron_hull = point_plane_distances <= 0
        point_faces_in_extruded_hull = point_plane_distances <= self.radius
        point_faces_to_check = (
            point_faces_in_extruded_hull & ~point_faces_in_polyhedron_hull
        )

        # Exit early if there are no intersections to check between points
        # and rounded faces
        if not np.any(point_faces_to_check):
            return in_polyhedron

        def check_face(point_id, face_id):
            """Check for intersection of the point with rounded faces."""
            point = points[point_id]
            extruded_face = extruded_faces[face_id]
            in_extruded_face = extruded_face.is_inside(point)[0]

            # Exit early if the point is found in the extruded face
            if in_extruded_face:
                return True

            # Check spherocylinders around the edges (excluding spherical caps)
            face_points = self.polyhedron.vertices[self.polyhedron.faces[face_id]]

            # Vectors along the face edges
            face_edges = np.roll(face_points, -1, axis=0) - face_points

            # Normalized vectors along the face edges
            face_edge_lengths = np.linalg.norm(face_edges, axis=-1)
            face_edges_norm = face_edges / face_edge_lengths[:, np.newaxis]

            # Vectors from point to the edge starts
            point_to_edge_starts = point - face_points

            # Compute the vector rejection (perpendicular projection) of point
            # along edge vectors and determine if the cylinders contain it
            edge_projections = np.sum(point_to_edge_starts * face_edges_norm, axis=1)
            perpendicular_projections = (
                point_to_edge_starts - edge_projections[:, np.newaxis] * face_edges_norm
            )
            cylinder_distances = np.linalg.norm(perpendicular_projections, axis=-1)
            in_cylinders = np.any(
                (cylinder_distances <= self.radius)
                & (edge_projections >= 0)
                & (edge_projections <= face_edge_lengths)
            )

            # Exit early if the point is found in the cylinders
            if in_cylinders:
                return True

            # Check the spherical caps
            cap_distances = np.linalg.norm(point_to_edge_starts, axis=-1)
            in_caps = np.any(cap_distances <= self.radius)
            return in_caps

        # Check the faces whose rounded portion could contain the point
        in_sphero_shape = np.zeros(len(points), dtype=bool)
        for point_id, face_id in zip(*np.where(point_faces_to_check)):
            if not in_sphero_shape[point_id]:
                in_sphero_shape[point_id] = check_face(point_id, face_id)

        return in_polyhedron | in_sphero_shape

    def __repr__(self):
        return (
            f"coxeter.shapes.ConvexSpheropolyhedron(vertices={self.vertices.tolist()}, "
            f"radius={self.radius})"
        )
