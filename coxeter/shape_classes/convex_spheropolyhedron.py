import numpy as np
from .convex_polyhedron import ConvexPolyhedron


class ConvexSpheropolyhedron(object):
    def __init__(self, vertices, radius):
        """A convex spheropolyhedron.

        A convex spheropolyhedron is defined as a convex polyhedron plus a
        rounding radius. All properties of the underlying polyhedron (the
        vertices, the faces and their neighbors, etc) can be accessed directly
        through :attr:`.polyhedron`.

        Args:
            vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
                The vertices of the underlying polyhedron.
            radius (float):
                The rounding radius of the spheropolyhedron.
        """
        self._polyhedron = ConvexPolyhedron(vertices)
        self._radius = radius

    @property
    def gsd_shape_spec(self):
        """dict: A complete description of this shape corresponding to the
        shape specification in the GSD file format as described
        `here <https://gsd.readthedocs.io/en/stable/shapes.html>`_."""
        return {'type': 'ConvexPolyhedron',
                'vertices': self.polyhedron._vertices.tolist(),
                'rounding_radius': self._radius}

    @property
    def polyhedron(self):
        """:class:`~coxeter.shape_classes.convex_polyhedron.ConvexPolyhedron`:
        The underlying polyhedron."""
        return self._polyhedron

    @property
    def vertices(self):
        """Get the vertices of the spheropolyhedron."""
        return self._polyhedron.vertices

    @property
    def volume(self):
        """float: The volume."""
        V_poly = self.polyhedron.volume
        V_sphere = (4/3)*np.pi*self._radius**3
        V_cyl = 0

        # For every pair of faces, find the dihedral angle, divide by 2*pi to
        # get the fraction of a cylinder it includes, then multiply by the edge
        # length to get the cylinder contribution.
        for i, j, edge in self.polyhedron._get_face_intersections():
            phi = self.polyhedron.get_dihedral(i, j)
            edge_length = np.linalg.norm(self.polyhedron.vertices[edge[0]] -
                                         self.polyhedron.vertices[edge[1]])
            V_cyl += (np.pi*self.radius**2)*(phi/(2*np.pi))*edge_length

        return V_poly + V_sphere + V_cyl

    @property
    def radius(self):
        """float: The rounding radius."""
        return self._radius

    @property
    def surface_area(self):
        """float: The surface area."""
        A_poly = self.polyhedron.surface_area
        A_sphere = 4*np.pi*self._radius**2
        A_cyl = 0

        # For every pair of faces, find the dihedral angle, divide by 2*pi to
        # get the fraction of a cylinder it includes, then multiply by the edge
        # length to get the cylinder contribution.
        for i, j, edge in self.polyhedron._get_face_intersections():
            phi = self.polyhedron.get_dihedral(i, j)
            edge_length = np.linalg.norm(self.polyhedron.vertices[edge[0]] -
                                         self.polyhedron.vertices[edge[1]])
            A_cyl += (2*np.pi*self.radius)*(phi/(2*np.pi))*edge_length

        return A_poly + A_sphere + A_cyl

    def is_inside(self, points):
        """Determine whether a set of points are contained in this
        spheropolyhedron.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                spheropolyhedron.
        """  # noqa: E501
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
        for face, normal in zip(self.polyhedron.faces,
                                self.polyhedron.normals):
            base_vertices = self.polyhedron.vertices[face]
            extruded_vertices = base_vertices + self.radius * normal
            extruded_faces.append(
                ConvexPolyhedron([*base_vertices, *extruded_vertices]))

        # Select the points between the inner polyhedron and extruded space
        # and then filter them using the point-face distances
        point_faces_in_polyhedron_hull = point_plane_distances <= 0
        point_faces_in_extruded_hull = point_plane_distances <= self.radius
        point_faces_to_check = \
            point_faces_in_extruded_hull & ~point_faces_in_polyhedron_hull

        # Exit early if there are no intersections to check between points
        # and rounded faces
        if not np.any(point_faces_to_check):
            return in_polyhedron

        def check_face(point_id, face_id):
            """Checks for intersection of the point with rounded faces."""
            point = points[point_id]
            extruded_face = extruded_faces[face_id]
            in_extruded_face = extruded_face.is_inside(point)[0]

            # Exit early if the point is found in the extruded face
            if in_extruded_face:
                return True

            # Check spherocylinders around the edges (excluding spherical caps)
            face_points = self.polyhedron.vertices[
                self.polyhedron.faces[face_id]]

            # Vectors along the face edges
            face_edges = np.roll(face_points, -1, axis=0) - face_points

            # Normalized vectors along the face edges
            face_edge_lengths = np.linalg.norm(face_edges, axis=-1)
            face_edges_norm = face_edges / face_edge_lengths[:, np.newaxis]

            # Vectors from point to the edge starts
            point_to_edge_starts = point - face_points

            # Compute the vector rejection (perpendicular projection) of point
            # along edge vectors and determine if the cylinders contain it
            edge_projections = np.sum(point_to_edge_starts * face_edges_norm,
                                      axis=1)
            perpendicular_projections = point_to_edge_starts - \
                edge_projections[:, np.newaxis] * face_edges_norm
            cylinder_distances = np.linalg.norm(
                perpendicular_projections, axis=-1)
            in_cylinders = np.any((cylinder_distances <= self.radius) &
                                  (edge_projections >= 0) &
                                  (edge_projections <= face_edge_lengths))

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
