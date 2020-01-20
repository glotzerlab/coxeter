from scipy.spatial import ConvexHull
import numpy as np
from .convex_polyhedron import ConvexPolyhedron


class ConvexSpheropolyhedron(object):
    def __init__(self, vertices, radius):
        """A convex spheropolyhedron.

        A convex spheropolyhedron is defined as a convex polyhedron plus a
        rounding radius. All properties of the underlying polyhedron (the
        vertices, the facets and their neighbors, etc) can be accessed directly
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
    def polyhedron(self):
        """:class:`~euclid.shape_classes.convex_polyhedron.ConvexPolyhedron`:
        The underlying polyhedron."""
        return self._polyhedron

    @property
    def volume(self):
        """float: The volume."""
        V_poly = self.polyhedron.volume
        V_sphere = (4/3)*np.pi*self._radius**3
        V_cyl = 0

        # For every pair of faces, find the dihedral angle, divide by 2*pi to
        # get the fraction of a cylinder it includes, then multiply by the edge
        # length to get the cylinder contribution.
        for i, j, edge in self.polyhedron._get_facet_intersections():
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
        for i, j, edge in self.polyhedron._get_facet_intersections():
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
        # bounded volume of facets extruded by the rounding radius
        hull = ConvexHull(self.polyhedron.vertices)
        points = np.atleast_2d(points)

        point_facet_checks = hull.equations[:, :3] @ points.T
        point_facet_checks += hull.equations[:, 3, np.newaxis]
        in_polyhedron = np.all(point_facet_checks <= 0, axis=0)

        # Compute convex hulls of the extruded facets
        extruded_hulls = []
        for i, eq in enumerate(hull.equations):
            base_vertices = hull.points[hull.simplices[i]]
            normal = eq[:3]
            normal /= np.linalg.norm(normal)
            extruded_vertices = base_vertices + self.radius * normal
            extruded_hulls.append(
                ConvexHull([*base_vertices, *extruded_vertices]))

        # Select the points between the convex hull facet and extruded hull
        # facet and then filter them using the point-facet checks
        point_facets_in_convex_hull = point_facet_checks <= 0
        point_facets_in_extruded_hull = point_facet_checks <= self.radius
        point_facets_to_check = \
            point_facets_in_convex_hull ^ point_facets_in_extruded_hull

        def check_facet(point_id, facet_id):
            """Checks for intersection of the point with rounded facets."""
            point = points[point_id]
            extruded_hull = extruded_hulls[facet_id]
            facet_check = extruded_hull.equations[:, :3] @ point
            facet_check += extruded_hull.equations[:, 3]
            in_extruded_hull = np.all(facet_check <= 0)

            # Exit early if the point is found in the extruded hull
            if in_extruded_hull:
                return True

            # Check spherocylinders around the edges (excluding spherical caps)
            facet_points = hull.points[hull.simplices[facet_id]]

            # Vectors along the facet edges
            facet_edges = np.roll(facet_points, -1, axis=0) - facet_points

            # Normalized vectors along the facet edges
            facet_edge_lengths = np.linalg.norm(facet_edges, axis=-1)
            facet_edges_norm = facet_edges / facet_edge_lengths[:, np.newaxis]

            # Vectors from point to the edge starts
            point_to_edge_starts = point - facet_points

            # Compute the vector rejection (perpendicular projection) of point
            # along edge vectors and determine if the cylinders contain it
            edge_projections = np.einsum(
                'ij,ij->i', point_to_edge_starts, facet_edges_norm)
            perpendicular_projections = point_to_edge_starts - \
                edge_projections[:, np.newaxis] * facet_edges_norm
            cylinder_distances = np.linalg.norm(
                perpendicular_projections, axis=-1)
            in_cylinders = np.any((cylinder_distances <= self.radius) &
                                  (edge_projections >= 0) &
                                  (edge_projections <= facet_edge_lengths))

            # Exit early if the point is found in the cylinders
            if in_cylinders:
                return True

            # Check the spherical caps
            cap_distances = np.linalg.norm(point_to_edge_starts, axis=-1)
            in_caps = np.any(cap_distances <= self.radius)
            return in_caps

        # Check the facets whose rounded portion could contain the point
        in_sphero_shape = np.zeros(len(points), dtype=bool)
        for facet_id, point_id in zip(*np.where(point_facets_to_check)):
            if not in_sphero_shape[point_id]:
                in_sphero_shape[point_id] = check_facet(point_id, facet_id)

        return in_polyhedron | in_sphero_shape
