import numpy as np
import rowan


class Polygon(object):
    def __init__(self, vertices, normal=[0, 0, 1]):
        """A simple (i.e. non-self-overlapping) polygon.

        The polygon may be embedded in 3-dimensions, in which case the normal
        vector determines which way is "up". Note that the precise normal
        vector is not important to the winding number because the winding
        number is invariant to projection, all we need to know is the sign of
        each component of the normal vector.

        Args:
            vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
                The vertices of the polygon.
            normal (list):
                #Whether the normal vector is in the positive direction in each
                #direction. Need to specify in all directions because the
                #polygon could be on any of the xy, xz, and yz planes, in which
                #case the orientation is degenerate with respect to that plane
                #(Default value: [True, True, True]).

        """
        vertices = np.asarray(vertices, dtype=np.float64)
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError(
                "Vertices must be specified as an Nx3 array.")

        if len(vertices) < 3:
            raise ValueError(
                "A polygon must be composed of at least 3 vertices.")
        self._vertices = vertices
        self._normal = normal

        # The polygon must be oriented in order for the area calculation to
        # work, so we always sort on construction. Users can alter the sorting
        # later if desired, but we cannot have unsorted vertices.
        self.reorder_verts()

    def reorder_verts(self, clockwise=False, ref_index=0,
                      increasing_length=True):
        """Sort the vertices such that the polygon is oriented with respect to
        the normal.

        The default ordering is counterclockwise and preserves the vertex in
        the 0th position. A different ordering may be requested; however,
        note that clockwise ordering will result in a negative signed area of
        the polygon.

        The reordering is performed by rotating the polygon onto the :math:`xy`
        plane, then computing the angles of all vertices. The vertices are then
        sorted by this angle.  Note that if two points are at the same angle,
        the ordering is arbitrary and determined by the output of
        :func:`numpy.argsort`, which using an unstable quicksort algorithm by
        default.

        Args:
            clockwise (bool):
                If True, sort in clockwise order (Default value: False).
            ref_index (int):
                Index indicating which vertex should be placed first in the
                sorted list (Default value: 0).
            increasing_length (bool):
                If two vertices are at the same angle relative to the
                center, when this flag is True the point closer to the center
                comes first, otherwise the point further away comes first
                (Default value: True).
        """
        # Center vertices at the origin.
        verts = self._vertices - self.center

        # Rotate shape so that normal vector coincides with z-axis
        rotation, _ = rowan.mapping.kabsch(self._normal, [0, 0, 1])
        verts = np.dot(verts, rotation.T)

        # Compute the angle of each vertex, shift so that the chosen
        # reference_index has a value of zero, then move into the [0, 2pi]
        # range. The secondary sorting level is in terms of distance from the
        # origin.
        angles = np.arctan2(verts[:, 1], verts[:, 0])
        angles = np.mod(angles - angles[ref_index], 2*np.pi)
        distances = np.linalg.norm(verts, axis=1)
        if not increasing_length:
            distances *= -1
        if clockwise:
            angles = np.mod(2*np.pi - angles, 2*np.pi)
        vert_order = np.lexsort((angles, distances))
        self._vertices = self._vertices[vert_order, :]

    @property
    def vertices(self):
        """Get the vertices of the polyhedron."""
        return self._vertices

    @property
    def signed_area(self):
        """Get the polygon's area.

        To support polygons embedded in 3 dimensional space, we employ a
        projection- and rescaling-based algorithm described
        `here <https://geomalgorithms.com/a01-_area.html>`_. Specifically, the
        polygon is projected onto the plane it is "most parallel" to, the area
        of the projected polygon is computed, then the area is rescaled by the
        component of the normal in the projected dimension.
        """
        # Choose the dimension to project out based on the largest magnitude
        # component of the normal vector.
        abs_norm = np.abs(self._normal)
        proj_coord = np.argmax(abs_norm)
        an = np.linalg.norm(abs_norm)

        coord1 = np.mod(proj_coord + 1, 3)
        coord2 = np.mod(proj_coord + 2, 3)

        area = np.sum(
            np.roll(self.vertices, shift=-1, axis=0)[:, coord1] * (
                np.roll(self.vertices, shift=-2, axis=0)[:, coord2] -
                self.vertices[:, coord2])
        ) * (an/(2*self._normal[proj_coord]))

        return area

    @property
    def area(self):
        """Get or set the polygon's area (setting rescales vertices).

        To get the area, we simply compute the signed area and take the
        absolute value.
        """
        return np.abs(self.signed_area)

    @area.setter
    def area(self, value):
        pass

    @property
    def planar_moments_inertia(self):
        R"""Get the planar moments with respect to the x and y axis as well as
        the product of inertia.

        The `planar moments <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`_
        and the
        `product moment <https://en.wikipedia.org/wiki/Second_moment_of_area#Product_moment_of_area>`_
        are defined by the formulas:

        .. math::
            \begin{align}
                I_x &= {\int \int}_A y^2 dA \\
                I_y &= {\int \int}_A z^2 dA
                I_{xy} &= {\int \int}_A xy dA
            \end{align}

        To compute this for a polygon, we discretize the sum:

        .. math::
            \begin{align}
                I_x &= \frac{1}{12} \sum_{i=1}^N (x_i y_{i+1} - x_{i+1} y_i) (y_i^2 + y_i*y_{i+1} + y_{i+1}^2) \\
                I_y &= \frac{1}{12} \sum_{i=1}^N (x_i y_{i+1} - x_{i+1} y_i) (x_i^2 + x_i*x_{i+1} + x_{i+1}^2) \\
                I_xy &= \frac{1}{12} \sum_{i=1}^N (x_i y_{i+1} - x_{i+1} y_i) (x_i y_{i+1} + 2 x_i y_i + 2 x_{i+1} y_{i+1} + x_{i+1} y_i) \\
            \end{align}

        These formulas can be derived as described
        `here <https://physics.stackexchange.com/questions/493736/moment-of-inertia-for-a-random-polygon>`_.

        Note that the moments are always calculated about an axis perpendicular
        to the polygon, i.e. the normal vector is aligned with the z axis
        before the moments are calculated. This alignment should be
        considered when computing the moments for polygons embedded in
        three-dimensional space that are rotated out of the :math:`xy` plane,
        since the planar moments are invariant to this orientation.
        """  # noqa: E501
        # Rotate shape so that normal vector coincides with z-axis
        rotation, _ = rowan.mapping.kabsch(self._normal, [0, 0, 1])
        verts = np.dot(self._vertices, rotation.T)

        shifted_verts = np.roll(verts, shift=-1, axis=0)

        xi_yip1 = verts[:, 0] * shifted_verts[:, 1]
        xip1_yi = verts[:, 1] * shifted_verts[:, 0]

        areas = xi_yip1 - xip1_yi

        # These are the terms in the formulas for Ix and Iy, which are computed
        # simulataneously since they're identical except that they use either
        # the x or y coordinates.
        sv_sq = shifted_verts**2
        verts_sq = verts**2
        prod = verts * shifted_verts

        # This accounts for the x_i*y_{i+1} and x_{i+1}*y_i terms in Ixy.
        xi_yi = verts[:, 0] * verts[:, 1]
        xip1_yip1 = shifted_verts[:, 0] * shifted_verts[:, 1]

        diag_sums = areas[:, np.newaxis]*(verts_sq + prod + sv_sq)
        Iy, Ix, _ = np.sum(diag_sums, axis=0)/12

        xy_sums = areas*(xi_yip1 + 2*(xi_yi + xip1_yip1) + xip1_yi)
        Ixy = np.sum(xy_sums)/24

        return Ix, Iy, Ixy

    @property
    def polar_moment_inertia(self):
        """The `polar moment of inertia <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`_.

        The moment is always calculated about an axis perpendicular to the
        polygon (i.e. the normal vector) placed at the centroid of the polygon.

        The polar moment is computed as the sum of the two planar moments of inertia.
        """
        return np.sum(self.planar_moments_inertia[:2])


    @property
    def center(self):
        """Get or set the polyhedron's centroid (setting rescales vertices)."""
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self._vertices += (np.asarray(value) - self.center)

    @property
    def incircle_radius(self):
        """Get or set the polyhedron's insphere radius (setting rescales
        vertices)."""
        pass

    @incircle_radius.setter
    def incircle_radius(self, value):
        pass

    @property
    def circumcircle_radius(self):
        """Get or set the polyhedron's circumsphere radius (setting rescales
        vertices)."""
        pass

    @circumcircle_radius.setter
    def circumcircle_radius(self, value):
        pass
