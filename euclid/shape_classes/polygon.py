import numpy as np
import rowan


class Polygon(object):
    def __init__(self, vertices, normal=[0, 0, 1]):
        """A simple (i.e. non-self-overlapping) polygon.

        The polygon is assumed to be of constant unit density.

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
        vertices = np.asarray(vertices)
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
    def area(self):
        """Get or set the polygon's area (setting rescales vertices).

        To support polygons embedded in 3 dimensional space, we employ a
        projection- and rescaling-based algorithm described
        `here <https://geomalgorithms.com/a01-_area.html>`_. Specifically, the
        polygon is projected onto the plane it is "most parallel" to, the area
        of the projected polygon is computed, then the area is rescaled by the
        component of the normal in the projected dimension.
        """
        # Choose the dimension to project out based on the smallest component
        # of the normal vector.
        abs_norm = np.abs(self._normal)
        # Should be able to replace the below logic with
        # coord = np.argmax(abs_norm)
        coord = 2
        if abs_norm[0] > abs_norm[1]:
            if abs_norm[0] > abs_norm[2]:
                coord = 0
        elif abs_norm[1] > abs_norm[2]:
            coord = 1

        an = np.linalg.norm(abs_norm)

        area = 0
        i = 1
        j = 2
        k = 0

        num_verts = len(self.vertices)
        if coord == 0:
            while i < num_verts:
                area += (self.vertices[i][1] * (
                    self.vertices[np.mod(j, num_verts)][2] -
                    self.vertices[k][2]))
                i += 1
                j += 1
                k += 1
            area += (self.vertices[0][1] * (self.vertices[1][2] -
                                            self.vertices[-1][2]))
            area *= (an/(2*self._normal[0]))
        elif coord == 1:
            while i < len(self.vertices):
                area += (self.vertices[i][2] * (
                    self.vertices[np.mod(j, num_verts)][0] -
                    self.vertices[k][0]))
                i += 1
                j += 1
                k += 1
            area += (self.vertices[0][2] * (self.vertices[1][0] -
                                            self.vertices[-1][0]))
            area *= (an/(2*self._normal[1]))
        elif coord == 2:
            while i < len(self.vertices):
                area += (self.vertices[i][0] * (
                    self.vertices[np.mod(j, num_verts)][1] -
                    self.vertices[k][1]))
                i += 1
                j += 1
                k += 1
            area += (self.vertices[0][0] * (self.vertices[1][1] -
                                            self.vertices[-1][1]))
            area *= (an/(2*self._normal[2]))

        return area

    @area.setter
    def area(self, value):
        pass

    @property
    def moment_inertia(self):
        """The moment of inertia.

        Compute using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """
        pass

    @property
    def center(self):
        """Get or set the polyhedron's centroid (setting rescales vertices)."""
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self.vertices += (np.asarray(value) - self.center)

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
