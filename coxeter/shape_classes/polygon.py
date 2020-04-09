import numpy as np
import rowan
from ..polytri import polytri
from ..bentley_ottman import poly_point_isect

try:
    import miniball
    MINIBALL = True
except ImportError:
    MINIBALL = False


def _align_points_by_normal(normal, points):
    """Given a normal vector and a set of points, find a rotation to align the
    normal with the z-axis and rotate all points by that rotation.

    The primary utility of this function is to bring a set of vertices into the
    xy plane. Note that this function will work for any arbitrary set of
    points; it does no checks to ensure that they are in fact planar, or that
    the provided normal vector is in fact normal to the plane defined by the
    points.

    Args:
        normal (:math:`(3, )` :class:`numpy.ndarray`):
            The normal vector to make coincide with [1, 0, 0].
        points (:math:`(N, 3)` :class:`numpy.ndarray`):
            The points that will be rotated and returned.

    Returns:
       :math:`(N, 3)` :class:`numpy.ndarray`: The rotated points.
    """
    # Since we are considering just a single vector, to avoid getting a pure
    # translation we need to consider mapping both the vector and its opposite
    # (which defines an oriented coordinate system).
    rotation, _ = rowan.mapping.kabsch([normal, -normal],
                                       [[0, 0, 1], [0, 0, -1]])
    return np.dot(points, rotation.T)


def _is_simple(vertices):
    """Check if the vertices define a simple polygon.

    This code directly calls through to an external implementation
    (https://github.com/ideasman42/isect_segments-bentley_ottmann) of the
    Bentley-Ottman algorithm to check for intersections between the line
    segments."""
    return len(poly_point_isect.isect_polygon(vertices)) == 0


class Polygon(object):
    def __init__(self, vertices, normal=None, planar_tolerance=1e-5,
                 test_simple=True):
        """A simple (i.e. non-self-overlapping) polygon.

        The polygon is embedded in 3-dimensions, so the normal
        vector determines which way is "up".

        .. note::
            This class is designed for polygons without self-intersections, so
            the internal sorting will automatically result in such
            intersections being removed.

        Args:
            vertices (:math:`(N, 3)` or :math:`(N, 2)` :class:`numpy.ndarray`):
                The vertices of the polygon.
            normal (sequence of length 3 or None):
                The normal vector to the polygon. If :code:`None`, the normal
                is computed by taking the cross product of the vectors formed
                by the first three vertices :code:`np.cross(vertices[2, :] -
                vertices[1, :], vertices[0, :] - vertices[1, :])`. This choice
                is made so that if the provided vertices are in the :math:`xy`
                plane and are specified in counterclockwise order, the
                resulting normal is the :math:`z` axis. Since this arbitrary
                choice may not preserve the orientation of the provided
                vertices, users may provide a normal instead
                (Default value: None).
            planar_tolerance (float):
                The tolerance to use to verify that the vertices are planar.
                Providing this argument may be necessary if you have a large
                number of vertices and are rotated significantly out of the
                plane.
            test_simple (bool):
                If ``True``, perform a sanity check on construction that the
                provided vertices constitute a simple polygon. If this check is
                omitted, the class may produce invalid results if the user
                inputs incorrect coordinates, so this flag should be set to
                ``False`` with care.
        """
        vertices = np.array(vertices, dtype=np.float64)

        if len(vertices.shape) != 2 or vertices.shape[1] not in (2, 3):
            raise ValueError(
                "Vertices must be specified as an Nx2 or Nx3 array.")
        if len(vertices) < 3:
            raise ValueError(
                "A polygon must be composed of at least 3 vertices.")

        _, indices = np.unique(vertices, axis=0, return_index=True)
        if len(indices) != vertices.shape[0]:
            raise ValueError("Found duplicate vertices.")

        # For convenience, we support providing vertices without z components,
        # but the stored vertices are always Nx3.
        if vertices.shape[1] == 2:
            self._vertices = np.hstack((vertices,
                                        np.zeros((vertices.shape[0], 1))))
        else:
            self._vertices = vertices

        # Note: Vertices do not yet need to be ordered for the purpose of
        # determining the normal, this check can be performed irrespective of
        # ordering since any cross product of vectors will provide a normal.
        computed_normal = np.cross(self._vertices[2, :] - self._vertices[1, :],
                                   self._vertices[0, :] - self._vertices[1, :])
        computed_normal /= np.linalg.norm(computed_normal)
        if normal is None:
            self._normal = computed_normal
        else:
            norm_normal = np.asarray(normal, dtype=np.float64)
            norm_normal /= np.linalg.norm(normal)

            if not np.isclose(np.abs(np.dot(computed_normal, norm_normal)), 1):
                raise ValueError("The provided normal vector is not "
                                 "orthogonal to the polygon.")
            self._normal = norm_normal

        d = self._normal.dot(self.vertices[0, :])
        # If this simple check of coplanarity is not robust enough for a
        # desired polygon, it might be necessary to implement more robust
        # checks based on something like
        # http://www.cs.cmu.edu/~quake/robust.html
        for v in self.vertices:
            if not np.isclose(self._normal.dot(v), d, planar_tolerance):
                raise ValueError("Not all vertices are coplanar.")

        if test_simple:
            planar_vertices = _align_points_by_normal(
                self._normal, self._vertices)
            if not _is_simple(planar_vertices):
                raise ValueError(
                    "The vertices must be passed in counterclockwise order. "
                    "Note that the Polygon class only supports simple "
                    "polygons, so self-intersecting polygons are not "
                    "permitted.")

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
        :func:`numpy.argsort`, which uses an unstable quicksort algorithm by
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
        verts = _align_points_by_normal(self._normal,
                                        self._vertices - self.center)

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
        vert_order = np.lexsort((distances, angles))
        self._vertices = self._vertices[vert_order, :]

    @property
    def normal(self):
        """The normal vector."""
        return self._normal

    @property
    def vertices(self):
        """Get the vertices of the polygon."""
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
    def area(self, new_area):
        scale_factor = np.sqrt(new_area/self.area)
        self._vertices *= scale_factor

    @property
    def planar_moments_inertia(self):
        R"""Get the planar moments with respect to the x and y axis as well as
        the product of inertia.

        The `planar moments <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        and the
        `product moment <https://en.wikipedia.org/wiki/Second_moment_of_area#Product_moment_of_area>`__
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
        `here <https://physics.stackexchange.com/questions/493736/moment-of-inertia-for-a-random-polygon>`__.

        Note that the moments are always calculated about an axis perpendicular
        to the polygon, i.e. the normal vector is aligned with the :math:`z`
        axis before the moments are calculated. This alignment should be
        considered when computing the moments for polygons embedded in
        three-dimensional space that are rotated out of the :math:`xy` plane,
        since the planar moments are invariant to this orientation.
        """  # noqa: E501
        # Rotate shape so that normal vector coincides with z-axis
        verts = _align_points_by_normal(self._normal,
                                        self._vertices)

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

        # Need to take absolute values in case vertices are ordered clockwise.
        diag_sums = areas[:, np.newaxis]*(verts_sq + prod + sv_sq)
        Iy, Ix, _ = np.abs(np.sum(diag_sums, axis=0)/12)

        xy_sums = areas*(xi_yip1 + 2*(xi_yi + xip1_yip1) + xip1_yi)
        Ixy = np.abs(np.sum(xy_sums)/24)

        return Ix, Iy, Ixy

    @property
    def polar_moment_inertia(self):
        """The polar moment of inertia.

        The `polar moment of inertia <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        is always calculated about an axis perpendicular to the polygon (i.e. the
        normal vector) placed at the centroid of the polygon.

        The polar moment is computed as the sum of the two planar moments of inertia.
        """  # noqa: E501
        return np.sum(self.planar_moments_inertia[:2])

    @property
    def center(self):
        """Get or set the polygon's centroid (setting shifts the vertices)."""
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self._vertices += (np.asarray(value) - self.center)

    def _triangulation(self):
        """Generate a triangulation of the polygon.

        Yields tuples of indices where each tuple corresponds to the vertex
        indices of a single triangle.

        Since the polygon may be embedded in 3D, we must rotate the polygon
        into the plane to get a triangulation.
        """
        yield from polytri.triangulate(self.vertices)

    @property
    def iq(self):
        """The isoperimetric quotient."""
        raise NotImplementedError

    def plot(self, ax, center=False, plot_verts=False, label_verts=False):
        """Plot the polygon.

        Note that the polygon is always rotated into the xy plane and plotted
        in two dimensions.

        Args:
            plot_verts (bool):
                If True, scatter points will be added at the vertices (Default
                value: False).
            label_verts (bool):
                If True, vertex indices will be added next to the vertices
                (Default value: False).
        """
        # TODO: Generate axis if one is not provided.
        verts = self._vertices - self.center if center else self._vertices
        verts = _align_points_by_normal(self._normal, verts)
        verts = np.concatenate((verts, verts[[0]]))
        x = verts[:, 0]
        y = verts[:, 1]
        ax.plot(x, y)

        if plot_verts:
            ax.scatter(x, y)
        if label_verts:
            # Typically a good shift for plotting the labels
            shift = (np.max(y) - np.min(y))*0.025
            for i, vert in enumerate(verts[:-1]):
                ax.text(vert[0], vert[1] + shift, '{}'.format(i), fontsize=10)

    @property
    def bounding_circle(self):
        """The bounding circle of the polygon, given by a center and a
        radius."""
        if not MINIBALL:
            raise ImportError("The miniball module must be installed. It can "
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

        if attempt == max_attempts:
            raise RuntimeError("Unable to solve for a bounding sphere.")

        # The center must be rotated back to undo any rotation.
        center = rowan.rotate(rowan.conjugate(current_rotation), center)

        return center, np.sqrt(r2)

    @property
    def circumcircle(self):
        """float: Get the polygon's circumcircle."""
        # Solves a linear system of equations to find a point equidistant from
        # all the vertices if it exists. Since the polygon is embedded in 3D,
        # we must constrain our solutions to the plane of the polygon.
        points = np.concatenate((
            self.vertices[1:] - self.vertices[0], self.normal[np.newaxis]))
        half_point_lengths = np.concatenate((
            np.sum(points[:-1]*points[:-1], axis=1)/2, [0]))
        x, resids, _, _ = np.linalg.lstsq(points, half_point_lengths, None)
        if len(self.vertices) > 3 and not np.isclose(resids, 0):
            raise RuntimeError("No circumcircle for this polygon.")

        return x + self.vertices[0], np.linalg.norm(x)
