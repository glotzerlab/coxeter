# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines a polygon."""

import warnings

import numpy as np
import rowan

from ..extern.bentley_ottmann import poly_point_isect
from ..extern.polyhedron.polygon import Polygon as PolyInside
from ..extern.polytri import polytri
from .base_classes import Shape2D
from .circle import Circle
from .utils import _generate_ax, rotate_order2_tensor, translate_inertia_tensor

try:
    import miniball

    MINIBALL = True
except ImportError:
    MINIBALL = False


def _align_points_by_normal(normal, points):
    """Rotate points to align the normal with the z-axis.

    Given a normal vector and a set of points, this method finds the rotation
    that aligns the normal with the z-axis and rotate all points by that
    rotation. The primary utility of this function is to bring a set of
    vertices into the :math:`xy` plane. Note that this function will work for any
    arbitrary set of points; it does no checks to ensure that they are in fact
    planar, or that the provided normal vector is in fact normal to the plane
    defined by the points.

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
    rotation, _ = rowan.mapping.kabsch([normal, -normal], [[0, 0, 1], [0, 0, -1]])
    return np.dot(points, rotation.T), rotation


def _is_simple(vertices):
    """Check if the vertices define a simple polygon.

    This code directly calls through to an external implementation
    (https://github.com/ideasman42/isect_segments-bentley_ottmann) of the
    Bentley-Ottmann algorithm to check for intersections between the line
    segments.
    """
    return len(poly_point_isect.isect_polygon(vertices)) == 0


class Polygon(Shape2D):
    """A simple (non-self-overlapping) polygon.

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

    Example:
        >>> triangle = coxeter.shapes.Polygon([[-1, 0], [0, 1], [1, 0]])
        >>> import numpy as np
        >>> assert np.isclose(triangle.area, 1.0)
        >>> bounding_circle = triangle.minimal_bounding_circle
        >>> assert np.isclose(bounding_circle.radius, 1.0)
        >>> assert np.allclose(triangle.center, [0., 1. / 3., 0.])
        >>> circumcircle = triangle.circumcircle
        >>> assert np.isclose(circumcircle.radius, 1.0)
        >>> triangle.gsd_shape_spec
        {'type': 'Polygon', 'vertices': [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]]}
        >>> assert np.allclose(
        ...   triangle.inertia_tensor,
        ...   np.diag([1. / 9., 0., 1. / 3.]))
        >>> triangle.normal
        array([ 0., -0., -1.])
        >>> assert np.allclose(
        ...   triangle.planar_moments_inertia,
        ...   (1. / 6., 1. / 6., 0.))
        >>> assert np.isclose(triangle.polar_moment_inertia, 1. / 3.)
        >>> assert np.isclose(triangle.signed_area, 1.0)

    """

    def __init__(self, vertices, normal=None, planar_tolerance=1e-5, test_simple=True):
        vertices = np.array(vertices, dtype=np.float64)

        if len(vertices.shape) != 2 or vertices.shape[1] not in (2, 3):
            raise ValueError("Vertices must be specified as an Nx2 or Nx3 array.")
        if len(vertices) < 3:
            raise ValueError("A polygon must be composed of at least 3 vertices.")

        _, indices = np.unique(vertices, axis=0, return_index=True)
        if len(indices) != vertices.shape[0]:
            raise ValueError("Found duplicate vertices.")

        # For convenience, we support providing vertices without z components,
        # but the stored vertices are always Nx3.
        if vertices.shape[1] == 2:
            self._vertices = np.hstack((vertices, np.zeros((vertices.shape[0], 1))))
        else:
            self._vertices = vertices

        # Note: Vertices do not yet need to be ordered for the purpose of
        # determining the normal, this check can be performed irrespective of
        # ordering since any cross product of vectors will provide a normal.
        computed_normal = np.cross(
            self._vertices[2, :] - self._vertices[1, :],
            self._vertices[0, :] - self._vertices[1, :],
        )
        computed_normal /= np.linalg.norm(computed_normal)
        if normal is None:
            self._normal = computed_normal
        else:
            norm_normal = np.asarray(normal, dtype=np.float64)
            norm_normal /= np.linalg.norm(normal)

            if not np.isclose(np.abs(np.dot(computed_normal, norm_normal)), 1):
                raise ValueError(
                    "The provided normal vector is not orthogonal to the polygon."
                )
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
            planar_vertices, _ = _align_points_by_normal(self._normal, self._vertices)
            if not _is_simple(planar_vertices):
                raise ValueError(
                    "The vertices must be passed in counterclockwise order. "
                    "Note that the Polygon class only supports simple "
                    "polygons, so self-intersecting polygons are not "
                    "permitted."
                )

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {"type": "Polygon", "vertices": self.vertices.tolist()}

    @property
    def normal(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get the normal vector."""
        return self._normal

    @property
    def vertices(self):
        """:math:`(N_{verts}, 3)` :class:`numpy.ndarray` of float: Get the vertices of the polygon."""  # noqa: E501
        return self._vertices

    @property
    def num_vertices(self):
        """int: Get the number of vertices."""
        return self.vertices.shape[0]

    def _rescale(self, scale):
        """Multiply length scale.

        Args:
            scale (float):
                Scale factor.
        """
        self._vertices *= scale

    @property
    def perimeter(self):
        """float: Get the perimeter of the polygon."""
        return np.sum(
            np.linalg.norm(
                np.roll(self.vertices, axis=0, shift=-1) - self.vertices, axis=-1
            )
        )

    @perimeter.setter
    def perimeter(self, value):
        if value > 0:
            scale = value / self.perimeter
            self._rescale(scale)
        else:
            raise ValueError("Perimeter must be greater than zero.")

    @property
    def signed_area(self):
        """float: Get the polygon's area.

        To support polygons embedded in 3 dimensional space, we employ a
        projection- and rescaling-based algorithm described `here
        <https://geomalgorithms.com/a01-_area.html#3D-Planar-Polygons>`__.
        Specifically, the polygon is projected onto the plane it is "most
        parallel" to, the area of the projected polygon is computed, then the
        area is rescaled by the component of the normal in the projected
        dimension.
        """
        # Choose the dimension to project out based on the largest magnitude
        # component of the normal vector.
        abs_norm = np.abs(self._normal)
        proj_coord = np.argmax(abs_norm)
        an = np.linalg.norm(abs_norm)

        coord1 = np.mod(proj_coord + 1, 3)
        coord2 = np.mod(proj_coord + 2, 3)

        area = np.sum(
            np.roll(self.vertices, shift=-1, axis=0)[:, coord1]
            * (
                np.roll(self.vertices, shift=-2, axis=0)[:, coord2]
                - self.vertices[:, coord2]
            )
        ) * (an / (2 * self._normal[proj_coord]))

        return area

    @property
    def area(self):
        """float: Get or set the polygon's area.

        To get the area, we simply compute the signed area and take the
        absolute value.
        """
        return np.abs(self.signed_area)

    @area.setter
    def area(self, value):
        scale = np.sqrt(value / self.area)
        self._rescale(scale)

    @property
    def planar_moments_inertia(self):
        r"""list[float, float, float]: Get the planar and product moments of inertia.

        Moments are computed with respect to the :math:`x` and :math:`y`
        axes. In addition to the two planar moments, this property also
        provides the product of inertia.

        The `planar moments <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        and the
        `product <https://en.wikipedia.org/wiki/Second_moment_of_area#Product_moment_of_area>`__
        of inertia are defined by the formulas:

        .. math::
            \begin{align}
                I_x &= {\int \int}_A y^2 dA \\
                I_y &= {\int \int}_A x^2 dA \\
                I_{xy} &= {\int \int}_A xy dA \\
            \end{align}

        To compute this for a polygon, we discretize the sum:

        .. math::
            \begin{align}
                I_x &= \frac{1}{12} \sum_{i=1}^N (x_i y_{i+1} - x_{i+1} y_i) (y_i^2 + y_i*y_{i+1} + y_{i+1}^2) \\
                I_y &= \frac{1}{12} \sum_{i=1}^N (x_i y_{i+1} - x_{i+1} y_i) (x_i^2 + x_i*x_{i+1} + x_{i+1}^2) \\
                I_{xy} &= \frac{1}{12} \sum_{i=1}^N (x_i y_{i+1} - x_{i+1} y_i) (x_i y_{i+1} + 2 x_i y_i + 2 x_{i+1} y_{i+1} + x_{i+1} y_i) \\
            \end{align}

        These formulas can be derived as described
        `here <https://physics.stackexchange.com/questions/493736/moment-of-inertia-for-a-random-polygon>`__.

        Note that the moments are always calculated about an axis perpendicular
        to the polygon, i.e. the normal vector is aligned with the :math:`z`
        axis before the moments are calculated. This alignment should be
        considered when computing the moments for polygons embedded in
        three-dimensional space that are rotated out of the :math:`xy` plane,
        since the planar moments are invariant to this orientation. The exact
        rotation used for this computation (i.e. changes in the :math:`x` and
        :math:`y` position) should not be relied upon.
        """  # noqa: E501
        # Rotate shape so that normal vector coincides with z-axis.
        verts, _ = _align_points_by_normal(self._normal, self._vertices)

        shifted_verts = np.roll(verts, shift=-1, axis=0)

        xi_yip1 = verts[:, 0] * shifted_verts[:, 1]
        xip1_yi = verts[:, 1] * shifted_verts[:, 0]

        areas = xi_yip1 - xip1_yi

        # These are the terms in the formulas for Ix and Iy, which are computed
        # simulataneously since they're identical except that they use either
        # the x or y coordinates.
        sv_sq = shifted_verts ** 2
        verts_sq = verts ** 2
        prod = verts * shifted_verts

        # This accounts for the x_i*y_{i+1} and x_{i+1}*y_i terms in Ixy.
        xi_yi = verts[:, 0] * verts[:, 1]
        xip1_yip1 = shifted_verts[:, 0] * shifted_verts[:, 1]

        # Need to take absolute values in case vertices are ordered clockwise.
        diag_sums = areas[:, np.newaxis] * (verts_sq + prod + sv_sq)
        i_y, i_x, _ = np.abs(np.sum(diag_sums, axis=0) / 12)

        xy_sums = areas * (xi_yip1 + 2 * (xi_yi + xip1_yip1) + xip1_yi)
        i_xy = np.abs(np.sum(xy_sums) / 24)

        return i_x, i_y, i_xy

    @property
    def inertia_tensor(self):
        r""":math:`(3, 3)` :class:`numpy.ndarray`: Get the inertia tensor.

        The inertia tensor is computed for the polygon embedded in
        :math:`\mathbb{R}^3`. This computation proceeds by first computing the
        polar moment of inertia for the polygon in the :math:`xy`-plane
        relative to its centroid. The tensor is then rotated back to the
        orientation of the polygon and shifted to the original centroid.
        """
        # Save the original configuration as we translate and rotate it to the
        # origin so that we can reset after (since we're modifying the internal
        # state in order to use self.polar_moment_inertia). The sequence here
        # is important: we must translate before rotating so that the parallel
        # axis theorem can be applied in the reverse direction (rotating about
        # the origin before translating to the actual centroid).
        original_center = self.center.copy()
        original_vertices = self._vertices.copy()
        original_normal = self._normal.copy()

        self.center = (0, 0, 0)
        mat, _ = rowan.mapping.kabsch(
            [self.normal, -self.normal], [[0, 0, 1], [0, 0, -1]]
        )
        self._vertices = self._vertices.dot(mat.T)
        self._normal = np.asarray([0, 0, 1])

        inertia_tensor = np.diag([0, 0, self.polar_moment_inertia])
        shifted_inertia_tensor = translate_inertia_tensor(
            original_center, rotate_order2_tensor(mat, inertia_tensor), self.area
        )

        self.center = original_center
        self._vertices = original_vertices
        self._normal = original_normal

        return shifted_inertia_tensor

    @property
    def centroid(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape.

        The centroid of a polygon is calculated according to
        `this formula <https://en.wikipedia.org/wiki/Centroid#Of_a_polygon>`__.
        """  # noqa: E501
        verts, rotation = _align_points_by_normal(self.normal, self.vertices)
        verts_shifted = np.roll(verts, shift=-1, axis=0)

        delta_term = (
            verts[:, 0] * verts_shifted[:, 1] - verts_shifted[:, 0] * verts[:, 1]
        )
        c_x = np.sum((verts[:, 0] + verts_shifted[:, 0]) * delta_term)
        c_y = np.sum((verts[:, 1] + verts_shifted[:, 1]) * delta_term)

        in_plane_centroid = np.array([c_x, c_y, 0]) / (6 * self.area)

        # We've rotated into the plane, so the z position of all vertices
        # should be equal. We take the average to improve numerical stablity.
        in_plane_centroid[2] = np.mean(verts[:, 2])

        # Revert the rotation into the plane.
        centroid = rotation.T.dot(in_plane_centroid)

        return centroid

    @centroid.setter
    def centroid(self, value):
        self._vertices += np.asarray(value) - self.centroid

    def _triangulation(self):
        """Generate a triangulation of the polygon.

        Yields tuples of indices where each tuple corresponds to the vertex
        indices of a single triangle.

        Since the polygon may be embedded in 3D, we must rotate the polygon
        into the plane to get a triangulation.
        """
        yield from polytri.triangulate(self.vertices)

    def plot(self, ax=None, center=False, plot_verts=False, label_verts=False):
        """Plot the polygon.

        Note that the polygon is always rotated into the :math:`xy` plane and
        plotted in two dimensions.

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes on which to draw the polygon. Axes will be created
                if this is None (Default value: None).
            center (bool):
                If True, the polygon vertices are plotted relative to its
                center (Default value: False).
            plot_verts (bool):
                If True, scatter points will be added at the vertices
                (Default value: False).
            label_verts (bool):
                If True, vertex indices will be added next to the vertices
                (Default value: False).
        """
        ax = _generate_ax(ax)
        verts = self._vertices - self.center if center else self._vertices
        verts, _ = _align_points_by_normal(self._normal, verts)
        verts = np.concatenate((verts, verts[[0]]))
        x = verts[:, 0]
        y = verts[:, 1]
        ax.plot(x, y)

        if plot_verts:
            ax.scatter(x, y)
        if label_verts:
            # Typically a good shift for plotting the labels
            shift = (np.max(y) - np.min(y)) * 0.025
            for i, vert in enumerate(verts[:-1]):
                ax.text(vert[0], vert[1] + shift, "{}".format(i), fontsize=10)

    @property
    def bounding_circle(self):
        """:class:`~.Circle`: Get the minimal bounding circle."""
        warnings.warn(
            "The bounding_circle property is deprecated, use "
            "minimal_bounding_circle instead",
            DeprecationWarning,
        )

        return self.minimal_bounding_circle

    @property
    def minimal_bounding_circle(self):
        """:class:`~.Circle`: Get the minimal bounding circle."""
        if not MINIBALL:
            raise ImportError(
                "The miniball module must be installed. It can "
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
            raise RuntimeError("Unable to solve for a bounding circle.")

        # The center must be rotated back to undo any rotation.
        center = rowan.rotate(rowan.conjugate(current_rotation), center)

        return Circle(np.sqrt(r2), center)

    @property
    def circumcircle(self):
        """:class:`~.Circle`: Get the polygon's circumcircle.

        A `circumcircle
        <https://en.wikipedia.org/wiki/Circumscribed_circle>`__ must touch
        all the points of the polygon. A circumcircle exists if and only if
        there is a point equidistant from all the vertices. The circumcircle is
        found by finding the least squares solution of the overdetermined system
        of linear equations defined by this constraint, and the circumcircle
        only exists if the resulting solution has no residual.

        Raises:
            RuntimeError: If no circumcircle exists for this polygon.
        """
        # The circumsphere is defined by center C and radius r. For vertex i
        # with position r_i, dot(r_i - C, r_i - C) = r^2, which is equivalent
        # to dot(r_i, r_i) - 2 dot(C, r_i) + dot(C, C) = r^2, a system of
        # quadratic equations. If we choose r_0 as the origin, then dot(C, C) =
        # r^2 and we instead have the linear equations dot(C, p_i) =
        # dot(p_i, p_i) / 2 where p_i = r_i - r_0. This is the set of equations
        # that we solve. The polygon is embedded in 3D, which imposes the
        # additional constraint that the circumcircle must lie in the plane of
        # the polygon.
        points = np.concatenate(
            (self.vertices[1:] - self.vertices[0], self.normal[np.newaxis])
        )
        half_point_lengths = np.concatenate(
            (np.sum(points[:-1] * points[:-1], axis=1) / 2, [0])
        )
        x, resids, _, _ = np.linalg.lstsq(points, half_point_lengths, None)
        if len(self.vertices) > 3 and not np.isclose(resids, 0):
            raise RuntimeError("No circumcircle for this polygon.")

        return Circle(np.linalg.norm(x), x + self.vertices[0])

    @property
    def circumcircle_radius(self):
        """float: Get the radius of the polygon's circumcircle."""
        return self.circumcircle.radius

    @circumcircle_radius.setter
    def circumcircle_radius(self, value):
        self._rescale(value / self.circumcircle_radius)

    @property
    def incircle(self):
        """:class:`~.Sphere`: Get the polygon's incircle.

        Note:
            The incircle of a polygon is defined as the circle contained within
            the polygon that is tangent to all its faces. This condition
            uniquely defines the circle, if it exists. The set of equations
            defined by this equation is solved using a least squares approach,
            with the magnitude of the residual used to determine whether or not
            the incircle exists.

        """
        # The incircle is defined by center C and radius r. For face i
        # defined by its unit normal n_i and any point in the plane (choose a
        # vertex v_i for convenience), we must have dot(C + r n_i - v_i, n_i) = 0.
        # Defining the vector Cr = (C_x, C_y, C_z, r), and the augmented
        # normals m = (n_x, n_y, n_z, 1), rearranging gives the equations that
        # we solve: dot(m_i, cr) = dot(n_i, v_i). The polygon is embedded in
        # 3D, which imposes the additional constraint that the incircle must
        # lie in the plane of the polygon.

        outward_normals = np.cross(
            # Order is important here to get the outward facing normal.
            np.roll(self.vertices, axis=0, shift=-1) - self.vertices,
            self.normal,
        )
        outward_normals /= np.linalg.norm(outward_normals, axis=-1)[:, np.newaxis]

        # vstack the row corresponding to the constraint equation
        a = np.vstack(
            (
                # hstack on the coefficient of 1 for the radius
                np.hstack((outward_normals, np.ones((self.num_vertices, 1)))),
                # The coefficient of the radius in the plane constraint is 0.
                np.append(self.normal, 0),
            )
        )

        b = np.concatenate(
            (
                # Arbitrarily choose the first vertex in every edge for the dot product.
                np.sum(outward_normals * self.vertices, axis=-1),
                # Can use the dot product of the normal with any of the vertices.
                [np.dot(self.normal, self.vertices[0])],
            )
        )

        x, resids, _, _ = np.linalg.lstsq(a, b, None)
        if len(self.vertices) > 4 and not np.isclose(resids, 0):
            raise RuntimeError("No incircle for this polygon.")

        return Circle(x[3], x[:3])

    @property
    def incircle_radius(self):
        """float: Get the radius of the polygon's incircle."""
        return self.incircle.radius

    @incircle_radius.setter
    def incircle_radius(self, value):
        self._rescale(value / self.incircle_radius)

    def compute_form_factor_amplitude(self, q, density=1.0):  # noqa: D102
        """Calculate the form factor intensity.

        The form factor amplitude of a polygon is computed according to the
        derivation provided in this dissertation:
        https://deepblue.lib.umich.edu/handle/2027.42/120906.
        The Kelvin-Stokes theorem allows reducing the surface integral to a line
        integral around the boundary.

        For more generic information about form factors, see
        `Shape.compute_form_factor_amplitude`.
        """
        form_factor = np.zeros((len(q),), dtype=np.complex128)

        # All the q vectors must be projected onto the plane of the polygon before they
        # can be calculated. Note that the orientation of the polygon is implicit in its
        # vertices, otherwise we would need to rotate the q vectors appropriately.
        q_dot_norm = np.dot(q, self.normal)
        q = q - q_dot_norm[:, np.newaxis] * self.normal
        q_sqs = np.sum(q * q, axis=-1)
        zero_q = np.isclose(q_sqs, 0)
        form_factor[zero_q] = self.area

        # Add the contribution over all edges of the face.
        verts = self._vertices
        verts_shifted = np.roll(verts, axis=0, shift=-1)
        edges = verts_shifted - verts
        midpoints = (verts + verts_shifted) / 2

        q_nonzero_broadcast = q[np.newaxis, ~zero_q, :]
        edges_cross_qs = np.cross(edges[:, np.newaxis, :], q_nonzero_broadcast)
        # Due to oddities of numpy broadcasting, many singleton dimensions can persist
        # and must be squeezed out.
        midpoints_dot_qs = np.inner(
            midpoints[:, np.newaxis, :], q_nonzero_broadcast
        ).squeeze()
        edges_dot_qs = np.inner(edges[:, np.newaxis, :], q_nonzero_broadcast).squeeze()
        f_ns = (
            np.dot(edges_cross_qs, self.normal)
            # Note that np.sinc(x) gives sin(pi*x)/(pi*x)
            * np.sinc(0.5 * edges_dot_qs / np.pi)
            / q_sqs[~zero_q]
        )
        # Apply translational shift relative to the center of the
        # polygonal face relative to its centroid.
        form_factor[~zero_q] = -np.sum(
            f_ns * 1j * np.exp(-1j * midpoints_dot_qs), axis=0
        )
        form_factor *= density
        return form_factor

    def is_inside(self, points):
        """Determine whether points are contained in this polygon.

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
        # Rotate both the vertices and the points into the plane.
        verts, rotation = _align_points_by_normal(self.normal, self.vertices)
        points_in_plane = np.dot(points, rotation.T)

        winding_number_calculator = PolyInside(verts)

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
                ):
                    return True
                else:
                    raise

        return np.array([_check_inside(p) for p in np.atleast_2d(points_in_plane)])

    def __repr__(self):
        return (
            f"coxeter.shapes.Polygon(vertices={self.vertices.tolist()}, "
            f"normal={self.normal.tolist()})"
        )
