# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Defines a convex polygon."""

import warnings

import numpy as np
from scipy.spatial import ConvexHull

from .circle import Circle
from .polygon import Polygon, _align_points_by_normal


def _is_convex(vertices, normal):
    """Check if the vertices provided define a convex shape.

    This algorithm makes no assumptions about ordering of the vertices, it
    simply constructs the convex hull of the points and checks that all of the
    vertices are on the convex hull.

    Args:
        vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
            The polygon vertices.
        normal (:math:`(3, )` :class:`numpy.ndarray`):
            The normal to the vertices.

    Returns:
        bool: ``True`` if ``vertices`` define a convex polygon.

    """
    # TODO: Add a tolerance check in case a user provides collinear vertices on
    # the boundary of a convex hull.
    verts_2d, _ = _align_points_by_normal(normal, vertices)
    hull = ConvexHull(verts_2d[:, :2])
    return len(hull.vertices) == len(vertices)


class ConvexPolygon(Polygon):
    """A convex polygon.

    The polygon is embedded in 3-dimensions, so the normal
    vector determines which way is "up".

    Args:
        vertices (:math:`(N, 3)` or :math:`(N, 2)` :class:`numpy.ndarray`):
            The vertices of the polygon. They need not be sorted since the
            order will be determined by the hull.
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

    Example:
        >>> square = coxeter.shapes.ConvexPolygon(
        ...   [[1, 1], [-1, -1], [1, -1], [-1, 1]])
        >>> import numpy as np
        >>> assert np.isclose(square.area, 4.0)
        >>> assert np.isclose(
        ...   square.minimal_bounding_circle.radius,
        ...   np.sqrt(2.))
        >>> square.center
        array([0., 0., 0.])
        >>> assert np.isclose(
        ...   square.circumcircle.radius,
        ...   np.sqrt(2.))
        >>> square.gsd_shape_spec
        {'type': 'Polygon', 'vertices': [[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]]}
        >>> assert np.isclose(square.maximal_centered_bounded_circle.radius, 1.0)
        >>> assert np.allclose(
        ...   square.inertia_tensor,
        ...   [[0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 0., 8. / 3.]])
        >>> square.normal
        array([0., 0., 1.])
        >>> assert np.allclose(
        ...   square.planar_moments_inertia,
        ...   (4. / 3., 4. / 3., 0.))
        >>> assert np.isclose(square.polar_moment_inertia, 8. / 3.)
        >>> assert np.isclose(square.signed_area, 4.0)
        >>> square.vertices
        array([[ 1.,  1.,  0.],
               [-1.,  1.,  0.],
               [-1., -1.,  0.],
               [ 1., -1.,  0.]])

    """

    def __init__(self, vertices, normal=None, planar_tolerance=1e-5):
        super(ConvexPolygon, self).__init__(vertices, normal, planar_tolerance, False)
        if _is_convex(self.vertices, self.normal):
            # If points form a convex set, then we can order the vertices. We
            # cannot directly use the output of scipy's convex hull because our
            # polygon may be embedded in 3D, so we sort ourselves.
            self._reorder_verts()
        else:
            # If the shape is nonconvex, the user must provide ordered vertices
            # to uniquely identify the polygon. We must check if there are any
            # intersections to avoid complex (self-intersecting) polygons.
            raise ValueError("The provided vertices do not form a convex polygon.")

    def _reorder_verts(self, clockwise=False, ref_index=0, increasing_length=True):
        """Sort the vertices.

        Sorting is done with respect to the direction of the normal vector. The
        default ordering is counterclockwise and preserves the vertex in the
        0th position. A different ordering may be requested; however, note that
        clockwise ordering will result in a negative signed area of the
        polygon.

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
        # The centroid cannot be computed until vertices are ordered, but for
        # convex polygons the mean of the vertices will be contained within the
        # shape so we can sort relative to that.
        verts, _ = _align_points_by_normal(
            self._normal, self._vertices - np.mean(self.vertices, axis=0)
        )

        # Compute the angle of each vertex, shift so that the chosen
        # reference_index has a value of zero, then move into the [0, 2pi]
        # range. The secondary sorting level is in terms of distance from the
        # origin.
        angles = np.arctan2(verts[:, 1], verts[:, 0])
        angles = np.mod(angles - angles[ref_index], 2 * np.pi)
        distances = np.linalg.norm(verts, axis=1)
        if not increasing_length:
            distances *= -1
        vert_order = np.lexsort((distances, angles))
        self._vertices = self._vertices[vert_order, :]

    @property
    def incircle_from_center(self):
        """:class:`~.Circle`: Get the largest concentric inscribed circle."""
        warnings.warn(
            "The incircle_from_center property is deprecated, use "
            "maximal_centered_bounded_circle instead.",
            DeprecationWarning,
        )
        return self.maximal_centered_bounded_circle

    @property
    def minimal_centered_bounding_circle(self):
        """:class:`~.Circle`: Get the smallest bounding concentric circle."""
        # The radius is determined by the furthest vertex from the center.
        return Circle(
            np.linalg.norm(self.vertices - self.center, axis=-1).max(), self.center
        )

    @property
    def maximal_centered_bounded_circle(self):
        """:class:`~.Circle`: Get the largest bounded concentric circle."""
        # The radius is determined by the furthest vertex from the center.
        v1s = self.vertices
        v2s = np.roll(self.vertices, shift=1, axis=0)
        deltas = v1s - v2s
        deltas /= np.linalg.norm(deltas, axis=-1)[:, np.newaxis]
        points = self.center[np.newaxis, :] - v1s[:, :]

        distances = np.linalg.norm(np.cross(points, deltas), axis=-1)

        radius = np.min(distances)
        return Circle(radius, self.center)

    def distance_to_surface(self, angles):  # noqa: D102
        # Bring the angles into the range for testing (also handles an
        # np.asarray for us).
        angles = np.mod(angles, 2 * np.pi)
        num_verts = len(self.vertices)

        # Rearrange the verts so that we start with the lowest angle
        verts, _ = _align_points_by_normal(self.normal, self.vertices - self.center)
        angles_to_vertices = np.arctan2(verts[:, 1], verts[:, 0])
        np.mod(angles_to_vertices, 2 * np.pi, out=angles_to_vertices)

        # find and reorganize to start with smallest angle
        shift = np.argmin(angles_to_vertices)
        verts = np.roll(verts, shift=-shift, axis=0)
        angles_to_vertices = np.roll(angles_to_vertices, shift=-shift, axis=0)

        # Pair vertices with numpy roll
        p1 = verts
        p2 = np.roll(p1, shift=-1, axis=0)

        # get the slopes
        slopes = np.ones(num_verts) * np.inf
        finite_slopes = (p1[:, 0] - p2[:, 0]) != 0.0
        slopes[finite_slopes] = (p1[finite_slopes, 1] - p2[finite_slopes, 1]) / (
            p1[finite_slopes, 0] - p2[finite_slopes, 0]
        )

        # Get the y_intercepts
        y_int = np.ones(num_verts) * np.inf
        y_int[finite_slopes] = (
            p1[finite_slopes, 1] - slopes[finite_slopes] * p1[finite_slopes, 0]
        )

        # Partition all angles into the angle ranges defined by the edges. For
        # the edge verts[-1]->verts[0] need to agument the final value so that
        # values > angles_to_vertices[-1] will work.
        angles_shifted = np.roll(angles_to_vertices, shift=-1, axis=0)
        eps = 1e-6  # Need angles_shifted[-1] > 2*pi so that 2*pi is in range.
        angles_shifted[-1] = 2 * np.pi + eps

        # Make the distances:
        distances = np.empty_like(angles)

        for i in range(num_verts):
            inside_range = (angles >= angles_to_vertices[i]) & (
                angles < angles_shifted[i]
            )
            if i == num_verts - 1:
                inside_range |= np.logical_and(
                    angles >= (angles_to_vertices[-1] - 2 * np.pi),
                    angles < angles_to_vertices[0],
                )

            if slopes[i] == 0:
                cos = np.cos(angles[inside_range])
                distances[inside_range] = np.sqrt(y_int[i] * y_int[i] / (1 - cos * cos))
            elif slopes[i] == np.inf or y_int[i] == np.inf:
                sin = np.sin(angles[inside_range])
                distances[inside_range] = np.sqrt(p1[i, 0] * p1[i, 0] / (1 - sin * sin))
            else:
                sl_k = np.tan(angles[inside_range])
                x = y_int[i] / (sl_k - slopes[i])
                y = sl_k * x
                distances[inside_range] = np.sqrt(x * x + y * y)

        return distances
