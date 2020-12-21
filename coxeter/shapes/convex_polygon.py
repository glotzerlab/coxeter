"""Defines a convex polygon."""

import numpy as np
from scipy import interpolate
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
    verts_2d = _align_points_by_normal(normal, vertices)
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
        ...   square.bounding_circle.radius,
        ...   np.sqrt(2.))
        >>> square.center
        array([0., 0., 0.])
        >>> assert np.isclose(
        ...   square.circumcircle.radius,
        ...   np.sqrt(2.))
        >>> square.gsd_shape_spec
        {'type': 'Polygon', 'vertices': [[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]]}
        >>> assert np.isclose(square.incircle_from_center.radius, 1.0)
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
        if _is_convex(self._vertices, self._normal):
            # If points form a convex set, then we can order the vertices. We
            # cannot directly use the output of scipy's convex hull because our
            # polygon may be embedded in 3D, so we sort ourselves.
            self.reorder_verts()
        else:
            # If the shape is nonconvex, the user must provide ordered vertices
            # to uniquely identify the polygon. We must check if there are any
            # intersections to avoid complex (self-intersecting) polygons.
            raise ValueError("The provided vertices do not form a convex " "polygon.")

    @property
    def incircle_from_center(self):
        """:class:`~.Circle`: Get the largest concentric inscribed circle.

        The requirement that the circle be centered at the centroid of the
        shape distinguishes this circle from most typical incircle
        calculations.
        """
        v1s = self.vertices
        v2s = np.roll(self.vertices, shift=1, axis=0)
        deltas = v1s - v2s
        deltas /= np.linalg.norm(deltas, axis=-1)[:, np.newaxis]
        points = self.center[np.newaxis, :] - v1s[:, :]

        distances = np.linalg.norm(np.cross(points, deltas), axis=-1)

        radius = np.min(distances)
        return Circle(radius, self.center)

    def shape_kernel(self, value):
        """Shape kernel from -pi to pi.

        This algorithm assumes the vertices are ordered
        counterclockwise and that they start in the first quadrant.

        Args:
            value (array):
                Points over which to calculate the shape kernel
                which can only be from negative pi to pi.

        Returns:
            kernel (function: cubic spline to give shape kernel)
        """
        # This is fine spacing that is good enough for most problems
        # The kernel is needed for
        spacing = 50000
        verts = self.vertices[:, :2]
        # Center vertices
        verts[:, 0] = verts[:, 0] - np.average(verts[:, 0])
        verts[:, 1] = verts[:, 1] - np.average(verts[:, 1])
        theta = np.linspace(0, 2 * np.pi, spacing)
        theta_component = np.arctan(verts[:, 1] / verts[:, 0])
        angle_between = []
        pairs = []
        slopes = []
        kernel = np.zeros((len(theta), 1))
        y_int = []
        angle_range = np.zeros((len(verts), 2))
        for i in range(len(verts)):
            if i < len(verts) - 1:
                pairs.append([verts[i, :], verts[i + 1, :]])
            else:
                pairs.append([verts[i, :], verts[0, :]])
        for i in range(len(pairs)):
            norm = np.sqrt(np.dot(pairs[i][0], pairs[i][0])) * np.sqrt(
                np.dot(pairs[i][1], pairs[i][1])
            )
            angle_between.append(np.arccos(np.dot(pairs[i][0], pairs[i][1]) / norm))

        # get the slopes
        for i in range(len(pairs)):
            # if the slope is infinite the radial
            # solution goes one way otherwise it goes the other
            if pairs[i][0][0] - pairs[i][1][0] == 0.0:
                slopes.append("inf")
            else:
                slopes.append(
                    (pairs[i][0][1] - pairs[i][1][1])
                    / (pairs[i][0][0] - pairs[i][1][0])
                )

        # Get the y_intercept
        for i in range(len(pairs)):
            if slopes[i] == "inf" or abs(slopes[i] > 10 ** 5):
                y_int.append("inf")
            else:
                y_int.append(pairs[i][0][1] - slopes[i] * pairs[i][0][0])

        # Get the ranges of theta for the parameterization
        theta_initial = theta_component[0]
        for i in range(len(verts)):
            if i < len(verts) - 1:
                min_angle = theta_initial
                max_angle = theta_initial + angle_between[i]
                theta_initial = max_angle
                angle_range[i, 0] = min_angle
                angle_range[i, 1] = max_angle
            else:
                angle_range[i, 0] = max_angle
                angle_range[i, 1] = theta_component[0]

        # apply the equation of the line that applies in each region
        # we know that y = mx+b and we know that tan(theta) = y/x

        for i in range(len(theta)):
            for j in range(len(angle_range)):
                if j < len(angle_range) - 1:
                    if theta[i] > angle_range[j, 0] and theta[i] <= angle_range[j, 1]:
                        if slopes[j] == "inf" or abs(slopes[j] > 10 ** 5):
                            kernel[i] = np.sqrt(
                                verts[j, 0] ** 2 * (1 + np.tan(theta[i]) ** 2)
                            )
                            break
                        else:
                            kernel[i] = np.sqrt(
                                (y_int[j] / (np.tan(theta[i]) - slopes[j])) ** 2
                                * (np.tan(theta[i]) ** 2 + 1)
                            )
                            break

                else:
                    if slopes[j] == "inf" or abs(slopes[j]) > 10 ** 5:

                        kernel[i] = np.sqrt(
                            verts[j, 0] ** 2 * (1 + np.tan(theta[i]) ** 2)
                        )
                    else:
                        kernel[i] = np.sqrt(
                            (y_int[j] / (np.tan(theta[i]) - slopes[j])) ** 2
                            * (np.tan(theta[i]) ** 2 + 1)
                        )
        kernel = kernel[:, 0]
        theta = theta - np.pi
        kernel = interpolate.splrep(theta, kernel, s=0)

        # Now we input this into an cubic spline

        return interpolate.splev(value, kernel, der=0)
