"""Defines a convex polygon."""

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
        if _is_convex(self.vertices, self.normal):
            # If points form a convex set, then we can order the vertices. We
            # cannot directly use the output of scipy's convex hull because our
            # polygon may be embedded in 3D, so we sort ourselves.
            self.reorder_verts()
        else:
            # If the shape is nonconvex, the user must provide ordered vertices
            # to uniquely identify the polygon. We must check if there are any
            # intersections to avoid complex (self-intersecting) polygons.
            raise ValueError("The provided vertices do not form a convex polygon.")

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

    def distance_to_surface(self, angles):
        """Distance to the surface of the shape.

        This calculation assumes vertices are ordered counterclockwise

        For more generic information about this calculation, see
        `Shape.distance_to_surface`.
        """
        verts = self.vertices[:, :2]
        theta = angles
        verts -= np.average(verts, axis=0)

        # Rearrange the verts so that we start with the lowest angle
        theta_component = np.arctan2(verts[:, 1], verts[:, 0])

        # Go from 0 to 2 pi
        theta_component[theta_component < 0] += 2 * np.pi

        # find and reorganize to start with smallest angle
        verts = np.roll(
            verts,
            len(theta_component)
            - np.where((np.min(theta_component) == theta_component))[0],
            axis=0,
        )

        theta_component = np.roll(
            theta_component,
            len(theta_component)
            - np.where((np.min(theta_component) == theta_component))[0],
            axis=0,
        )

        # Pair vertices with numpy roll
        p1 = verts
        p2 = np.roll(verts, -1, axis=0)

        slopes = np.zeros(len(p1), dtype=object)
        y_int = np.zeros(len(p1), dtype=object)
        angle_range = np.zeros((len(verts), 2))

        norm = np.sqrt(np.sum(p1 * p1, axis=1) * np.sum(p2 * p2, axis=1))
        angle_between = np.arccos(np.sum(p1 * p2, axis=1) / norm)

        # get the slopes

        sl_inf = np.where((p1[:, 0] - p2[:, 0] == 0.0))
        sl_ninf = np.where((p1[:, 0] - p2[:, 0] != 0.0))

        slopes[sl_inf] = "inf"
        slopes[sl_ninf] = (p1[sl_ninf, 1] - p2[sl_ninf, 1]) / (
            p1[sl_ninf, 0] - p2[sl_ninf, 0]
        )

        # Get the y_intercepts
        y_inf1 = np.where((slopes == "inf"))
        y_ninf = np.where((slopes != "inf"))

        y_int[y_inf1] = "inf"
        y_int[y_ninf] = p1[y_ninf, 1] - slopes[y_ninf] * p1[y_ninf, 0]

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
                angle_range[i, 1] = 2 * np.pi

        # Make the kernel:

        kernel = np.zeros_like(theta)

        for i in range(len(angle_range)):
            if slopes[i] == 0:
                wh = np.where(
                    (theta >= angle_range[i, 0]) & (theta <= angle_range[i, 1])
                )
                kernel[wh] = np.sqrt(
                    y_int[i] * y_int[i] / (1 - np.cos(theta[wh]) * np.cos(theta[wh]))
                )
            elif slopes[i] == "inf" or y_int[i] == "inf":
                if i != len(angle_range) - 1:
                    wh = np.where(
                        (theta >= angle_range[i, 0]) & (theta <= angle_range[i, 1])
                    )
                    x_int = p1[i, 0]
                    kernel[wh] = np.sqrt(
                        x_int * x_int / (1 - np.sin(theta[wh]) * np.sin(theta[wh]))
                    )
                else:
                    x_int = p1[i, 0]
                    wh = np.where(
                        ((theta >= angle_range[i, 0]) & (theta <= angle_range[i, 1]))
                        | ((theta >= 0) & (theta <= angle_range[0, 0]))
                    )
                    kernel[wh] = np.sqrt(
                        x_int * x_int / (1 - np.sin(theta[wh]) * np.sin(theta[wh]))
                    )
            else:
                if i != len(angle_range) - 1:
                    wh = np.where(
                        (theta >= angle_range[i, 0]) & (theta <= angle_range[i, 1])
                    )
                else:
                    wh = np.where(
                        ((theta >= angle_range[i, 0]) & (theta <= angle_range[i, 1]))
                        | ((theta >= 0) & (theta <= angle_range[0, 0]))
                    )
                sl_k = np.tan(theta[wh])
                x = y_int[i] / (sl_k - slopes[i])
                y = sl_k * x
                kernel[wh] = np.sqrt(x * x + y * y)

        return kernel
