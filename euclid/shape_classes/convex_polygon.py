from .polygon import Polygon, _align_points_by_normal
from scipy.spatial import ConvexHull
import numpy as np


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
    def __init__(self, vertices, normal=None, planar_tolerance=1e-5):
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
        """
        super(ConvexPolygon, self).__init__(vertices, normal, planar_tolerance,
                                            False)
        if _is_convex(self._vertices, self._normal):
            # If points form a convex set, then we can order the vertices. We
            # cannot directly use the output of scipy's convex hull because our
            # polygon may be embedded in 3D, so we sort ourselves.
            self.reorder_verts()
        else:
            # If the shape is nonconvex, the user must provide ordered vertices
            # to uniquely identify the polygon. We must check if there are any
            # intersections to avoid complex (self-intersecting) polygons.
            raise ValueError("The provided vertices do not form a convex "
                             "polygon.")

    @property
    def incircle_from_center(self):
        """The largest circle centered at the centroid that fits inside the
        convex polygon, given by a center and a radius."""
        v1s = self.vertices
        v2s = np.roll(self.vertices, shift=1, axis=0)
        deltas = v1s - v2s
        deltas /= np.linalg.norm(deltas, axis=-1)[:, np.newaxis]
        points = self.center[np.newaxis, :] - v1s[:, :]

        distances = np.linalg.norm(np.cross(points, deltas), axis=-1)

        radius = np.min(distances)
        return self.center, radius
