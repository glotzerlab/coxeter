"""Defines a convex spheropolygon.

A convex spheropolygon is defined by the Minkowski sum of a convex polygon and
a circle of some radius.
"""

import numpy as np

from .base_classes import Shape2D
from .convex_polygon import ConvexPolygon, _is_convex


class ConvexSpheropolygon(Shape2D):
    """A convex spheropolygon.

    Args:
        vertices (:math:`(N, 3)` or :math:`(N, 2)` :class:`numpy.ndarray`):
            The vertices of the polygon.
        radius (float):
            The rounding radius of the spheropolygon.
        normal (sequence of length 3 or None):
            The normal vector to the polygon. If :code:`None`, the normal
            is computed by taking the cross product of the vectors formed
            by the first three vertices :code:`np.cross(vertices[2, :] -
            vertices[1, :], vertices[0, :] - vertices[1, :])`. Since this
            arbitrary choice may not preserve the orientation of the
            provided vertices, users may provide a normal instead
            (Default value: None).

    Example:
        >>> rounded_tri = coxeter.shapes.ConvexSpheropolygon(
        ...   [[1, 0], [0, 1], [-1, 0]], radius=.1)
        >>> rounded_tri.area
        1.5142...
        >>> rounded_tri.gsd_shape_spec
        {'type': 'Polygon', 'vertices': [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        'rounding_radius': 0.1}
        >>> rounded_tri.radius
        0.1
        >>> rounded_tri.signed_area
        1.5142...
        >>> rounded_tri.vertices
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [-1.,  0.,  0.]])

    """

    def __init__(self, vertices, radius, normal=None):
        self.radius = radius
        self._polygon = ConvexPolygon(vertices, normal)
        if not _is_convex(self.vertices, self._polygon.normal):
            raise ValueError("The vertices do not define a convex polygon.")

    def _require_xy_plane(self, allow_negative_z=False):
        normal = self.polygon.normal
        if allow_negative_z:
            normal = np.abs(normal)
        if not np.array_equal(normal, np.array([0, 0, 1])):
            class_name = type(self).__name__
            raise ValueError(
                f"This method requires the {class_name} to be embedded in the xy "
                "plane with a normal vector pointing along the positive z "
                f"direction. The normal of this {class_name} is "
                f"{self.polygon.normal}."
            )

    @property
    def polygon(self):
        """:class:`~coxeter.shapes.ConvexPolygon`: The underlying polygon."""
        return self._polygon

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        self._require_xy_plane()
        return {
            "type": "Polygon",
            "vertices": self.polygon.vertices[:, :2].tolist(),
            "rounding_radius": self.radius,
        }

    @property
    def vertices(self):
        """:math:`(N_{verts}, 3)` :class:`numpy.ndarray` of float: Get the vertices of the spheropolygon."""  # noqa: E501
        return self.polygon.vertices

    @property
    def radius(self):
        """float: Get or set the rounding radius."""
        return self._radius

    @radius.setter
    def radius(self, value):
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius must be greater than or equal to zero.")

    def _rescale(self, scale):
        """Multiply length scale.

        Args:
            scale (float):
                Scale factor.
        """
        self.polygon._vertices *= scale
        self.radius *= scale

    @property
    def signed_area(self):
        """Get the signed area of the spheropolygon.

        The area is computed as the sum of the underlying polygon area and the
        area added by the rounding radius.
        """
        poly_area = self.polygon.signed_area

        drs = self.vertices - np.roll(self.vertices, shift=-1, axis=0)
        edge_area = np.sum(np.linalg.norm(drs, axis=1)) * self.radius
        cap_area = np.pi * self.radius * self.radius
        sphero_area = edge_area + cap_area

        if poly_area < 0:
            return poly_area - sphero_area
        else:
            return poly_area + sphero_area

    @property
    def area(self):
        """Get or set the polygon's area.

        To get the area, we simply compute the signed area and take the
        absolute value.
        """
        return np.abs(self.signed_area)

    @area.setter
    def area(self, value):
        if value > 0:
            scale = np.sqrt(value / self.area)
            self._rescale(scale)
        else:
            raise ValueError("Area must be greater than zero.")

    @property
    def perimeter(self):
        """float: Get the perimeter of the spheropolygon."""
        return self.polygon.perimeter + 2 * np.pi * self.radius

    @perimeter.setter
    def perimeter(self, value):
        if value > 0:
            scale = value / self.perimeter
            self._rescale(scale)
        else:
            raise ValueError("Perimeter must be greater than zero.")

    def is_inside(self, points):
        """Determine whether points are contained in this spheropolygon.

        .. note::

            Points on the boundary of the shape will return :code:`True`.

        Args:
            points (:math:`(N, 3)` :class:`numpy.ndarray`):
                The points to test. The points must lie in the xy plane.

        Returns:
            :math:`(N, )` :class:`numpy.ndarray`:
                Boolean array indicating which points are contained in the
                spheropolygon.

        Example:
            >>> rounded_square = coxeter.shapes.ConvexSpheropolygon(
            ...   [[0, 0], [1, 0], [1, 1], [0, 1]], radius=0.5)
            >>> rounded_square.is_inside([[0, 0, 0], [1., 1.5, 0], [1.5, 1.5, 0]])
            array([ True, True, False])

        """
        self._require_xy_plane()
        points = np.atleast_2d(points)
        # For convenience, we support providing points without z components
        if points.shape[1] == 2:
            points = np.hstack((points, np.zeros((points.shape[0], 1))))
        if not np.all(points[:, 2] == 0):
            raise ValueError(
                "All provided points must be in the xy plane (points[:, 2] == 0)."
            )

        # Determine which points are in the polygon
        in_polygon = self.polygon.is_inside(points)

        # Exit early if all points are inside the convex polygon
        if np.all(in_polygon):
            return in_polygon

        def _line_segment_to_point_distance(point, segment_start, segment_end):
            """Finds minimal distance between a point and a line segment."""
            start_to_end = segment_end - segment_start
            start_to_point = point - segment_start
            end_to_point = point - segment_end
            past_end = np.dot(start_to_end, end_to_point) > 0
            if past_end:
                return np.linalg.norm(end_to_point)
            past_start = np.dot(start_to_end, start_to_point) < 0
            if past_start:
                return np.linalg.norm(start_to_point)
            perpendicular_distance = np.linalg.norm(
                np.cross(start_to_end, start_to_point)
            ) / np.linalg.norm(start_to_end)
            return perpendicular_distance

        in_sphero_shape = np.zeros(len(points), dtype=bool)
        for point_index in np.where(~in_polygon)[0]:
            point = points[point_index]
            for v_i, v_j in zip(
                self.vertices, np.roll(self.vertices, shift=-1, axis=0)
            ):
                if _line_segment_to_point_distance(point, v_i, v_j) <= self.radius:
                    in_sphero_shape[point_index] = True
                    break

        return in_polygon | in_sphero_shape

    def __repr__(self):
        return (
            f"coxeter.shapes.ConvexSpheropolygon(vertices={self.vertices.tolist()}, "
            f"radius={self.radius}, normal={self.polygon.normal.tolist()})"
        )
