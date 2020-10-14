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
        ...   [[-1, 0], [0, 1], [1, 0]], radius=.1)
        >>> rounded_tri.area
        1.5142...
        >>> rounded_tri.center
        array([0.        , 0.333..., 0.        ])
        >>> rounded_tri.gsd_shape_spec
        {'type': 'Polygon', 'vertices': [[-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], 'rounding_radius': 0.1}
        >>> rounded_tri.polygon
        <coxeter.shapes.convex_polygon.ConvexPolygon object at 0x...>
        >>> rounded_tri.radius
        0.1
        >>> rounded_tri.signed_area
        1.5142...
        >>> rounded_tri.vertices
        array([[-1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  0.,  0.]])

    """

    def __init__(self, vertices, radius, normal=None):
        self.radius = radius
        self._polygon = ConvexPolygon(vertices, normal)
        if not _is_convex(self.vertices, self._polygon.normal):
            raise ValueError("The vertices do not define a convex polygon.")

    def reorder_verts(self, clockwise=False, ref_index=0, increasing_length=True):
        """Sort the vertices.

        For more information see
        :meth:`~coxeter.shapes.Polygon.reorder_verts`.

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

        Example:
            >>> rounded_tri = coxeter.shapes.ConvexSpheropolygon(
            ...   [[-1, 0], [0, 1], [1, 0]], radius=0.1)
            >>> rounded_tri.vertices
            array([[-1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 1.,  0.,  0.]])
            >>> rounded_tri.reorder_verts(clockwise=True)
            >>> rounded_tri.vertices
            array([[-1.,  0.,  0.],
                   [ 1.,  0.,  0.],
                   [ 0.,  1.,  0.]])

        """
        self._polygon.reorder_verts(clockwise, ref_index, increasing_length)

    @property
    def polygon(self):
        """:class:`~coxeter.shapes.ConvexPolygon`: The underlying polygon."""
        return self._polygon

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {
            "type": "Polygon",
            "vertices": self._polygon._vertices.tolist(),
            "rounding_radius": self._radius,
        }

    @property
    def vertices(self):
        """:math:`(N_{verts}, 3)` :class:`numpy.ndarray` of float: Get the vertices of the spheropolygon."""  # noqa: E501
        return self._polygon.vertices

    @property
    def radius(self):
        """float: Get or set the rounding radius."""
        return self._radius

    @radius.setter
    def radius(self, value):
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius must be greater or equal to zero.")

    @property
    def signed_area(self):
        """Get the signed area of the spheropolygon.

        The area is computed as the sum of the underlying polygon area and the
        area added by the rounding radius.
        """
        poly_area = self._polygon.signed_area

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
            scale_factor = np.sqrt(value / self.area)
            self.polygon._vertices *= scale_factor
            self.radius *= scale_factor
        else:
            raise ValueError("Area must be greater than zero.")

    @property
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self._polygon.center

    @center.setter
    def center(self, new_center):
        self._polygon.center = new_center

    @property
    def perimeter(self):
        """float: Get the perimeter of the spheropolygon."""
        return self._polygon.perimeter + 2 * np.pi * self.radius

    @perimeter.setter
    def perimeter(self, value):
        if value > 0:
            scale_factor = value / self.perimeter
            self.polygon._vertices *= scale_factor
            self.radius *= scale_factor
        else:
            raise ValueError("Perimeter must be greater than zero.")
