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
    """

    def __init__(self, vertices, radius, normal=None):
        if radius < 0:
            raise ValueError("The radius must be positive.")
        self._polygon = ConvexPolygon(vertices, normal)
        if not _is_convex(self.vertices, self._polygon.normal):
            raise ValueError("The vertices do not define a convex polygon.")
        self._radius = radius

    def reorder_verts(self, clockwise=False, ref_index=0, increasing_length=True):
        """Sort the vertices.

        For more information see
        :meth:`~coxeter.shape_classes._polygon.Polygon.reorder_verts`.

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
        self._polygon.reorder_verts(clockwise, ref_index, increasing_length)

    @property
    def polygon(self):
        """:class:`~coxeter.shape_classes.ConvexPolygon`: The underlying polygon."""  # noqa: E501
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
        """:math:`(N_{verts}, 3) :class:`numpy.ndarray` of float: Get the vertices of the spheropolygon."""  # noqa: E501
        return self._polygon.vertices

    @property
    def radius(self):
        """float: Get or set the rounding radius."""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

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
        # TODO: area setter for spheropolygon
        return np.abs(self.signed_area)

    @property
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self._polygon.center

    @center.setter
    def center(self, new_center):
        self._polygon.center = new_center
