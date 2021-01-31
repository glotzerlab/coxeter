"""Defines a convex spheropolygon.

A convex spheropolygon is defined by the Minkowski sum of a convex polygon and
a circle of some radius.
"""

import numpy as np

from .base_classes import Shape2D
from .circle import Circle
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

        For more information see `Polygon.reorder_verts`.

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
        self.polygon.reorder_verts(clockwise, ref_index, increasing_length)

    @property
    def polygon(self):
        """:class:`~coxeter.shapes.ConvexPolygon`: The underlying polygon."""
        return self._polygon

    @property
    def gsd_shape_spec(self):
        """dict: Get a :ref:`complete GSD specification <shapes>`."""  # noqa: D401
        return {
            "type": "Polygon",
            "vertices": self.polygon.vertices.tolist(),
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
    def center(self):
        """:math:`(3, )` :class:`numpy.ndarray` of float: Get or set the centroid of the shape."""  # noqa: E501
        return self.polygon.center

    @center.setter
    def center(self, value):
        self.polygon.center = value

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

    def _get_outward_unit_normal(self, vector, point):
        """Get the outward unit normal vector to the given vector going through point."""  # noqa: E501
        # get an outward unit normal for all kinds of slopes
        if vector[0] == 0:  # infinte slope
            xint = point[0]
            nvecu = np.array([np.sign(xint) * 1, 0])
            return nvecu

        slope = vector[1] / vector[0]
        yint = point[1] - slope * point[0]
        if slope == 0:
            nvecu = np.array([0, np.sign(yint) * 1])
        else:  # now do the math for non-weird slopes
            normal_vector = np.array([-slope, 1])
            nvecu = normal_vector / np.linalg.norm(normal_vector)
            # make sure unit normal is OUTWARD
            if slope > 0:
                if yint > 0 and nvecu[0] > 0:
                    nvecu *= -1
                elif yint < 0 and nvecu[0] < 0:
                    nvecu *= -1
            else:
                if yint > 0 and nvecu[0] < 0:
                    nvecu *= -1
                elif yint < 0 and nvecu[0] > 0:
                    nvecu *= -1
        return nvecu

    def _get_polar_angle(self, vector):
        """Get the polar angle for the given vector from 0 to 2pi."""
        if vector[0] == 0:
            return (1 - 1 * np.sign(vector[1]) / 2) * np.pi
        angle = np.arctan(vector[1] / vector[0])
        if vector[0] < 0:  # 2nd/3rd quadrant
            angle += np.pi
        elif vector[0] > 0 and vector[1] < 0:  # 4th quadrant
            angle += 2 * np.pi
        return angle

    def distance_to_surface(self, angles):
        """Distance to the surface of this shape.

        This algorithm assumes vertices are ordered counterclockwise.

        For more general information about this calculation, see
        `Shape.distance_to_surface`.
        """
        num_verts = len(self._polygon.vertices)
        verts = self._polygon.vertices[:, :2]
        verts -= np.average(verts, axis=0)
        # intermediate data
        new_verts = np.zeros_like(verts)  # expanded vertices
        angle_ranges = []  # angle ranges where we need to round

        # compute intermediates
        v1 = np.roll(verts, 1, axis=0)
        v2 = verts
        v3 = np.roll(verts, -1, axis=0)
        v12 = v1 - v2
        v32 = v3 - v2
        v12n = np.zeros_like(v12)
        v32n = np.zeros_like(v32)
        for i in range(num_verts):
            v12n[i] = self._get_outward_unit_normal(v12[i], v2[i])
            v32n[i] = self._get_outward_unit_normal(v32[i], v2[i])

        # get the new vertex corresponding to the old one
        v12norm = np.linalg.norm(v12, axis=1)
        v32norm = np.linalg.norm(v32, axis=1)
        dot = np.multiply(v32, v12).sum(1)
        phi = np.arccos(dot / (v32norm * v12norm))
        uvec = v12n + v32n
        uvec /= np.linalg.norm(uvec, axis=1)[:, None]
        new_verts = v2 + uvec * self.radius / (np.sin(phi / 2)[:, None])

        # define the angle range for rounding
        pt1 = v2 + v12n * self.radius
        pt3 = v2 + v32n * self.radius
        for i in range(num_verts):
            angle_ranges.append(
                (self._get_polar_angle(pt1[i]), self._get_polar_angle(pt3[i]))
            )

        # compute shape kernel for the new set of vertices
        kernel = ConvexPolygon(new_verts).distance_to_surface(angles)

        # get the shape kernel for this shape by adjusting indices of shape kernel
        # for the new vertices
        for i in range(len(angle_ranges)):
            theta1, theta2 = angle_ranges[i]
            if theta2 < theta1:  # case the angle range crosses the 2pi boundary
                indices = np.where((angles >= theta1) | (angles <= theta2))
            else:
                indices = np.where((angles >= theta1) & (angles <= theta2))
            v = verts[i]
            norm_v = np.linalg.norm(v)
            phi = self._get_polar_angle(v)
            a = 1
            b = -2 * norm_v * np.cos(angles[indices] - phi)
            c = norm_v ** 2 - self.radius ** 2
            kernel[indices] = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        return kernel

    @property
    def minimal_bounding_circle(self):
        """:class:`~.Circle`: Get the minimal bounding circle."""
        polygon_circle = self.polygon.minimal_bounding_circle
        return Circle(polygon_circle.radius + self.radius, polygon_circle.center)

    @property
    def minimal_centered_bounding_circle(self):
        """:class:`~.Circle`: Get the minimal concentric bounding circle."""
        polygon_circle = self.polygon.minimal_centered_bounding_circle
        return Circle(polygon_circle.radius + self.radius, polygon_circle.center)
