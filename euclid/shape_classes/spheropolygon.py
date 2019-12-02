import numpy as np

from .polygon import Polygon


class Spheropolygon(object):
    def __init__(self, vertices, radius, normal=None):
        """A simple (i.e. non-self-overlapping) spheropolygon.

        The polygon may be embedded in 3-dimensions, in which case the normal
        vector determines which way is "up". Note that the precise normal
        vector is not important to the winding number because the winding
        number is invariant to projection, all we need to know is the sign of
        each component of the normal vector.

        Args:
            vertices (:math:`(N, 3)` or :math:`(N, 2)` :class:`numpy.ndarray`):
                The vertices of the polygon.
            normal (sequence of length 3 or None):
                The normal vector to the polygon. If :code:`None`, the normal
                is computed by taking the cross product of the vectors formed
                by the first three vertices :code:`np.cross(vertices[2, :] -
                vertices[1, :], vertices[0, :] - vertices[1, :])`. Since this
                arbitrary choice may not preserve the orientation of the
                provided vertices, users may provide a normal instead
                (Default value: None).

        """
        if radius < 0:
            raise ValueError("The radius must be positive.")
        self.polygon = Polygon(vertices, normal)
        self._radius = radius

    @property
    def radius(self):
        """The rounding radius."""
        return self._radius

    @property
    def signed_area(self):
        """Get the signed area of the spheropolygon.

        The area is computed as the sum of the underlying polygon area and the
        area added by the rounding radius.
        """
        poly_area = self.polygon.signed_area

        if poly_area < 0:
            return poly_area - self._sphero_area
        else:
            return poly_area + self._sphero_area

    @property
    def _sphero_area(self):
        """The area added to the core polygon area by the rounding radius.

        The area is computed as the sum of area added to the edges and the area
        of the circular caps.
        """
        drs = self.polygon.vertices - np.roll(self.polygon.vertices,
                                              shift=-1, axis=0)
        edge_area = np.sum(np.linalg.norm(drs, axis=1)) * self.radius
        cap_area = np.pi * self.radius * self.radius
        return edge_area + cap_area

    @property
    def area(self):
        """Get or set the polygon's area (setting rescales vertices).

        To get the area, we simply compute the signed area and take the
        absolute value.
        """
        return np.abs(self.signed_area)

    @property
    def center(self):
        return self.polygon.center

    @center.setter
    def center(self, new_center):
        self.polygon.center = new_center
