"""Certain common shape families that can be analytically generated."""

from ..shape_classes import ConvexPolygon
from .shape_family import ShapeFamily
import numpy as np


class RegularNGonFamily(ShapeFamily):
    """A family of regular polygons.

    The following parameters are required by this class:

      - :math:`n`: The number of vertices of the polygon
    """
    def __call__(self, n):
        return ConvexPolygon(self.make_vertices(n))

    def make_vertices(self, n):
        r = 1  # The radius of the circle
        theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        pos = np.array([np.cos(theta), np.sin(theta)])

        # First normalize to guarantee that the limiting case of an infinite
        # number of vertices produces a circle of area r^2.
        pos /= (np.sqrt(np.pi)/r)

        # Area of an n-gon inscribed in a circle
        # A_poly = ((n*r**2)/2)*np.sin(2*np.pi/n)
        # A_circ = np.pi*r**2
        # pos *= np.sqrt(A_circ/A_poly)
        A_circ_A_poly_sq = np.pi/((n/2)*np.sin(2*np.pi/n))
        pos *= np.sqrt(A_circ_A_poly_sq)

        return pos.T
