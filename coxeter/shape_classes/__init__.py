"""Define shape classes.

This subpackage is the core of coxeter and defines various shapes in two and
three dimensions. Shapes support standard calculations like volume and area,
and they take care of various conveniences such as orienting polyhedron faces
and automatically identifying convex hulls of points.
"""

from .circle import Circle
from .base_classes import Shape, Shape2D, Shape3D
from .convex_polygon import ConvexPolygon
from .convex_polyhedron import ConvexPolyhedron
from .convex_spheropolyhedron import ConvexSpheropolyhedron
from .ellipse import Ellipse
from .ellipsoid import Ellipsoid
from .polygon import Polygon
from .polyhedron import Polyhedron
from .sphere import Sphere
from .convex_spheropolygon import ConvexSpheropolygon

__all__ = [
    'Circle',
    'ConvexPolyhedron',
    'ConvexSpheropolyhedron',
    'Ellipse',
    'Ellipsoid',
    'ConvexPolygon',
    'Polygon',
    'Polyhedron',
    'Shape',
    'Shape2D',
    'Shape3D',
    'Sphere',
    'ConvexSpheropolygon',
]
