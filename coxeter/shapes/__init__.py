# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Define shape classes.

This subpackage is the core of coxeter and defines various shapes in two and
three dimensions. Shapes support standard calculations like volume and area,
and they take care of various conveniences such as orienting polyhedron faces
and automatically identifying convex hulls of points.
"""

from .base_classes import Shape, Shape2D, Shape3D
from .circle import Circle
from .convex_polygon import ConvexPolygon
from .convex_polyhedron import ConvexPolyhedron
from .convex_spheropolygon import ConvexSpheropolygon
from .convex_spheropolyhedron import ConvexSpheropolyhedron
from .ellipse import Ellipse
from .ellipsoid import Ellipsoid
from .polygon import Polygon
from .polyhedron import Polyhedron
from .sphere import Sphere

__all__ = [
    "Circle",
    "ConvexPolygon",
    "ConvexPolyhedron",
    "ConvexSpheropolygon",
    "ConvexSpheropolyhedron",
    "Ellipse",
    "Ellipsoid",
    "Polygon",
    "Polyhedron",
    "Shape",
    "Shape2D",
    "Shape3D",
    "Sphere",
]
