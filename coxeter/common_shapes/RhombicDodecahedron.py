from numpy import sqrt
import numpy
from coxeter.shape_classes.convex_polyhedron import ConvexPolyhedron

points = [
    (-sqrt(2 / 3), -sqrt(2 / 3), 0),
    (-sqrt(2 / 3), 0, -(1 / sqrt(3))),
    (-sqrt(2 / 3), 0, 1 / sqrt(3)),
    (-sqrt(2 / 3), sqrt(2 / 3), 0),
    (0, -sqrt(2 / 3), -(1 / sqrt(3))),
    (0, -sqrt(2 / 3), 1 / sqrt(3)),
    (0, 0, -2 / sqrt(3)),
    (0, 0, 2 / sqrt(3)),
    (0, sqrt(2 / 3), -(1 / sqrt(3))),
    (0, sqrt(2 / 3), 1 / sqrt(3)),
    (sqrt(2 / 3), -sqrt(2 / 3), 0),
    (sqrt(2 / 3), 0, -(1 / sqrt(3))),
    (sqrt(2 / 3), 0, 1 / sqrt(3)),
    (sqrt(2 / 3), sqrt(2 / 3), 0),
]

shape = ConvexPolyhedron(numpy.array(points))
