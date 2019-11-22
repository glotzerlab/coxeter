from numpy import sqrt
import numpy
from euclid.polyhedron import ConvexPolyhedron

points = [
    (-(1 / sqrt(2)), 0, 0),
    (0, 1 / sqrt(2), 0),
    (0, 0, -(1 / sqrt(2))),
    (0, 0, 1 / sqrt(2)),
    (0, -(1 / sqrt(2)), 0),
    (1 / sqrt(2), 0, 0),
]

shape = ConvexPolyhedron(numpy.array(points))
