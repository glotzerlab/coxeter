from numpy import sqrt
import numpy
from coxeter.shape_classes.convex_polyhedron import ConvexPolyhedron

points = [
    (-1, 0, -1 / 2),
    (-1, 0, 1 / 2),
    (-1 / 2, -sqrt(3) / 2, -1 / 2),
    (-1 / 2, -sqrt(3) / 2, 1 / 2),
    (-1 / 2, sqrt(3) / 2, -1 / 2),
    (-1 / 2, sqrt(3) / 2, 1 / 2),
    (1 / 2, -sqrt(3) / 2, -1 / 2),
    (1 / 2, -sqrt(3) / 2, 1 / 2),
    (1 / 2, sqrt(3) / 2, -1 / 2),
    (1 / 2, sqrt(3) / 2, 1 / 2),
    (1, 0, -1 / 2),
    (1, 0, 1 / 2),
]

shape = ConvexPolyhedron(numpy.array(points))
