import numpy
from euclid.polyhedron import ConvexPolyhedron

points = [
    (1, 0.1, 0.1),
    (0.1, 1, 0.1),
    (0.1, 0.1, 1),
    (1, -0.1, -0.1),
    (0.1, -0.1, -1),
    (0.1, -1, -0.1),
    (-0.1, 1, -0.1),
    (-0.1, 0.1, -1),
    (-1, 0.1, -0.1),
    (-0.1, -0.1, 1),
    (-0.1, -1, 0.1),
    (-1, -0.1, 0.1)
]

shape = ConvexPolyhedron(numpy.array(points))
