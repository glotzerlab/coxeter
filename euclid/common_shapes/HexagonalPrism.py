from numpy import sqrt
import numpy
from euclid.polyhedron import ConvexPolyhedron

# Example:
# from euclid.polyhedron.HexagonalPrism import shape
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
