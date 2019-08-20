from numpy import sqrt
import numpy
from euclid.polyhedron import ConvexPolyhedron

# Example:
# from euclid.polyhedron.TriangularDipyramid import shape
points = [
    (0, 0, -sqrt(2/3)),
    (0, 0, sqrt(2/3)),
    (-1/(2*sqrt(3)), -1/2, 0),
    (-1/(2*sqrt(3)), 1/2, 0),
    (1/sqrt(3), 0, 0),
]

shape = ConvexPolyhedron(numpy.array(points))
