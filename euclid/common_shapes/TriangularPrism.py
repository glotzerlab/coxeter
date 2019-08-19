from numpy import sqrt
import numpy
from euclid.polyhedron import ConvexPolyhedron

# Example:
# from euclid.polyhedron.TriangularPrism import shape
points = [
    (-1/(2*sqrt(3)), -1/2, -1/2),
    (-1/(2*sqrt(3)), -1/2, 1/2),
    (-1/(2*sqrt(3)), 1/2, -1/2),
    (-1/(2*sqrt(3)), 1/2, 1/2),
    (1/sqrt(3), 0, -1/2),
    (1/sqrt(3), 0, 1/2),
]

shape = ConvexPolyhedron(numpy.array(points))
