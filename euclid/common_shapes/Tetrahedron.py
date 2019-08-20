import numpy
from euclid.polyhedron import ConvexPolyhedron

# Example:
# from euclid.polyhedron.Tetrahedron import shape

points = [(1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1)]

shape = ConvexPolyhedron(numpy.array(points))
