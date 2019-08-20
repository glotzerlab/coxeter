import numpy
from euclid.polyhedron import ConvexPolyhedron

# Example:
# from euclid.FreudShape.Cube import shape

points = [(1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1),
          (1, 1, -1), (1, -1, -1), (-1, -1, -1), (-1, 1, -1)]

shape = ConvexPolyhedron(numpy.array(points))
