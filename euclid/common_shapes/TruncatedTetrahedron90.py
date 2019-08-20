import numpy
from euclid.polyhedron import ConvexPolyhedron

# Example:
# from euclid.polyhedron.TruncatedTetrahedron90 import shape
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
