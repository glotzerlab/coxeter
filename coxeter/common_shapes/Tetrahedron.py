import numpy
from coxeter.shape_classes.convex_polyhedron import ConvexPolyhedron


points = [(1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1)]

shape = ConvexPolyhedron(numpy.array(points))
