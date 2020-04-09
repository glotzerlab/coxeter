from numpy import sqrt
import numpy
from coxeter.shape_classes.convex_polyhedron import ConvexPolyhedron

phi = (1. + sqrt(5.)) / 2.
inv = 2. / (1. + sqrt(5.))
points = [
    (-1, -1, -1),
    (-1, -1, 1),
    (-1, 1, -1),
    (-1, 1, 1),
    (1, -1, -1),
    (1, -1, 1),
    (1, 1, -1),
    (1, 1, 1),
    (0, -inv, -phi),
    (0, -inv, phi),
    (0, inv, -phi),
    (0, inv, phi),
    (-inv, -phi, 0),
    (-inv, phi, 0),
    (inv, -phi, 0),
    (inv, phi, 0),
    (-phi, 0, -inv),
    (-phi, 0, inv),
    (phi, 0, -inv),
    (phi, 0, inv)
]
# produces a dodecahedron with circumradius sqrt(3)

shape = ConvexPolyhedron(numpy.array(points))
