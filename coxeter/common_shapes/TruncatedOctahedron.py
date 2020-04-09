from numpy import sqrt
import numpy
from coxeter.shape_classes.convex_polyhedron import ConvexPolyhedron


points = [[-(3 / 2), -(1 / 2), 0],
          [-(3 / 2), 1 / 2, 0],
          [-1, -1, -(1 / sqrt(2))],
          [-1, -1, 1 / sqrt(2)],
          [-1, 1, -(1 / sqrt(2))],
          [-1, 1, 1 / sqrt(2)],
          [-(1 / 2), -(3 / 2), 0],
          [-(1 / 2), -(1 / 2), -sqrt(2)],
          [-(1 / 2), -(1 / 2), sqrt(2)],
          [-(1 / 2), 1 / 2, -sqrt(2)],
          [-(1 / 2), 1 / 2, sqrt(2)],
          [-(1 / 2), 3 / 2, 0],
          [1 / 2, -(3 / 2), 0],
          [1 / 2, -(1 / 2), -sqrt(2)],
          [1 / 2, -(1 / 2), sqrt(2)],
          [1 / 2, 1 / 2, -sqrt(2)],
          [1 / 2, 1 / 2, sqrt(2)],
          [1 / 2, 3 / 2, 0],
          [1, -1, -(1 / sqrt(2))],
          [1, -1, 1 / sqrt(2)],
          [1, 1, -(1 / sqrt(2))],
          [1, 1, 1 / sqrt(2)],
          [3 / 2, -(1 / 2), 0],
          [3 / 2, 1 / 2, 0]
          ]

shape = ConvexPolyhedron(numpy.array(points))
