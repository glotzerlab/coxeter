from __future__ import division
from numpy import sqrt
import numpy
from euclid.FreudShape import ConvexPolyhedron

# Example:
# from euclid.FreudShape.TriangularDipyramid import shape
points = [ 
          (0, 0, -sqrt(2/3)),
          (0, 0, sqrt(2/3)),
          (-1/(2*sqrt(3)), -1/2, 0),
          (-1/(2*sqrt(3)), 1/2, 0),
          (1/sqrt(3), 0, 0),
         ]

shape = ConvexPolyhedron(numpy.array(points))
