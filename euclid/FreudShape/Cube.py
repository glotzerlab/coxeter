from __future__ import division
from numpy import sqrt
import numpy
from euclid.FreudShape import ConvexPolyhedron

# Example:
# from euclid.FreudShape.Cube import shape

points = [ (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
           (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ]

shape = ConvexPolyhedron(numpy.array(points))

