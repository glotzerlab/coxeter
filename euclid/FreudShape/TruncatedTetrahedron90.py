from __future__ import division
from numpy import sqrt
import numpy
from euclid.FreudShape import ConvexPolyhedron

# Example:
# from euclid.FreudShape.TruncatedTetrahedron90 import shape
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
