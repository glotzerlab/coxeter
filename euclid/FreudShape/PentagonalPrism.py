from __future__ import division
from numpy import sqrt
import numpy
from euclid.FreudShape import ConvexPolyhedron

# Example:
# from euclid.FreudShape.PentagonalPrism import shape
points = [ 
          (1/(2*sqrt(5/8 - sqrt(5)/8)), 0, -1/2),
          (1/(2*sqrt(5/8 - sqrt(5)/8)), 0, 1/2),
          ((-1 - sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), -1/2, -1/2),
          ((-1 - sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), -1/2, 1/2),
          ((-1 - sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), 1/2, -1/2),
          ((-1 - sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), 1/2, 1/2),
          ((-1 + sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), -sqrt((5/8 + sqrt(5)/8)/(5/8 - sqrt(5)/8))/2, -1/2),
          ((-1 + sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), -sqrt((5/8 + sqrt(5)/8)/(5/8 - sqrt(5)/8))/2, 1/2),
          ((-1 + sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), sqrt((5/8 + sqrt(5)/8)/(5/8 - sqrt(5)/8))/2, -1/2),
          ((-1 + sqrt(5))/(8*sqrt(5/8 - sqrt(5)/8)), sqrt((5/8 + sqrt(5)/8)/(5/8 - sqrt(5)/8))/2, 1/2),
         ]

shape = ConvexPolyhedron(numpy.array(points))
