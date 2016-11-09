from __future__ import division
from numpy import sqrt
import numpy
from euclid.FreudShape import ConvexPolyhedron

# Example:
# from euclid.FreudShape.TruncatedCube import shape
points = [ 
          (-1/2, 1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2)),
          (-1/2, 1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1)),
          (-1/2, (2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2)),
          (-1/2, (2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1)),
          (1/2, 1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2)),
          (1/2, 1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1)),
          (1/2, (2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2)),
          (1/2, (2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1)),
          (1/2 + 1/sqrt(2), -1/2, 1/2 + 1/sqrt(2)),
          (1/2 + 1/sqrt(2), -1/2, (2 - 2*sqrt(2))**(-1)),
          (1/2 + 1/sqrt(2), 1/2, 1/2 + 1/sqrt(2)),
          (1/2 + 1/sqrt(2), 1/2, (2 - 2*sqrt(2))**(-1)),
          (1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2), -1/2),
          (1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2), 1/2),
          (1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1), -1/2),
          (1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1), 1/2),
          ((2 - 2*sqrt(2))**(-1), -1/2, 1/2 + 1/sqrt(2)),
          ((2 - 2*sqrt(2))**(-1), -1/2, (2 - 2*sqrt(2))**(-1)),
          ((2 - 2*sqrt(2))**(-1), 1/2, 1/2 + 1/sqrt(2)),
          ((2 - 2*sqrt(2))**(-1), 1/2, (2 - 2*sqrt(2))**(-1)),
          ((2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2), -1/2),
          ((2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2), 1/2),
          ((2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1), -1/2),
          ((2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1), 1/2),
         ]

shape = ConvexPolyhedron(numpy.array(points))
