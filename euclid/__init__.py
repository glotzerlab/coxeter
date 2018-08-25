import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import json
import collections

from euclid import shapes
from euclid import utils
from euclid import damasceno
from euclid import quaternion_tools as qt
from euclid import symmetry
from euclid import form_factors
