# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""The coxeter package simplifies working with shapes in 2D and 3D.

While many computational geometry tools exist, they almost exclusively aim to
solve problems on the more complex end that are the subject of much study, such
as the construction of convex hulls, Delaunay triangulations, and Voronoi
tessellations. The coxeter package is instead aimed at providing transparent
APIs for the generation of shapes and the calculation of quantities for which
formulas are well-known but nontrivial to implement in robust ways. For
instance, given a set of vertices defining a convex polyhedron, coxeter can
automatically compute the faces of the polyhedron and keep face indices sorted
appropriately for use in most computational geometry applications. The package
is especially designed for calculations of interest in physics-based
applications such as inertia tensors.
"""

__version__ = "0.9.0"

from . import families, io, shapes
from .shape_getters import from_gsd_type_shapes

__all__ = ["families", "shapes", "from_gsd_type_shapes", "io"]
