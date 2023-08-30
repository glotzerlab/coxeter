# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Certain common shape families that can be analytically generated."""

import os

import numpy as np

from ..shapes import ConvexPolygon
from .doi_data_repositories import _DATA_FOLDER
from .shape_family import ShapeFamily
from .tabulated_shape_family import TabulatedGSDShapeFamily


class RegularNGonFamily(ShapeFamily):
    """The family of convex regular polygons.

    This class generates the set of convex regular polygons with :math:`n`
    sides. The polygons are normalized to be unit area by default, and the
    initial vertex always lies on the :math:`x` axis (so, for example, a
    4-sided shape generated by this will look like a diamond, i.e. a square
    rotated by 45 degrees).

    The following parameters are required by this class:

      - :math:`n`: The number of vertices of the polygon
    """

    @classmethod
    def get_shape(cls, n):
        """Generate a unit area n-gon.

        Args:
            n (int):
                The number of vertices (greater than or equal to 3).

        Returns:
             :class:`~.ConvexPolygon`: The corresponding regular polygon.
        """
        return ConvexPolygon(cls.make_vertices(n))

    @classmethod
    def make_vertices(cls, n):
        """Generate vertices of a unit area n-gon.

        Args:
            n (int):
                An integer greater than or equal to 3.

        Returns:
            :math:`(n, 3)` :class:`numpy.ndarray` of float: The vertices of the polygon.
        """
        if n < 3:
            raise ValueError("Cannot generate an n-gon with fewer than 3 vertices.")
        r = 1  # The radius of the circle
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        pos = np.array([np.cos(theta), np.sin(theta)]).T

        # First normalize to guarantee that the limiting case of an infinite
        # number of vertices produces a circle of area r^2.
        pos /= np.sqrt(np.pi) / r

        # The area of an n-gon inscribed in a circle is given by:
        # \frac{n r^2}{2} \sin(2\pi / n)
        # The ratio of that n-gon area to its circumscribed circle area is:
        a_circ_a_poly = np.pi / ((n / 2) * np.sin(2 * np.pi / n))

        # Rescale the positions so that the final shape has area 1.
        pos *= np.sqrt(a_circ_a_poly)

        return pos


PlatonicFamily = TabulatedGSDShapeFamily.from_json_file(
    os.path.join(_DATA_FOLDER, "platonic.json"),
    classname="PlatonicFamily",
    docstring="""The family of Platonic solids (5 total).

The following parameters are required by this class:

    - name: The name of the Platonic solid. Options are "Cube", "Dodecahedron", \
            "Icosahedron", "Octahedron", and "Tetrahedron".
""",
)

ArchimedeanFamily = TabulatedGSDShapeFamily.from_json_file(
    os.path.join(_DATA_FOLDER, "archimedean.json"),
    classname="ArchimedeanFamily",
    docstring="""The family of Archimedean solids (13 total).

The following parameters are required by this class:
    - name: The name of the ArchimedeanFamily solid. Options are "Cuboctahedron", \
            "Icosidodecahedron", "Truncated Tetrahedron", "Truncated Octahedron", \
            "Truncated Cube", "Truncated Icosahedron", "Truncated Dodecahedron", \
            "Rhombicuboctahedron", "Rhombicosidodecahedron", "Truncated \
            Cuboctahedron", "Truncated Icosidodecahedron", "Snub Cuboctahedron", \
            and "Snub Icosidodecahedron".
""",
)

CatalanFamily = TabulatedGSDShapeFamily.from_json_file(
    os.path.join(_DATA_FOLDER, "catalan.json"),
    classname="CatalanFamily",
    docstring="""The family of Catalan solids, also known as Archimedean duals \
    (13 total).

The following parameters are required by this class:
    - name: The name of the CatalanFamily solid. Options are "Deltoidal  \
            Hexecontahedron", "Deltoidal Icositetrahedron", "Disdyakis \
            Dodecahedron", "Disdyakis Triacontahedron", "Pentagonal \
            Hexecontahedron", "Pentagonal Icositetrahedron", "Pentakis \
            Dodecahedron", "Rhombic Dodecahedron", "Rhombic \
            Triacontahedron", "Triakis Octahedron", "Tetrakis \
            Hexahedron", "Triakis Icosahedron", and "Triakis Tetrahedron".
""",
)

JohnsonFamily = TabulatedGSDShapeFamily.from_json_file(
    os.path.join(_DATA_FOLDER, "johnson.json"),
    classname="JohnsonFamily",
    docstring="""The family of Johnson solids, as enumerated in \
    :cite:`Johnson1966` (92 total).

The following parameters are required by this class:
    - name: The name of the JohnsonFamily solid. A full list is available in \
            :cite:`Johnson1966`. In general, shape names \
            should have the first character of each word capitalized, with spaces \
            between words (e.g. "Elongated Triangular Cupola"). Pyramids and \
            dipyramids are named from their base polygon (e.g. "Square Pyramid" \
            or "Elongated Pentagonal Dipyramid").
""",
)

PyramidDipyramidFamily = TabulatedGSDShapeFamily.from_json_file(
    os.path.join(_DATA_FOLDER, "pyramid_dipyramid.json"),
    classname="PyramidDipyramidFamily",
    docstring="""The family of regular equilateral pyramids and dipyramids (6 total).

    The following parameters are required by this class:
    - name: The name of the pyramid or dipyramid. Options are "Triangular Pyramid", \
            "Square Pyramid", "Pentagonal Pyramid", "Triangular Dipyramid", \
            "Square Dipyramid", and "Pentagonal Dipyramid".
""",
)

PrismAntiprismFamily = TabulatedGSDShapeFamily.from_json_file(
    os.path.join(_DATA_FOLDER, "prism_antiprism.json"),
    classname="PrismAntiprismFamily",
    docstring="""The family of uniform n-prisms and n-antiprisms with n∈[3,10] \
    (16 total).

    The following parameters are required by this class:
    - name: The name of the prism or antiprism. Options for prisms are  \
            "Triangular Prism", "Square Prism", "Pentagonal Prism", "Hexagonal Prism", \
            "Heptagonal Prism", "Octagonal Prism", "Nonagonal Prism", and \
            "Decagonal Prism". Options for antiprisms are "Triangular Antiprism", \
            "Square Antiprism", "Pentagonal Antiprism", "Hexagonal Antiprism", \
            "Heptagonal Antiprism", "Octagonal Antiprism","Nonagonal Antiprism", \
            and "Decagonal Antiprism".
""",
)
