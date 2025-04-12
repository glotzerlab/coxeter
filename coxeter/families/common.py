# Copyright (c) 2015-2025 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Certain common shape families that can be analytically generated."""

import os
import warnings
from functools import wraps
from math import cos, sin, sqrt, tan

import numpy as np
from numpy import cbrt, pi

from ..shapes import ConvexPolygon, ConvexPolyhedron
from .doi_data_repositories import _DATA_FOLDER
from .shape_family import ShapeFamily
from .tabulated_shape_family import TabulatedGSDShapeFamily


def csc(theta):
    """Compute the cosecant of a value.

    :meta private:
    """
    return 1 / sin(theta)


# Allows us to monkeypatch an existing method with our choice of warning
def _deprecated_method(func, deprecated="", replacement="", reason=""):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            (f"{deprecated} has been deprecated in favor of {replacement}. {reason}"),
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


def sec(theta):
    """Return the secant of an angle.

    :meta private:
    """
    return 1 / cos(theta)


def cot(theta):
    """Return the cotangent of an angle.

    :meta private:
    """
    return 1 / tan(theta)


def _make_ngon(n, z=0.0, area=1, angle=0):
    """Make a regular n-gon with a given area, z height, and rotation angle.

    The initial vertex  lies on the :math:`x` axis by default, but can be rotated by
    the angle parameter.

    Args:
        n (int): Number of vertices
        z (int|float): z value for the polygon. Defaults to 0.
        area (int|float|None, optional): Area of polygon. Defaults to 1.
        angle (int|float, optional): Rotation angle, in radians. Defaults to 0.

    Returns
    -------
        np.array: n-gon vertices
    """
    if n < 3:
        raise ValueError("Cannot generate an n-gon with fewer than 3 vertices.")

    theta = np.linspace(0, 2 * pi, num=n, endpoint=False) + angle
    ngon_vertices = np.array([np.cos(theta), np.sin(theta), np.full_like(theta, z)]).T
    if area is not None:
        area_0 = 0.5 * n * sin(2 * pi / n)  # Area of the shape with circumradius = 1
        ngon_vertices[:, :2] *= sqrt(area / area_0)  # Rescale coords to correct area

    return ngon_vertices


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
        r"""Generate a unit area n-gon.

        Args:
            n (int):
                The number of vertices (:math:`n \geq 3`).

        Returns
        -------
             :class:`~.ConvexPolygon`: The corresponding regular polygon.
        """
        return ConvexPolygon(cls.make_vertices(n))

    @classmethod
    def make_vertices(cls, n):
        r"""Generate vertices of a unit area n-gon.

        Args:
            n (int):
                The number of vertices of the polygon (:math:`n \geq 3`).

        Returns
        -------
            :math:`(n, 3)` :class:`numpy.ndarray` of float: The vertices of the polygon.
        """
        return _make_ngon(n, area=1, angle=0)


class UniformPrismFamily(ShapeFamily):
    """The infinite family of uniform right prisms with unit volume.

    As with the :class:`~.RegularNGonFamily`, the initial vertex lies on the
    :math:`x` axis.
    """

    @classmethod
    def get_shape(cls, n):
        r"""Generate a uniform right n-prism of unit volume.

        Args:
            n (int):
                The number of vertices of the base polygons (:math:`n \geq 3`).

        Returns
        -------
             :class:`~.ConvexPolyhedron`: The corresponding convex polyhedron.
        """
        return ConvexPolyhedron(cls.make_vertices(n))

    @classmethod
    def make_vertices(cls, n):
        r"""Generate the vertices of a uniform right n-prism with unit volume.

        Args:
            n (int):
                The number of vertices of the base polygons (:math:`n \geq 3`).

        Returns
        -------
             :math:`(n*2, 3)` :class:`numpy.ndarray` of float:
                 The vertices of the prism.
        """
        volume = 1
        h = cbrt(volume * 4 / n * tan(pi / n))
        area = volume / h  # Top and bottom face areas
        vertices = np.concatenate(
            [_make_ngon(n, z=-h / 2, area=area), _make_ngon(n, z=h / 2, area=area)]
        )
        return vertices


class UniformAntiprismFamily(ShapeFamily):
    r"""The infinite family of uniform right antiprisms with unit volume.

    As with the :class:`~.RegularNGonFamily`, the initial vertex lies on the
    :math:`x` axis. The bottom face of each antiprism is rotated :math:`\pi/n` radians
    relative to the top face.
    """

    @classmethod
    def get_shape(cls, n):
        r"""Generate a uniform right n-antiprism of unit volume.

        Args:
            n (int):
                The number of vertices of the base polygons (:math:`n \geq 3`).

        Returns
        -------
             :class:`~.ConvexPolyhedron`: The corresponding convex polyhedron.
        """
        return ConvexPolyhedron(cls.make_vertices(n))

    @classmethod
    def make_vertices(cls, n):
        r"""Generate the vertices of a uniform right n-antiprism with unit volume.

        Args:
            n (int):
                The number of vertices of the base polygons (:math:`n \geq 3`).

        Returns
        -------
             :math:`(n*2, 3)` :class:`numpy.ndarray` of float:
                 The vertices of the antiprism.
        """
        volume = 1
        s = cbrt(
            24
            * volume
            / (n * (cot(pi / (2 * n)) + cot(pi / n)) * sqrt(4 - sec(pi / (2 * n)) ** 2))
        )
        area = n / 4 * (cot(pi / n)) * s**2

        h = sqrt(1 - 0.25 * sec(pi / (2 * n)) ** 2) * s
        vertices = np.concatenate(
            [_make_ngon(n, -h / 2, area, angle=pi / n), _make_ngon(n, h / 2, area)]
        )
        return vertices


class UniformPyramidFamily(ShapeFamily):
    """The family of uniform right pyramids with unit volume.

    As with the :class:`~.RegularNGonFamily`, the initial vertex of the base polygon
    lies on the :math:`x` axis.
    """

    @classmethod
    def get_shape(cls, n):
        r"""Generate a uniform right n-pyramid of unit volume.

        Args:
            n (int):
                The number of vertices of the base polygon (:math:`3 \leq n \leq 5`).

        Returns
        -------
             :class:`~.ConvexPolyhedron`: The corresponding convex polyhedron.
        """
        return ConvexPolyhedron(cls.make_vertices(n))

    @classmethod
    def make_vertices(cls, n):
        r"""Generate the vertices of a uniform right n-pyramid with unit volume.

        Args:
            n (int):
                The number of vertices of the base polygon (:math:`3 \leq n \leq 5`).

        Returns
        -------
             :math:`(n+1, 3)` :class:`numpy.ndarray` of float:
                 The vertices of the pyramid.
        """
        volume = 1
        h = cbrt((3 * volume * (4 - sin(pi / n) ** -2)) / (n * cot(pi / n)))
        area = 3 * volume / h

        # The centroid of a pyramid is 1/4 of the height offset from the base.
        base = _make_ngon(n, z=-h / 4, area=area)
        apex = [[0, 0, 3 * h / 4]]
        vertices = np.concatenate([base, apex])
        return vertices


class UniformDipyramidFamily(ShapeFamily):
    """The family of uniform right dipyramids with unit volume.

    As with the :class:`~.RegularNGonFamily`, the initial vertex of the base polygon
    lies on the :math:`x` axis.
    """

    @classmethod
    def get_shape(cls, n):
        r"""Generate a uniform right n-dipyramid of unit volume.

        Args:
            n (int):
                The number of vertices of the base polygon (:math:`3 \leq n \leq 5`).

        Returns
        -------
             :class:`~.ConvexPolyhedron`: The corresponding convex polyhedron.
        """
        return ConvexPolyhedron(cls.make_vertices(n))

    @classmethod
    def make_vertices(cls, n):
        r"""Generate the vertices of a uniform right n-dipyramid with unit volume.

        Args:
            n (int):
                The number of vertices of the base polygon (:math:`3 \leq n \leq 5`).

        Returns
        -------
             :math:`(n+2, 3)` :class:`numpy.ndarray` of float:
                 The vertices of the dipyramid.
        """
        volume = 1
        # h = cbrt((3 * volume * (4 - sin(pi / n) ** -2)) / (n * cot(pi / n)))
        # area = 3.0 * volume / h

        h = cbrt((3 * volume * (4 - sin(pi / n) ** -2) / 2) / (n * cot(pi / n)))
        area = 1.5 * volume / h

        base = _make_ngon(n, z=0, area=area)
        apexes = [[0, 0, h], [0, 0, -h]]
        vertices = np.concatenate([base, apexes])
        return vertices


class CanonicalTrapezohedronFamily(ShapeFamily):
    r"""The infinite family of canonical n-trapezohedra (antiprism duals).

    Formulas for vertices are derived from :cite:`Rajpoot_2015` rather than via explicit
    canonicalization to ensure the method is deterministic and fast.
    """

    @classmethod
    def get_shape(cls, n: int):
        r"""Generate a canonical n-antiprism of unit volume.

        Args:
            n (int): The number of vertices of the base polygons (:math:`n \geq 3`).

        Returns
        -------
             :class:`~.ConvexPolyhedron`: The corresponding convex polyhedron.
        """
        return ConvexPolyhedron(cls.make_vertices(n))

    @classmethod
    def make_vertices(cls, n):
        r"""Generate the vertices of a uniform right n-antiprism with unit volume.

        Args:
            n (int): The number of vertices of the base polygons (:math:`n \geq 3`).

        Returns
        -------
             :math:`(n*2 + 2, 3)` :class:`numpy.ndarray` of float:
                 The vertices of the trapezohedron.
        """
        r = 0.5 * csc(np.pi / n)  # Midradius for canonical trapezohedron
        area = r**2 * n / 2 * sin(2 * np.pi / n)  # Area of center polygons

        # Height of center polygons
        c = sqrt(4 - sec(np.pi / (2 * n)) ** 2) / (4 + 8 * cos(np.pi / n))

        # Height of apexes
        z = (
            0.25
            * cos(np.pi / (2 * n))
            * cot(np.pi / (2 * n))
            * csc(3 * np.pi / (2 * n))
            * sqrt(4 - sec(np.pi / (2 * n)) ** 2)
        )

        top_center_polygon = _make_ngon(n, c, area, angle=np.pi / n)
        bottom_center_polygon = _make_ngon(n, -c, area)

        vertices = np.concatenate(
            [
                top_center_polygon,
                bottom_center_polygon,
                [[0, 0, z], [0, 0, -z]],
            ]
        )

        # Compute height of (canonical) trapezohedron
        h = 1 / 4 * csc(np.pi / (2 * n)) * sqrt(2 * cos(np.pi / n) + 1) / 2

        # Compute edge lengths (required to compute the volume)
        edge_length_ratio = 1 / (2 - 2 * cos(np.pi / n))
        short_edge_length = np.linalg.norm(
            top_center_polygon[0] - bottom_center_polygon[0]
        )
        long_edge_length = short_edge_length * edge_length_ratio

        vo = 2 * n / 3 * short_edge_length * long_edge_length * h
        vertices *= cbrt(1 / vo)  # Rescale to unit volume
        return vertices


PlatonicFamily = TabulatedGSDShapeFamily._from_json_file(
    os.path.join(_DATA_FOLDER, "platonic.json"),
    classname="PlatonicFamily",
    docstring="""The family of Platonic solids (5 total).

    Options can be found in the :doc:`Platonic Solids Table<table-platonic>`.
""",
)

ArchimedeanFamily = TabulatedGSDShapeFamily._from_json_file(
    os.path.join(_DATA_FOLDER, "archimedean.json"),
    classname="ArchimedeanFamily",
    docstring="""The family of Archimedean solids (13 total).

    Options can be found in the :doc:`Archimedean Solids Table<table-archimedean>`.
""",
)

CatalanFamily = TabulatedGSDShapeFamily._from_json_file(
    os.path.join(_DATA_FOLDER, "catalan.json"),
    classname="CatalanFamily",
    docstring="""The family of Catalan solids, also known as Archimedean duals \
    (13 total).

    Options can be found in the :doc:`Catalan Solids Table<table-catalan>`.
""",
)

JohnsonFamily = TabulatedGSDShapeFamily._from_json_file(
    os.path.join(_DATA_FOLDER, "johnson.json"),
    classname="JohnsonFamily",
    docstring="""The family of Johnson solids, as enumerated in \
    :cite:`Johnson1966` (92 total).

    Options can be found in the :doc:`Johnson Solids Table<table-johnson>`.
""",
)

PyramidDipyramidFamily = TabulatedGSDShapeFamily._from_json_file(
    os.path.join(_DATA_FOLDER, "pyramid_dipyramid.json"),
    classname="PyramidDipyramidFamily",
    docstring="""The family of regular equilateral pyramids and dipyramids (6 total).

    Options can be found in the :doc:`Pyramid-Dipyramid Table<table-pyramid-dipyramid>`.
""",
)

PrismAntiprismFamily = TabulatedGSDShapeFamily._from_json_file(
    os.path.join(_DATA_FOLDER, "prism_antiprism.json"),
    classname="PrismAntiprismFamily",
    docstring="""The family of uniform n-prisms and n-antiprisms with n∈[3,10] \
    (16 total).


    .. warning::

        This class has been deprecated in favor of the :class:`~.UniformPrismFamily`
        and :class:`~.UniformAntiprismFamily`, as the new classes have a simplified API
        and support the entire infinite shape family. Please transfer existing code to
        use the new classes.

    Options for prisms can be found in the \
    :doc:`Prism-Antiprism Table<table-prism-antiprism>`.
""",
)

PrismAntiprismFamily.get_shape = _deprecated_method(
    PrismAntiprismFamily.get_shape,
    deprecated="PrismAntiprismFamily",
    replacement="UniformPrismFamily and UniformAntiprismFamily",
    reason=(
        "These alternate classes have a simplified interface and support the "
        "entire infinite family of geometries."
    ),
)
PyramidDipyramidFamily.get_shape = _deprecated_method(
    PyramidDipyramidFamily.get_shape,
    deprecated="PyramidDipyramidFamily",
    replacement="UniformPyramidFamily and UniformDipyramidFamily",
    reason=(
        "These alternate classes have a simplified interface and better match the "
        "naming conventions of coxeter."
    ),
)
