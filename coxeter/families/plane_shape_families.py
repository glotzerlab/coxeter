# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Shape families defined by the intersection of half spaces.

This module defines a set of shape families that are defined as the
intersection of half spaces defined by a set of planes. The families here are
generally taken from :cite:`Chen2014` and :cite:`Damasceno2012`.
"""

import numpy as np
from scipy.constants import golden_ratio

from ..shapes import ConvexPolyhedron
from .shape_family import ShapeFamily


class TruncationPlaneShapeFamily(ShapeFamily):
    """A shape famly defined by plane half-space intersections.

    This family of shapes is defined in :cite:`Chen2014`. A set of planes are
    symmetrically placed about a central point, and shapes are defined by the
    intersection of the half spaces defined by these planes. Depending on the
    symmetry group chosen to define the planes, different shapes can result.

    The following parameters are required by this class:

      - :math:`a`
      - :math:`b`
      - :math:`c`

    See :cite:`Chen2014` for descriptions of these parameters. The bounds of
    each parameter are set by the subclasses.
    """

    # Documentation for developers:
    # Subclasses of this class must define the :attr:`~._planes` and
    # :attr:`~._plane_types` class attributes to define the planes and which distance
    # parameter is used to define those truncations.

    @classmethod
    def get_planes(cls):
        """Get the set of planes used to truncate the shape.

        Returns:
            (:math:`N_{planes}`, 3) :class:`numpy.ndarray` of float:
                The planes defining this family
        """
        return cls._planes

    @classmethod
    def get_plane_types(cls):
        """Get the types of the planes.

        The types are encoded via the following integer mapping:

        * type 0 corresponds to the parameter a.
        * type 1 corresponds to the parameter b.
        * type 2 corresponds to the parameter c.

        Returns:
            (:math:`N_{planes}`, ) :class:`numpy.ndarray` of int:
                The plane types.
        """
        return cls._plane_types

    @classmethod
    def make_vertices(cls, a, b, c):
        """Generate vertices from the a, b, and c parameters.

        Args:
            a (float): The a parameter.
            b (float): The b parameter.
            c (float): The c parameter.

        Returns:
            (:math:`N_{vertices}`, 3) :class:`numpy.ndarray` of float:
                The vertices of the shape generated by the provided parameters.
        """
        # Vectorize the plane distances.
        dists = np.array([a, b, c])

        thresh = 1e-6

        planetypes = cls._plane_types
        planelist = cls._planes

        # Generate all unique combinations of planes.
        num_planes = len(planetypes)
        indices = [
            (i, j, k)
            for i in range(num_planes)
            for j in range(i + 1, num_planes)
            for k in range(j + 1, num_planes)
        ]

        # To identify the vertices of the shape, we set up a linear system of
        # equations that finds points that simultaneously satisfy multiple
        # plane equations, i.e. points of intersection of all planes at the
        # specified distances.
        coeffs = planelist[indices]
        alltypes = planetypes[indices]
        bs = dists[alltypes]

        # A determinant of zero for the coefficient matrix indicates that the
        # matrix is not full rank, meaning no solution exists, so we ignore
        # those cases.
        dets = np.linalg.det(coeffs)
        solution_indices = np.abs(dets) > thresh
        xs = np.linalg.solve(coeffs[solution_indices], bs[solution_indices])

        # Reject any solutions that are intersections that lie beyond at least
        # one of the bounding planes.
        dots = np.einsum("ik,jk", xs, planelist, optimize=True)
        alldists = dists[planetypes]
        dist_filter = (dots <= alldists[np.newaxis, :] + thresh).all(axis=1)
        passed_plane_test = xs[dist_filter]

        # Identify unique vertices.  We don't want to lose precision in the
        # vertices to ensure that the convex hull ends up finding the right
        # faces, so get the unique indices based on rounding but then use the
        # original vertices.
        _, verts_indices = np.unique(
            passed_plane_test.round(6), axis=0, return_index=True
        )
        verts = passed_plane_test[verts_indices]

        return verts


class Family323Plus(TruncationPlaneShapeFamily):
    r"""The 323+ shape family defined in :cite:`Chen2014`.

    This class requires the parameters

    :math:`a \in [1, 3]`

    :math:`c \in [1, 3]`

    The :math:`b` parameter is always equal to 1 for this family.

    The extremal shapes in this shape family are an octahedron at (1, 1), a
    tetrahedron at (3, 1) and (1, 3), and a cube at (3, 3).
    """

    _planes = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    _plane_types = np.array([2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    @classmethod
    def get_shape(cls, a, c):
        r"""Generate a shape for the provided parameters.

        Args:
            a (float):
                The parameter :math:`a \in [1, 3]`.
            c (float):
                The parameter :math:`c \in [1, 3]`.

        Returns:
            :class:`~coxeter.shapes.ConvexPolyhedron`:
                The desired shape.
        """
        if not 1 <= a <= 3:
            raise ValueError("The a parameter must be between 1 and 3.")
        if not 1 <= c <= 3:
            raise ValueError("The c parameter must be between 1 and 3.")
        return ConvexPolyhedron(cls.make_vertices(a, 1, c))


class Family423(TruncationPlaneShapeFamily):
    r"""The 423 shape family defined in :cite:`Chen2014`.

    This class requires the parameters

    :math:`a \in [1, 2]`

    :math:`c \in [2, 3]`

    The :math:`b` parameter is always equal to 2 for this family.

    The extremal shapes in this shape family are a cuboctahedron at (1, 2), an
    octahedron at (2, 2), a cube at (1, 3), and a rhombic dodecahedron at
    (2, 3).
    """

    _planes = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
            [0.0, -1.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    _plane_types = np.array(
        [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    )

    @classmethod
    def get_shape(cls, a, c):
        r"""Generate a shape for the provided parameters.

        Args:
            a (float):
                The parameter :math:`a \in [1, 2]`.
            c (float):
                The parameter :math:`c \in [2, 3]`.

        Returns:
            :class:`~coxeter.shapes.ConvexPolyhedron`:
                The desired shape.
        """
        if not 1 <= a <= 2:
            raise ValueError("The a parameter must be between 1 and 2.")
        if not 2 <= c <= 3:
            raise ValueError("The c parameter must be between 2 and 3.")
        return ConvexPolyhedron(cls.make_vertices(a, 2, c))


class Family523(TruncationPlaneShapeFamily):
    r"""The 523 shape family defined in :cite:`Chen2014`.

    This class requires the parameters

    :math:`a \in [1, s\sqrt{5}]`

    :math:`c \in [S^2, 3]`

    where :math:`S = \frac{1}{2}\left(\sqrt{5} + 1\right)` is the golden ratio and
    :math:`s = \frac{1}{2}\left(\sqrt{5} - 1\right)` is its inverse. The :math:`b`
    parameter is always equal to 2 for this family.

    The extremal shapes in this shape family are an icosidodecahedron at
    (:math:`1`, :math:`S^2`), an icosahedron at (:math:`s\sqrt{5}`, :math:`S^2`), a
    dodecahedron at (:math:`1`, :math:`3`), and a rhombic triacontahedron at
    (:math:`s\sqrt{5}`, :math:`3`).
    """

    s = 1 / golden_ratio
    """The constant s (the inverse of the golden ratio)."""

    S = golden_ratio
    """The constant S (the golden ratio)."""

    _planes = np.array(
        [
            [1.0, 0.0, s],
            [-1.0, 0.0, -s],
            [-1.0, 0.0, s],
            [1.0, 0.0, -s],
            [0.0, -s, -1.0],
            [0.0, s, 1.0],
            [0.0, s, -1.0],
            [0.0, -s, 1.0],
            [-s, -1.0, 0.0],
            [s, 1.0, 0.0],
            [s, -1.0, 0.0],
            [-s, 1.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, 2.0],
            [S, S, S],
            [-S, S, S],
            [S, -S, S],
            [S, S, -S],
            [S, -S, -S],
            [-S, -S, S],
            [-S, S, -S],
            [-S, -S, -S],
            [1.0, 0.0, S ** 2],
            [-1.0, 0.0, -(S ** 2)],
            [-1.0, 0.0, S ** 2],
            [1.0, 0.0, -(S ** 2)],
            [0.0, -(S ** 2), -1.0],
            [0.0, S ** 2, 1.0],
            [0.0, -(S ** 2), 1.0],
            [0.0, S ** 2, -1.0],
            [-(S ** 2), -1.0, 0.0],
            [S ** 2, 1.0, 0.0],
            [S ** 2, -1.0, 0.0],
            [-(S ** 2), 1.0, 0.0],
            [S, -1.0, -s],
            [-S, 1.0, -s],
            [-S, -1.0, s],
            [S, 1.0, s],
            [S, -1.0, s],
            [S, 1.0, -s],
            [-S, 1.0, s],
            [-S, -1.0, -s],
            [s, S, 1.0],
            [s, -S, -1.0],
            [-s, -S, 1.0],
            [-s, S, -1.0],
            [-s, -S, -1.0],
            [s, -S, 1.0],
            [-s, S, 1.0],
            [s, S, -1.0],
            [1.0, -s, -S],
            [-1.0, s, -S],
            [-1.0, -s, S],
            [1.0, s, S],
            [1.0, s, -S],
            [-1.0, s, S],
            [1.0, -s, S],
            [-1.0, -s, -S],
        ]
    )

    _plane_types = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )

    @classmethod
    def get_shape(cls, a, c):
        r"""Generate a shape for the provided parameters.

        Args:
            a (float):
                The parameter :math:`a \in [1, s\sqrt{5}]`.
            c (float):
                The parameter :math:`c \in [S^2, 3]`.

        Returns:
            :class:`~coxeter.shapes.ConvexPolyhedron`:
                The desired shape.
        """
        if not 1 <= a <= (cls.s * np.sqrt(5)):
            raise ValueError(
                "The a parameter must be between 1 and s\u221A5 "
                "(where s is the inverse of the golden ratio)."
            )
        if not cls.S ** 2 <= c <= 3:
            raise ValueError(
                "The c parameter must be between S^2 and 3 "
                "(where S is the golden ratio)."
            )
        return ConvexPolyhedron(cls.make_vertices(a, 2, c))


class TruncatedTetrahedronFamily(Family323Plus):
    r"""The truncated tetrahedron family used in :cite:`Damasceno2012`.

    The following parameters are required by this class:

      - truncation :math:`\in [0, 1]`

    This family is constructed as a limiting case of :class:`~.Family323Plus`
    with a = 1. The c value is then directly related to a linear interpolation
    over truncations. In particular, :math:`c = 3 - 2(\text{truncation})`.
    """

    @classmethod
    def get_shape(cls, truncation):
        r"""Generate a shape for a given truncation value.

        Args:
            truncation (float):
                The parameter :math:`truncation \in [0, 1]`.

        Returns:
            :class:`~coxeter.shapes.ConvexPolyhedron`:
                The desired truncated tetrahedron.
        """
        if not 0 <= truncation <= 1:
            raise ValueError("The truncation must be between 0 and 1.")
        c = 3 - 2 * truncation
        return super().get_shape(1, c)
