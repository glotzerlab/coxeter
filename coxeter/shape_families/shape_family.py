from abc import ABC, abstractmethod
import numpy as np
from ..shape_classes import ConvexPolyhedron


class ShapeFamily(ABC):
    """A shape family encapsulates the ability to generate a set of shapes of
    of type :class:`~coxeter.shape_classes.Shape`.
    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """:class:`~coxeter.shape_classes.Shape`: Generate a shape based on the
        provided parameters."""
        pass


class TruncationPlaneShapeFamily(ShapeFamily):
    """A family of shapes defined by a set of truncation planes"""
    def __call__(self, a=1, b=1, c=1):
        # vectorize the plane distances
        dists = np.array([a, b, c])

        verts = []

        thresh = 1e-6

        planetypes = self.plane_types
        planelist = self.planes

        num_planes = len(planetypes)
        indices = [(i, j, k) for i in range(num_planes) for j in
                   range(i+1, num_planes) for k in range(j+1, num_planes)]

        As = planelist[indices]
        dets = np.linalg.det(As)

        alltypes = planetypes[indices]
        bs = dists[alltypes]

        #  xs = np.zeros(bs.shape)
        solution_indices = np.abs(dets) > thresh

        xs = np.linalg.solve(As[solution_indices], bs[solution_indices])

        # Get for each x whether any of the planes fail.
        # Need to squeeze because broadcasting generates odd singleton
        # dimensions.
        dots = np.inner(
            xs[:, np.newaxis, :], planelist[np.newaxis, :, :]).squeeze()
        alldists = dists[planetypes]
        dist_filter = (dots <= alldists[np.newaxis, :] + thresh).all(axis=1)

        passed_plane_test = xs[dist_filter]

        # We don't want to lose precision in the vertices to ensure that the
        # convex hull ends up finding the right faces, so get the unique
        # indices based on rounding but then use the original vertices.
        _, verts_indices = np.unique(
            passed_plane_test.round(6), axis=0, return_index=True)
        verts = passed_plane_test[verts_indices]

        return ConvexPolyhedron(verts)

    @property
    @abstractmethod
    def planes(self):
        pass

    @property
    @abstractmethod
    def plane_types(self):
        pass


class Family332(TruncationPlaneShapeFamily):
    """The 332 shape family defined in :cite:`Chen2014`"""
    @property
    def planes(self):
        return np.array([
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
            [0.0, 0.0, -1.0]])

    @property
    def plane_types(self):
        return np.array([2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


class Family532(TruncationPlaneShapeFamily):
    """The 532 shape family defined in :cite:`Chen2014`"""
    @property
    def planes(self):
        s = ((5**0.5)-1.0)/2.
        S = ((5**0.5)+1.0)/2.
        return np.array([
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
            [1.0, 0.0, S**2],
            [-1.0, 0.0, -S**2],
            [-1.0, 0.0, S**2],
            [1.0, 0.0, -S**2],
            [0.0, -S**2, -1.0],
            [0.0, S**2, 1.0],
            [0.0, -S**2, 1.0],
            [0.0, S**2, -1.0],
            [-S**2, -1.0, 0.0],
            [S**2, 1.0, 0.0],
            [S**2, -1.0, 0.0],
            [-S**2, 1.0, 0.0],
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
            [-1.0, -s, -S]
            ])

    @property
    def plane_types(self):
        return np.array([0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0,  1, 1, 1, 1, 1,
                         1,  2, 2, 2, 2, 2,  2, 2, 2, 2, 2,  2, 2, 2, 2, 2,  2,
                         2, 2, 2, 2,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1,
                         1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1])


class Family432(TruncationPlaneShapeFamily):
    """The 432 shape family defined in :cite:`Chen2014`"""
    @property
    def planes(self):
        return np.array([
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
             [0.0,  0.0,  1.0],
             [0.0, 0.0, -1.0]])

    @property
    def plane_types(self):
        return np.array([2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 0, 0, 0, 0, 0, 0])
