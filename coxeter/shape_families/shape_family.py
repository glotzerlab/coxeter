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
        dists = [a, b, c]

        verts = []

        thresh = 1e-6

        planetypes = self.plane_types
        planelist = self.planes
        # for each triplet of planes
        for i in range(len(planetypes)):
            for j in range(i+1, len(planetypes)):
                for k in range(j+1, len(planetypes)):
                    A = np.array([planelist[i], planelist[j], planelist[k]])
                    # compute the determinant
                    det = np.linalg.det(A)
                    # if it is non-zero, the find the intersection point
                    if abs(det) > thresh:
                        b = np.array([dists[planetypes[i]],
                                      dists[planetypes[j]],
                                      dists[planetypes[k]]])
                        x = np.linalg.solve(A, b)

                        # once the intersection point has been found, check it
                        # against the planes
                        try:
                            for (n, t) in zip(planelist, planetypes):
                                assert(np.dot(n, x) <= dists[t] + thresh)
                        except:  # noqa
                            pass
                        else:
                            # then check whether it has alerady been found
                            # within some tolerance
                            try:
                                if len(verts) > 0:
                                    for v in verts:
                                        diff = x - v
                                        numer = np.dot(diff, diff)
                                        denom = np.sqrt(
                                            np.dot(x, x)*np.dot(v, v))
                                        assert(numer/denom > thresh)
                                else:
                                    pass
                            except:  # noqa
                                pass
                            else:
                                verts.append(x)

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
        return [
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
            [0.0, 0.0, -1.0]]

    @property
    def plane_types(self):
        return [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


class Family532(TruncationPlaneShapeFamily):
    """The 532 shape family defined in :cite:`Chen2014`"""
    @property
    def planes(self):
        s = ((5**0.5)-1.0)/2.
        S = ((5**0.5)+1.0)/2.
        return [
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
            ]

    @property
    def plane_types(self):
        return [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0,  1, 1, 1, 1, 1,  1,  2,
                2, 2, 2, 2,  2, 2, 2, 2, 2,  2, 2, 2, 2, 2,  2, 2, 2, 2, 2,  1,
                1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1,
                1, 1, 1]


class Family432(TruncationPlaneShapeFamily):
    """The 432 shape family defined in :cite:`Chen2014`"""
    @property
    def planes(self):
        return [
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
             [0.0, 0.0, -1.0]]

    @property
    def plane_types(self):
        return [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0]
