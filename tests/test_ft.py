import unittest

import numpy as np
import numpy.testing as npt

import coxeter.ft
from coxeter.shape_families import PlatonicFamily


class TestFormFactors(unittest.TestCase):
    def setUp(self):
        self.K = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 2, 3],
                [-2, 4, -5.2],
            ],
            dtype=np.float,
        )

    def test_ft_defaults(self):
        """Ensures default quantities are set properly."""
        for ft_class in [
            coxeter.ft._FTbase,
            coxeter.ft.FTdelta,
            coxeter.ft.FTpolyhedron,
        ]:
            ft = ft_class()
            self.assertEqual(len(ft.K), 1)
            npt.assert_array_equal(ft.K, np.zeros((1, 3)))
            self.assertEqual(ft.S.shape, (0, 0))
            self.assertEqual(ft.density, 1.0)
            npt.assert_array_equal(
                ft.orientation, np.array([[1, 0, 0, 0]], dtype=np.float)
            )
            npt.assert_array_equal(ft.position, np.array([[0, 0, 0]], dtype=np.float))
            self.assertEqual(ft.scale, 1.0)

    def test_ft_delta(self):
        ft = coxeter.ft.FTdelta(k=self.K)

        positions = np.array([[0, 0, 0]], dtype=np.float)
        orientations = np.array([[1, 0, 0, 0]] * len(positions), dtype=np.float)
        ft.set_rq(positions, orientations)
        ft.compute()
        npt.assert_array_equal(ft.S, np.ones(len(self.K)))

        positions = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float,
        )
        orientations = np.array([[1, 0, 0, 0]] * len(positions), dtype=np.float)
        ft.set_rq(positions, orientations)
        ft.compute()
        npt.assert_almost_equal(
            ft.S,
            [
                6.0,
                5.08060461,
                5.08060461,
                3.16770633,
                5.08060461,
                5.08060461,
                -1.73167405,
                -1.20254791,
            ],
            decimal=6,
        )

    def test_ft_convex_polyhedron(self):
        # TODO: Currently using this to test FTpolyhedron indirectly
        cube = PlatonicFamily()("Cube")
        cube.volume = 8
        npt.assert_almost_equal(cube.volume, 8)
        ft = coxeter.ft.FTconvexPolyhedron(cube, k=self.K)

        positions = np.array([[0, 0, 0]], dtype=np.float)
        orientations = np.array([[1, 0, 0, 0]] * len(positions), dtype=np.float)
        ft.set_rq(positions, orientations)
        ft.compute()
        npt.assert_almost_equal(ft.S[0], 8, decimal=6)
        npt.assert_almost_equal(
            ft.S,
            [
                8.0,
                6.73176788,
                6.73176788,
                3.63718971,
                6.73176788,
                6.73176788,
                0.14397014,
                0.1169148,
            ],
        )


if __name__ == "__main__":
    unittest.main()
