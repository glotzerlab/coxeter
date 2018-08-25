import unittest
import numpy as np
import numpy.testing as npt
import euclid
import euclid.FreudShape
import euclid.FreudShape.Cube

class TestFormFactors(unittest.TestCase):

    def setUp(self):
        self.K = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [-1, 0, 0]], dtype=np.float)

    def test_FTdefaults(self):
        """Ensures default quantities are set properly."""
        for ft_class in [euclid.ft._FTbase,
                         euclid.ft.FTdelta,
                         euclid.ft.FTsphere,
                         euclid.ft.FTpolyhedron]:
            ft = ft_class()
            self.assertEqual(ft.NK, 0)
            npt.assert_array_equal(ft.K, np.zeros((1, 3)))
            self.assertEqual(ft.S.shape, (0, 0))
            self.assertEqual(ft.density, 1.)
            npt.assert_array_equal(ft.orientation,
                                   np.array([[1, 0, 0, 0]], dtype=np.float))
            npt.assert_array_equal(ft.position,
                                   np.array([[0, 0, 0]], dtype=np.float))
            self.assertEqual(ft.scale, 1.)

    def test_FTdelta(self):
        ft = euclid.ft.FTdelta()
        ft.set_K(self.K)

        positions = np.array([[0, 0, 0]], dtype=np.float)
        orientations = np.array([[1, 0, 0, 0]] * len(positions),
                                dtype=np.float)
        ft.set_rq(positions, orientations)
        ft.compute()
        npt.assert_array_equal(ft.S, np.ones(len(self.K)))

        # TODO: Need some test using nontrivial k and r vectors that is
        #       analytically verified.
        """
        positions = np.array([[1, 0, 0],
                              [-1, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0],
                              [0, 0, 1],
                              [0, 0, -1]], dtype=np.float)
        orientations = np.array([[1, 0, 0, 0]] * len(positions),
                                dtype=np.float)
        ft.set_rq(positions, orientations)
        ft.compute()
        npt.assert_array_almost_equal(ft.S, np.ones(len(self.K)))
        """

    def test_FTsphere(self):
        ft = euclid.ft.FTsphere()
        self.assertEqual(ft.get_radius(), 0.5)
        ft.set_K(self.K)

        positions = np.array([[0, 0, 0]], dtype=np.float)
        orientations = np.array([[1, 0, 0, 0]] * len(positions),
                                dtype=np.float)
        ft.set_rq(positions, orientations)
        ft.compute()
        npt.assert_equal(ft.S[0], 4./3. * np.pi * ft.get_radius()**3)
        # TODO: Need some test using nontrivial k and r vectors that is
        #       analytically verified.

    def test_FTconvexPolyhedron(self):
        # TODO: Currently using this to test FTpolyhedron indirectly
        cube = euclid.FreudShape.Cube.shape
        npt.assert_almost_equal(cube.getVolume(), 8)
        ft = euclid.ft.FTconvexPolyhedron(cube)
        ft.set_K(self.K)

        positions = np.array([[0, 0, 0]], dtype=np.float)
        orientations = np.array([[1, 0, 0, 0]] * len(positions),
                                dtype=np.float)
        ft.set_rq(positions, orientations)
        ft.compute()
        npt.assert_almost_equal(ft.S[0], 8)
        # TODO: Need some test using nontrivial k and r vectors that is
        #       analytically verified.

if __name__ == '__main__':
    unittest.main()
