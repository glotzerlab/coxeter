import unittest
import numpy as np
from euclid.polyhedron import ConvexPolyhedron, ConvexSpheropolyhedron

tetrahedron = np.array([[0.5, -0.5, -0.5], [0.5, 0.5, 0.5],
                        [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]])
cube = np.concatenate((tetrahedron, -tetrahedron))
cubepoly = ConvexPolyhedron(cube)
tolerance = 1e-6


class TestShape(unittest.TestCase):
    pass


class TestConvexPolyhedron(unittest.TestCase):

    def test_mergeFacets(self):
        # Check ConvexPolyhedron.mergeFacets
        assert cubepoly.nfacets == 6, \
            ("ConvexPolyhedron.mergeFacets did not produce the right number ",
             "faces for a cube")

    def test_rhFace(self):
        # Check ConvexPolyhedron.rhFace
        for i in range(cubepoly.nfacets):
            normal = cubepoly.equations[i, 0:3]
            verts = cubepoly.facets[i]
            v0 = cubepoly.points[verts[0]]
            for j in range(1, cubepoly.nverts[i] - 1):
                v1 = cubepoly.points[verts[j]] - v0
                v2 = cubepoly.points[verts[j+1]] - v0
                cp = np.cross(v1, v2)
                assert np.dot(normal, cp) >= 0, \
                    ('For face {i}, rays {a} and {b} do not produce an ',
                     'outward facing cross product'.format(
                         i=i,
                         a=[verts[0], verts[j]],
                         b=[verts[0], verts[j+1]]))

    def test_rhNeighbor(self):
        # Check ConvexPolyhedron.rhNeighbor
        # The kth neighbor of facet i should share vertices
        # cubepoly.facets[i, [k, k+1]]
        for i in range(cubepoly.nfacets):
            facet = list(cubepoly.facets[i])
            # Apply periodic boundary for convenience
            facet.append(facet[0])
            for k in range(cubepoly.nverts[i]):
                edge = [facet[k], facet[k+1]]
                edge_set = set(edge)
                neighbor = cubepoly.neighbors[i, k]
                neighbor_set = set(cubepoly.facets[neighbor])
                # Check if edge points are a subset of the neighbor points
                assert edge_set < neighbor_set, \
                    ('Face {i} has neighboring facet {k} that does not share '
                     'vertices {a} and {b}.'.format(
                         k=neighbor,
                         i=i,
                         a=edge[0],
                         b=edge[1]))

    def test_getArea(self):
        # Check ConvexPolyhedron.getArea
        area = cubepoly.getArea()
        assert abs(area - 6.0) <= tolerance
        area = cubepoly.getArea(1)
        assert abs(area - 1.0) <= tolerance

    def test_getVolume(self):
        # Check ConvexPolyhedron.getVolume
        volume = cubepoly.getVolume()
        assert abs(volume - 1.0) < tolerance, \
            ('ConvexPolyhedron.getVolume found volume {v} when it should be ',
             '1.0'.format(v=volume))

    def test_getDihedral(self):
        # Check Polyhedron.getDihedral
        tetrahedronpoly = ConvexPolyhedron(tetrahedron)
        for i in range(1, tetrahedronpoly.nfacets):
            dihedral = tetrahedronpoly.getDihedral(0, i)
            assert 0 <= dihedral <= np.pi/2, \
                ('Polyhedron.getDihedral found one or more bogus angles for '
                 'tetrahedron')

    def test_getInsphereRadius(self):
        # Check ConvexPolyhedron.getInsphereRadius
        rectangularBox = np.array(cube)
        rectangularBox[:, 2] *= 2
        isrShouldBe = 0.5
        boxpoly = ConvexPolyhedron(rectangularBox)
        isr = boxpoly.getInsphereRadius()
        assert abs(isr - isrShouldBe) < tolerance, \
            ('ConvexPolyhedron.getInsphereRadius found {r1} when it should '
             'be 0.5'.format(r1=isr))

    def test_getCircumsphereRadius(self):
        # Check ConvexPolyhedron.getCircumsphereRadius
        rectangularBox = np.array(cube)
        rectangularBox[:, 2] *= 2
        osrShouldBe = np.sqrt(1.0*1.0 + 0.5*0.5 + 0.5*0.5)
        boxpoly = ConvexPolyhedron(rectangularBox)
        osr = boxpoly.getCircumsphereRadius()
        assert abs(osr - osrShouldBe) < tolerance, \
            ('ConvexPolyhedron.getCircumsphereRadius found {0} when it '
             'should be {1}'.format(osr, osrShouldBe))

    def test_setInsphereRadius(self):
        # Check ConvexPolyhedron.setInsphereRadius
        rectangularBox = np.array(cube)
        rectangularBox[:, 2] *= 2
        boxpoly = ConvexPolyhedron(rectangularBox)
        boxpoly.setInsphereRadius(1.0)
        boxpoly.setInsphereRadius(3.33)
        boxpoly.setInsphereRadius(1.0)
        isr = boxpoly.getInsphereRadius()
        assert abs(isr - 1.0) < tolerance, \
            ('ConvexPolyhedron.setInsphereRadius resulted in {r1} when it '
             'should be 1.0'.format(r1=isr))

    def test_setCircumsphereRadius(self):
        # Check ConvexPolyhedron.setCircumsphereRadius
        rectangularBox = np.array(cube)
        rectangularBox[:, 2] *= 2
        boxpoly = ConvexPolyhedron(rectangularBox)
        boxpoly.setCircumsphereRadius(1.0)
        boxpoly.setCircumsphereRadius(4.0)
        boxpoly.setCircumsphereRadius(1.0)
        osr = boxpoly.getCircumsphereRadius()
        assert abs(osr - 1.0) < tolerance, \
            ('ConvexPolyhedron.setCircumsphereRadius resulted in {r1} when '
             'it should be 1.0'.format(r1=osr))

    def test_isInside(self):
        # Check ConvexPolyhedron.isInside
        v1 = (-0.4, 0.1, 0.49)
        v2 = (0.5, 0.1, 0.51)
        assert cubepoly.isInside(v1), \
            ('ConvexPolyhedron.isInside does not return True when it should')
        assert not cubepoly.isInside(v2), \
            ('ConvexPolyhedron.isInside does not return False when it should')

    def test_asphericity(self):
        # Check Polyhedron curvature and asphericity determination
        t_points = np.array([[0.5, -0.5, -0.5], [0.5, 0.5, 0.5],
                             [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]])
        tetrahedronpoly = ConvexPolyhedron(t_points)
        alpha = tetrahedronpoly.getAsphericity()
        target = 2.23457193395116
        assert abs(alpha - target) < tolerance, \
            ("Polyhedron.getAsphericity for tetrahedron found {0}. "
             "Should be {1}.".format(alpha, target))

    def test_pathological(self):
        # Check some pathologically tricky point sets from which to wrap
        # convex polyhedra
        try:
            ConvexPolyhedron([[0.91362386,  1.10105279,  4.18922237],
                              [3.83002807,  2.04126867,  1.02440944],
                              [4.57425055,  0.43286285,  1.79130228],
                              [-2.6605048,   2.43464971, -1.1060311],
                              [0.45252156, -4.03371108, -4.82946401],
                              [-2.89015607,  4.85837971, -3.56173295],
                              [1.61194889, -1.49216365, -2.98212742],
                              [-1.67820421, -1.86887374,  4.48610106],
                              [-0.82725764, -1.37828095, -3.60469154],
                              [-1.5991328,  -4.4569862,   2.42236848],
                              [0.24897384, -3.84177112, -2.19968814],
                              [-0.84937492, -4.03836328, -0.90140948]])
        except Exception as e:
            print("Pathological polyhedron 1 could not be processed")
            print(e.message)

        try:
            ConvexPolyhedron([[0.91362386,  1.10105279,  4.18922237],
                              [3.83002807,  2.04126867,  1.02440944],
                              [4.57425055,  0.43286285,  1.79130228],
                              [-2.6605048,   2.43464971, -1.1060311],
                              [0.45252156, -4.03371108, -4.82946401],
                              [-2.89015607,  4.85837971, -3.56173295],
                              [1.61194889, -1.49216365, -2.98212742],
                              [-1.67820421, -1.86887374,  4.48610106],
                              [-0.82725764, -1.37828095, -3.60469154],
                              [-1.5991328,  -4.4569862,   2.42236848],
                              [0.24897384, -3.84177112, -2.19968814],
                              [-0.84937492, -4.03836328, -0.90140948]])
        except Exception as e:
            print("Pathological polyhedron 2 could not be processed")
            print(e.message)


class TestConvexSpheropolyhedron(unittest.TestCase):

    def test_getArea(self):
        # Check ConvexSpheropolyhedron.getArea
        R = 1.0
        L = 1.0
        spherocubepoly = ConvexSpheropolyhedron(cube, R)
        ConvexPolyhedron.setInsphereRadius(spherocubepoly, L/2.)
        Aface = L*L
        Asphere = 4.0 * np.pi * R * R
        Acyl = L * 2.0 * np.pi * R
        area_should_be = 6 * Aface + 3 * Acyl + Asphere
        area = spherocubepoly.getArea()
        assert abs(area - area_should_be) < tolerance, \
            ('ConvexSpheropolyhedron.getArea found area {0} when it should '
             'be {1}'.format(area, area_should_be))

    def test_getVolume(self):
        # Check ConvexSpheropolyhedron.getVolume
        R = 1.0
        L = 1.0
        spherocubepoly = ConvexSpheropolyhedron(cube, R)
        ConvexPolyhedron.setInsphereRadius(spherocubepoly, L/2.)
        Vpoly = L*L*L
        Vplate = L*L*R
        Vcyl = L * np.pi * R * R
        Vsphere = 4.0 * np.pi * R * R * R / 3.0
        volume_should_be = Vpoly + 6 * Vplate + 3 * Vcyl + Vsphere
        volume = spherocubepoly.getVolume()
        assert abs(volume - volume_should_be) < tolerance, \
            ('ConvexSpheroolyhedron.getVolume found volume {0} when it '
             'should be {1}'.format(volume, volume_should_be))

    def test_setInsphereRadius(self):
        # Check ConvexSpheropolyhedron.setInsphereRadius
        R = 1.0
        R_target = R*2
        spherocubepoly = ConvexSpheropolyhedron(cube, R)
        insphereR = spherocubepoly.getInsphereRadius()
        isr_target = insphereR * 2
        spherocubepoly.setInsphereRadius(1.0)
        spherocubepoly.setInsphereRadius(3.33)
        spherocubepoly.setInsphereRadius(insphereR * 2)
        isr = spherocubepoly.getInsphereRadius()
        assert abs(isr - isr_target) < tolerance, \
            ('ConvexSpheropolyhedron.setInsphereRadius produces isr={0} when '
             'it should be {1}'.format(isr, isr_target))
        assert abs(spherocubepoly.R - R_target) < tolerance, \
            ('ConvexSpheropolyhedron.setInsphereRadius produces R={0} when '
             'it should be {1}'.format(R, R_target))

    def test_setCircumsphereRadius(self):
        # Check ConvexSpheropolyhedron.setCircumsphereRadius
        R = 1.0
        R_target = R*2
        spherocubepoly = ConvexSpheropolyhedron(cube, R)
        osphereR = spherocubepoly.getCircumsphereRadius()
        osr_target = osphereR * 2
        spherocubepoly.setCircumsphereRadius(1.0)
        spherocubepoly.setCircumsphereRadius(3.33)
        spherocubepoly.setCircumsphereRadius(osphereR * 2)
        osr = spherocubepoly.getCircumsphereRadius()
        assert abs(osr - osr_target) < tolerance, \
            ('ConvexSpheropolyhedron.setCircumsphereRadius produces osr={0} '
             'when it should be {1}'.format(osr, osr_target))
        assert abs(spherocubepoly.R - R_target) < tolerance, \
            ('ConvexSpheropolyhedron.setCircumsphereRadius produces R={0} '
             'when it should be {1}'.format(R, R_target))

    def test_isInside(self):
        # Check ConvexSpheropolyhedron.isInside
        spherocubepoly = ConvexSpheropolyhedron(cube)
        v1 = (-0.4, 0.1, 0.49)
        v2 = (0.5, 0.1, 0.51)
        assert spherocubepoly.isInside(v1), \
            ('ConvexSpheropolyhedron.isInside does not return True when it '
             'should')
        assert not spherocubepoly.isInside(v2), \
            ('ConvexSpheropolyhedron.isInside does not return False when it '
             'should')


if __name__ == '__main__':
    unittest.main()
