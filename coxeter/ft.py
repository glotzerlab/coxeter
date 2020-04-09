"""Calculates Fourier transforms of collections of delta peaks, spheres, and
(convex) polyhedra at specified k-space vectors.

Ported from freud.kspace by Bradley Dice.
Original authors: Eric Irrgang, Jens Glaser.
"""

import numpy as np
import rowan


class _FTbase(object):
    """Compute the Fourier transform of a set of delta peaks at a list of
    :math:`K` points.

    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    Attributes:
        FT (:class:`np.ndarray`): The Fourier transform.
    """

    def __init__(self):
        self.scale = 1.0
        self.density = 1.0
        self.S = np.zeros((0, 0), dtype=np.complex128)
        self.K = np.zeros((1, 3))
        self.position = np.zeros((1, 3))
        self.orientation = np.zeros((1, 4))
        self.orientation[0][0] = 1.0
        self.NK = 0

    def _computeFT(self):
        pass

    def compute(self):
        self._computeFT()
        return self

    def getFT(self):
        """Return Fourier Transform.

        Returns:
            :class:`numpy.ndarray`: Fourier Transform.
        """

        return self.S

    def set_K(self, K):
        """Set the :math:`K` values to evaluate.

        Args:
            K((:math:`N_{K}`, 3) :class:`numpy.ndarray`):
                :math:`K` values to evaluate.
        """
        K = np.asarray(K)
        if K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')

        self.NK = K.shape[0]
        self.K = K

    def set_scale(self, scale):
        """Set scale.

        Args:
            scale (float): Scale.
        """
        self.scale = scale

    def get_scale(self):
        """Get scale.

        Returns:
            float: Scale.
        """
        return self.scale

    def set_rq(self, position, orientation):
        """Set particle positions and orientations.

        Args:
            position ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Particle position vectors.
            orientation ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Particle orientation quaternions.
        """
        position = np.asarray(position)
        orientation = np.asarray(orientation)
        if position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')
        if orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')
        if position.shape[0] != orientation.shape[0]:
            raise TypeError(
                'position and orientation should have the same length')

        self.position = position
        self.orientation = orientation

    def set_density(self, density):
        """Set scattering density.

        Args:
            density (complex): Complex value of scattering density.
        """
        self.density = density

    def get_density(self):
        """Get density.

        Returns:
            complex: Density.
        """
        return self.density


class FTdelta(_FTbase):
    """Compute the Fourier transform of a set of delta peaks at a list of
    :math:`K` points.

    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    Attributes:
        FT (:class:`np.ndarray`): The Fourier transform.
    """

    def __init__(self):
        super(FTdelta, self).__init__()

    def _computeFT(self):
        self.S = np.zeros((len(self.K),), dtype=np.complex128)
        for i, k in enumerate(self.K):
            for r in self.position:
                self.S[i] += self.density * np.exp(-1j * np.dot(k, r))


class FTsphere(_FTbase):
    """
    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    Attributes:
        FT (:class:`np.ndarray`): The Fourier transform.
    """

    def __init__(self):
        super(FTsphere, self).__init__()
        self.radius = 0.5

    def _computeFT(self):
        self.S = np.zeros((len(self.K),), dtype=np.complex128)
        for i, k in enumerate(self.K):
            for r in self.position:
                k_sq = np.dot(k, k)
                if k_sq == 0:
                    f = (4. / 3. * np.pi * self.radius**3)
                else:
                    kr = np.sqrt(k_sq) * self.radius
                    # Note that np.sinc(x) gives sin(pi*x)/(pi*x)
                    f = 4. * np.pi * self.radius / k_sq * \
                        (np.sinc(kr / np.pi) - np.cos(kr))
                self.S[i] += self.density * f * np.exp(-1j * np.dot(k, r))

    def set_radius(self, radius):
        """Set particle radius.

        Args:
            radius (float): Particle radius.
        """
        self.radius = radius

    def get_radius(self):
        """Get radius parameter.

        If appropriate, return value should be scaled by
        :py:meth:`~.get_scale` for interpretation.

        Returns:
            float: Unscaled radius.
        """
        return self.radius


class FTpolyhedron(_FTbase):
    """
    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    Attributes:
        FT (:class:`np.ndarray`): The Fourier transform.
    """

    def __init__(self):
        super(FTpolyhedron, self).__init__()

    def _computeFT(self):
        self.S = np.zeros((len(self.K),), dtype=np.complex128)
        for i, k in enumerate(self.K):
            for r, q in zip(self.position, self.orientation):
                k_sq = np.dot(k, k)
                """
                The FT of an object with orientation q at a given k-space point
                is the same as the FT of the unrotated object at a k-space
                point rotated the opposite way. The opposite of the rotation
                represented by a quaternion is the conjugate of the quaternion,
                found by inverting the sign of the imaginary components.
                """
                k = rowan.rotate(rowan.inverse(q), k)
                f = 0
                if k_sq == 0:
                    f = self.volume
                else:
                    for facet_id, facet in enumerate(self.facets):
                        norm = self.norms[facet_id]
                        k_dot_norm = np.dot(norm, k)
                        k_projected = k - k_dot_norm * norm
                        k_projected_sq = np.dot(k_projected, k_projected)
                        f_2D = 0
                        if k_projected_sq == 0:
                            f_2D = self.areas[facet_id]
                        else:
                            n_verts = len(facet)
                            for edge_id in range(n_verts):
                                r0 = self.verts[facet[edge_id]]
                                r1 = self.verts[facet[(edge_id + 1) % n_verts]]
                                edge_vec = r1 - r0
                                edge_center = 0.5 * (r0 + r1)
                                edge_cross_k = np.cross(edge_vec, k_projected)
                                k_dot_center = np.dot(k_projected, edge_center)
                                k_dot_edge = np.dot(k_projected, edge_vec)
                                # Note that np.sinc(x) gives sin(pi*x)/(pi*x)
                                f_n = np.dot(norm, edge_cross_k) * \
                                    np.sinc(0.5 * k_dot_edge / np.pi) / \
                                    k_projected_sq
                                f_2D -= f_n * 1j * np.exp(-1j * k_dot_center)
                        d = self.d[facet_id]
                        exp_kr = np.exp(-1j * k_dot_norm * d)
                        f += k_dot_norm * (1j * f_2D * exp_kr)
                    # end loop over facets
                    f /= k_sq
                # end if/else, f is now calculated
                # S += rho * f * exp(-i k r)
                self.S[i] += self.density * f * \
                    np.exp(-1j * np.dot(k, rowan.rotate(rowan.inverse(q), r)))

    def set_params(self, verts, facets, norms, d, areas, volume):
        """Set polyhedron geometry.

        Args:
            verts ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vertex coordinates.
            facets ((:math:`N_{facets}`, 3) :class:`numpy.ndarray`):
                Facet vertex indices.
            norms ((:math:`N_{facets}`, 3) :class:`numpy.ndarray`):
                Facet normals.
            d ((:math:`N_{facets}`) :class:`numpy.ndarray`):
                Facet distances.
            area ((:math:`N_{facets}`) :class:`numpy.ndarray`):
                Facet areas.
            volume (:class:`numpy.float64`):
                Polyhedron volume.
        """
        verts = np.asarray(verts)
        if verts.shape[1] != 3:
            raise TypeError('verts should be an Nx3 array')

        facet_offs = np.zeros((len(facets) + 1), dtype=np.uint32)
        for i, f in enumerate(facets):
            facet_offs[i + 1] = facet_offs[i] + len(f)

        facet_offs = np.asarray(facet_offs)

        facets = np.asarray(facets)

        norms = np.asarray(norms)
        if norms.shape[1] != 3:
            raise TypeError('norms should be an Nx3 array')

        d = np.asarray(d)

        areas = np.asarray(areas)

        if norms.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError(
                ('Length of norms should be equal to number of facet offsets'
                    '- 1'))
        if d.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError(
                ('Length of facet distances should be equal to number of facet'
                    'offsets - 1'))
        if areas.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError(
                ('Length of areas should be equal to number of facet offsets'
                    '- 1'))
        self.verts = verts
        self.facet_offs = facet_offs
        self.facets = facets
        self.norms = norms
        self.d = d
        self.areas = areas
        self.volume = volume


class FTconvexPolyhedron(FTpolyhedron):
    """Fourier Transform for convex polyhedra.

    Args:
        hull (:class:`coxeter.shape_classes.ConvexPolyhedron`):
            Convex polyhedron object.
    """

    def __init__(self, hull):
        super(FTconvexPolyhedron, self).__init__()
        self.hull = hull

        # set convex hull geometry
        verts = self.hull.vertices * self.scale
        facets = self.hull.facets
        norms = self.hull._equations[:, 0:3]
        d = -self.hull._equations[:, 3] * self.scale
        areas = [self.hull.get_facet_area(i) * self.scale**2.0
                 for i in range(len(facets))]
        volume = self.hull.volume * self.scale**3.0
        self.set_params(verts, facets, norms, d, areas, volume)

    def set_radius(self, radius):
        """Set radius of in-sphere.

        Args:
            radius (float):
                Radius of inscribed sphere without scale applied.
        """
        # Find original in-sphere radius, determine necessary scale factor,
        # and scale vertices and surface distances
        self.hull.setInsphereRadius(float(radius))

    def get_radius(self):
        """Get radius parameter.

        If appropriate, return value should be scaled by
        get_parambyname('scale') for interpretation.

        Returns:
            float: Unscaled radius.
        """
        # Find current in-sphere radius
        return self.hull.getInsphereRadius()

    def Spoly2D(self, i, k):
        """Calculate Fourier transform of polygon.

        Args:
            i (float):
                Face index into self.hull simplex list.
            k (:class:`numpy.ndarray`):
                Angular wave vector at which to calculate
                :math:`S\\left(i\\right)`.
        """
        if np.dot(k, k) == 0.0:
            S = self.hull.getArea(i) * self.scale**2
        else:
            S = 0.0
            nverts = self.hull.nverts[i]
            verts = list(self.hull.facets[i, 0:nverts])
            # apply periodic boundary condition for convenience
            verts.append(verts[0])
            points = self.hull.points * self.scale
            n = self.hull.equations[i, 0:3]
            for j in range(self.hull.nverts[i]):
                v1 = points[verts[j + 1]]
                v0 = points[verts[j]]
                edge = v1 - v0
                centrum = np.array((v1 + v0) / 2.)
                # Note that np.sinc(x) gives sin(pi*x)/(pi*x)
                x = np.dot(k, edge) / np.pi
                cpedgek = np.cross(edge, k)
                S += np.dot(n, cpedgek) * np.exp(
                    -1.j * np.dot(k, centrum)) * np.sinc(x)
            S *= (-1.j / np.dot(k, k))
        return S

    def Spoly3D(self, k):
        """Calculate Fourier transform of polyhedron.

        Args:
            k (int):
                Angular wave vector at which to calculate
                :math:`S\\left(i\\right)`.
        """
        if np.dot(k, k) == 0.0:
            S = self.hull.getVolume() * self.scale**3
        else:
            S = 0.0
            # for face in faces
            for i in range(self.hull.nfacets):
                # need to project k into plane of face
                ni = self.hull.equations[i, 0:3]
                di = - self.hull.equations[i, 3] * self.scale
                dotkni = np.dot(k, ni)
                k_proj = k - ni * dotkni
                S += dotkni * np.exp(-1.j * dotkni * di) * \
                    self.Spoly2D(i, k_proj)
            S *= 1.j / (np.dot(k, k))
        return S
