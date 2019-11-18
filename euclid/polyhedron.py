'''\package euclid.FreudShape

Classes to manage shape data.


Provides data structures and calculation methods for working with polyhedra with
nice data structures.

Attributes:

 - npoints number of input points
 - ndim number of dimensions of input points (should be 3)
 - points ndarray (npoints, ndim) input points
 - nfacets number of facets
 - nverts ndarray (nfacets,) of number of vertices and neighbors per facet
 - facets ndarray (nfacets, max(nverts)) vertex indices for each facet. values
   for facets[i, j > nverts[i]] are undefined
 - neighbors (nfacets, max(nverts)) neighbor k shares vertices k and k+1 with
   face. values for neighbors[i, k > nverts[i] - 1] are undefined
 - equations (nfacets, ndim+1) [n, d] for corresponding facet where n is the 3D
   normal vector and d the offset from the origin.
   Satisfies the hyperplane equation \f$ \bar v \cdot \hat n + d < 0 \f$ for
   points v enclosed by the surface.
 - simplicial referece to another Polygon object containing data based on
   simplicial facets

 Example:
 Some shapes are already provided in the module
 >>> from euclid.FreudShape.Cube import shape
 these are the vertices for a cube
 >>> shape.points
 array([
       [ 1.,  1.,  1.],
       [ 1., -1.,  1.],
       [-1., -1.,  1.],
       [-1.,  1.,  1.],
       [ 1.,  1., -1.],
       [ 1., -1., -1.],
       [-1., -1., -1.],
       [-1.,  1., -1.]])

 these are the hyperplane equations enumerating the facets
 >>> shape.equations
 array([
       [-0., -0.,  1., -1.],
       [ 1.,  0., -0., -1.],
       [-0.,  1.,  0., -1.],
       [-0., -1.,  0., -1.],
       [-1.,  0.,  0., -1.],
       [ 0.,  0., -1., -1.]])


 these are right-handed lists of vertex indices defining each facet
 >>> shape.facets
 array([
       [0, 3, 2, 1],
       [4, 0, 1, 5],
       [4, 7, 3, 0],
       [1, 2, 6, 5],
       [3, 7, 6, 2],
       [5, 6, 7, 4]])

 these are right-handed lists of facet indices for the facets which border each
 facet
 >>> shape.neighbors
 array([
       [2, 4, 3, 1],
       [2, 0, 3, 5],
       [5, 4, 0, 1],
       [0, 4, 5, 1],
       [2, 5, 3, 0],
       [3, 4, 2, 1]])

 The Polyhedron methods assume facet vertices and neighbor lists have
 right-handed ordering. If input data is not available at instantiation, you
 can use some helper functions to reorder the data.

 Example:
 \code
 mypoly = Polyhedron(points, nverts, facets, neighbors, equations)
 for i in range(mypoly.nfacets):
   mypoly.facets[i, 0:mypoly.nverts[i]] = mypoly.rhFace(i)
 for i in range(mypoly.nfacets):
   mypoly.neighbors[i, 0:mypoly.nverts[i]] = mypoly.rhNeighbor(i)
 \endcode
 '''
import rowan
import logging
import numpy as np
logger = logging.getLogger(__name__)
try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None
    msg = "scipy.spatial.ConvexHull is not available, so "
    msg += "euclid.FreudShape.ConvexPolyhedron is not available."
    logger.warning(msg)


class Polyhedron:
    '''Create a Polyhedron object from precalculated data structures.
     Args:
        \param points (Np, 3) list of vertices ordered such that indices are
            used by other data structures
        \param nverts (Nf,) list of numbers of vertices for correspondingly indexed facet
        \param facets (Nf, max(nverts)) array of vertex indices associated with each facet
        \param neighbors (Nf, max(nverts)) array of facet neighbor information. For
            neighbors[i,k], neighbor k shares points[[k, k+1]] with facet i.
        \param equations (Nf, ndim + 1) list of lists of hyperplane parameters of the
            form [[n[0], n[1], n[2], d], ...] where n, d satisfy the hyperplane
            equation \f$ \bar v \cdot \hat n + d < 0 \f$ for points v enclosed by
            the surface.
        \param simplicial_facets (Nsf, 3) List of simplices (triangular facets in 3D)
        \param simplicial_neighbors (Nsf, 3) List of neighboring simplices for each simplicial facet
        \param simplicial_equations (Nsf, ndim+1) hyperplane equation coefficients for simplicial facets
    '''

    def __init__(
            self,
            points,
            nverts,
            facets,
            neighbors,
            equations,
            simplicial_facets=None,
            simplicial_neighbors=None,
            simplicial_equations=None):
        self.points = np.array(points)
        self.npoints = len(self.points)
        pshape = points.shape
        if (len(pshape) != 2) or pshape[1] != 3:
            raise ValueError("points parameter must be an Nx3 array of points")
        self.ndim = pshape[1]

        self.nverts = np.array(nverts, dtype=int)
        self.facets = np.array(facets, dtype=int)
        self.nfacets = len(facets)
        self.neighbors = np.array(neighbors, dtype=int)
        self.equations = np.array(equations)
        # Should put in some error checking here...

        self.originalpoints = np.array(self.points)
        self.originalequations = np.array(self.equations)

        if not (simplicial_facets is None or
                simplicial_equations is None or
                simplicial_neighbors is None):
            nfacets = len(simplicial_facets)
            self.simplicial = Polyhedron(points,
                                         [self.ndim] * nfacets,
                                         simplicial_facets,
                                         simplicial_neighbors,
                                         simplicial_equations)
        else:
            self.simplicial = None

    def mergeFacets(self):
        '''Merge coplanar simplicial facets

        Requires multiple iterations when many non-adjacent coplanar facets
        exist.

        Example:
        \code
        old_nfacets = 0
        new_nfacets = self.nfacets
        while new_nfacets != old_nfacets:
        mypoly.mergeFacets()
        old_nfacets = new_nfacets
        new_nfacets = mypoly.nfacets
        '''
        # If performance is ever an issue, this should really all be replaced
        # with our own qhull wrapper...

        if ConvexHull is None:
            logger.error(
                "Cannot safely merge coplanar facets because "
                "scipy.spatial.ConvexHull is not available.")
            return

        def convexPolygon3D(normal, points):
            '''Helper function to identify the exterior points of a convex polygon.

            Projects the input points onto a plane and finds the 2D convex hull.
            \param normal the normal vector defining the plane of the face
            \param points list of 3D points (presumed to be) in the plane of the
                face
            \returns indices of points on the exterior of the polygon
            '''
            # Rotate the plane and points to be parallel to the x,y plane
            # The easiest way to do this is to find the quaternion that rotates
            # by pi about the unit vector that bisects the input normal vector
            # and the z unit vector.
            nhat = np.asarray(normal, dtype=np.float64)
            zhat = np.array([0., 0., 1.], dtype=np.float64)
            # check if normal is antialigned with zhat, in which case we'll
            # just rotate around xhat
            if np.dot(zhat, nhat) + 1.0 < 1e-12:
                u = np.array([1., 0., 0.], dtype=np.float64)
            else:
                v = nhat + zhat
                vmag2 = np.dot(v, v)
                u = v / np.sqrt(vmag2)
            q = np.concatenate(
                ([np.cos(np.pi / 2)], np.sin(np.pi / 2) * u))
            # Rotate the points and drop the z dimension to get a set of 2D
            # points
            p3D = [rowan.rotate(q, p) for p in points]
            p2D = np.array(p3D)[:, :2]
            # Find the 2D convex hull of the points
            hull = ConvexHull(p2D)
            # Collect the indices of the exterior points and return the list of
            # indices
            return hull.vertices
        Nf = self.nfacets
        facet_verts = [set(self.facets[i, 0:self.nverts[i]])
                       for i in range(self.nfacets)]
        neighbors = [set(self.neighbors[i, 0:self.nverts[i]])
                     for i in range(self.nfacets)]
        equations = list(self.equations)
        normals = list(self.equations[:, 0:3])
        nverts = list(self.nverts)
        face = 0
        # go in order through the faces. For each face, check to see which of
        # its neighbors should be merged.
        while face < Nf:
            # Since we're using the dot product to detect coplanar facets, be
            # precise.
            n0 = np.asarray(normals[face], dtype=np.float64)
            merge_list = list()
            for neighbor in neighbors[face]:
                n1 = np.asarray(normals[neighbor], dtype=np.float64)
                d = np.dot(n0, n1)
                if abs(d - 1.0) < 1e-13:
                    merge_list.append(neighbor)
            # for each neighbor in merge_list:
            #  merge points in simplices
            #  merge (and prune) neighbors
            #  update other neighbor lists
            #  prune neighbors, equations, normals, facet_verts, nverts
            #  update Nf
            #  update neighbors and merge_list references
            #  check next face
            # afterwards, update nverts
            for m in range(len(merge_list)):
                merged_neighbor = merge_list[m]
                # merge in points from neighboring facet
                facet_verts[face] |= facet_verts[merged_neighbor]
                # merge in neighbors from neighboring facet
                neighbors[face] |= neighbors[merged_neighbor]
                # remove self and neighbor from neighbor list
                neighbors[face].remove(merged_neighbor)
                neighbors[face].remove(face)
                # update other neighbor lists: replace occurrences of neighbor
                # with face
                for i in range(len(neighbors)):
                    if merged_neighbor in neighbors[i]:
                        neighbors[i].remove(merged_neighbor)
                        neighbors[i].add(face)
                # prune neighbors, equations, normals, face_verts, nverts
                del neighbors[merged_neighbor]
                del equations[merged_neighbor]
                del normals[merged_neighbor]
                del facet_verts[merged_neighbor]
                del nverts[merged_neighbor]
                # correct for changing face list length
                Nf -= 1
                # Deal with changed indices for merge_list and neighbors
                # update merge_list
                for i in range(m + 1, len(merge_list)):
                    if merge_list[i] > merged_neighbor:
                        merge_list[i] -= 1
                # update neighbors
                # note that all facet indices > merged_neighbor have to be
                # decremented. This is going to be slow...  Maybe optimize by
                # instead making a translation table during processing to be
                # applied later.  A better optimization would be a c++ module
                # to access qhull directly rather than through scipy.spatial
                if merged_neighbor < face:
                    face -= 1
                for i in range(len(neighbors)):
                    nset = neighbors[i]
                    narray = np.array(list(nset))
                    mask = narray > merged_neighbor
                    narray[mask] -= 1
                    neighbors[i] = set(narray)
            # The new face may now contain interior points not part of any
            # edge. These need to be removed and nverts updated.  Project the
            # vertices onto the plane of the face and find the convex hull.
            polygon_points = list(facet_verts[face])
            # ext_points is a list of indices into polygon_points
            ext_points = convexPolygon3D(
                normals[face], self.points[polygon_points])
            facet_verts[face] = set([polygon_points[i] for i in ext_points])
            nverts[face] = len(facet_verts[face])
            face += 1  # proceed to next face
        # write updated data to self.facets, self.equations, self.neighbors,
        # self.nfacets, self.nverts
        self.nfacets = len(facet_verts)
        self.nverts = np.array(nverts)
        self.facets = np.empty((self.nfacets, max(self.nverts)), dtype=int)
        self.neighbors = np.empty(
            (self.nfacets, max(self.nverts)), dtype=int)
        for i in range(self.nfacets):
            self.facets[i, :self.nverts[i]] = np.array(list(facet_verts[i]))
            self.neighbors[i, :(self.nverts[i])] = np.array(
                list(neighbors[i]))
        self.equations = np.array(list(equations))

    def rhFace(self, iface):
        '''Use a list of vertices and a outward face normal and return a
        right-handed ordered list of vertex indices.

        \param iface index of facet to process
        '''
        # n = np.asarray(normal)
        Ni = self.nverts[iface]  # number of vertices in facet
        n = self.equations[iface, 0:3]
        facet = self.facets[iface, 0:Ni]
        points = self.points

        z = np.array([0., 0., 1.])
        theta = np.arccos(n[2])
        if np.dot(n, z) == 1.0:
            # face already aligned in z direction
            q = np.array([1., 0., 0., 0.])
        elif np.dot(n, z) == -1.0:
            # face anti-aligned in z direction
            q = np.array([0., 1., 0., 0.])
        else:
            cp = np.cross(n, z)
            k = cp / np.sqrt(np.dot(cp, cp))
            q = np.concatenate(
                ([np.cos(theta / 2.)], np.sin(theta / 2.) * k))
        vertices = points[facet]  # 3D vertices
        for i in range(Ni):
            v = vertices[i]
            # print("rotating {point} with {quat}".format(point=v, quat=q))
            vertices[i] = rowan.rotate(q, v)
        vertices = vertices[:, 0:2]  # 2D vertices, discarding the z axis
        # The indices of vertices correspond to the indices of facet
        centrum = vertices.sum(axis=0) / Ni
        # sort vertices
        idx_srt = list()
        a_srt = list()
        for i in range(Ni):
            r = vertices[i] - centrum
            a = np.arctan2(r[1], r[0])
            if a < 0.0:
                a += np.pi * 2.0
            new_i = 0
            for j in range(len(idx_srt)):
                if a <= a_srt[j]:
                    break
                else:
                    new_i = j + 1
            idx_srt.insert(new_i, facet[i])
            a_srt.insert(new_i, a)
        return np.array(idx_srt)

    def rhNeighbor(self, iface):
        '''Use the list of vertices for a face to order a list of neighbors,
        given their vertices.

        \param iface index of facet to process
        '''
        Ni = self.nverts[iface]
        facet = list(self.facets[iface, 0:Ni])
        # for convenience, apply the periodic boundary condition
        facet.append(facet[0])

        # get a list of sets of vertices for each neighbor
        old_neighbors = list(self.neighbors[iface, 0:Ni])
        neighbor_verts = list()
        for i in range(Ni):
            neighbor = old_neighbors[i]
            verts_set = set(self.facets[neighbor, 0:self.nverts[neighbor]])
            neighbor_verts.append(verts_set)

        new_neighbors = list()
        for i in range(Ni):
            # Check each pair of edge points in turn
            edge = set([facet[i], facet[i + 1]])
            for j in range(len(neighbor_verts)):
                # If edge points are also in neighboring face then we have
                # found the corresponding neighbor
                if edge < neighbor_verts[j]:
                    new_neighbors.append(old_neighbors[j])
                    # del old_neighbors[j]
                    # del neighbor_verts[j]
                    break
        return np.array(new_neighbors)

    def getArea(self, iface=None):
        '''Find surface area of polyhedron or a face

        \param iface index of facet to calculate area of (default sum all facet
            area)
        '''
        if iface is None:
            facet_list = range(self.nfacets)
        else:
            facet_list = list([iface])
        A = 0.0
        # for each face
        for i in facet_list:
            face = self.facets[i]
            # print(face)
            n = self.equations[i, 0:3]
            # print(n)
            Ni = self.nverts[i]  # number of points on the facet)
            # for each triangle on the face
            for j in range(1, Ni - 1):
                r1 = self.points[face[j]] - self.points[face[0]]
                r2 = self.points[face[j + 1]] - self.points[face[0]]
                cp = np.cross(r1, r2)
                # print(cp)
                A += abs(np.dot(cp, n)) / 2.0
        return A

    def getVolume(self):
        '''Find the volume of the polyhedron'''
        V = 0.0
        # for each face, calculate area -> volume, and accumulate
        for i in range(len(self.facets)):
            d = -1 * self.equations[i, 3]  # distance from centroid
            A = Polyhedron.getArea(self, i)
            V += d * A / 3.0
        return V

    def getCircumsphereRadius(self, original=False):
        '''Get circumsphere radius

        \param original True means to retrieve the original points before any
        subsequent rescaling (default False)
        '''
        # get R2[i] = dot(points[i], points[i]) by getting the diagonal (i=j)
        # of the array of dot products dot(points[i], points[j])
        if original:
            points = self.originalpoints
        else:
            points = self.points
        R2 = np.diag(np.dot(points, points.T))
        return np.sqrt(R2.max())

    def getInsphereRadius(self, original=False):
        '''Get insphere radius

        \param original True means to retrieve the original points before any
            subsequent rescaling (default False)
        '''
        if original:
            equations = self.originalequations
        else:
            equations = self.equations
        facetDistances = equations[:, 3]
        return abs(facetDistances.max())

    def setCircumsphereRadius(self, radius):
        '''Scale polyhedron to fit a given circumsphere radius
        \param radius new circumsphere radius
        '''
        # use unscaled data from original to avoid accumulated errors
        oradius = Polyhedron.getCircumsphereRadius(self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:, 3] = self.originalequations[:, 3] * scale_factor
        if self.simplicial is not None:
            self.simplicial.points = self.simplicial.originalpoints * scale_factor
            self.simplicial.equations[:,
                                      3] = self.simplicial.originalequations[:,
                                                                             3] * scale_factor

    def setInsphereRadius(self, radius):
        '''Scale polyhedron to fit a given circumsphere radius

        \param radius new insphere radius
        '''
        oradius = Polyhedron.getInsphereRadius(self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:, 3] = self.originalequations[:, 3] * scale_factor
        if self.simplicial is not None:
            self.simplicial.points = self.simplicial.originalpoints * scale_factor
            self.simplicial.equations[:,
                                      3] = self.simplicial.originalequations[:,
                                                                             3] * scale_factor

    def isInside(self, point):
        '''Test if a point is inside the shape

        \param point 3D coordinates of test point
        '''
        v = np.asarray(point)
        for i in range(self.nfacets):
            d = np.dot(v, self.equations[i, 0:3])
            if d + self.equations[i, 3] > 0:
                return False
        return True

    def getSharedEdge(self, a, b):
        '''Identify the index of facet b as a neighbor of facet a

        The index of neighbor b also corresponds to the index of the first of
        two right-hand-ordered vertices of the shared edge

        \returns the index of b in the neighbor list of a or None if they are
        not neighbors

        \par Example
        from euclid.FreudShape.Cube import shape
        a, b = 0, 1
        edge_i = shape.getSharedEdge(a,b)
        edge_j = (edge_i + 1) % shape.nverts[a]
        point_coords = shape.points[[shape.facets[a, edge_i], shape.facets[a, edge_j]]]
        '''
        # Note that facet only has as many neighbors as it does vertices
        neighbors = list(self.neighbors[a, 0:self.nverts[a]])
        try:
            k = neighbors.index(b)
        except ValueError:
            k = None
        return k

    def getDihedral(self, a, b):
        '''Get the signed dihedral angle between two facets.

        Theta == 0 implies faces a and b form a convex blade.  Theta == pi
        implies faces a and b are parallel. Theta == 2 pi implies faces a and b
        form a concave blade.

        \param a index of first facet
        \param b index of second facet (must be a neighbor of a)
        \returns theta angle on [0, 2 pi)
        '''
        # Find which neighbor b is
        k = self.getSharedEdge(a, b)
        if k is None:
            raise ValueError("b must be a neighbor of a")

        # Find path e1 -> e2 -> e3, where e2 is an edge shared by both faces,
        # e1 lies in a and e3 lies in b.  Note that to find interior angle,
        # e1 -> e2 are in the left-handed direction of a while e2 -> e3 are in
        # the right-handed direction of b.  Denote the path as
        # points p0 -> p1 -> p2 -> p3
        nextk = k + 1
        if nextk >= self.nverts[a]:
            nextk = 0
        nextnextk = nextk + 1
        if nextnextk >= self.nverts[a]:
            nextnextk = 0
        p0 = self.facets[a, nextnextk]
        p1 = self.facets[a, nextk]
        p2 = self.facets[a, k]
        k_new = list(self.facets[b]).index(self.facets[a, k])
        # Check for logic error
        if (p2 != self.facets[b, k_new]):
            raise RuntimeError("Logic error finding k_new from p2")
        nextk = k_new + 1
        if nextk >= self.nverts[b]:
            nextk = 0
        p3 = self.facets[b, nextk]

        # Get vectors along path
        v1 = self.points[p1] - self.points[p0]
        v2 = self.points[p2] - self.points[p1]
        v3 = self.points[p3] - self.points[p2]

        cp12 = np.cross(v1, v2)
        cp23 = np.cross(v2, v3)
        x1_vec = np.cross(cp12, cp23)
        x1 = np.sqrt(np.dot(x1_vec, x1_vec))
        x2 = np.dot(cp12, cp23)
        return np.arctan2(x1, x2)

    def getMeanCurvature(self):
        '''Get the mean curvature

        Mean curvature R for a polyhedron is determined from the edge lengths
        L_i and dihedral angles \phi_i and is given by $\sum_i (1/2) L_i (\pi -
        \phi_i) / (4 \pi)$
        \returns R
        '''
        R = 0.0
        # check each pair of faces i,j such that i < j
        nfacets = self.nfacets
        for i in range(nfacets - 1):
            for j in range(i + 1, nfacets):
                # get the length of the shared edge, if there is one
                k = self.getSharedEdge(i, j)  # index of first vertex
                if k is not None:
                    nextk = k + 1  # index of second vertex
                    if nextk == self.nverts[i]:
                        nextk = 0
                    # get point indices corresponding to vertex indices
                    p0 = self.facets[i, k]
                    p1 = self.facets[i, nextk]
                    v0 = self.points[p0]
                    v1 = self.points[p1]
                    r = v1 - v0
                    Li = np.sqrt(np.dot(r, r))
                    # get the dihedral angle
                    phi = self.getDihedral(i, j)
                    R += Li * (np.pi - phi)
        R /= 8 * np.pi
        return R

    def getAsphericity(self):
        '''Get asphericity

        Asphericity alpha is defined as RS/3V where R is the mean curvature, S
        is surface area, V is volume
        \returns alpha
        '''
        R = self.getMeanCurvature()
        S = self.getArea()
        V = self.getVolume()
        return R * S / (3 * V)

    def getQ(self):
        '''Get isoperimetric quotient

        Isoperimetric quotient is a unitless measure of sphericity defined as Q
        = 36 \pi \frac{V^2}{S^3}
        \returns isoperimetric quotient
        '''
        V = self.getVolume()
        S = self.getArea()
        Q = np.pi * 36 * V * V / (S * S * S)
        return Q

    def getTau(self):
        '''Get tau = 4\pi R^2 / S (ref: Naumann and Leland)

        Tau is a measure of asphericity that appears to be relevant to the
        third and fourth virial coefficients

        \returns tau
        '''
        R = self.getMeanCurvature()
        S = self.getArea()
        return 4. * np.pi * R * R / S


# Inherits from class Polyhedron
class ConvexPolyhedron(Polyhedron):
    '''Create a ConvexPolyhedron object from a list of points.

    Store and compute data associated with a convex polyhedron, calculated as
    the convex hull of a set of input points.  ConvexPolyhedron objects are a
    modification to the scipy.spatial.ConvexHull object with data in a form more
    useful to operations involving polyhedra.

    \note euclid.FreudShape.ConvexPolyhedron requires scipy.spatil.ConvexHull
    (as of scipy 0.12.0).

     \param points Nx3 list of vertices from which to construct the convex hull
     \param mergeFacets automatically try to merge coplanar simplicial facets
        (default True)
    '''

    def __init__(self, points, mergeFacets=True):
        if ConvexHull is None:
            logger.error(
                'Cannot initialize ConvexPolyhedron because scipy.spatial.ConvexHull is not available.')

        simplicial = ConvexHull(points)
        facets = simplicial.simplices
        neighbors = simplicial.neighbors
        equations = simplicial.equations

        points = simplicial.points
        pshape = points.shape
        if (len(pshape) != 2) or pshape[1] != 3:
            raise ValueError("points parameter must be an Nx3 array of points")

        nfacets = len(facets)
        ndim = pshape[1]
        nverts = [ndim] * nfacets

        # Call base class constructor
        Polyhedron.__init__(self,
                            points,
                            nverts,
                            facets,
                            neighbors,
                            equations,
                            facets,
                            neighbors,
                            equations)

        # mergeFacets does not merge all coplanar facets when there are a lot of neighboring coplanar facets,
        # but repeated calls will do the job.
        # If performance is ever an issue, this should really all be replaced
        # with our own qhull wrapper...
        if mergeFacets:
            old_nfacets = 0
            new_nfacets = self.nfacets
            while new_nfacets != old_nfacets:
                self.mergeFacets()
                old_nfacets = new_nfacets
                new_nfacets = self.nfacets
        for i in range(self.nfacets):
            self.facets[i, 0:self.nverts[i]] = self.rhFace(i)
        for i in range(self.nfacets):
            self.neighbors[i, 0:self.nverts[i]] = self.rhNeighbor(i)
        self.originalpoints = np.array(self.points)
        self.originalequations = np.array(self.equations)


# Store and compute data associated with a convex spheropolyhedron, calculated
# as the convex hull of a set of input points plus a rounding radius.
# Inherits from ConvexPolyhedron but replaces several methods.
class ConvexSpheropolyhedron(ConvexPolyhedron):
    '''Create a ConvexPolyhedron object from a list of points and a rounding radius.

     \param points Nx3 list of vertices from which to construct the convex hull
     \param R rounding radius by which to extend the polyhedron boundary
     \param mergeFacets automatically try to merge coplanar simplicial facets
        of the polyhedron (default True)
     '''

    def __init__(self, points, R=0.0, mergeFacets=True):
        ConvexPolyhedron.__init__(self, points, mergeFacets)
        self.R = float(R)
        self.originalR = self.R

    def getArea(self):
        '''Find surface area of spheropolyhedron.'''
        R = self.R
        facet_list = range(self.nfacets)
        Aface = 0.0
        Acyl = 0.0
        Asphere = 4. * np.pi * R * R
        # for each face
        for i in facet_list:
            face = self.facets[i]
            n = self.equations[i, 0:3]
            Ni = self.nverts[i]  # number of points on the facet)
            # for each triangle on the face, sum up the area
            for j in range(1, Ni - 1):
                r1 = self.points[face[j]] - self.points[face[0]]
                r2 = self.points[face[j + 1]] - self.points[face[0]]
                cp = np.cross(r1, r2)
                Aface += abs(np.dot(cp, n)) / 2.0
            # for each edge on the face get length and dihedral to calculate
            # cylinder contribution
            for j in range(0, Ni):
                p1 = self.points[face[j]]
                if j >= Ni - 1:
                    p2 = self.points[face[0]]
                else:
                    p2 = self.points[face[j + 1]]
                edge = p2 - p1
                edge_length = np.sqrt(np.dot(edge, edge))
                angle = np.pi - self.getDihedral(i, self.neighbors[i, j])
                # divide partial cylinder area by 2 because edges are
                # double-counted
                Acyl += edge_length * angle * R / 2.0
        return Aface + Acyl + Asphere

    def getVolume(self):
        '''Find the volume of the spheropolyhedron'''
        R = self.R
        Vpoly = 0.0
        Vcyl = 0.0
        Vsphere = 4. * np.pi * R * R * R / 3.
        # for each face, calculate area -> volume, and accumulate
        for i in range(len(self.facets)):
            face = self.facets[i]
            Ni = self.nverts[i]
            d = -1 * self.equations[i, 3]  # distance from centroid
            A = Polyhedron.getArea(self, i)
            # add volume of polyhedral wedge for the interior polyhedron
            Vpoly += d * A / 3.0
            # add volume for the polygonal plate due to R
            Vpoly += R * A
            # for each edge on the face get length and dihedral to calculate
            # cylinder contribution
            for j in range(0, Ni):
                p1 = self.points[face[j]]
                if j >= Ni - 1:
                    p2 = self.points[face[0]]
                else:
                    p2 = self.points[face[j + 1]]
                edge = p2 - p1
                edge_length = np.sqrt(np.dot(edge, edge))
                angle = np.pi - self.getDihedral(i, self.neighbors[i, j])
                # divide partial cylinder volume by 2 because edges are
                # double-counted
                Vcyl += edge_length * angle * R * R / 4.0
        return Vpoly + Vcyl + Vsphere

    def getCircumsphereRadius(self, original=False):
        '''Get circumsphere radius

        get R2[i] = dot(points[i], points[i]) by getting the diagonal (i=j) of
        the array of dot products dot(points[i], points[j])

        \param original True means to retrieve the original points before any
        subsequent rescaling (default False)
        '''
        if original:
            points = self.originalpoints
        else:
            points = self.points
        R2 = np.diag(np.dot(points, points.T))
        d = np.sqrt(R2.max())
        if original:
            d += self.originalR
        else:
            d += self.R
        return d

    def getInsphereRadius(self, original=False):
        '''Get insphere radius

        \param original True means to retrieve the original points before any
        subsequent rescaling (default False)
        '''
        if original:
            equations = self.originalequations
        else:
            equations = self.equations
        facetDistances = equations[:, 3]
        d = abs(facetDistances.max())
        if original:
            d += self.originalR
        else:
            d += self.R
        return d

    def setCircumsphereRadius(self, radius):
        '''Scale spheropolyhedron to fit a given circumsphere radius.

        \param radius new circumsphere radius
            Scales points and R. To scale just the underlying polyhedron, use
            the base class method.
        '''
        # use unscaled data from original to avoid accumulated errors
        oradius = ConvexSpheropolyhedron.getCircumsphereRadius(
            self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:, 3] = self.originalequations[:, 3] * scale_factor
        self.R = self.originalR * scale_factor
        self.simplicial.points = self.simplicial.originalpoints * scale_factor
        self.simplicial.equations[:,
                                  3] = self.simplicial.originalequations[:,
                                                                         3] * scale_factor

    def setInsphereRadius(self, radius):
        '''Scale polyhedron to fit a given circumsphere radius

        \param radius new insphere radius
        '''
        oradius = ConvexSpheropolyhedron.getInsphereRadius(self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:, 3] = self.originalequations[:, 3] * scale_factor
        self.R = self.originalR * scale_factor
        self.simplicial.points = self.simplicial.originalpoints * scale_factor
        self.simplicial.equations[:,
                                  3] = self.simplicial.originalequations[:,
                                                                         3] * scale_factor

    def isInside(self, point):
        '''Test if a point is inside the shape

        \param point 3D coordinates of test point
        '''
        v = np.asarray(point)
        for i in range(self.nfacets):
            d = np.dot(v, self.equations[i, 0:3])
            if d + self.equations[i, 3] > self.R:
                return False
        return True

    def getMeanCurvature(self):
        raise RuntimeError("Not implemented")

    def getAsphericity(self):
        raise RuntimeError("Not implemented")
