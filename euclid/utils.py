from . import np
from . import ConvexHull, Delaunay
from collections import Counter, defaultdict, deque, namedtuple
from itertools import chain

thresh = 1e-5

### Functions contributed by Carl Simon Adorf

def getEdges(normal,vertices) :
  # take the normal to be the z axis
  n = np.array(normal)
  n /= np.sqrt(np.dot(n,n))

  _edges = []

  # for each of the vertices
  for v in vertices :
    for w in vertices :
      # pick the first two vertices in the list and define them as an x axis
      diff = w-v
      dmag = np.sqrt(np.dot(diff,diff))
      if dmag > thresh :
        diff /= dmag
        # compute the y axis by taking the cross-product with the normal
        yhat = np.cross(n,diff)

        # loop over all of the vertices and check whether they are in the upper
        # half-plane
        okay = True
        for x in vertices :
          ddiff = x-v
          ddmag = np.sqrt(np.dot(ddiff,ddiff))
          if ddmag > thresh :
            okay = okay and (np.dot(yhat,ddiff)/ddmag > -thresh )
        # if they all do, then append this edge
        if okay :
          _edges.append((v,w))

  #  sort the edges
  __edges = [_edges[0]]
  _edges.remove(_edges[0])
  while  len(_edges) > 0 :
    for e in range(len(_edges)) :
      if np.dot(_edges[e][0]-__edges[-1][1],_edges[e][0]-__edges[-1][1])\
          < thresh :
        break
    __edges.append(_edges[e])
    del _edges[e]

  # return the sorted list
  return __edges

def mapInertiaTensor(tetrahedron) :
  # This is based on this paper:
  # F. Tonon, J. Math. Stat., 1 p. 8-11 2004
  J = np.linalg.det(np.array(tetrahedron[1:]))
  (x2,y2,z2) = tuple(tetrahedron[1])
  (x3,y3,z3) = tuple(tetrahedron[2])
  (x4,y4,z4) = tuple(tetrahedron[3])
  Ta = J*(y2**2+y2*y3+y3**2+y2*y4+y3*y4+y4**2\
         +z2**2+z2*z3+z3**2+z2*z4+z3*z4+z4**2)
  Tb = J*(x2**2+x2*x3+x3**2+x2*x4+x3*x4+x4**2\
         +z2**2+z2*z3+z3**2+z2*z4+z3*z4+z4**2)
  Tc = J*(x2**2+x2*x3+x3**2+x2*x4+x3*x4+x4**2\
         +y2**2+y2*y3+y3**2+y2*y4+y3*y4+y4**2)
  Tap = J*(2*y2*z2+y3*z2+y4*z2+y2*z3+2*y3*z3+y4*z3+y2*z4+y3*z4+2*y4*z4)
  Tbp = J*(2*x2*z2+x3*z2+x4*z2+x2*z3+2*x3*z3+x4*z3+x2*z4+x3*z4+2*x4*z4)
  Tcp = J*(2*x2*y2+x3*y2+x4*y2+x2*y3+2*x3*y3+x4*y3+x2*y4+x3*y4+2*x4*y4)
  return np.array([[Ta,-Tbp,-Tcp],[-Tbp,Tb,-Tap],[-Tcp,-Tap,Tc]])

  # get a normalization factor
  if unit :
    scale = getVolume(a=a,b=b,c=c)**(-5.0/3)
  else :
    scale = 1.0

  # get the tetrahedral decomposition of the polyhedron
  tets = getTetrahedra(a=a,b=b,c=c)

  II = np.zeros((3,3))
  # for each of the tetrahedra, compute the moment of inertia
  for tet in tets :
    II += mapInertiaTensor(tet)

  # return the sum all of the moments
  return II*scale

  # get the tetrahedral decomposition of the polyhedron
  tets = getTetrahedra(a=a,b=b,c=c)

  total = 0
  # compute the volume of each of the tetrahedra
  for tet in tets :
    m = np.array(tet[1:])
    total += np.linalg.det(m)/6

  # return the sum
  return total

### end CSA

### Functions contributed by Bryan van Saders

# Plots a frame of lines based on the simplicies of a set of verts. ax is a
# matplotlib axis object, and color should be a string or vector interpretable by matplotlib
# as a plot color for the lines
def plot_skeleton(verts, color, ax):
    hull = ConvexHull(verts)
    for simplex in hull.simplices:
        ax.plot(*zip(hull.points[simplex[0]],hull.points[simplex[1]]), c=color)
        ax.plot(*zip(hull.points[simplex[1]],hull.points[simplex[2]]), c=color)
        ax.plot(*zip(hull.points[simplex[2]],hull.points[simplex[0]]), c=color)
    return None

# Finds the intersection point of a plane and a line
# line given as [p0,p], plane as [n0,n]
def line_plane_intersection(line, plane):
    line = np.asarray(line)
    plane = np.asarray(plane)
    t = np.dot(plane[1],(plane[0]-line[0]))/np.dot(plane[1],(line[1]-line[0]))
    return line[0] + t*(line[1]-line[0])

# Returns the verts of a shape that is input verts sliced by the input plane
# Verts on the positive side of the plane are retained and negative are discarded
def shape_slice(verts, plane):
    # get all the lines of the shape
    lines = []
    hull = ConvexHull(verts)
    for simplex in hull.simplices:
        for point1 in simplex:
            for point2 in simplex:
            # if the points straddle the plane
                if(np.dot(plane[1],hull.points[point1]-plane[0])*np.dot(plane[1],hull.points[point2]-plane[0])<0):
                    lines.append([hull.points[point1],hull.points[point2]])
    intersections = []
    for line in lines:
        intersections.append(line_plane_intersection(line,plane))
    # add any points that are already on the plane
    for vert in verts:
        if np.abs(np.dot(plane[1],vert-plane[0]))<1e-8:
            intersections.append(vert)
    intersections = np.asarray(intersections)
    # remove nan's if some lines were parallel
    intersections = intersections[~np.any((np.isnan(intersections)), axis=1),:]
    # discard old points that are below the plane
    verts = verts[np.dot(verts, plane[1])>0,:]
    # append the new intersections
    verts = np.append(verts, intersections, axis=0)

    # remove duplicate points
    inds = np.ones(verts.shape[0]).astype(bool)
    for i in range(verts.shape[0]):
        if inds[i]:
            for j in range(i+1,verts.shape[0]):
                if np.linalg.norm(verts[i]-verts[j])<1e-8:
                    inds[j] = False

    return verts[inds,:]

def simplical_content(verts):
    if len(verts)==4:
        edges=verts-verts[0,:]
        return np.abs(np.dot(edges[1],np.cross(edges[2],edges[3])))/6

    if len(verts)==3:
        dist=pdist(verts,'euclidean')
        s=(dist[0]+dist[1]+dist[2])/2
        return np.sqrt(s*(s-dist[0])*(s-dist[1])*(s-dist[2]))

def convex_content(verts):
    trian = Delaunay(verts)
    total_vol = 0
    for simplex in trian.simplices:
        simp_verts = verts[simplex]
        total_vol += simplical_content(simp_verts)
    return total_vol

# Returns a list of tuples of cap angles and lengths (angle, length)
def cylinder_caps(verts):
    cap_list = []
    hull = ConvexHull(verts)
    for i in range(hull.simplices.shape[0]):
        neighbor_list = np.argwhere(hull.neighbors==i)
        for j in neighbor_list[:,0]:
            if j>i:
                eq1 = hull.equations[i][0:3]
                eq2 = hull.equations[j][0:3]
                # Angle of the cylindrical edge
                angle = np.arccos(np.dot(eq1, eq2)/(np.linalg.norm(eq1)*np.linalg.norm(eq2)))
                # Shared points of the neighboring simplices
                shared = np.intersect1d(hull.simplices[i], hull.simplices[j])
                # Length of the cylindrical edge
                length = np.linalg.norm(hull.points[shared[0]]-hull.points[shared[1]])
                if angle<2*np.pi:
                    cap_list.append((angle, length))
    return cap_list

# Returns the areas of all the simplices of a convex hull
def simplex_area(verts):
    areas = []
    hull = ConvexHull(verts)
    for simplex in hull.simplices:
        areas.append(simplical_content(hull.points[simplex]))

    return areas

# Must be convex. return_area flag triggers return of (volume, area)
# tuple
def spheropolyhedra_volume(verts, R=1.0, return_area=False):
    assert(R>=0)
    hull = ConvexHull(verts)
    # Base volume
    convex_vol = hull.volume
    # Rectangular section volume
    rect_vol = hull.area*R
    # Cylinder cap volume
    cyl_vol = 0
    cyl_area = 0
    for (ang, length) in cylinder_caps(verts):
        cyl_vol += R**2*length*np.pi*(ang/(2*np.pi))
        cyl_area += 2*np.pi*R*length*(ang/(2*np.pi))

    assert((convex_vol>=0)*(rect_vol>=0)*(cyl_vol>=0))
    vol = convex_vol + rect_vol + cyl_vol + (4*np.pi*R**3)/3
    area = hull.area + cyl_area + 4*np.pi*R**2
    if return_area:
        return (vol, area)
    else:
        return vol


# Returns rescaled vertices and a rounding radius that
# can be used to create a spheropolyhedron consistent
# with the s parameter [0,1) and target volume.
# s = 0.0 is a polyhedron, s = 1.0 is a sphere
def sphero_shape(verts, s, target_vol=1):

    # Find the rounding amout by finding the equivilent radius of a sphere
    # with equal volume
    initial_vol = ConvexHull(verts).volume
    r_eq = np.power(3*initial_vol/(4*np.pi),1/3)

    sphero_rounding = r_eq*s/(1-s)

    vol = spheropolyhedra_volume(verts, R=sphero_rounding)

    factor = target_vol/vol

    final_shape = verts*np.power(factor, 1/3)
    final_rounding = sphero_rounding*np.power(factor, 1/3)

    return final_shape, final_rounding

### end BVS

### Functions contributed by Matthew Spellings

def _normalize(vector):
    """Returns a normalized version of a numpy vector."""
    return vector/np.sqrt(np.dot(vector, vector));

def _polygonNormal(vertices):
    """Returns the unit normal vector of a planar set of vertices."""
    return -_normalize(np.cross(vertices[1] - vertices[0], vertices[0] - vertices[-1]));

def area(vertices, factor=1.):
    """Computes the signed area of a polygon in 2 or 3D.

    Args:
        vertices (list): (x, y) or (x, y, z) coordinates for each vertex
        factor (float): Factor to scale the resulting area by

    """
    vertices = np.asarray(vertices);
    shifted = np.roll(vertices, -1, axis=0);

    crosses = np.sum(np.cross(vertices, shifted), axis=0);

    return np.abs(np.dot(crosses, _polygonNormal(vertices))*factor/2);

def spheroArea(vertices, radius=1., factor=1.):
    """Computes the area of a spheropolygon.

    Args:
        vertices (list): List of (x, y) coordinates, in right-handed (counterclockwise) order
        radius (float): Rounding radius of the disk to expand the polygon by
        factor (float): Factor to scale the resulting area by

    """
    vertices = list(vertices);

    if not len(vertices) or len(vertices) == 1:
        return np.pi*radius*radius;

    # adjust for concave vertices
    adjustment = 0.;
    shifted = vertices[1:] + [vertices[0]];
    delta = [(x2 - x1, y2 - y1) for ((x1, y1), (x2, y2)) in zip(vertices, shifted)];
    lastDelta = [delta[-1]] + delta[:-1];
    thetas = [np.arctan2(y, x) for (x, y) in delta];
    dthetas = [(theta2 - theta1)%(2*np.pi) for (theta1, theta2) in
               zip([thetas[-1]] + thetas[:-1], thetas)];

    # non-rounded component of the given polygon + sphere
    polygonSkeleton = [];

    for ((x, y), dtheta, dr1, dr2) in zip(vertices, dthetas, lastDelta, delta):

        if dtheta > np.pi: # this is a concave vertex
            # subtract the rounded segment we'll add later
            theta = dtheta - np.pi;
            adjustment += radius*radius*theta/2;

            # add a different point to the skeleton
            h = radius/np.sin(theta/2);

            bisector = _negBisector(dr1, (-dr2[0], -dr2[1]));
            point = (x + bisector[0]*h, y + bisector[1]*h);
            polygonSkeleton.append(point);

        else:
            (dr1, dr2) = _normalize(dr1), _normalize(dr2);

            polygonSkeleton.append((x + dr1[1]*radius, y - dr1[0]*radius));
            polygonSkeleton.append((x, y));
            polygonSkeleton.append((x + dr2[1]*radius, y - dr2[0]*radius));

    # Contribution from rounded corners
    sphereContribution = (sum([theta % np.pi for theta in dthetas]))/2.*radius**2;

    return (area(polygonSkeleton) + sphereContribution - adjustment)*factor;

def rmax(vertices, radius=0., factor=1.):
    """Compute the maximum distance among a set of vertices

    Args:
        vertices (list): list of (x, y) or (x, y, z) coordinates
        factor (float): Factor to scale the result by

    """
    return (np.sqrt(np.max(np.sum(np.asarray(vertices)*vertices, axis=1))) + radius)*factor;

def _fanTriangles(vertices, faces=None):
    """Create triangles by fanning out from vertices. Returns a
    generator for vertex triplets. If faces is None, assume that
    vertices are planar and indicate a polygon; otherwise, use the
    face indices given in faces."""
    vertices = np.asarray(vertices);

    if faces is None:
        if len(vertices) < 3:
            return;
        for tri in ((vertices[0], verti, vertj) for (verti, vertj) in
                    zip(vertices[1:], vertices[2:])):
            yield tri;
    else:
        for face in faces:
            for tri in ((vertices[face[0]], vertices[i], vertices[j]) for (i, j) in
                        zip(face[1:], face[2:])):
                yield tri;

def massProperties(vertices, faces=None, factor=1.):
    """Compute the mass, center of mass, and inertia tensor of a polygon or polyhedron

    Args:
        vertices (list): List of (x, y) or (x, y, z) coordinates in 2D or 3D, respectively
        faces (list): List of vertex indices for 3D polyhedra, or None for 2D. Faces should be in right-hand order.
        factor (float): Factor to scale the resulting results by

    Returns (mass, center of mass, moment of inertia tensor in (xx,
    xy, xz, yy, yz, zz) order) specified by the given list of vertices
    and faces. Note that the faces must be listed in a consistent
    order so that normals are all pointing in the correct direction
    from the face. If given a list of 2D vertices, return the same but
    for the 2D polygon specified by the vertices.

    .. warning::
        All faces should be specified in right-handed order.

    The computation for the 3D case follows "Polyhedral Mass
    Properties (Revisited) by David Eberly, available at:

    http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf

    """
    vertices = np.array(vertices, dtype=np.float64);

    # Specially handle 2D
    if len(vertices[0]) == 2:
        # First, calculate the center of mass and center the vertices
        shifted = list(vertices[1:]) + [vertices[0]];
        a_s = [(x1*y2 - x2*y1) for ((x1, y1), (x2, y2))
               in zip(vertices, shifted)];
        triangleCOMs = [(v0 + v1)/3 for (v0, v1) in zip(vertices, shifted)];
        COM = np.sum([a*com for (a, com) in zip(a_s, triangleCOMs)],
                     axis=0)/np.sum(a_s);
        vertices -= COM;

        shifted = list(vertices[1:]) + [vertices[0]];
        f = lambda x1, x2: x1*x1 + x1*x2 + x2*x2;
        Ixyfs = [(f(y1, y2), f(x1, x2)) for ((x1, y1), (x2, y2))
                 in zip(vertices, shifted)];

        Ix = sum(I*a for ((I, _), a) in zip(Ixyfs, a_s))/12.;
        Iy = sum(I*a for ((_, I), a) in zip(Ixyfs, a_s))/12.;

        I = np.array([Ix, 0, 0, Iy, 0, Ix + Iy]);

        return area(vertices)*factor, COM, factor*I;

    # multiplicative factors
    factors = 1./np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120]);

    # order: 1, x, y, z, x^2, y^2, z^2, xy, yz, zx
    intg = np.zeros(10);

    for (v0, v1, v2) in _fanTriangles(vertices, faces):
        # (xi, yi, zi) = vi
        abc1 = v1 - v0;
        abc2 = v2 - v0;
        d = np.cross(abc1, abc2);

        temp0 = v0 + v1;
        f1 = temp0 + v2;
        temp1 = v0*v0;
        temp2 = temp1 + v1*temp0;
        f2 = temp2 + v2*f1;
        f3 = v0*temp1 + v1*temp2 + v2*f2;
        g0 = f2 + v0*(f1 + v0);
        g1 = f2 + v1*(f1 + v1);
        g2 = f2 + v2*(f1 + v2);

        intg[0] += d[0]*f1[0];
        intg[1:4] += d*f2;
        intg[4:7] += d*f3;
        intg[7] += d[0]*(v0[1]*g0[0] + v1[1]*g1[0] + v2[1]*g2[0]);
        intg[8] += d[1]*(v0[2]*g0[1] + v1[2]*g1[1] + v2[2]*g2[1]);
        intg[9] += d[2]*(v0[0]*g0[2] + v1[0]*g1[2] + v2[0]*g2[2]);

    intg *= factors;

    mass = intg[0];
    com = intg[1:4]/mass;

    moment = np.zeros(6);

    moment[0] = intg[5] + intg[6] - mass*np.sum(com[1:]**2);
    moment[1] = -(intg[7] - mass*com[0]*com[1]);
    moment[2] = -(intg[9] - mass*com[0]*com[2]);
    moment[3] = intg[4] + intg[6] - mass*np.sum(com[[0, 2]]**2);
    moment[4] = -(intg[8] - mass*com[1]*com[2]);
    moment[5] = intg[4] + intg[5] - mass*np.sum(com[:2]**2);

    return mass*factor, com, moment*factor;

def center(vertices, faces=None):
    """Centers shapes in 2D or 3D.

    Args:
        vertices (list): List of (x, y) or (x, y, z) coordinates in 2D or 3D, respectively
        faces (list): List of vertex indices for 3D polyhedra, or None for 2D. Faces should be in right-hand order.

    Returns a list of vertices shifted to have the center of mass of
    the given points at the origin. Shapes should be specified in
    right-handed order. If the input shape has no mass, return the
    input.

    .. warning::
        All faces should be specified in right-handed order.

    """
    (mass, COM, _) = massProperties(vertices, faces);
    if mass > 1e-6:
        return np.asarray(vertices) - COM[np.newaxis, :];
    else:
        return np.asarray(vertices);

def _negBisector(p1, p2):
    """Return the negative bisector of an angle given by points p1 and p2"""
    return -_normalize(_normalize(p1) + _normalize(p2));

def convexHull(vertices, tol=1e-6):
    """Compute the 3D convex hull of a set of vertices and merge coplanar faces.

    Args:
        vertices (list): List of (x, y, z) coordinates
        tol (float): Floating point tolerance for merging coplanar faces


    Returns an array of vertices and a list of faces (vertex
    indices) for the convex hull of the given set of vertice.

    .. note::
        This method uses scipy's quickhull wrapper and therefore requires scipy.

    """
    from scipy.spatial import cKDTree, ConvexHull;
    from scipy.sparse.csgraph import connected_components;

    hull = ConvexHull(vertices);
    # Triangles in the same face will be defined by the same linear equalities
    dist = cKDTree(hull.equations);
    trianglePairs = dist.query_pairs(tol);

    connectivity = np.zeros((len(hull.simplices), len(hull.simplices)), dtype=np.int32);

    for (i, j) in trianglePairs:
        connectivity[i, j] = connectivity[j, i] = 1;

    # connected_components returns (number of faces, cluster index for each input)
    (_, joinTarget) = connected_components(connectivity, directed=False);
    faces = defaultdict(list);
    norms = defaultdict(list);
    for (idx, target) in enumerate(joinTarget):
        faces[target].append(idx);
        norms[target] = hull.equations[idx][:3];

    # a list of sets of all vertex indices in each face
    faceVerts = [set(hull.simplices[faces[faceIndex]].flat) for faceIndex in sorted(faces)];
    # normal vector for each face
    faceNorms = [norms[faceIndex] for faceIndex in sorted(faces)];

    # polygonal faces
    polyFaces = [];
    for (norm, faceIndices) in zip(faceNorms, faceVerts):
        face = np.array(list(faceIndices), dtype=np.uint32);
        N = len(faceIndices);

        r = hull.points[face];
        rcom = np.mean(r, axis=0);

        # plane_{a, b}: basis vectors in the plane
        plane_a = r[0] - rcom;
        plane_a /= np.sqrt(np.sum(plane_a**2));
        plane_b = np.cross(norm, plane_a);

        dr = r - rcom[np.newaxis, :];

        thetas = np.arctan2(dr.dot(plane_b), dr.dot(plane_a));

        sortidx = np.argsort(thetas);

        face = face[sortidx];
        polyFaces.append(face);

    return (hull.points, polyFaces);

ConvexDecomposition = namedtuple('ConvexDecomposition', ['vertices', 'edges', 'faces'])

def convexDecomposition(vertices):
    """Decompose a convex polyhedron specified by a list of vertices into
    vertices, faces, and edges. Returns a ConvexDecomposition object.
    """
    (vertices, faces) = convexHull(vertices)
    edges = set()

    for face in faces:
        for (i, j) in zip(face, np.roll(face, -1)):
            edges.add((min(i, j), max(i, j)))

    return ConvexDecomposition(vertices, edges, faces)

def fanTriangleIndices(faces):
    """Returns the indices needed to break the faces of a polyhedron into
    a set of triangle faces"""
    for face in faces:
        for (i, j) in zip(face[1:], face[2:]):
            yield (face[0], i, j)

def fanTriangles(vertices, faces=None):
    """Create triangles by fanning out from vertices. Returns a
    generator for vertex triplets. If faces is None, assume that
    vertices are planar and indicate a polygon; otherwise, use the
    face indices given in faces."""
    vertices = np.asarray(vertices)

    if faces is None:
        if len(vertices) < 3:
            return
        for tri in ((vertices[0], verti, vertj) for (verti, vertj) in
                    zip(vertices[1:], vertices[2:])):
            yield tri
    else:
        for (i, j, k) in fanTriangleIndices(faces):
            yield (vertices[i], vertices[j], vertices[k])

### end MS

# This function is for quickly finding unique rows (respects ordering)
# of a numpy array. Cribbed from stack exchange:
# http://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

# Given a set of vertices, Delaunay triangulate and
# pair down to unique bonds between points
def minimum_bonds(verts, maxdist=None):
    tri = Delaunay(verts)
    bond_list = []
    for i in range(verts.shape[0]):
        # All the simplices with the given vertex
        simps = tri.simplices[np.where(tri.simplices==i)[0],:].flatten()
        for s in simps[simps!=i]:
            bond_list.append((i,s))

    # Sort the rows to remove flipped duplicates
    b = [np.sort(bond) for bond in bond_list]

    a = unique_rows(b)

    if maxdist is not None:
        c = []
        for bond in a:
            if np.linalg.norm(verts[bond[0]]-verts[bond[1]]) < maxdist:
                c.append(bond)
        return np.asarray(c)
    else:
        return a

# Gets the outsphere radius of a collection
# of points. Does NOT center the points on
# zero, this is assumed to already be done
def get_outsphere_radius(verts):
    verts = np.asarray(verts)
    return np.amax(np.power(np.power(verts,2).sum(axis=1),0.5))

# Contributed by Erin Teich
def rtoConvexHull(vertices):
    """Compute the convex hull of a set of points, just like scipy.spatial.ConvexHull.
    Ensure that the given simplices are in right-handed order."""
    hull = ConvexHull(vertices)
    expanded = hull.points[hull.simplices]

    crossProduct = np.cross(expanded[:, 1, :] - expanded[:, 0, :], expanded[:, 2, :] - expanded[:, 0, :])
    faceNormals = crossProduct/np.sqrt(np.sum(crossProduct**2, axis=-1))[:, np.newaxis]
    flipped = np.sum(faceNormals*np.mean(expanded, axis=1), axis=-1) < 0
    temp = hull.simplices[flipped, 0].copy()
    hull.simplices[flipped, 0] = hull.simplices[flipped, 1]
    hull.simplices[flipped, 1] = temp

    return hull

# finds the height and width of an isoceles triangle that would fit
# with tangent sides within a larger isoceles triangle that was decorated
# with circles of radius radius. point is the endpoint of one of the sides
# of the large triangle ([0,0]->point). sigma is from the WCA potential,
# it further reduces the size of the triangle
#
# Contributed by BVS
def find_triangle(point, radius, sigma=0):
    point = np.array(point)
    # Projected overlap of the circles
    t = np.linalg.norm(point) - 2*radius
    # The director that points towards the face normal of the inner triangle
    cos_term = np.arccos((2*radius + t/2)/(2*radius))
    if np.isnan(cos_term):
        cos_term = 0
    n2 = np.array([1, np.tan((np.arctan(point[1]/point[0]) - cos_term)/2)])
    n2 = n2/np.linalg.norm(n2)

    # a perpendicular vector, this is the circle tangent or side of the triangle
    s2 = 1/n2
    s2[0] = -s2[0]
    s2 = s2/np.linalg.norm(s2)
    # This is the tangent point for the diagonal side of the triangle
    s2p = (radius + (sigma/2)*2**(1/6))*n2

    # This is the tangent point for the base side of the triangle
    s1p = np.array([point[0], point[1] - radius - (sigma/2)*2**(1/6)])

    # Solve for the point of intersection of the midline and the long side
    midn = np.array([0,1])
    midp = np.array([point[0],0])

    sol2 = np.linalg.solve(np.array([s2, -midn]).T, midp-s2p)
    point2 = midp + midn*sol2[1]

    height = (s1p - point2)[1]

    # solve for the point of intersection of the base and the long side
    s1 = np.array([1,0])
    sol1 = np.linalg.solve(np.array([s2, -s1]).T, s1p-s2p)
    point1 = s1p + s1*sol1[1]

    width = 2*(s1p - point1)[0]
    # the center of mass of the triangle points
    center = np.array([point[0],point2[1]+2*height/3])

    return (height, width, center)
