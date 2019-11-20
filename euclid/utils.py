import numpy as np
from scipy.spatial import ConvexHull
from .polyhedron import ConvexPolyhedron, ConvexSpheropolyhedron
import logging

logger = logging.getLogger(__name__)
# Returns rescaled vertices and a rounding radius that
# can be used to create a spheropolyhedron consistent
# with the s parameter [0,1) and target volume.
# s = 0.0 is a polyhedron, s = 1.0 is a sphere


def _normalize(vector):
    """Returns a normalized version of a numpy vector."""
    return vector/np.sqrt(np.dot(vector, vector))


def _polygonNormal(vertices):
    """Returns the unit normal vector of a planar set of vertices."""
    return -_normalize(np.cross(vertices[1] - vertices[0],
                                vertices[0] - vertices[-1]))


def area(vertices, factor=1.):
    """Computes the signed area of a polygon in 2 or 3D.

    Args:
        vertices (list): (x, y) or (x, y, z) coordinates for each vertex
        factor (float): Factor to scale the resulting area by

    """
    vertices = np.asarray(vertices)
    shifted = np.roll(vertices, -1, axis=0)

    crosses = np.sum(np.cross(vertices, shifted), axis=0)

    return np.abs(np.dot(crosses, _polygonNormal(vertices))*factor/2)


def sphero_shape(verts, s, target_vol=1):

    # Find the rounding amout by finding the equivilent radius of a sphere
    # with equal volume
    poly = ConvexPolyhedron(verts) if not isinstance(
        verts, ConvexPolyhedron) else verts
    initial_vol = poly.getVolume()
    r_eq = np.power(3 * initial_vol / (4 * np.pi), 1 / 3)

    sphero_rounding = r_eq * s / (1 - s)

    vol = ConvexSpheropolyhedron(poly.points, R=sphero_rounding).getVolume()

    factor = target_vol / vol

    final_shape = poly.points * np.power(factor, 1 / 3)
    final_rounding = sphero_rounding * np.power(factor, 1 / 3)

    return ConvexSpheropolyhedron(final_shape, R=final_rounding)


def _fanTriangles(vertices, faces=None):
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
        for face in faces:
            for tri in ((vertices[face[0]], vertices[i], vertices[j])
                        for (i, j) in zip(face[1:], face[2:])):
                yield tri


def massProperties(vertices, faces=None, factor=1.):
    """Compute the mass, center of mass, and inertia tensor of a polygon or
    polyhedron

    .. warning::
        All faces should be specified in right-handed order.

    The computation for the 3D case follows "Polyhedral Mass
    Properties (Revisited) by David Eberly, available at:

    http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf

    Args:
        vertices (list):
            List of (x, y) or (x, y, z) coordinates in 2D or 3D, respectively
        faces (list):
            List of vertex indices for 3D polyhedra, or None for 2D. Faces
            should be in right-hand order.
        factor (float):
            Factor to scale the resulting results by

    Returns:
        (mass, center of mass, moment of inertia tensor in (xx, xy, xz, yy, yz,
        zz) order) specified by the given list of vertices and faces. Note that
        the faces must be listed in a consistent order so that normals are all
        pointing in the correct direction from the face. If given a list of 2D
        vertices, return the same but for the 2D polygon specified by the
        vertices.
    """
    vertices = np.array(vertices, dtype=np.float64)

    # Specially handle 2D
    if len(vertices[0]) == 2:
        # First, calculate the center of mass and center the vertices
        shifted = list(vertices[1:]) + [vertices[0]]
        a_s = [(x1 * y2 - x2 * y1) for ((x1, y1), (x2, y2))
               in zip(vertices, shifted)]
        triangleCOMs = [(v0 + v1) / 3 for (v0, v1) in zip(vertices, shifted)]
        COM = np.sum([a * com for (a, com) in zip(a_s, triangleCOMs)],
                     axis=0) / np.sum(a_s)
        vertices -= COM

        shifted = list(vertices[1:]) + [vertices[0]]

        def f(x1, x2):
            return x1 * x1 + x1 * x2 + x2 * x2
        Ixyfs = [(f(y1, y2), f(x1, x2)) for ((x1, y1), (x2, y2))
                 in zip(vertices, shifted)]

        Ix = sum(I * a for ((I, _), a) in zip(Ixyfs, a_s)) / 12.
        Iy = sum(I * a for ((_, I), a) in zip(Ixyfs, a_s)) / 12.

        moi = np.array([Ix, 0, 0, Iy, 0, Ix + Iy])

        return area(vertices) * factor, COM, factor * moi

    # multiplicative factors
    factors = 1. / np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120])

    # order: 1, x, y, z, x^2, y^2, z^2, xy, yz, zx
    intg = np.zeros(10)

    for (v0, v1, v2) in _fanTriangles(vertices, faces):
        # (xi, yi, zi) = vi
        abc1 = v1 - v0
        abc2 = v2 - v0
        d = np.cross(abc1, abc2)

        temp0 = v0 + v1
        f1 = temp0 + v2
        temp1 = v0 * v0
        temp2 = temp1 + v1 * temp0
        f2 = temp2 + v2 * f1
        f3 = v0 * temp1 + v1 * temp2 + v2 * f2
        g0 = f2 + v0 * (f1 + v0)
        g1 = f2 + v1 * (f1 + v1)
        g2 = f2 + v2 * (f1 + v2)

        intg[0] += d[0] * f1[0]
        intg[1:4] += d * f2
        intg[4:7] += d * f3
        intg[7] += d[0] * (v0[1] * g0[0] + v1[1] * g1[0] + v2[1] * g2[0])
        intg[8] += d[1] * (v0[2] * g0[1] + v1[2] * g1[1] + v2[2] * g2[1])
        intg[9] += d[2] * (v0[0] * g0[2] + v1[0] * g1[2] + v2[0] * g2[2])

    intg *= factors

    mass = intg[0]
    com = intg[1:4] / mass

    moment = np.zeros(6)

    moment[0] = intg[5] + intg[6] - mass * np.sum(com[1:]**2)
    moment[1] = -(intg[7] - mass * com[0] * com[1])
    moment[2] = -(intg[9] - mass * com[0] * com[2])
    moment[3] = intg[4] + intg[6] - mass * np.sum(com[[0, 2]]**2)
    moment[4] = -(intg[8] - mass * com[1] * com[2])
    moment[5] = intg[4] + intg[5] - mass * np.sum(com[:2]**2)

    return mass * factor, com, moment * factor


# Contributed by Erin Teich
def rtoConvexHull(vertices):
    """Compute the convex hull of a set of points, just like
    :class:scipy.spatial.ConvexHull`.  Ensure that the given simplices are in
    right-handed order."""
    hull = ConvexHull(vertices)
    expanded = hull.points[hull.simplices]

    crossProduct = np.cross(expanded[:, 1, :] -
                            expanded[:, 0, :], expanded[:, 2, :] -
                            expanded[:, 0, :])
    faceNormals = crossProduct / \
        np.sqrt(np.sum(crossProduct**2, axis=-1))[:, np.newaxis]
    flipped = np.sum(faceNormals * np.mean(expanded, axis=1), axis=-1) < 0
    temp = hull.simplices[flipped, 0].copy()
    hull.simplices[flipped, 0] = hull.simplices[flipped, 1]
    hull.simplices[flipped, 1] = temp

    return hull
