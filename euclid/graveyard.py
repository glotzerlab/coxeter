import numpy as np
from scipy.spatial import ConvexHull


# 8/30 comments: probably duplicated in euclid.polyhedron, change
# be deleted based on further testing on the class method isinside
# Author: BVS
# Find the indices of points which lie within a convex sphero polyhedron
# hull: convex hull of the core polyhedron
# r: rounding radius of the spheropolyhedron
# points: points to check against (Nx3)
# rp: radius of the points (sphere check)
# Returns a list of row indices of points that are inside the shape


def find_idx_hull_overlap(hull, r, points, rp=0):
    # Speed things up by only keeping points that
    # overlap with the bounding convex polyhedra
    suspect_idx = find_in_idx(points, hull, rp=rp + r)
    orig_idx = np.arange(points.shape[0])[suspect_idx]
    suspects = points[suspect_idx, :]

    # Check the base shape
    idx = orig_idx[find_in_idx(suspects, hull, rp=rp)]
    if r > 0:
        # Check the face segments
        face_hulls = extrude_facets(hull, r)
        for fhull in face_hulls:
            idx = np.append(idx, orig_idx[find_in_idx(
                suspects, fhull, rp=rp)], axis=0)

        # Check the cylinders around the ridges
        for i in range(len(hull.simplices)):
            verts = hull.points[hull.simplices[i]]
            lines = np.array([[verts[0], verts[1]],
                              [verts[1], verts[2]],
                              [verts[2], verts[0]]])
            planes = np.array([line[1] - line[0] for line in lines])
            planes = planes / np.linalg.norm(planes, axis=1)

            otherplanes = -np.copy(planes)
            planes = np.append(
                planes,
                -((planes * lines[:, 0]).sum(axis=1)).reshape((-1, 1)), axis=1)
            otherplanes = np.append(
                otherplanes,
                -((otherplanes * lines[:, 1]).sum(axis=1)).reshape((-1, 1)),
                axis=1)

            for line, plane, oplane in zip(lines, planes, otherplanes):
                cond = (
                    (line_point_distance(line, suspects) <= r + rp) *
                    (
                        (np.dot(
                            suspects[:, 0:3],
                            np.reshape(plane[0:3], (3, 1))) + plane[3] >= rp
                         ).flatten()
                     ) *
                    (
                        (np.dot(
                            suspects[:, 0:3],
                            np.reshape(oplane[0:3], (3, 1))) + oplane[3] >= rp
                         ).flatten()
                     )
                    )
                idx = np.append(idx, orig_idx[cond])

        # Check the vertex caps
        for vertex in hull.vertices:
            idx = np.append(
                idx,
                orig_idx[
                    (np.linalg.norm(hull.points[vertex] - suspects, axis=1) <=
                     r + rp).flatten()
                ], axis=0)

    return idx

# 8/30 comments: helper function for find_idx_hull_overlap
# Creates a list of convex hulls that are the facets of the given hull
# extruded outwards by the given r
# Author: BVS


def extrude_facets(hull, r):
    hull_list = []
    for i in range(len(hull.equations)):
        eq = hull.equations[i]
        verts = hull.points[hull.simplices[i]]
        exverts = verts + r * eq[0:3] / np.linalg.norm(eq[0:3])
        hull_list.append(ConvexHull(np.append(verts, exverts, axis=0)))
    return hull_list

# 8/30 comments: helper function for find_idx_hull_overlap
# line should be given as two points
# Author: BVS


def line_point_distance(line, points):
    return np.linalg.norm(np.cross(
        points - line[0],
        points - line[1]), axis=1) / np.linalg.norm(line[1] - line[0])

# 8/30 comments: helper function for find_idx_hull_overlap
# Returns the indices of points that are inside the convex hull
# rp is the radius of the point, for sphere checking
# Author: BVS


def find_in_idx(points, hull, rp=0):
    in_idx = np.arange(points.shape[0])
    for plane in hull.equations:
        truth = (np.dot(points[in_idx, 0:3], np.reshape(
            plane[0:3], (3, 1))) + plane[3] <= rp).flatten()
        in_idx = in_idx[truth]
    return in_idx

# 8/30 comments: potentially useful code for slicing shapes
# Returns the verts of a shape that is input verts sliced by the input plane
# Verts on the positive side of the plane are retained and negative are
# discarded


def shape_slice(verts, plane):
    # get all the lines of the shape
    lines = []
    hull = ConvexHull(verts)
    for simplex in hull.simplices:
        for point1 in simplex:
            for point2 in simplex:
                # if the points straddle the plane
                if(np.dot(plane[1], hull.points[point1] - plane[0]) *
                   np.dot(plane[1], hull.points[point2] - plane[0]) < 0):
                    lines.append([hull.points[point1], hull.points[point2]])
    intersections = []
    for line in lines:
        intersections.append(line_plane_intersection(line, plane))
    # add any points that are already on the plane
    for vert in verts:
        if np.abs(np.dot(plane[1], vert - plane[0])) < 1e-8:
            intersections.append(vert)
    intersections = np.asarray(intersections)
    # remove nan's if some lines were parallel
    intersections = intersections[~np.any(
        (np.isnan(intersections)), axis=1), :]
    # discard old points that are below the plane
    verts = verts[np.dot(verts, plane[1]) > 0, :]
    # append the new intersections
    verts = np.append(verts, intersections, axis=0)

    # remove duplicate points
    inds = np.ones(verts.shape[0]).astype(bool)
    for i in range(verts.shape[0]):
        if inds[i]:
            for j in range(i + 1, verts.shape[0]):
                if np.linalg.norm(verts[i] - verts[j]) < 1e-8:
                    inds[j] = False

    return verts[inds, :]

# 8/30 comments: helper function for shape_slice
# Finds the intersection point of a plane and a line
# line given as [p0,p], plane as [n0,n]


def line_plane_intersection(line, plane):
    line = np.asarray(line)
    plane = np.asarray(plane)
    t = np.dot(plane[1], (plane[0] - line[0])) / \
        np.dot(plane[1], (line[1] - line[0]))
    return line[0] + t * (line[1] - line[0])
