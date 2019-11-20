import numpy as np


class Polygon:
    '''Compute basic properties of a polygon, stored as a list of adjacent
    vertices.

    Attributes:
        vertices nx2 numpy array of adjacent vertices
        n number of vertices in the polygon
        triangles cached numpy array of constituent triangles
    '''

    def __init__(self, verts):
        """Initialize a polygon with a counterclockwise list of 2D
        points and checks that they are ordered counter-clockwise"""
        self.vertices = np.array(verts, dtype=np.float32)

        self.rmax = np.sqrt(np.max(np.sum(self.vertices**2, axis=-1)))

        if len(self.vertices) < 3:
            raise TypeError("a polygon must have at least 3 vertices")
        if len(self.vertices[1]) != 2:
            raise TypeError("positions must be an Nx2 array")
        self.n = len(self.vertices)

        # This actually checks that the majority of the polygon is
        # listed in counter-clockwise order, but seems like it should
        # be sufficient for common use cases. Non-simple polygons can
        # still sneak in clockwise vertices.
        if self.area() < 0:
            raise RuntimeError(
                "Polygon was given with some clockwise vertices, "
                "but it requires that vertices be listed in "
                "counter-clockwise order")

    def area(self):
        """Calculate and return the signed area of the polygon with
        counterclockwise shapes having positive area"""
        shifted = np.roll(self.vertices, -1, axis=0)

        # areas is twice the signed area of each triangle in the shape
        areas = self.vertices[:, 0] * shifted[:, 1] - \
            shifted[:, 0] * self.vertices[:, 1]

        return np.sum(areas) / 2

    def center(self):
        """Center this polygon around (0, 0)"""
        self.vertices -= np.mean(self.vertices, axis=0)

    def getRounded(self, radius=1.0, granularity=5):
        """Approximate a spheropolygon by adding rounding to the
        corners. Returns a new Polygon object."""
        # Make 3D unit vectors drs from each vertex i to its neighbor i+1
        drs = np.roll(self.vertices, -1, axis=0) - self.vertices
        drs /= np.sqrt(np.sum(drs * drs, axis=1))[:, np.newaxis]
        drs = np.hstack([drs, np.zeros((drs.shape[0], 1))])

        # relStarts and relEnds are the offsets relative to the first and
        # second point of each line segment in the polygon.
        rvec = np.array([[0, 0, -1]]) * radius
        relStarts = np.cross(rvec, drs)[:, :2]
        relEnds = np.cross(rvec, drs)[:, :2]

        # absStarts and absEnds are the beginning and end points for each
        # straight line segment.
        absStarts = self.vertices + relStarts
        absEnds = np.roll(self.vertices, -1, axis=0) + relEnds

        relStarts = np.roll(relStarts, -1, axis=0)

        # We will join each of these segments by a round cap; this will be
        # done by tracing an arc with the given radius, centered at each
        # vertex from an end of a line segment to a beginning of the next
        theta1s = np.arctan2(relEnds[:, 1], relEnds[:, 0])
        theta2s = np.arctan2(relStarts[:, 1], relStarts[:, 0])
        dthetas = (theta2s - theta1s) % (2 * np.pi)

        # thetas are the angles at which we'll place points for each
        # vertex; curves are the points on the approximate curves on the
        # corners.
        thetas = np.zeros((self.vertices.shape[0], granularity))
        for i, (theta1, dtheta) in enumerate(zip(theta1s, dthetas)):
            thetas[i] = theta1 + \
                np.linspace(0, dtheta, 2 + granularity)[1:-1]
        curves = radius * \
            np.vstack([np.cos(thetas).flat, np.sin(thetas).flat]).T
        curves = curves.reshape((-1, granularity, 2))
        curves += np.roll(self.vertices, -1, axis=0)[:, np.newaxis, :]

        # Now interleave the pieces
        result = []
        for (end, curve, start, vert, dtheta) in zip(
                absEnds, curves, np.roll(absStarts, -1, axis=0),
                np.roll(self.vertices, -1, axis=0), dthetas):
            # Don't round a vertex if it is degenerate
            skip = dtheta < 1e-6 or np.abs(2 * np.pi - dtheta) < 1e-6

            # convex case: add the end of the last straight line segment, the
            # curved edge, then the start of the next straight line segment.
            if dtheta <= np.pi and not skip:
                result.append(end)
                result.append(curve)
                result.append(start)
            # concave case: don't use the curved region, just find the
            # intersection and add that point.
            elif not skip:
                l = radius / np.cos(dtheta / 2)  # noqa: E741
                p = 2 * vert - start - end
                p /= np.sqrt(np.dot(p, p))
                result.append(vert + p * l)

        result = np.vstack(result)

        return Polygon(result)

    @property
    def triangles(self):
        """A cached property of an Ntx3x2 numpy array of points, where
        Nt is the number of triangles in this polygon."""
        try:
            return self._triangles
        except AttributeError:
            self._triangles = self._triangulation()
        return self._triangles

    @property
    def normalizedTriangles(self):
        """A cached property of the same shape as triangles, but
        normalized such that all coordinates are bounded on [0, 1]."""
        try:
            return self._normalizedTriangles
        except AttributeError:
            self._normalizedTriangles = self._triangles.copy()
            self._normalizedTriangles -= np.min(self._triangles)
            self._normalizedTriangles /= np.max(self._normalizedTriangles)
        return self._normalizedTriangles

    def _triangulation(self):
        """Return a numpy array of triangles with shape (Nt, 3, 2) for
        the 3 2D points of Nt triangles."""

        if self.n <= 3:
            return [tuple(self.vertices)]

        result = []
        remaining = self.vertices

        # step around the shape and grab ears until only 4 vertices are left
        while len(remaining) > 4:
            signs = []
            for vert in (remaining[-1], remaining[1]):
                arms1 = remaining[2:-2] - vert
                arms2 = vert - remaining[3:-1]
                signs.append(np.sign(arms1[:, 1] * arms2[:, 0] -
                                     arms2[:, 1] * arms1[:, 0]))
            for rest in (remaining[2:-2], remaining[3:-1]):
                arms1 = remaining[-1] - rest
                arms2 = rest - remaining[1]
                signs.append(np.sign(arms1[:, 1] * arms2[:, 0] -
                                     arms2[:, 1] * arms1[:, 0]))

            cross = np.any(np.bitwise_and(signs[0] != signs[1],
                                          signs[2] != signs[3]))
            if not cross and twiceTriangleArea(remaining[-1], remaining[0],
                                               remaining[1]) > 0.:
                # triangle [-1, 0, 1] is a good one, cut it out
                result.append((remaining[-1].copy(), remaining[0].copy(),
                               remaining[1].copy()))
                remaining = remaining[1:]
            else:
                remaining = np.roll(remaining, 1, axis=0)

        # there must now be 0 or 1 concave vertices left; find the
        # concave vertex (or a vertex) and fan out from it
        vertices = remaining
        shiftedUp = vertices - np.roll(vertices, 1, axis=0)
        shiftedBack = np.roll(vertices, -1, axis=0) - vertices

        # signed area for each triangle (i-1, i, i+1) for vertex i
        areas = shiftedBack[:, 1] * shiftedUp[:, 0] - \
            shiftedUp[:, 1] * shiftedBack[:, 0]

        concave = np.where(areas < 0.)[0]

        fan = (concave[0] if len(concave) else 0)
        fanVert = remaining[fan]
        remaining = np.roll(remaining, -fan, axis=0)[1:]

        result.extend([(fanVert, remaining[0], remaining[1]),
                       (fanVert, remaining[1], remaining[2])])

        return np.array(result, dtype=np.float32)


class ConvexSpheropolygon:
    """Basic class to hold a set of points for a 2D Convex Spheropolygon.
       The behavior for concave inputs is not defined"""

    def __init__(self, verts, radius):
        """Initialize a polygon with a counterclockwise list of 2D
        points and checks that they are ordered counter-clockwise"""
        self.vertices = np.array(verts, dtype=np.float32)

        if len(self.vertices[0]) != 2:
            raise TypeError("positions must be an Nx2 array")
        self.n = len(self.vertices)
        self.radius = radius

        # This actually checks that the majority of the polygon is
        # listed in counter-clockwise order, but seems like it should
        # be sufficient for common use cases. Non-simple polygons can
        # still sneak in clockwise vertices.
        if self.getArea() < 0:
            raise RuntimeError("Spheropolygon was given with some clockwise "
                               "vertices, but it requires that vertices be "
                               "listed in counter-clockwise order")

    def getArea(self):
        """Calculate and return the signed area of the polygon with
        counterclockwise shapes having positive area"""
        # circle
        if (self.n <= 1):
            return np.pi * (self.radius**2)
        # circly-rod
        elif (self.n == 2):
            dr = self.vertices[0] - self.vertices[1]
            return np.pi * (self.radius**2) + \
                np.sqrt(np.dot(dr, dr)) * self.radius * 2.0
        # proper spheropolygon
        else:
            # first calculate the area of the underlying polygon
            shifted = np.roll(self.vertices, -1, axis=0)
            # areas is twice the signed area of each triangle in the shape
            areas = self.vertices[:, 0] * shifted[:, 1] - \
                shifted[:, 0] * self.vertices[:, 1]

            poly_area = np.sum(areas) / 2

            drs = shifted - self.vertices
            edge_area = np.sum(np.sqrt(np.diag(
                np.dot(drs, drs.transpose())))) * self.radius
            # add edge, poly and vertex area
            return poly_area + edge_area + np.pi * self.radius**2

    def center(self):
        """Center this polygon around (0, 0)"""
        self.vertices -= np.mean(self.vertices, axis=0)

    @property
    def triangles(self):
        """A cached property of an Ntx3x2 numpy array of points, where
        Nt is the number of triangles in this polygon."""
        try:
            return self._triangles
        except AttributeError:
            self._triangles = self._triangulation()
        return self._triangles

    @property
    def normalizedTriangles(self):
        """A cached property of the same shape as triangles, but
        normalized such that all coordinates are bounded on [0, 1]."""
        try:
            return self._normalizedTriangles
        except AttributeError:
            self._normalizedTriangles = self._triangles.copy()
            self._normalizedTriangles -= np.min(self._triangles)
            self._normalizedTriangles /= np.max(self._normalizedTriangles)
        return self._normalizedTriangles

    # left over from Polygon, I assume this is for freud viz
    def _triangulation(self):
        """Return a numpy array of triangles with shape (Nt, 3, 2) for
        the 3 2D points of Nt triangles."""

        if self.n <= 3:
            return [tuple(self.vertices)]

        result = []
        remaining = self.vertices

        # step around the shape and grab ears until only 4 vertices are left
        while len(remaining) > 4:
            signs = []
            for vert in (remaining[-1], remaining[1]):
                arms1 = remaining[2:-2] - vert
                arms2 = vert - remaining[3:-1]
                signs.append(np.sign(arms1[:, 1] * arms2[:, 0] -
                                     arms2[:, 1] * arms1[:, 0]))
            for rest in (remaining[2:-2], remaining[3:-1]):
                arms1 = remaining[-1] - rest
                arms2 = rest - remaining[1]
                signs.append(np.sign(arms1[:, 1] * arms2[:, 0] -
                                     arms2[:, 1] * arms1[:, 0]))

            cross = np.any(np.bitwise_and(signs[0] != signs[1],
                                          signs[2] != signs[3]))
            if not cross and twiceTriangleArea(remaining[-1], remaining[0],
                                               remaining[1]) > 0.:
                # triangle [-1, 0, 1] is a good one, cut it out
                result.append((remaining[-1].copy(), remaining[0].copy(),
                               remaining[1].copy()))
                remaining = remaining[1:]
            else:
                remaining = np.roll(remaining, 1, axis=0)

        # there must now be 0 or 1 concave vertices left; find the
        # concave vertex (or a vertex) and fan out from it
        vertices = remaining
        shiftedUp = vertices - np.roll(vertices, 1, axis=0)
        shiftedBack = np.roll(vertices, -1, axis=0) - vertices

        # signed area for each triangle (i-1, i, i+1) for vertex i
        areas = shiftedBack[:, 1] * shiftedUp[:, 0] - \
            shiftedUp[:, 1] * shiftedBack[:, 0]

        concave = np.where(areas < 0.)[0]

        fan = (concave[0] if len(concave) else 0)
        fanVert = remaining[fan]
        remaining = np.roll(remaining, -fan, axis=0)[1:]

        result.extend([(fanVert, remaining[0], remaining[1]),
                       (fanVert, remaining[1], remaining[2])])

        return np.array(result, dtype=np.float32)


def twiceTriangleArea(p0, p1, p2):
    """Returns twice the signed area of the triangle specified by the
    2D numpy points (p0, p1, p2)."""
    p1 = p1 - p0
    p2 = p2 - p0
    return p1[0] * p2[1] - p2[0] * p1[1]
