import numpy as np


class Polygon(object):
    def __init__(self, vertices, normal=[0, 0, 1]):
        """A simple (i.e. non-self-overlapping) polygon.

        The polygon is assumed to be of constant unit density.

        The polygon may be embedded in 3-dimensions, in which case the normal
        vector determines which way is "up". Note that the precise normal
        vector is not important to the winding number because the winding
        number is invariant to projection, all we need to know is the sign of
        each component of the normal vector.

        Args:
            vertices (:math:`(N, 3)` :class:`numpy.ndarray`):
                The vertices of the polygon.
            normal (list):
                #Whether the normal vector is in the positive direction in each
                #direction. Need to specify in all directions because the
                #polygon could be on any of the xy, xz, and yz planes, in which
                #case the orientation is degenerate with respect to that plane
                #(Default value: [True, True, True]).

        """
        vertices = np.asarray(vertices)
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError(
                "Vertices must be specified as an Nx3 array.")

        if len(vertices) < 3:
            raise ValueError(
                "A polygon must be composed of at least 3 vertices.")
        self._vertices = vertices
        self._normal = normal

    def reorder_verts(self, cw=False, ref_index=0):
        """Sort the vertices in order with respect to the normal.

        The default ordering is counterclockwise.

        Args:
            cw (bool):
                If True, sort in clockwise order (Default value: False).
            ref_index (int):
                Index indicating which vertex should be placed first in the
                sorted list (Default value: 0).
        """
        pass

    @property
    def vertices(self):
        """Get the vertices of the polyhedron."""
        return self._vertices

    @property
    def area(self):
        """Get or set the polygon's area (setting rescales vertices)."""
        pass

    @area.setter
    def area(self, value):
        pass

    @property
    def moment_inertia(self):
        """The moment of inertia.

        Compute using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """
        pass

    @property
    def center(self):
        """Get or set the polyhedron's centroid (setting rescales vertices)."""
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self.vertices += (np.asarray(value) - self.center)

    @property
    def incircle_radius(self):
        """Get or set the polyhedron's insphere radius (setting rescales
        vertices)."""
        pass

    @incircle_radius.setter
    def incircle_radius(self, value):
        pass

    @property
    def circumcircle_radius(self):
        """Get or set the polyhedron's circumsphere radius (setting rescales
        vertices)."""
        pass

    @circumcircle_radius.setter
    def circumcircle_radius(self, value):
        pass
