from scipy.spatial import ConvexHull
import numpy as np
from .polygon import Polygon


class Polyhedron(object):
    def __init__(self, vertices, facets=None):
        """A general polyhedron.

        If only vertices are passed in, the result is a convex polyhedron
        defined by these vertices. If facets are provided, the resulting
        polyhedron may be nonconvex.

        The polyhedron is assumed to be of unit mass and constant density.

        """
        self._vertices = vertices
        if facets is None:
            hull = ConvexHull(vertices)
            self._facets = [facet for facet in hull.simplices]
        else:
            self._facets = facets

    def merge_facets(self, tolerance=1e-6):
        """Merge facets of a polyhedron.

        For convex polyhedra, facets will automatically be merged to an
        appropriate degree.  However, the merging of facets must be based on a
        tolerance (we may need to provide two such parameters depending on how
        we perform the merge), so we need to expose this method to allow the
        user to redo the merge with a different tolerance."""
        pass

    @property
    def vertices(self):
        """Get the vertices of the polyhedron."""
        return self._vertices

    @property
    def facets(self):
        """Get the polyhedron's facets."""
        return self._facets

    @property
    def volume(self):
        """Get or set the polyhedron's volume (setting rescales vertices)."""
        pass

    @volume.setter
    def volume(self, value):
        pass

    def get_facet_area(self, facets=None):
        """Get the total surface area of a set of facets.

        Args:
            facets (int, sequence, or None):
                The index of a facet or a set of facet indices for which to
                find the area. If None, finds the area of all facets (Default
                value: None).

        Returns:
            list: The area of each facet.
        """
        if facets is None:
            facets = range(len(self.facets))

        areas = []
        for facet_index in facets:
            facet = self.facets[facet_index]
            poly = Polygon(self.vertices[facet])
            areas.append(poly.area)

        return areas

    @property
    def surface_area(self):
        """The surface area."""
        return np.sum(self.get_facet_area())

    @property
    def inertia_tensor(self):
        """The inertia tensor.

        Computed using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """
        try:
            return self._inertia_tensor
        except AttributeError:
            centered_vertices = self.vertices - self.center
            simplices = centered_vertices[self.faces]

            volumes = np.abs(np.linalg.det(simplices)/6)

            fxx = lambda triangles: triangles[:, 1]**2 + triangles[:, 2]**2 # noqa
            fxy = lambda triangles: -triangles[:, 0]*triangles[:, 1] # noqa
            fxz = lambda triangles: -triangles[:, 0]*triangles[:, 2] # noqa
            fyy = lambda triangles: triangles[:, 0]**2 + triangles[:, 2]**2 # noqa
            fyz = lambda triangles: -triangles[:, 1]*triangles[:, 2] # noqa
            fzz = lambda triangles: triangles[:, 0]**2 + triangles[:, 1]**2 # noqa

            def compute(f):
                return f(simplices[:, 0, :]) + f(simplices[:, 1, :]) + \
                    f(simplices[:, 2, :]) + f(simplices[:, 0, :] +
                                              simplices[:, 1, :] +
                                              simplices[:, 2, :])

            Ixx = (compute(fxx)*volumes/20).sum()
            Ixy = (compute(fxy)*volumes/20).sum()
            Ixz = (compute(fxz)*volumes/20).sum()
            Iyy = (compute(fyy)*volumes/20).sum()
            Iyz = (compute(fyz)*volumes/20).sum()
            Izz = (compute(fzz)*volumes/20).sum()

            self._inertia_tensor = np.array([[Ixx, Ixy, Ixz],
                                            [Ixy,   Iyy, Iyz],
                                            [Ixz,   Iyz,   Izz]])

            return self._inertia_tensor

    @property
    def center(self):
        """Get or set the polyhedron's centroid (setting rescales vertices)."""
        return np.mean(self.vertices, axis=0)

    @center.setter
    def center(self, value):
        self.vertices += (np.asarray(value) - self.center)

    @property
    def insphere_radius(self):
        """Get or set the polyhedron's insphere radius (setting rescales
        vertices)."""
        pass

    @insphere_radius.setter
    def insphere_radius(self, value):
        pass

    @property
    def circumsphere_radius(self):
        """Get or set the polyhedron's circumsphere radius (setting rescales
        vertices)."""
        pass

    @circumsphere_radius.setter
    def circumsphere_radius(self, value):
        pass

    @property
    def asphericity(self):
        """The asphericity."""
        pass

    @property
    def iq(self):
        """The isoperimetric quotient."""
        pass

    @property
    def tau(self):
        """The sphericity measure defined in
        https://www.sciencedirect.com/science/article/pii/0378381284800199."""
        pass

    @property
    def facet_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        facets."""
        pass

    @property
    def vertex_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        vertex."""
        pass
