from scipy.spatial import ConvexHull
import numpy as np

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

    @property
    def vertices(self):
        """Get the vertices of the polyhedron."""
        return self._vertices

    @property
    def facets(self):
        """Get the polyhedron's facets."""
        self._facets

    @property
    def volume(self):
        """Get or set the polyhedron's volume (setting rescales vertices)."""

    @volume.setter
    def volume(self, value):

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

        for facet_index in facets:
            facet = self.facets[facet_index]

    @property
    def surface_area(self):
        """The surface area."""
        return get_facet_area(None)

    @property
    def moment_inertia(self):
        """The moment of inertia.

        Compute using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """

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

    @insphere_radius.setter(self, value):
        pass

    @property
    def circumsphere_radius(self):
        """Get or set the polyhedron's circumsphere radius (setting rescales
        vertices)."""

    @circumsphere_radius.setter(self, value):
        pass

    @property
    def asphericity(self):
        """The asphericity."""

    @property
    def iq(self):
        """The isoperimetric quotient."""

    @property
    def tau(self):
        """The sphericity measure defined in
        https://www.sciencedirect.com/science/article/pii/0378381284800199."""

    @property
    def facet_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        facets."""

    @property
    def vertex_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        vertex."""
