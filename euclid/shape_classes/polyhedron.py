class Polyhedron(object):
    def __init__(self, vertices, faces=None):
        """A general polyhedron.

        If only vertices are passed in, the result is a convex polyhedron
        defined by these vertices. If facets are provided, the resulting
        polyhedron may be nonconvex.

        The polyhedron is assumed to be of unit mass and constant density.

        """
        pass

    def merge_faces(self, tolerance=1e-6):
        """Merge faces of a polyhedron.

        For convex polyhedra, faces will automatically be merged to an
        appropriate degree.  However, the merging of faces must be based on a
        tolerance (we may need to provide two such parameters depending on how
        we perform the merge), so we need to expose this method to allow the
        user to redo the merge with a different tolerance."""

    @property
    def vertices(self):
        """Get the vertices of the polyhedron."""

    @property
    def faces(self):
        """Get the polyhedron's faces."""

    @property
    def volume(self):
        """Get or set the polyhedron's volume (setting rescales vertices)."""

    @volume.setter
    def volume(self, value):

    @property
    def surface_area(self):
        """The surface area."""

    @property
    def moment_inertia(self):
        """The moment of inertia.

        Compute using the method described in
        https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129220
        """

    @property
    def center(self):
        """Get or set the polyhedron's centroid (setting rescales vertices)."""

    @center.setter
    def center(self, value):

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
    def face_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        faces."""

    @property
    def vertex_neighbors(self):
        """An Nx2 NumPy array containing indices of pairs of neighboring
        vertex."""
