The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`__.
This project adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`__.


v0.x.x - 20xx-xx-xx
-------------------

Added
~~~~~

- ``edges``, ``edge_vectors``, and ``edge_lengths`` properties to ``Polygon``

Changed
~~~~~~~

- The minimum required Python version is now 3.9.

v0.10.0 - 2025-02-24
-------------------

Added
~~~~~

- Support for Python 3.13
- Rendered shape family tables in ReadTheDocs

Fixed
~~~~~

- ``to_hoomd`` method now properly exports convex polygons
- Warnings raised in test cases are now handled properly

v0.9.0 - 2024-09-13
-------------------

Breaking
~~~~~

- The private ``TabulatedShapeFamily`` class has been removed, with functionality moved to ``TabulatedGSDShapeFamily``.

Added
~~~~~

- ``TabulatedGSDShapeFamilies`` are now iterable, allowing easier access to shapes.
- New ``UniformPrismFamily``, ``UniformAntiprismFamily``, ``UniformPyramidFamily``, and ``UniformDipyramidFamily``
- New documentation to help users initialize common geometries.
- New methods to export Polyhedra as OBJ, OFF, STL, PLY, VTK, X3D, and HTML files.
- New documentation section: "Shape Family Tables"

Changed
~~~~~~~

- The data in ``DOI_SHAPE_REPOSITORIES`` for source :cite:`Damasceno2012` is now sorted to match the order described in the paper.

Deprecated
~~~~~~~~~~

- The ``PrismAntiprismFamily`` and ``PyramidDipyramidFamily`` have been deprecated in favor of the new families added above, which are faster, more consistent, and present a simplified interface.
  The deprecated code was retained for backwards compatibility, but is no longer included in the documentation.

v0.8.0 - 2024-02-21
-------------------

Added
~~~~~

- New ``edge_lengths`` method.
- ``combine_simplices``, ``find_simplex_equations``, ``_find_face_centroids``,
  ``find_coplanar_simplices``, ``_find_face_centroids``, and ``calculate_signed_volume``
  methods for the ConvexPolyhedron class.
- ``simplices``, ``equations``, and ``face_centroids`` properties for the
  ConvexPolyhedron class.
- Additional pytests for surface area, volume, centroid, moment of inertia, and equations properties.
- New ``to_hoomd`` and ``to_json`` export methods for use with simulation tools

Changed
~~~~~~~

- Pre-commit now uses ruff instead of flake8, pydocstyle, pyupgrade and isort.
- CI now uses GitHub Actions.
- Docs ported to furo theme.
- Reimplemented ``find_equations``, ``_volume``, ``surface_area``, ``centroid``,
  ``_compute_inertia_tensor``, ``rescale``, and ``get_face_area`` methods for convex
  polyhedra using NumPy vectorized operations and polyhedron simplices.
- [breaking] ``ConvexPolyhedron._surface_triangulation`` now returns sorted simplices,
  rather than running polytri. This can change the order of vertices and/or triangles.
- [breaking] ``faces`` may return faces in a different order than previously. Faces are still sorted with ``sort_faces``, and will still be ordered such that curl and divergence theorems work properly.
- ``volume``, ``surface_area``, and ``centroid`` properties now return stored values, rather than computing the quantity at each call.
- ``rescale`` now computes the centroid to ensure the correct value is available when ``centroid`` is called.
- Optimized pytest configurations for more efficient use of local and remote resources.

v0.7.0 - 2023-09-18
-------------------

Added
~~~~~

- New `edges` and `edge_vectors` properties for polyhedra.
- New shape families for Archimedean, Catalan, and Johnson shapes.
- New shape families for regular pyramids and dipyramids, and a selection of regular prisms and antiprisms.

Changed
~~~~~~~

- The minimum required Python version is now 3.8.
- The minimum required NumPy version is now 1.19.
- [breaking] Sped up point in polygon and point in polyhedron using NumPy.
- Migrated to pyproject.toml.

Fixed
~~~~~

- Numerical precision issues in tests.
- GSD spec correctly outputs for `Polyhedron` objects.
- Error in `__repr__` for polyhedra with multiple face degrees.
- ReadTheDocs build errors resulting from `conda` memory usage.

v0.6.1 - 2021-07-15
-------------------

Fixed
~~~~~

- Typos in JOSS paper.

v0.6.0 - 2021-07-14
-------------------

Added
~~~~~

- Plotting and other graphical rendering of shapes using `plato <https://plato-draw.readthedocs.io/>`__.
- Notebooks with example use-cases for the package.
- A quickstart tutorial.

v0.5.0 - 2021-02-23
-------------------

Added
~~~~~

- Ellipse area setter and Ellipsoid volume setter.
- Point in circle checks.
- Point in ellipse checks.
- Inertia tensors for 2D shapes that implement moments of inertia.
- Add minimal bounding sphere for all shapes.
- Add minimal centered bounding sphere calculations for all shapes except general polygons, general polyhedra, spheropolygons, and spheropolyhedra.
- Enable getting and setting the circumsphere or bounding sphere radius of a polyhedron (for both types of bounding sphere).
- Add maximal bounded sphere for all shapes.
- Add maximal centered bounded sphere calculations for all shapes except general polygons, general polyhedra, spheropolygons, and spheropolyhedra.
- Enable getting and setting the insphere or bounded sphere radius of a polyhedron (for both types of bounding sphere).
- Point in polygon checks for general (nonconvex) polygons.
- Point in polyhedron checks for general (nonconvex) polyhedrons.
- Minimal bounding sphere for all shapes except spheropolygons and spheropolyhedra.
- Add minimal centered bounding sphere calculations for all shapes except general polygons, general polyhedra, spheropolygons, and spheropolyhedra.
- Getters and setters for the circumsphere or bounding sphere radius of a polyhedron (for both types of bounding sphere).
- A ``repr`` for all shapes.

Changed
~~~~~~~

- Ensure that hypothesis-based tests don't implicitly reuse pytest fixtures.

Deprecated
~~~~~~~~~~

- The circumsphere from center calculations (replaced by minimal centered bounding sphere).
- The bounding_sphere property is deprecated in favor of minimal_bounding_sphere.
- The insphere from center calculations (replaced by maximal centered bounded sphere).

Fixed
~~~~~

- Centroid calculations for polygon and polyhedron use the full integrals rather than simple averages of vertices.

v0.4.0 - 2020-10-14
-------------------

Added
~~~~~

- Circumsphere and insphere from center calculations for ConvexSpheropolyhedron.
- Form factors amplitudes for sphere, polygons, and polyhedra.
- Shape families associated with a DOI can be directly accessed via a dictionary.
- Expected abstract interface for shapes (both 2D and 3D) has expanded.
- Plotting polygons or polyhedra can automatically create matplotlib axes.
- Perimeter calculation for polygons.
- Area and perimeter setters for spheropolygons.

Changed
~~~~~~~

- Shape family API is now entirely based on class methods rather than a call operator.
- The parent ShapeFamily class is now part of the public API.
- Doctests are now run as part of pytest.
- Subpackages have been renamed: shape_classes is now shapes, and shape_families is now families.
- The common_families submodule of shape_families is now just common.

Fixed
~~~~~

- Volume calculation for ConvexSpheropolyhedron includes area of extruded faces in addition to vertices and edges.
- Documentation has been revised and edited.

Removed
~~~~~~~

- The symmetry.py module.
- The ft.py module.
- The symmetry.py module.
- The get_params method of TabulatedShapeFamily.
- The family_from_doi method (the underlying data dictionary is now directly exposed).

v0.3.0 - 2020-06-18
-------------------

Added
~~~~~

- Calculation of circumsphere from center for convex polyhedra.
- Simple name-based shape getter for damasceno SHAPES dictionary.
- Polygons moment of inertia calculation.
- Interoperability with the GSD shape specification.
- Shape families and stored data for well-known families.
- All shapes can be centered anywhere in 3D Euclidean space.
- Extensive style checking using black, isort, and various other flake8 plugins.
- Make Circle area settable.
- 3D shapes can be oriented by their principal axes.
- Make Sphere volume settable.

Changed
~~~~~~~

- Inertia tensors for polyhedra and moments of inertia for polygons are calculated in global coordinates rather than the body frame.
- Modified testing of convex hulls to generate points on ellipsoids to avoid degenerate simplices.
- All insphere, circumsphere, and bounding sphere calculations now return the appropriate classes instead of tuples.

Removed
~~~~~~~

- The common_shapes subpackage.

v0.2.0 - 2020-04-09
-------------------

Added
~~~~~

- Continuous integrated testing on CircleCI.
- New Polygon class with property-based API.
- New ConvexSpheropolygon class with property-based API.
- New Polyhedron class with property-based API and robust facet sorting and merging.
- New ConvexPolyhedron class with property-based API.
- New ConvexSpheropolyhedron class with property-based API.
- Ability to plot Polyhedra and Polygons.
- Can now check whether points lie inside a ConvexPolyhedron or ConvexSpheropolyhedron.
- Added documentation.
- New Ellipsoid class with property-based API.
- New Sphere class with property-based API.
- New Ellipse class with property-based API.
- New Circle class with property-based API.
- Added insphere from center calculation for convex polyhedra.
- New ConvexPolygon class.
- Documentation is hosted on ReadTheDocs.

Changed
~~~~~~~

- Moved core shape classes from euclid.FreudShape into top-level package namespace.
- Moved common shape definitions into common_shapes subpackage.
- Shapes from Damasceno science 2012 paper are now stored in a JSON file that is loaded in the damasceno module.

Fixed
~~~~~

- Formatting now properly follows PEP8.

Removed
~~~~~~~

- Various unused or redundant functions in the utils module.
- The quaternion_tools module (uses rowan for quaternion math instead).
- The shapelib module.
- Old polygon.py and polyhedron.py modules, which contained old implementations of various poly\* and spheropoly\* classes.

v0.1.0
------

- Initial version of code base.
