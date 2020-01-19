# Change Log
The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## next

### Added

* Continuous integrated testing on CircleCI.
* New Polygon class with property-based API.
* New ConvexSpheropolygon class with property-based API.
* New Polyhedron class with property-based API and robust facet sorting and merging.
* New ConvexPolyhedron class with property-based API.
* New ConvexSpheropolyhedron class with property-based API.
* Ability to plot Polyhedra and Polygons.
* Can now check whether points lie inside a ConvexPolyhedron.
* Added documentation.

### Changed

* Moved core shape classes from euclid.FreudShape into top-level package namespace.
* Moved common shape definitions into common\_shapes subpackage.
* Shapes from Damasceno science 2012 paper are now stored in a JSON file that is loaded in the damasceno module.

### Fixed

* Formatting now properly follows PEP8.

### Removed

* Various unused or redundant functions in the utils module.
* The quaternion\_tools module (uses rowan for quaternion math instead).
* The shapelib module.

## v0.1.0

* Initial version of code base.
