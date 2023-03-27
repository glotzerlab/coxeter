coxeter Developers
------------------

The following people contributed to the development of coxeter.

Vyas Ramasubramani - **Creator and lead developer**

* Created documentation pages.
* Formalized contribution guidelines and contributor agreement.
* Cleaned up damasceno module and separated out shape information into a JSON file that is read on demand.
* Fixed code formatting to conform to PEP8 requirements.
* Implemented Polygon class.
* Implemented ConvexSpheropolygon class.
* Implemented Polyhedron class.
* Implemented ConvexPolyhedron class.
* Implemented ConvexSpheropolyhedron class.
* Add ability to check if points are contained in convex polyhedra.
* Fix calculation of circumsphere to work for non-regular polyhedra.
* Fix calculation of circumcircle to work for non-regular polygons.
* Add ability to calculate minimum bounding sphere/circle for polyhedra/polygons.
* Implemented ConvexPolygon class.
* Added ReadTheDocs support.
* Added circumsphere from center calculation for convex polyhedra.
* Added shape getter for damasceno shapes.
* Define proper inertia tensor calculations and transformations for polygons and polyhedra.
* Added interoperability with the GSD shape specification.
* Developed shape families and all associated shape repository APIs.
* Add ability to diagonalize the inertia tensors of shapes.
* Defined base classes for all shapes.
* Standardize usage of Sphere/Circle classes for circum, in, and bounding sphere/circle calculations.
* Moved form factor amplitude calculations from legacy ft module to shape classes, cleaned and added more tests.
* Added point-in-shape checks for circles and ellipses.
* Added generic inertia tensors for 2D shapes.
* Added minimal bounding sphere for all shapes.
* Added minimal centered bounding sphere calculations for all shapes except general polygons and polyhedra.
* Enabled getting and setting the circumsphere or bounding sphere/circle radius of a polyhedron/polygon (for both types of bounding sphere/circle).
* Added maximal bounding sphere for all shapes.
* Added maximal centered bounded sphere calculations for all shapes except general polygons and polyhedra.
* Enabled getting and setting the insphere or bounded sphere/circle radius of a polyhedron/polygon (for both types of bounding sphere/circle).
* Added point in polygon checks.
* Added point in polyhedron checks.
* Added repr for all shapes.
* Fixed centroid calculations for polygon and polyhedron to use integrals rather than simple averages of vertices.
* Wrote example notebooks.

Bryan VanSaders - **Original maintainer of legacy euclid package**

* Created package layout.
* Original port of classes and methods into package.
* Added some methods to the utils module.
* Added symmetry groups.

James Proctor

* Ported some damasceno code into coxeter from standalone module.

Bradley Dice

* Migrated ft code into coxeter from freud and added tests.
* Added CircleCI support.
* Add ability to check if points are contained in convex spheropolyhedra.
* Revised and edited all documentation.
* Updated doctests to be part of pytest suite.
* Added automatic axis creation for plotting.
* Added spheropolygon area and perimeter setters.
* Added ellipse area setter and ellipsoid volume setter.
* Added plato support.

Brandon Butler

* Removed old quat\_tools module and modified modules to use rowan.
* Moved logic in FreudShape module to top-level package namespace.
* Moved all common shape definitions into a common\_shapes module.

Eric Harper

* Migrated shape classes into coxeter from freud.

Jens Glaser

* Bug fix for convex hull finding.

M. Eric Irrgang

* Bugfixes to imports.
* Implemented core shape classes.
* Implemented the ft module.

Carl Simon Adorf

* Implemented the damasceno module.

Matthew Spellings

* Added some methods to the utils module.
* Triangulation of core shape classes.

William Zygmunt

* Helped clean up utils module.

Tobias Dwyer

* Added getter and setter tests to some of the shape classes.
* Added examples for the shape classes.

Jen Bradley

* Bug fixes for ``gsd_shape_spec`` to correctly comply with GSD specifications.
* Fixed error where ``__repr__`` would fail for polyhedra with multiple face types.
* Increased accuracy of stored data for PlatonicFamily solids
* Added shape families for Archimedean, Catalan, and Johnson solids.
* Added shape family for prisms and antiprisms.
* Added shape family for equilateral pyramids and dipyramids.

Source code
-----------

**coxeter** includes the source code of the following Python packages and
modules.

.. highlight:: none

The source of polytri (https://github.com/bjorkegeek/polytri) is included
directly into the **coxeter** package. The module implementing that code is
reproduced in its entirety along with an additional ``__init__`` file to enable
its import as a subpackage. It is used for the triangulation of polygons and
the surface triangulation of polyhedra. This software is made available under
the MIT license::

    The MIT License (MIT)

    Copyright (c) 2016 David Björkevik

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE

The source of isect_segments-bentley_ottmann
(https://github.com/ideasman42/isect_segments-bentley_ottmann) is included
directly into the **coxeter** package. The module implementing that code is
reproduced in its entirety along with an additional ``__init__`` file to enable
its import as a subpackage. It is used to check whether a set of vertices
defines a simple or a complex polygon. This software is made available under
the MIT license::

    Copyright (c) 2010 by Bart Kiers
    Copyright (c) 2015 by Campbell Barton

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use,
    copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.

The source of polyhedron (https://github.com/mdickinson/polyhedron) is included
directly into the **coxeter** package. It is used for point in polygon/polyhedron
checks for general polygons and polyhedra (specifically, to calculate the winding
number). This software is made available under the BSD-3 license::

    BSD 3-Clause License

    Copyright (c) 2019, Mark Dickinson
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
