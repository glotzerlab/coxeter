# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
import numpy as np
from pytest import approx

from scipy.spatial import ConvexHull
from coxeter import from_gsd_type_shapes
from coxeter.families import( 
    ArchimedeanFamily,
    CatalanFamily,
    JohnsonFamily,
    PlatonicFamily,
    PrismAntiprismFamily,
    PyramidDipyramidFamily
)

def test_shape_families():
    # Testdics contain
    platonic_solids = [
        "Tetrahedron",
        "Octahedron",
        "Cube",
        "Icosahedron",
        "Dodecahedron",
    ]
    archimedean_solids = [
        "Cuboctahedron",
        "Icosidodecahedron",
        "Truncated Tetrahedron",
        "Truncated Octahedron",
        "Truncated Cube",
        "Truncated Icosahedron",
        "Truncated Dodecahedron",
        "Rhombicuboctahedron",
        "Rhombicosidodecahedron",
        "Truncated Cuboctahedron",
        "Truncated Icosidodecahedron",
        "Snub Cuboctahedron",
        "Snub Icosidodecahedron",
    ]
    catalan_solids = [
        "Deltoidal Hexecontahedron",
        "Deltoidal Icositetrahedron",
        "Disdyakis Dodecahedron",
        "Disdyakis Triacontahedron",
        "Pentagonal Hexecontahedron",
        "Pentagonal Icositetrahedron",
        "Pentakis Dodecahedron",
        "Rhombic Dodecahedron",
        "Rhombic Triacontahedron",
        "Triakis Octahedron",
        "Tetrakis Hexahedron",
        "Triakis Icosahedron",
        "Triakis Tetrahedron",
    ]
    # A sample of the johnson solids are tested, rather than all 92.
    johnson_solids = [
        "Square Pyramid",
        "Trigyrate Rhombicosidodecahedron",
        "Bilunabirotunda",
        "Elongated Pentagonal Cupola",
        "Gyroelongated Pentagonal Pyramid",
        "Pentagonal Orthobirotunda",
        "Elongated Pentagonal Orthocupolarotunda",
        "Parabiaugmented Hexagonal Prism",
        "Tridiminished Icosahedron",
        "Augmented Truncated Dodecahedron",
        "Triangular Hebesphenorotunda",
        "Elongated Square Pyramid",
    ]
    prism_antiprism = [
        "Triangular Prism",
        "Cube",
        "Pentagonal Prism",
        "Hexagonal Prism",
        "Heptagonal Prism",
        "Octagonal Prism",
        "Nonagonal Prism",
        "Decagonal Prism",
        "Octahedron",
        "Square Antiprism",
        "Pentagonal Antiprism",
        "Hexagonal Antiprism",
        "Heptagonal Antiprism",
        "Octagonal Antiprism",
        "Nonagonal Antiprism",
        "Decagonal Antiprism",
    ]
    pyramid_dipyramid = [
        "Tetrahedron",
        "Square Pyramid",
        "Pentagonal Pyramid",
        "Triangular Dipyramid",
        "Octahedron",
        "Pentagonal Dipyramid",
    ]

    for solid in platonic_solids:
        poly = PlatonicFamily.get_shape(solid)
        vertices = poly.vertices
        hull = ConvexHull(vertices)
        assert np.isclose(poly.volume, hull.volume)
        assert np.isclose(poly.surface_area, hull.area)

    for solid in archimedean_solids:
        poly = ArchimedeanFamily.get_shape(solid)
        vertices = poly.vertices
        hull = ConvexHull(vertices)
        assert np.isclose(poly.volume, hull.volume)
        assert np.isclose(poly.surface_area, hull.area)

    for solid in catalan_solids:
        poly = CatalanFamily.get_shape(solid)
        vertices = poly.vertices
        hull = ConvexHull(vertices)
        assert np.isclose(poly.volume, hull.volume)
        assert np.isclose(poly.surface_area, hull.area)

    for solid in johnson_solids:
        poly = JohnsonFamily.get_shape(solid)
        vertices = poly.vertices
        hull = ConvexHull(vertices)
        assert np.isclose(poly.volume, hull.volume)
        assert np.isclose(poly.surface_area, hull.area)

    for solid in prism_antiprism:
        poly = PrismAntiprismFamily.get_shape(solid)
        vertices = poly.vertices
        hull = ConvexHull(vertices)
        assert np.isclose(poly.volume, hull.volume)
        assert np.isclose(poly.surface_area, hull.area)

    for solid in pyramid_dipyramid:
        poly = PyramidDipyramidFamily.get_shape(solid)
        vertices = poly.vertices
        hull = ConvexHull(vertices)
        assert np.isclose(poly.volume, hull.volume)
        assert np.isclose(poly.surface_area, hull.area)

def test_gsd_shape_getter():
    test_specs = [
        {"type": "Sphere", "diameter": 1},
        {"type": "Ellipsoid", "a": 1, "b": 2, "dimensions": 2},
        {"type": "Ellipsoid", "a": 1, "b": 2, "c": 2},
        {
            "type": "Polygon",
            "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0], [0, 1, 0]],
        },
        {"type": "Polygon", "vertices": [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]},
        {
            "type": "Polygon",
            "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            "rounding_radius": 1,
        },
        {
            "type": "ConvexPolyhedron",
            "vertices": [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
            ],
        },
        {
            "type": "ConvexPolyhedron",
            "vertices": [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
            ],
            "rounding_radius": 1,
        },
        {
            "type": "Mesh",
            "vertices": [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
            ],
            "indices": [
                [0, 1, 2, 3],
                [4, 7, 6, 5],
                [0, 3, 7, 4],
                [1, 5, 6, 2],
                [3, 2, 6, 7],
                [0, 4, 5, 1],
            ],
        },
    ]

    for shape_spec in test_specs:
        # First create and validate the shape.
        dimensions = shape_spec.pop("dimensions", 3)
        shape = from_gsd_type_shapes(shape_spec, dimensions=dimensions)
        for param, value in shape_spec.items():
            if param == "diameter":
                assert shape.radius == approx(value / 2)
            elif param == "rounding_radius":
                assert shape.radius == approx(value)
            elif param == "indices":
                np.testing.assert_allclose(shape.faces, value)
            elif param != "type":
                try:
                    assert getattr(shape, param) == value
                except ValueError as e:
                    if str(e) == (
                        "The truth value of an array with more than "
                        "one element is ambiguous. Use a.any() or "
                        "a.all()"
                    ):
                        np.testing.assert_allclose(getattr(shape, param), value)

        # Now convert back and make sure the conversion is lossless.
        assert shape.gsd_shape_spec == shape_spec
