# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import json

import numpy as np
import pytest
from pytest import approx

from conftest import data_filenames_mark
from coxeter import from_gsd_type_shapes
from coxeter.families import common


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
                assert shape.faces == value
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


@data_filenames_mark
def test_json_data_families(family):
    # Iterate through shape families stored in common
    for fam in [attr for attr in dir(common)]:
        # Set correct shape family for current test
        if family.title().replace("_", "") in fam:
            # Load the family via common
            module = getattr(common, fam)
            # Manually load the json data
            with open("coxeter/families/data/" + family + ".json") as file:
                fam_data = json.load(file)

            # Extract the stored shape keys
            shapes = list(fam_data.keys())
            for shape in shapes:
                if "pyramid" in family or "prism" in family:
                    with pytest.warns(
                        DeprecationWarning, match="deprecated in favor of"
                    ):
                        module_vertices = module.get_shape(shape).vertices
                else:
                    module_vertices = module.get_shape(shape).vertices
                json_vertices = fam_data[shape]["vertices"]
                assert np.all(module_vertices == json_vertices)
