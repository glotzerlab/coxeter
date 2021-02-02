import numpy as np

from coxeter import from_gsd_type_shapes
from coxeter.shapes import Shape2D


def test_gsd_shape_getter():
    test_specs = [
        {"type": "Sphere", "diameter": 1},
        {"type": "Ellipsoid", "a": 1, "b": 2, "dimensions": 2},
        {"type": "Ellipsoid", "a": 1, "b": 2, "c": 2},
        {
            "type": "Polygon",
            "vertices": [[0, 0], [1, 0], [1, 1], [0.5, 0.5], [0, 1]],
        },
        {"type": "Polygon", "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        {
            "type": "Polygon",
            "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
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
            "faces": [
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
                assert shape.radius == value / 2
            elif param == "rounding_radius":
                assert shape.radius == value
            elif param != "type":
                try:
                    if param == "vertices" and isinstance(shape, Shape2D):
                        check_value = getattr(shape, param)[:, :2]
                    else:
                        check_value = getattr(shape, param)
                    assert check_value == value
                except ValueError as e:
                    if str(e) == (
                        "The truth value of an array with more than "
                        "one element is ambiguous. Use a.any() or "
                        "a.all()"
                    ):
                        np.testing.assert_allclose(check_value, value)

        # Now convert back and make sure the conversion is lossless.
        assert shape.gsd_shape_spec == shape_spec
