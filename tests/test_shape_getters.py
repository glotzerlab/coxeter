from coxeter import from_gsd_type_shapes
import numpy as np


def test_gsd_shape_getter():
    test_specs = [
        {
            'type': 'Sphere',
            'diameter': 1
        },
        {
            'type': 'Ellipsoid',
            'a': 1,
            'b': 2,
            'c': 3,
            'ndim': 2
        },
        {
            'type': 'Ellipsoid',
            'a': 1,
            'b': 2,
            'c': 2
        },
        {
            'type': 'Polygon',
            'vertices': [[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0.5, 0.5, 0],
                         [0, 1, 0]],
        },
        {
            'type': 'Polygon',
            'vertices': [[0, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0],
                         [1, 0, 0]],
        },
        {
            'type': 'Polygon',
            'vertices': [[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0]],
            'rounding_radius': 1,
        },
        {
            'type': 'ConvexPolyhedron',
            'vertices': [[0, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1],
                         [1, 0, 1]],
        },
        {
            'type': 'ConvexPolyhedron',
            'vertices': [[0, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1],
                         [1, 0, 1]],
            'rounding_radius': 1
        },
        {
            'type': 'Mesh',
            'vertices': [[0, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1],
                         [1, 0, 1]],
            'faces': [[0, 1, 2, 3],
                      [4, 7, 6, 5],
                      [0, 3, 7, 4],
                      [1, 5, 6, 2],
                      [3, 2, 6, 7],
                      [0, 4, 5, 1]],

        },
    ]

    for shape_spec in test_specs:
        # First create and validate the shape.
        ndim = shape_spec.pop('ndim', 3)
        shape = from_gsd_type_shapes(shape_spec, ndim=ndim)
        for param, value in shape_spec.items():
            if param == 'diameter':
                assert shape.radius == value/2
            elif param == 'rounding_radius':
                assert shape.radius == value
            elif param != 'type':
                try:
                    assert getattr(shape, param) == value
                except ValueError as e:
                    if str(e) == ("The truth value of an array with more than "
                                  "one element is ambiguous. Use a.any() or "
                                  "a.all()"):
                        np.testing.assert_allclose(
                            getattr(shape, param), value)

        # Now convert back and make sure the conversion is lossless.
        assert shape.gsd_shape_spec == shape_spec
