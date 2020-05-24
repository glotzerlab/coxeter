import numpy as np
from coxeter.shape_classes import ConvexPolyhedron
import rowan
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from coxeter.damasceno import get_shape_by_name
from coxeter.shape_classes.utils import (translate_inertia_tensor,
                                         rotate_order2_tensor)
from utils import compute_inertia_mc


@given(arrays(np.float64, (4, 3), elements=floats(-10, 10, width=64),
              unique=True))
def test_rotate_inertia(points):
    # Use the input as noise rather than the base points to avoid precision and
    # degenerate cases provided by hypothesis.
    tet = get_shape_by_name('Tetrahedron')
    vertices = tet.vertices + points

    rotation = rowan.random.rand()
    shape = ConvexPolyhedron(vertices)
    rotated_shape = ConvexPolyhedron(rowan.rotate(rotation, vertices))

    mat = rowan.to_matrix(rotation)
    rotated_inertia = rotate_order2_tensor(mat, shape.inertia_tensor)

    assert np.allclose(rotated_inertia, rotated_shape.inertia_tensor)


# Use a small range of translations to ensure that the Delaunay triangulation
# used by the MC calculation will not break.
@given(arrays(np.float64, (3, ), elements=floats(-0.2, 0.2, width=64),
              unique=True))
def test_translate_inertia(translation):
    shape = get_shape_by_name('Cube')
    # Choose a volume > 1 to test the volume scaling, but don't choose one
    # that's too large because the uncentered polyhedral calculation has
    # massive error without fixing that.
    shape.volume = 2
    shape.center = (0, 0, 0)

    translated_shape = ConvexPolyhedron(shape.vertices + translation)

    translated_inertia = translate_inertia_tensor(
        translation, shape.inertia_tensor, shape.volume)
    mc_tensor = compute_inertia_mc(translated_shape.vertices, 1e4)

    assert np.allclose(translated_inertia,
                       translated_shape._compute_inertia_tensor(False),
                       atol=2e-1,
                       rtol=2e-1)
    assert np.allclose(mc_tensor,
                       translated_shape._compute_inertia_tensor(False),
                       atol=2e-1,
                       rtol=2e-1)

    assert np.allclose(mc_tensor,
                       translated_inertia,
                       atol=1e-2,
                       rtol=1e-2)
    assert np.allclose(mc_tensor,
                       translated_shape.inertia_tensor,
                       atol=1e-2,
                       rtol=1e-2)
