import numpy as np
from test_polyhedron import cube_points, convex_cube
from euclid.shape_classes.convex_spheropolyhedron import ConvexSpheropolyhedron

from hypothesis import given
from hypothesis.strategies import floats


@given(radius=floats(0.1, 1))
def test_volume(radius, cube_points):
    sphero_cube = ConvexSpheropolyhedron(cube_points, radius)
    V_cube = 1
    V_sphere = (4/3)*np.pi*radius**3
    V_cyl = 12*np.pi*radius**2/4
    assert np.isclose(sphero_cube.volume, V_cube + V_sphere + V_cyl)


def test_volume_polyhedron(convex_cube, cube_points):
    """Ensure that the base case of zero radius gives the same result as a
    polyhedron."""
    sphero_cube = ConvexSpheropolyhedron(cube_points, 0)
    assert sphero_cube.volume == convex_cube.volume


def test_surface_area():
    pass
