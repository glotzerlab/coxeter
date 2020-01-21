import numpy as np
from euclid.shape_classes.convex_spheropolyhedron import ConvexSpheropolyhedron

from hypothesis import given
from hypothesis.strategies import floats


@given(radius=floats(0.1, 1))
def test_volume(radius, cube_points):
    sphero_cube = ConvexSpheropolyhedron(cube_points, radius)
    V_cube = 1
    V_sphere = (4/3)*np.pi*radius**3
    V_cyl = 12*(np.pi*radius**2)/4
    assert np.isclose(sphero_cube.volume, V_cube + V_sphere + V_cyl)


def test_volume_polyhedron(convex_cube, cube_points):
    """Ensure that the base case of zero radius gives the same result as a
    polyhedron."""
    sphero_cube = ConvexSpheropolyhedron(cube_points, 0)
    assert sphero_cube.volume == convex_cube.volume


@given(radius=floats(0.1, 1))
def test_surface_area(radius, cube_points):
    sphero_cube = ConvexSpheropolyhedron(cube_points, radius)
    S_cube = 6
    S_sphere = 4*np.pi*radius**2
    S_cyl = 12*(2*np.pi*radius)/4
    assert np.isclose(sphero_cube.surface_area, S_cube + S_sphere + S_cyl)


def test_surface_area_polyhedron(convex_cube, cube_points):
    """Ensure that the base case of zero radius gives the same result as a
    polyhedron."""
    sphero_cube = ConvexSpheropolyhedron(cube_points, 0)
    assert sphero_cube.surface_area == convex_cube.surface_area


def test_inside_boundaries(cube_points):
    sphero_cube = ConvexSpheropolyhedron(cube_points, 1)

    points_inside = [[0, 0, 0], [1, 1, 1], [-0.01, -0.01, -0.01],
                     [2, 0.5, 0.5], [2, 1, 0.5], [0.5, -0.7, -0.7],
                     [-0.57, -0.57, -0.57]]
    points_outside = [[-0.99, -0.99, -0.99], [-1.01, -1.01, -1.01],
                      [2.01, 0.5, 0.5], [2.01, 1, 0.5],
                      [0.5, -0.99, -0.99], [0.5, -1.01, -1.01],
                      [2, -0.7, -0.7]]
    assert np.all(sphero_cube.is_inside(points_inside))
    assert np.all(~sphero_cube.is_inside(points_outside))

    assert np.all(sphero_cube.is_inside(sphero_cube.polyhedron.vertices))
    sphero_cube.polyhedron.center = [0, 0, 0]
    verts = sphero_cube.polyhedron.vertices
    # Points are inside the convex hull
    assert np.all(sphero_cube.is_inside(verts * 0.99))
    # Points are outside the convex hull but inside the spherical caps
    assert np.all(sphero_cube.is_inside(verts * 1.01))
    # Points are outside the spherical caps
    assert np.all(sphero_cube.is_inside(verts * 2))
    # Points are on the very corners of the spherical caps
    assert np.all(sphero_cube.is_inside(verts * (1 + 2*np.sqrt(1/3))))
    # Points are just outside the very corners of the spherical caps
    assert np.all(~sphero_cube.is_inside(verts * (1 + 2*np.sqrt(1/3) + 1e-6)))
