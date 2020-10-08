import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from conftest import make_sphero_cube


@given(radius=floats(0.1, 1))
def test_volume(radius):
    sphero_cube = make_sphero_cube(radius=radius)
    v_cube = 1
    v_sphere = (4 / 3) * np.pi * radius ** 3
    v_cyl = 12 * (np.pi * radius ** 2) / 4
    v_face = sphero_cube.polyhedron.surface_area * radius
    assert np.isclose(sphero_cube.volume, v_cube + v_sphere + v_face + v_cyl)


def test_volume_polyhedron(convex_cube, cube_points):
    """Ensure that zero radius gives the same result as a polyhedron."""
    sphero_cube = make_sphero_cube(radius=0)
    assert sphero_cube.volume == convex_cube.volume


@given(radius=floats(0.1, 1))
def test_surface_area(radius):
    sphero_cube = make_sphero_cube(radius=radius)
    sa_cube = 6
    sa_sphere = 4 * np.pi * radius ** 2
    sa_cyl = 12 * (2 * np.pi * radius) / 4
    assert np.isclose(sphero_cube.surface_area, sa_cube + sa_sphere + sa_cyl)


def test_surface_area_polyhedron(convex_cube):
    """Ensure that zero radius gives the same result as a polyhedron."""
    sphero_cube = make_sphero_cube(radius=0)
    assert sphero_cube.surface_area == convex_cube.surface_area


def test_inside_boundaries():
    sphero_cube = make_sphero_cube(radius=1)

    points_inside = [
        [0, 0, 0],
        [1, 1, 1],
        [-0.01, -0.01, -0.01],
        [2, 0.5, 0.5],
        [2, 1, 0.5],
        [0.5, -0.7, -0.7],
        [-0.57, -0.57, -0.57],
    ]
    points_outside = [
        [-0.99, -0.99, -0.99],
        [-1.01, -1.01, -1.01],
        [2.01, 0.5, 0.5],
        [2.01, 1, 0.5],
        [0.5, -0.99, -0.99],
        [0.5, -1.01, -1.01],
        [2, -0.7, -0.7],
    ]
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
    assert np.all(~sphero_cube.is_inside(verts * 3))
    # Points are on the very corners of the spherical caps
    assert np.all(sphero_cube.is_inside(verts * (1 + 2 * np.sqrt(1 / 3))))
    # Points are just outside the very corners of the spherical caps
    assert np.all(~sphero_cube.is_inside(verts * (1 + 2 * np.sqrt(1 / 3) + 1e-6)))
