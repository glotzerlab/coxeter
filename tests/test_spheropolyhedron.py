# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats

from conftest import make_sphero_cube
from coxeter.shapes import ConvexSpheropolyhedron


@given(radius=floats(0.1, 1))
def test_volume(radius):
    sphero_cube = make_sphero_cube(radius=radius)
    v_cube = 1
    v_sphere = (4 / 3) * np.pi * radius**3
    v_cyl = 12 * (np.pi * radius**2) / 4
    v_face = sphero_cube.polyhedron.surface_area * radius
    assert np.isclose(sphero_cube.volume, v_cube + v_sphere + v_face + v_cyl)


def test_volume_polyhedron(convex_cube, cube_points):
    """Ensure that zero radius gives the same result as a polyhedron."""
    sphero_cube = make_sphero_cube(radius=0)
    assert np.isclose(sphero_cube.volume, convex_cube.volume)


@given(value=floats(0.1, 1))
def test_set_volume(value):
    sphero_cube = make_sphero_cube(radius=0)
    sphero_cube.volume = value
    assert np.isclose(sphero_cube.volume, value)


@settings(deadline=1000)
@given(radius=floats(0.1, 1))
def test_surface_area(radius):
    sphero_cube = make_sphero_cube(radius=radius)
    sa_cube = 6
    sa_sphere = 4 * np.pi * radius**2
    sa_cyl = 12 * (2 * np.pi * radius) / 4
    assert np.isclose(sphero_cube.surface_area, sa_cube + sa_sphere + sa_cyl)


@given(value=floats(0.1, 1))
def test_set_surface_area(value):
    sphero_cube = make_sphero_cube(radius=0)
    sphero_cube.surface_area = value
    assert np.isclose(sphero_cube.surface_area, value)


def test_surface_area_polyhedron(convex_cube):
    """Ensure that zero radius gives the same result as a polyhedron."""
    sphero_cube = make_sphero_cube(radius=0)
    assert np.isclose(sphero_cube.surface_area, convex_cube.surface_area)


@given(radius=floats(0.1, 1))
def test_mean_curvature(radius):
    sphero_cube = make_sphero_cube(radius=radius)
    h_cube = 3 / 4
    h_sphere = radius
    assert np.isclose(sphero_cube.mean_curvature, h_cube + h_sphere)


def test_mean_curvature_polyhedron(convex_cube, cube_points):
    """Ensure that zero radius gives the same result as a polyhedron."""
    sphero_cube = make_sphero_cube(radius=0)
    assert np.isclose(sphero_cube.mean_curvature, convex_cube.mean_curvature)


@given(value=floats(0.1, 1))
def test_set_mean_curvature(value):
    sphero_cube = make_sphero_cube(radius=0)
    sphero_cube.mean_curvature = value
    assert np.isclose(sphero_cube.mean_curvature, value)


@given(r=floats(0, 1.0))
def test_radius_getter_setter(r):
    sphero_cube = make_sphero_cube(radius=r)
    assert sphero_cube.radius == r
    sphero_cube.radius = r + 1
    assert sphero_cube.radius == r + 1


@given(r=floats(-1000, -1))
def test_invalid_radius(r):
    with pytest.raises(ValueError):
        make_sphero_cube(radius=r)


@settings(deadline=1000)
@given(r=floats(-1000, -1))
def test_invalid_radius_setter(r):
    sphero_cube = make_sphero_cube(1)
    with pytest.raises(ValueError):
        sphero_cube.radius = r


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


def test_repr():
    sphero_cube = make_sphero_cube(radius=1)
    assert str(sphero_cube), str(eval(repr(sphero_cube)))


@pytest.mark.xfail(reason="Maximum rounding radius is shape-dependent.")
@given(r=floats(0.01, 1.0))
def test_to_hoomd(poly, r):
    poly.centroid = [0, 0, 0]
    poly = ConvexSpheropolyhedron(poly.vertices, r)
    dict_keys = ["vertices", "centroid", "sweep_radius", "volume"]
    dict_vals = [
        poly.vertices,
        [0, 0, 0],
        poly.radius,
        poly.volume,
    ]
    hoomd_dict = poly.to_hoomd()
    for key, val in zip(dict_keys, dict_vals):
        assert np.allclose(hoomd_dict[key], val), f"{key}"
