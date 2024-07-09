# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import tempfile

import pytest

import coxeter

try:
    import matplotlib

    matplotlib.use("Agg")
    import plato.draw.matplotlib  # noqa: F401

    MATPLOTLIB_PLATO_AVAILABLE = True
except ImportError:
    MATPLOTLIB_PLATO_AVAILABLE = False


@pytest.mark.skipif(
    not MATPLOTLIB_PLATO_AVAILABLE,
    reason="plato and matplotlib are required for this test.",
)
def test_draw_circle():
    circle = coxeter.shapes.Circle(1)
    scene = circle.to_plato_scene("matplotlib", scene_kwargs=dict(zoom=10))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        scene.save(tmp.name)


@pytest.mark.skipif(
    not MATPLOTLIB_PLATO_AVAILABLE,
    reason="plato and matplotlib are required for this test.",
)
def test_draw_polygon():
    polygon = coxeter.shapes.Polygon([[0, 0], [1, 0], [0, 1]])
    scene = polygon.to_plato_scene("matplotlib", scene_kwargs=dict(zoom=10))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        scene.save(tmp.name)


@pytest.mark.skipif(
    not MATPLOTLIB_PLATO_AVAILABLE,
    reason="plato and matplotlib are required for this test.",
)
def test_draw_convex_polygon():
    polygon = coxeter.shapes.ConvexPolygon([[0, 0], [1, 0], [0, 1]])
    scene = polygon.to_plato_scene("matplotlib", scene_kwargs=dict(zoom=10))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        scene.save(tmp.name)


@pytest.mark.skipif(
    not MATPLOTLIB_PLATO_AVAILABLE,
    reason="plato and matplotlib are required for this test.",
)
def test_draw_spheropolygon():
    spheropolygon = coxeter.shapes.ConvexSpheropolygon([[0, 0], [1, 0], [0, 1]], 0.3)
    scene = spheropolygon.to_plato_scene("matplotlib", scene_kwargs=dict(zoom=10))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        scene.save(tmp.name)


@pytest.mark.skipif(
    not MATPLOTLIB_PLATO_AVAILABLE,
    reason="plato and matplotlib are required for this test.",
)
def test_draw_sphere():
    sphere = coxeter.shapes.Sphere(1)
    scene = sphere.to_plato_scene("matplotlib", scene_kwargs=dict(zoom=10))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        scene.save(tmp.name)


@pytest.mark.skipif(
    not MATPLOTLIB_PLATO_AVAILABLE,
    reason="plato and matplotlib are required for this test.",
)
def test_draw_convex_polyhedron():
    cube = coxeter.shapes.ConvexPolyhedron(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ]
    )
    scene = cube.to_plato_scene("matplotlib", scene_kwargs=dict(zoom=10))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        scene.save(tmp.name)
