# Copyright (c) 2015-2025 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import numpy as np
import pytest

from coxeter.families import ArchimedeanFamily, PlatonicFamily
from coxeter.shapemoves import vertex_truncate


def shapeeq(
    poly1,
    poly2,
    test_vertices=False,
    test_inertia_tensor=True,
    test_surface_volume=True,
    test_curvature=True,
):
    vx1, vx2 = poly1.vertices, poly2.vertices
    assert poly1.num_vertices == poly2.num_vertices, (
        f"Polyhedra do not have the same number of vertices "
        f"({poly1.num_vertices} vs. {poly2.num_vertices})"
    )

    ex1, ex2 = poly1.edges, poly2.edges
    assert np.shape(ex1) == np.shape(ex2), (
        f"Polyhedra do not have the same number of edges "
        f"({np.shape(ex1)[0]} vs. {np.shape(ex2)[0]})"
    )

    fx1, fx2 = poly1.faces, poly2.faces
    assert poly1.num_faces == poly2.num_faces, (
        f"Polyhedra do not have the same number of faces "
        f"({poly1.num_faces} vs. {poly2.num_faces})"
    )

    if test_inertia_tensor:
        poly1.volume, poly2.volume = 1, 1
        assert np.allclose(poly1.inertia_tensor, poly2.inertia_tensor), (
            f"Inertia tensors do not match: {poly1.inertia_tensor.round(10)} vs. "
            f"{poly2.inertia_tensor.round(10)}."
        )

    if test_vertices:
        for vertex in vx1:
            assert np.any(
                [np.isclose(vertex, vertex2) for vertex2 in vx2]
            ), f"Vertex {vertex} not found in poly2."

    for edge in ex1:
        assert edge in ex2 or edge[::-1] in ex2, f"Edge {edge} not found in poly2."

    for face in fx1:
        found = any(
            len(face) == len(face2) and all(vert in face for vert in face2)
            for face2 in fx2
        )
        assert found, f"Face {face} not found in poly2."

    if test_surface_volume:
        assert np.isclose(
            poly1.volume / poly1.surface_area, poly2.volume / poly2.surface_area
        ), (
            f"{poly1.volume / poly1.surface_area} vs. "
            f"{poly2.volume / poly2.surface_area}"
        )

    if test_curvature:
        assert np.isclose(poly1.mean_curvature, poly2.mean_curvature)

    return True


@pytest.mark.parametrize(
    "poly_name", ["Tetrahedron", "Cube", "Octahedron", "Dodecahedron", "Icosahedron"]
)
def test_truncation(poly_name):
    uniform_truncation_depths = {
        "Tetrahedron": 1 / 3,
        "Cube": 1 / (2 + np.sqrt(2)),
        "Octahedron": 1 / 3,
        "Dodecahedron": 1 / (2 + (1 + np.sqrt(5)) / 2),
        "Icosahedron": 1 / 3,
    }
    untruncated_poly = PlatonicFamily.get_shape(poly_name)
    truncated_poly = ArchimedeanFamily.get_shape(f"Truncated {poly_name}")
    test_poly = vertex_truncate(
        untruncated_poly,
        t=uniform_truncation_depths[poly_name],
    )
    assert shapeeq(truncated_poly, test_poly)
