# Copyright (c) 2015-2025 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Shape-move operators, as defined by geometers John Conway and George Hart."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import HalfspaceIntersection

from coxeter.shapes import ConvexPolyhedron


def _truncate(poly: ConvexPolyhedron, t: float, degrees=None, filter_unique=False):
    # Define the distance along each edge between original vertices and new vertices
    edge_endpoint_offsets = (poly.edge_vectors[..., None] * [t, -t]).transpose(0, 2, 1)
    # Compute new point locations from offsets and original points
    new_vertices = poly.vertices[poly.edges] + edge_endpoint_offsets

    if degrees is not None:
        # Compute the degree of each vertex of the polyhedron
        vertex_degrees = np.bincount(poly.edges.ravel())
        # Mask vertices to be truncated (vertex degree in input degree set)
        degree_mask = np.any(vertex_degrees[:, None] == np.array(degrees), axis=1)
        vertices_to_truncate = np.where(degree_mask)[0]
        edge_endpoint_mask = np.isin(poly.edges, vertices_to_truncate)

        # Take only vertices of correct degree and merge in original vertices
        new_vertices = np.vstack(
            [new_vertices[edge_endpoint_mask], poly.vertices[~degree_mask]]
        )

    # Reshape back down to an N,3 array and prune duplicates
    return np.unique(new_vertices.reshape(-1, 3), axis=0)


def vertex_truncate(poly: ConvexPolyhedron, t: float, degrees=None):
    """Truncate the vertices of a polyhedron.

    .. important::

        Conway's original definition of vertex truncation is only well-defined for
        vertices connecting a single type of incoming edge (like the Platonic and
        Archimedean solids, for example). This method will work for arbitrary convex
        geometries, but may result in more than one face for each truncated vertex.

    Args:
        poly (ConvexPolyhedron): Shape to truncate.
        t (float):
            Truncation depth as a fraction of initial edge length. Must be in [0.0, 0.5]
        degrees (None | np.array | list, optional):
            Vertex degrees to truncate. Defaults to None, which truncates all vertices.

    Returns
    -------
        ConvexPolyhedron: Truncated polyhedron
    """
    new_vertices = _truncate(poly, t, degrees=degrees)
    new_poly = ConvexPolyhedron(new_vertices)
    new_poly.volume = poly.volume
    return new_poly


def dual(poly: ConvexPolyhedron):
    """Dual polyhedron as defined by Conway, computed using halfspaces.

    Args:
        poly (ConvexPolyhedron): Shape to compute dual of

    Returns
    -------
        ConvexPolyhedron: Dual polyhedron
    """
    dual = HalfspaceIntersection(
        halfspaces=poly._simplex_equations,
        interior_point=poly.centroid,
        qhull_options="H Qt",
    )
    try:
        dual_vertices = dual.dual_points[dual.dual_vertices]
    except ValueError:
        # If the halfspace intersection has double points, try manually
        dual_vertices = np.unique(dual.dual_points, axis=1)
    new_poly = ConvexPolyhedron(dual_vertices)
    new_poly.volume = poly.volume
    return new_poly


def kis(poly: ConvexPolyhedron, k: float, degrees=None | ArrayLike):
    """Pyramid-augment apolyhedron as defined by Conway's kis operator.

    This method is implemented as the sequence of operators `dtd` to ensure a pyramid of
    the proper height is raised on each face.

    Args:
        poly (ConvexPolyhedron):
            Shape to pyramid augment.
        k (float):
            Pyramid height as a fraction of initial edge length. Should be in [0, 0.5].
        degrees (None | ArrayLike, optional):
            Face degrees to augment. Defaults to None.
        normalize (bool, optional):
            Whether to normalize output volume to match input. Defaults to True.

    Returns
    -------
        ConvexPolyhedron: Augmented polyhedron polyhedron
    """
    new_poly = dual(ConvexPolyhedron(vertex_truncate(dual(poly), k, degrees=degrees)))
    new_poly.volume = poly.volume
    return new_poly
