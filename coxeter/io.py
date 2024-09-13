# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Import/Export utilities for shape classes.

This module contains functions for saving shapes to disk and creating shapes from
local files. Currently, the following formats are supported:
- Export: OBJ, OFF, STL, PLY, VTK, X3D, HTML

These functions currently only work with `Polyhedron` and its subclasses.
"""

import os
from copy import deepcopy
from xml.etree import ElementTree

import numpy as np

from coxeter import __version__


def to_obj(shape, filename):
    """Save shape to a wavefront OBJ file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Note:
        In OBJ files, vertices in face definitions are indexed from one.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    content = ""
    content += (
        f"# wavefront obj file written by Coxeter "
        f"version {__version__}\n"
        f"# {shape.__class__.__name__}\n\n"
    )

    for v in shape.vertices:
        content += f"v {' '.join([str(coord) for coord in v])}\n"

    content += "\n"

    for f in shape.faces:
        content += f"f {' '.join([str(v_index+1) for v_index in f])}\n"

    content = content[:-1]

    with open(filename, "w") as file:
        file.write(content)


def to_off(shape, filename):
    """Save shape to an Object File Format (OFF) file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    content = ""
    content += (
        f"OFF\n# OFF file written by Coxeter "
        f"version {__version__}\n"
        f"# {shape.__class__.__name__}\n"
    )

    content += f"{len(shape.vertices)} f{len(shape.faces)} " f"{len(shape.edges)}\n"

    for v in shape.vertices:
        content += f"{' '.join([str(coord) for coord in v])}\n"

    for f in shape.faces:
        content += f"{len(f)} {' '.join([str(v_index) for v_index in f])}\n"

    content = content[:-1]

    with open(filename, "w") as file:
        file.write(content)


def to_stl(shape, filename):
    """Save shape to a stereolithography (STL) file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Note:
        The output file is ASCII-encoded.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    with open(filename, "w") as file:
        # Ensure shape is not mutated
        shape = deepcopy(shape)

        # Shift vertices so all coordinates are positive
        mins = np.amin(a=shape.vertices, axis=0)
        for i, m in enumerate(mins):
            if m < 0:
                shape.centroid[i] -= m

        # Write data
        vs = shape.vertices
        file.write(f"solid {shape.__class__.__name__}\n")

        for f in shape.faces:
            # Decompose face into triangles
            # ref: https://stackoverflow.com/a/66586936/15426433
            triangles = [[vs[f[0]], vs[b], vs[c]] for b, c in zip(f[1:], f[2:])]

            for t in triangles:
                n = np.cross(t[1] - t[0], t[2] - t[1])  # order?

                file.write(f"facet normal {n[0]} {n[1]} {n[2]}\n" f"\touter loop\n")
                for point in t:
                    file.write(f"\t\tvertex {point[0]} {point[1]} {point[2]}\n")

                file.write("\tendloop\nendfacet\n")

        file.write(f"endsolid {shape.__class__.__name__}")


def to_ply(shape, filename):
    """Save shape to a Polygon File Format (PLY) file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Note:
        The output file is ASCII-encoded.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    content = ""
    content += (
        f"ply\nformat ascii 1.0\n"
        f"comment PLY file written by Coxeter version {__version__}\n"
        f"comment {shape.__class__.__name__}\n"
        f"element vertex {len(shape.vertices)}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"element face {len(shape.faces)}\n"
        f"property list uchar uint vertex_indices\n"
        f"end_header\n"
    )

    for v in shape.vertices:
        content += f"{' '.join([str(coord) for coord in v])}\n"

    for f in shape.faces:
        content += f"{len(f)} {' '.join([str(int(v_index)) for v_index in f])}\n"

    content = content[:-1]

    with open(filename, "w") as file:
        file.write(content)


def to_x3d(shape, filename):
    """Save shape to an Extensible 3D (X3D) file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    # TODO: translate shape so that its centroid is at the origin

    # Parent elements
    root = ElementTree.Element(
        "x3d",
        attrib={
            "profile": "Interchange",
            "version": "4.0",
            "xmlns:xsd": "http://www.w3.org/2001/XMLSchema-instance",
            "xsd:schemaLocation": "http://www.web3d.org/specifications/x3d-4.0.xsd",
        },
    )
    x3d_scene = ElementTree.SubElement(root, "Scene")
    x3d_shape = ElementTree.SubElement(
        x3d_scene, "shape", attrib={"DEF": f"{shape.__class__.__name__}"}
    )

    x3d_appearance = ElementTree.SubElement(x3d_shape, "Appearance")
    ElementTree.SubElement(
        x3d_appearance, "Material", attrib={"diffuseColor": "#6495ED"}
    )

    # Geometry data
    point_indices = list(range(sum([len(f) for f in shape.faces])))
    prev_index = 0
    for f in shape.faces:
        point_indices.insert(len(f) + prev_index, -1)
        prev_index += len(f) + 1

    points = [v for f in shape.faces for v_index in f for v in shape.vertices[v_index]]

    x3d_indexedfaceset = ElementTree.SubElement(
        x3d_shape,
        "IndexedFaceSet",
        attrib={"coordIndex": " ".join([str(c_index) for c_index in point_indices])},
    )
    ElementTree.SubElement(
        x3d_indexedfaceset,
        "Coordinate",
        attrib={"point": " ".join([str(p) for p in points])},
    )

    # Write to file
    ElementTree.ElementTree(root).write(filename, encoding="UTF-8")


def to_vtk(shape, filename):
    """Save shape to a legacy VTK file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    content = ""
    # Title and Header
    content += (
        f"# vtk DataFile Version 3.0\n"
        f"{shape.__class__.__name__} created by "
        f"Coxeter version {__version__}\n"
        f"ASCII\n"
    )

    # Geometry
    content += f"DATASET POLYDATA\n" f"POINTS {len(shape.vertices)} float\n"
    for v in shape.vertices:
        content += f"{v[0]} {v[1]} {v[2]}\n"

    num_points = len(shape.faces)
    num_connections = sum([len(f) for f in shape.faces])
    content += f"POLYGONS {num_points} {num_points + num_connections}\n"
    for f in shape.faces:
        content += f"{len(f)} {' '.join([str(v_index) for v_index in f])}\n"
    content = content.rstrip("\n")

    # Write file
    with open(filename, "wb") as file:
        file.write(content.encode("ascii"))


def to_html(shape, filename):
    """Save shape to an HTML file.

    This method calls shape.to_x3d to create a temporary X3D file, then
    parses that X3D file and creates an HTML file in its place.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    # Create, parse, and remove x3d file
    to_x3d(shape, filename)
    x3d = ElementTree.parse(filename)
    os.remove(filename)

    # HTML Head
    html = ElementTree.Element("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml"})
    head = ElementTree.SubElement(html, "head")
    script = ElementTree.SubElement(
        head,
        "script",
        attrib={"type": "text/javascript", "src": "http://x3dom.org/release/x3dom.js"},
    )
    script.text = " "  # ensures the tag is not shape-closing
    ElementTree.SubElement(
        head,
        "link",
        attrib={
            "rel": "stylesheet",
            "type": "text/css",
            "href": "http://x3dom.org/release/x3dom.css",
        },
    )

    # HTML body
    body = ElementTree.SubElement(html, "body")
    body.append(x3d.getroot())

    # Write file
    with open(filename, "w") as file:
        file.write("<!DOCTYPE html>")
        file.write(ElementTree.tostring(html, encoding="unicode"))
