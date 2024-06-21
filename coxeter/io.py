import os
import xml.etree.ElementTree as ET

import numpy as np

from coxeter import __version__


def to_obj(shape, filename):
    """Save Polyhedron to a wavefront OBJ file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Note:
        In OBJ files, vertices in face definitions are indexed from one.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    with open(filename, "w") as file:
        file.write(
            f"# wavefront obj file written by Coxeter "
            f"version {__version__}\n"
            f"# {shape.__class__.__name__}\n\n"
        )

        for v in shape.vertices:
            file.write(f"v {' '.join([str(i) for i in v])}\n")

        file.write("\n")

        for f in shape.faces:
            file.write(f"f {' '.join([str(i+1) for i in f])}\n")


def to_off(shape, filename):
    """Save Polyhedron to an Object File Format (OFF) file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    with open(filename, "w") as file:
        file.write(
            f"OFF\n# OFF file written by Coxeter "
            f"version {__version__}\n"
            f"# {shape.__class__.__name__}\n"
        )

        file.write(
            f"{len(shape.vertices)} f{len(shape.faces)} " f"{len(shape.edges)}\n"
        )

        for v in shape.vertices:
            file.write(f"{' '.join([str(i) for i in v])}\n")

        for f in shape.faces:
            file.write(f"{len(f)} {' '.join([str(i) for i in f])}\n")


def to_stl(shape, filename):
    """Save Polyhedron to a stereolithography (STL) file.

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
    """Save Polyhedron to a Polygon File Format (PLY) file.

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
        file.write(
            f"ply\nformat ascii 1.0\n"
            f"comment PLY file written by Coxeter "
            f"version {__version__}\n"
            f"comment {shape.__class__.__name__}\n"
            f"element vertex {len(shape.vertices)}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"element face {len(shape.faces)}\n"
            f"property list uchar uint vertex_indices\n"
            f"end_header\n"
        )

        for v in shape.vertices:
            file.write(f"{' '.join([str(i) for i in v])}\n")

        for f in shape.faces:
            file.write(f"{len(f)} {' '.join([str(int(i)) for i in f])}\n")


def to_x3d(shape, filename):
    """Save Polyhedron to an Extensible 3D (X3D) file.

    Args:
        filename (str, pathlib.Path, or os.PathLike):
            The name or path of the output file, including the extension.

    Raises
    ------
        OSError: If open() encounters a problem.
    """
    # TODO: translate shape so that its centroid is at the origin

    # Parent elements
    root = ET.Element(
        "x3d",
        attrib={
            "profile": "Interchange",
            "version": "4.0",
            "xmlns:xsd": "http://www.w3.org/2001/XMLSchema-instance",
            "xsd:noNamespaceSchemaLocation": "http://www.web3d.org/specifications/x3d-4.0.xsd",
        },
    )
    x3d_scene = ET.SubElement(root, "Scene")
    x3d_shape = ET.SubElement(
        x3d_scene, "shape", attrib={"DEF": f"{shape.__class__.__name__}"}
    )

    x3d_appearance = ET.SubElement(x3d_shape, "Appearance")
    x3d_material = ET.SubElement(
        x3d_appearance, "Material", attrib={"diffuseColor": "#6495ED"}
    )

    # Geometry data
    coordinate_indices = list(range(sum([len(f) for f in shape.faces])))
    prev_index = 0
    for f in shape.faces:
        coordinate_indices.insert(len(f) + prev_index, -1)
        prev_index += len(f) + 1

    coordinate_points = [v for f in shape.faces for i in f for v in shape.vertices[i]]

    x3d_indexedfaceset = ET.SubElement(
        x3d_shape,
        "IndexedFaceSet",
        attrib={"coordIndex": " ".join([str(i) for i in coordinate_indices])},
    )
    x3d_coordinate = ET.SubElement(
        x3d_indexedfaceset,
        "Coordinate",
        attrib={"point": " ".join([str(i) for i in coordinate_points])},
    )

    # Write to file
    ET.ElementTree(root).write(filename, encoding="UTF-8")


def to_vtk(shape, filename):
    """Save Polyhedron to a legacy VTK file.

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
        content += f"{len(f)} {' '.join([str(i) for i in f])}\n"

    # Write file
    with open(filename, "wb") as file:
        file.write(content.encode("ascii"))


def to_html(shape, filename):
    """Save Polyhedron to an HTML file.

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
    x3d = ET.parse(filename)
    os.remove(filename)

    # HTML Head
    html = ET.Element("html")
    head = ET.SubElement(html, "head")
    script = ET.SubElement(
        head,
        "script",
        attrib={"type": "text/javascript", "src": "http://x3dom.org/release/x3dom.js"},
    )
    script.text = " "  # ensures the tag is not shape-closing
    link = ET.SubElement(
        head,
        "link",
        attrib={
            "rel": "stylesheet",
            "type": "text/css",
            "href": "http://x3dom.org/release/x3dom.css",
        },
    )

    # HTML body
    body = ET.SubElement(html, "body")
    body.append(x3d.getroot())

    # Write file
    ET.ElementTree(html).write(filename, encoding="UTF-8")