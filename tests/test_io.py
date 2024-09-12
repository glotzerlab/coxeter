# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

# Copyright (c) 2024 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Regression tests for the I/O module."""

import filecmp
import tempfile
from pathlib import Path

import pytest

from coxeter import io
from coxeter.shapes import ConvexPolyhedron, Polyhedron

EXPORT_FUNCS_BY_FILE_TYPE = {
    "obj": io.to_obj,
    "off": io.to_off,
    "stl": io.to_stl,
    "ply": io.to_ply,
    "x3d": io.to_x3d,
    "vtk": io.to_vtk,
    "html": io.to_html,
}

SHAPES_BY_NAME = {
    "polyhedron": Polyhedron(
        vertices=[
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        faces=[
            [0, 1, 3, 2],
            [0, 2, 6, 4],
            [4, 6, 7, 5],
            [1, 5, 7, 3],
            [0, 4, 5, 1],
            [6, 2, 3, 7],
        ],
    ),
    "convex_polyhedron": ConvexPolyhedron(
        vertices=[
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    ),
}

CONTROL_DIR = Path("/home/joseph/GlotzerGroup/coxeter/tests/control")


@pytest.mark.parametrize(
    "file_type,export_func,shape_name",
    [
        (ft, func, name)
        for ft, func in EXPORT_FUNCS_BY_FILE_TYPE.items()
        for name in SHAPES_BY_NAME.keys()
    ],
)
def test_regression(file_type, export_func, shape_name):
    """Check that export functions yield files identical to the control."""
    control_file_path = CONTROL_DIR / f"{shape_name}.{file_type}"
    shape = SHAPES_BY_NAME[shape_name]

    with tempfile.TemporaryDirectory(dir=CONTROL_DIR) as tempdir:
        test_file_path = Path(tempdir) / f"test_{shape_name}.{file_type}"
        export_func(shape=shape, filename=test_file_path)

        assert filecmp.cmp(control_file_path, test_file_path, shallow=False), (
            f"During regression testing with the shape '{shape_name}' and "
            f"file type '{file_type}', {control_file_path.name} and "
            f"{test_file_path.name} were found to be not equivalent."
        )


if __name__ == "__main__":
    # Generate new control files
    for name in SHAPES_BY_NAME.keys():
        for ft, func in EXPORT_FUNCS_BY_FILE_TYPE.items():
            control_file_path = CONTROL_DIR / f"{name}.{ft}"
            func(shape=SHAPES_BY_NAME[name], filename=control_file_path)
