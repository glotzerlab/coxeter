# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

# Copyright (c) 2024 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Regression tests for the I/O module."""

import tempfile
from pathlib import Path

import pytest

from coxeter import io
from coxeter.families import JohnsonFamily
from coxeter.shapes import Polyhedron


def compare_text_files(file_path_1, file_path_2):
    """Check whether two text files have identical contents, ignoring different
    newline characters."""
    with open(file_path_1) as file1, open(file_path_2) as file2:
        file1_contents, file2_contents = file1.readlines(), file2.readlines()
        assert file1_contents == file2_contents


EXPORT_FUNCS_BY_FILE_TYPE = {
    "obj": io.to_obj,
    "off": io.to_off,
    "stl": io.to_stl,
    "ply": io.to_ply,
    "x3d": io.to_x3d,
    "vtk": io.to_vtk,
    "html": io.to_html,
}

epyr5 = JohnsonFamily.get_shape("Elongated Pentagonal Pyramid")

SHAPES_BY_NAME = {
    "polyhedron": Polyhedron(epyr5.vertices, epyr5.faces),
    "convex_polyhedron": epyr5,
}

CONTROL_DIR = Path("tests/control")


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

        compare_text_files(control_file_path, test_file_path)


if __name__ == "__main__":
    # Generate new control files
    for name in SHAPES_BY_NAME.keys():
        for ft, func in EXPORT_FUNCS_BY_FILE_TYPE.items():
            control_file_path = CONTROL_DIR / f"{name}.{ft}"
            func(shape=SHAPES_BY_NAME[name], filename=control_file_path)
