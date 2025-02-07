"""Sphinx extension for rendering X3D models inline."""

from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import roles


def x3d_model_role(name, rawtext, text, lineno, inliner, options={}, x3d_content=[]):
    """Define custom role for loarding X3D models from file into raw HTML nodes."""
    url = text.strip()
    with open(Path.cwd() / "build" / "html" / url, encoding="utf-8") as f:
        x3d_content = f.read()

    node = nodes.raw("", f'<div class="x3d-model">{x3d_content}</div>', format="html")
    return [node], []


def setup(app):
    """Connect the custom module to the Sphinx app."""
    roles.register_local_role("x3d-model", x3d_model_role)
    return {"version": "0.1"}
