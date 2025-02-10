"""Sphinx extension for rendering X3D models inline."""

from pathlib import Path
from functools import partial
from docutils import nodes
from docutils.parsers.rst import roles


def x3d_model_role(name, rawtext, text, lineno, inliner, options={}, x3d_content=[], app=None):
    """Define custom role for loarding X3D models from file into raw HTML nodes.

    Note: app is a required argument.
    """
    if app is not None:
        url = text.strip()
        builddir = Path(app.outdir)
        with open(builddir / url, encoding="utf-8") as f:
            x3d_content = f.read()

        node = nodes.raw("", f'<div class="x3d-model">{x3d_content}</div>', format="html")
        return [node], []
    else:
        raise ValueError("Pass the Sphinx application object as 'app'.")


def setup(app):
    """Connect the custom module to the Sphinx app."""
    roles.register_local_role("x3d-model", partial(x3d_model_role, app=app))
    return {"version": "0.1"}
