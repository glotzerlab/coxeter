# https://www.sphinx-doc.org/en/master/development/tutorials/extending_syntax.html
from docutils import nodes
from docutils.parsers.rst import roles
from pathlib import Path


def x3d_model_role(name, rawtext, text, lineno, inliner, options={}, x3d_content=[]):

    url = text.strip()
    with open(Path.cwd() / "build" / "html" / url, "r", encoding="utf-8") as f:
        x3d_content = f.read()
    
    node = nodes.raw('', f'<div class="x3d-model">{x3d_content}</div>', format="html")
    return [node], []


def setup(app):
    roles.register_local_role('x3d-model', x3d_model_role)
    return {'version': '0.1'}