# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "coxeter"
copyright = "2015-2026, The Regents of the University of Michigan"
author = "Vyas Ramasubramani"

# The full version, including alpha/beta/rc tags
version = "0.10.0"
release = "0.10.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
sys.path.insert(0, os.path.abspath("./_extensions"))
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "autodocsumm",
    "nbsphinx",
    "x3d_model_extension",
]

# For sphincontrib.bibtex (as of v2.0).
bibtex_bibfiles = ["coxeter.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["table-prism-antiprism.rst"]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "gsd": ("https://gsd.readthedocs.io/en/stable/", None),
    "plato": ("https://plato-draw.readthedocs.io/en/latest/", None),
}

autodoc_default_options = {
    "inherited-members": True,
    "show-inheritance": True,
    "autosummary": True,
}


# The reST default role to use for all documents.
default_role = "any"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def _is_stub(obj):
    """Check whether a member is a base-class stub tagged as unimplemented.

    Stubs are tagged in ``coxeter/shapes/base_classes.py`` by the
    ``_no_default_implementation`` decorator, which sets ``__unimplemented__``
    on the underlying function.
    """
    func = getattr(obj, "fget", None) or obj
    return getattr(func, "__unimplemented__", False)


# The autodoc-skip-member event receives bare member names with no reference
# to the class being documented, so the class currently being documented is
# recorded here while autodoc processes it. Sphinx passes every event argument
# positionally, so the handlers below must accept the full event signatures
# (unused parameters are prefixed with an underscore).
_current_class = {}


def _record_class(obj):
    if isinstance(obj, type):
        _current_class["cls"] = obj


def _record_class_from_bases(_app, _name, obj, _options, _bases):
    _record_class(obj)


def _record_class_from_docstring(_app, obj_type, _name, obj, _options, _lines):
    if obj_type == "class":
        _record_class(obj)


def skip_unimplemented(_app, what, name, obj, skip, _options):
    """Omit unimplemented base-class members from the docs of subclasses.

    The base shape classes define members that raise ``NotImplementedError``
    for shapes that lack an implementation (e.g. the minimal bounding sphere
    of a spheropolyhedron), which is misleading if shown on every subclass.
    Such members are skipped wherever they are merely inherited, along with
    any ``*_radius`` property delegating to an unimplemented member. They
    remain documented on the class that defines them, which is the canonical
    definition of the property.

    See https://github.com/glotzerlab/coxeter/issues/184.
    """
    if what != "class" or skip:
        return None
    cls = _current_class.get("cls")
    if cls is None:
        # No class context available; fall back to the default behavior.
        return None
    if name in vars(cls):
        # Defined by the documented class itself (an implementation or the
        # canonical definition); always keep it.
        return None
    if _is_stub(obj):
        return True
    if name.endswith("_radius"):
        core = getattr(cls, name[: -len("_radius")], None)
        if core is not None and _is_stub(core):
            return True
    return None


def setup(app):
    app.connect("autodoc-process-bases", _record_class_from_bases)
    app.connect("autodoc-process-docstring", _record_class_from_docstring)
    app.connect("autodoc-skip-member", skip_unimplemented)
