"""Sphinx extension for building interactive polyhedron tables from coxeter.

Examples
--------
A table of all 92 Johnson solids on a single page:

.. code-block:: rst

   .. polyhedron-table::
      :family: 10.1126/science.1220869

      J01-J92

A table from any (parametric) shape family:

.. code-block:: rst

   .. polyhedron-table::
      :family: coxeter.families.UniformPrismFamily

      3-8
      10 | Decagonal Prism

A single model embedded inline in text:

.. code-block:: rst

   :polyhedron:`coxeter.families.PlatonicFamily.get_shape("Cube")`
"""

import html
import json
import re
from importlib import import_module

import numpy as np
from docutils import nodes
from docutils.parsers.rst import Directive, directives, roles

try:
    import coxeter
except ImportError:  # pragma: no cover - docs of other projects never use this
    coxeter = None

# Number of decimal places kept in embedded vertex coordinates.
_COORDINATE_PRECISION = 5

# Columns supported by the polyhedron-table directive, mapped to header labels.
_COLUMNS = {
    "id": "ID",
    "name": "Name",
    "vertices": "Vertices",
    "faces": "Faces",
    "edges": "Edges",
    "model": "Model",
}

_DEFAULT_COLUMNS = ("id", "name", "vertices", "faces", "model")

# Prefixes used to name shapes of parametric families, e.g. a prism built
# from a 7-gon is a "Heptagonal Prism".
_NUMERALS = {
    3: "Triangular",
    4: "Square",
    5: "Pentagonal",
    6: "Hexagonal",
    7: "Heptagonal",
    8: "Octagonal",
    9: "Nonagonal",
    10: "Decagonal",
    11: "Hendecagonal",
    12: "Dodecagonal",
    13: "Tridecagonal",
    14: "Tetradecagonal",
    15: "Pentadecagonal",
    16: "Hexadecagonal",
    17: "Heptadecagonal",
    18: "Octadecagonal",
    19: "Enneadecagonal",
    20: "Icosagonal",
}

# Nouns inferred from family class names for automatic shape naming. Longer
# suffixes must be checked first.
_FAMILY_NOUNS = [
    ("antiprismfamily", "Antiprism"),
    ("dipyramidfamily", "Dipyramid"),
    ("trapezohedronfamily", "Trapezohedron"),
    ("prismfamily", "Prism"),
    ("pyramidfamily", "Pyramid"),
]

_RANGE_RE = re.compile(
    r"^(?P<prefix>[A-Za-z]*)(?P<start>\d+)-(?P<prefix2>[A-Za-z]*)?(?P<end>\d+)$"
)
_INT_RANGE_RE = re.compile(r"^\d+$")


def _shape_payload(shape, name):
    """Serialize a shape into the compact JSON payload consumed by the viewer.

    Args:
        shape: Any coxeter shape with ``vertices`` and ``faces`` properties.
        name (str): A human-readable name used for accessibility labels.

    Returns
    -------
        dict: The payload passed to the client-side viewer.
    """
    vertices = np.asarray(shape.vertices, dtype=float)
    # Round for compactness and normalize negative zeros (which serialize as
    # "-0.0" and bloat the payload without adding information).
    vertices = np.round(vertices, _COORDINATE_PRECISION) + 0.0
    return {
        "n": name,
        "v": vertices.tolist(),
        "f": [[int(i) for i in face] for face in shape.faces],
    }


def _model_html(payload, size=None):
    """Build the raw HTML placeholder for a single interactive model.

    Args:
        payload (dict): A payload produced by :func:`_shape_payload`.
        size (int): Optional cell size in pixels.

    Returns
    -------
        str: An HTML snippet containing the payload and a fallback label.
    """
    # "</" cannot appear in these payloads, but escape anyway so that the
    # JSON can never terminate the enclosing <script> element early.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    style = f' style="--coxeter-model-size: {size}px"' if size else ""
    name = html.escape(str(payload["n"]))
    return (
        f'<span class="polyhedron-model"{style}>'
        f'<script type="application/json" class="polyhedron-data">{data}</script>'
        f'<span class="polyhedron-fallback">{name}</span>'
        "</span>"
    )


def _expand_specs(specs):
    """Expand a list of shape specs into (identifier, label) pairs.

    Each spec is a single identifier (e.g. ``J04`` or ``7``), an inclusive
    range (e.g. ``J01-J08`` or ``3-8``), or an identifier followed by a pipe
    and a display name (e.g. ``7 | Heptagonal Prism``).

    Args:
        specs (list[str]): The content lines of the directive.

    Returns
    -------
        list[tuple[str, str | None]]: Identifiers and optional labels.
    """
    expanded = []
    for spec in specs:
        spec = spec.strip()
        if not spec or spec.startswith("#"):
            continue
        label = None
        if "|" in spec:
            spec, label = (part.strip() for part in spec.split("|", 1))
        match = _RANGE_RE.match(spec)
        if match is not None:
            prefix, prefix2 = match.group("prefix"), match.group("prefix2") or ""
            if prefix.lower() != prefix2.lower():
                raise ValueError(f"Range '{spec}' mixes different prefixes.")
            start, end = int(match.group("start")), int(match.group("end"))
            if start > end or end - start > 1000:
                raise ValueError(f"Invalid or suspicious range '{spec}'.")
            width = len(match.group("start"))
            identifiers = [f"{prefix}{i:0{width}d}" for i in range(start, end + 1)]
        elif _INT_RANGE_RE.match(spec):
            identifiers = [spec]
        elif re.match(r"^[A-Za-z]*\d+$", spec):
            identifiers = [spec]
        else:
            raise ValueError(f"Cannot parse shape spec '{spec}'.")
        expanded.extend((identifier, label) for identifier in identifiers)
    return expanded


class PolyhedronTableDirective(Directive):
    """Generate a table of interactive polyhedron models from a shape family.

    The directive instantiates each requested shape with coxeter at build
    time and renders a standard documentation table whose model column
    contains an interactive viewer cell.
    """

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "family": directives.unchanged_required,
        "columns": directives.unchanged,
        "id-header": directives.unchanged,
        "size": directives.positive_int,
        "class": directives.class_option,
    }

    def _resolve_family(self, spec):
        """Resolve the :family: option to a shape family object.

        Args:
            spec (str): Either a DOI known to
                :data:`coxeter.families.DOI_SHAPE_REPOSITORIES` or the dotted
                path of a shape family class.

        Returns
        -------
            tuple: The family object and whether it is tabulated by DOI.
        """
        if coxeter is None:
            raise self.error(
                "The coxeter package is required to build polyhedron tables."
            )
        if "/" in spec:
            try:
                family = coxeter.families.DOI_SHAPE_REPOSITORIES[spec][0]
            except KeyError:
                raise self.error(
                    f"No shape repository is known for DOI '{spec}'."
                ) from None
            return family, True
        module_name, _, class_name = spec.rpartition(".")
        try:
            family = getattr(import_module(module_name), class_name)
        except (ImportError, AttributeError):
            raise self.error(f"Cannot import shape family '{spec}'.") from None
        return family, False

    def _default_name(self, family, identifier):
        """Guess a display name for a shape of a parametric family.

        Args:
            family: The family class.
            identifier (str): The parameter value (e.g. ``"7"``).

        Returns
        -------
            str or None: A name such as "Heptagonal Prism", if inferable.
        """
        class_name = family.__name__.lower() if isinstance(family, type) else ""
        noun = next((n for s, n in _FAMILY_NOUNS if class_name.endswith(s)), None)
        numeral = _NUMERALS.get(int(identifier))
        if noun and numeral:
            return f"{numeral} {noun}"
        return None

    def run(self):
        """Build the table node for the requested shapes."""
        family_spec = self.options.get("family")
        if family_spec is None:
            raise self.error(
                "The polyhedron-table directive requires a :family: option."
            )
        try:
            specs = _expand_specs(list(self.content))
        except ValueError as error:
            raise self.error(str(error)) from None
        if not specs:
            raise self.error(
                "The polyhedron-table directive requires shape specs "
                "(e.g. 'J01-J92' or '3-8') as its content."
            )
        family, tabulated = self._resolve_family(family_spec)

        column_names = self.options.get("columns", " ".join(_DEFAULT_COLUMNS))
        columns = []
        for token in re.split(r"[,\s]+", column_names.strip()):
            if not token:
                continue
            if token.lower() not in _COLUMNS:
                raise self.error(
                    f"Unknown column '{token}'. Valid columns are: "
                    f"{', '.join(_COLUMNS.values())}."
                )
            columns.append(token.lower())
        if "model" not in columns:
            raise self.error(
                "A polyhedron-table without a Model column renders no models."
            )

        rows = []
        for identifier, label in specs:
            if tabulated:
                if identifier not in family.data:
                    raise self.error(
                        f"Family {family_spec} has no shape '{identifier}'."
                    )
                name = family.data[identifier]["name"]
                argument = identifier
            else:
                try:
                    value = int(identifier)
                except ValueError:
                    value = identifier
                argument = value
                name = label or self._default_name(family, identifier) or str(value)
            try:
                shape = family.get_shape(argument)
            except Exception as error:
                raise self.error(
                    f"Family {family_spec} cannot build shape '{identifier}': {error}"
                ) from None
            cells = {
                "id": identifier,
                "name": name,
                "vertices": shape.num_vertices,
                "faces": shape.num_faces,
                "edges": shape.num_edges,
                "model": _model_html(
                    _shape_payload(shape, name), self.options.get("size")
                ),
            }
            rows.append([cells[column] for column in columns])

        header_labels = [_COLUMNS[column] for column in columns]
        if "id" in columns and self.options.get("id-header"):
            header_labels[columns.index("id")] = self.options["id-header"]
        return [
            _build_table(
                header_labels,
                rows,
                classes=list(self.options.get("class", [])),
                wide=[column == "model" for column in columns],
            )
        ]


def _build_table(headers, rows, classes=(), wide=()):
    """Assemble a docutils table node.

    Args:
        headers (list[str]): Header labels.
        rows (list[list]): One list of cell contents (strings or raw HTML)
            per row.
        classes (list[str]): Additional classes for the table element.
        wide (list[bool]): Whether each column is wide (drives colwidths).

    Returns
    -------
        docutils.nodes.table: The assembled table.
    """
    table = nodes.table(classes=["polyhedron-table", *classes])
    tgroup = nodes.tgroup(cols=len(headers))
    table += tgroup
    for is_wide in wide:
        tgroup += nodes.colspec(colwidth=3 if is_wide else 1)
    thead = nodes.thead()
    tgroup += thead
    header_row = nodes.row()
    thead += header_row
    for header in headers:
        entry = nodes.entry()
        entry += nodes.paragraph(text=header)
        header_row += entry
    tbody = nodes.tbody()
    tgroup += tbody
    for cells in rows:
        row = nodes.row()
        tbody += row
        for cell in cells:
            entry = nodes.entry()
            if isinstance(cell, str) and cell.startswith("<"):
                entry += nodes.raw("", cell, format="html")
            else:
                entry += nodes.paragraph(text=str(cell))
            row += entry
    return table


def polyhedron_role(
    name,
    rawtext,
    text,
    lineno,
    inliner,
    options={},  # noqa: B006 (signature required by docutils)
    content=[],  # noqa: B006
):
    """Render a single shape as an interactive inline model.

    The role text is a Python expression, evaluated at build time with
    ``coxeter`` in scope, that evaluates to a shape, e.g.
    ``:polyhedron:`coxeter.families.UniformAntiprismFamily.get_shape(4)``.
    """
    expression = text.strip()
    try:
        # The role only runs on trusted documentation source files.
        shape = eval(expression, {"coxeter": coxeter})  # noqa: S307, B307
        payload = _shape_payload(shape, expression)
    except Exception as error:
        message = inliner.reporter.error(
            f"Cannot evaluate polyhedron expression '{expression}': {error}",
            line=lineno,
        )
        return [nodes.literal(rawtext, expression)], [message]
    return [nodes.raw("", _model_html(payload), format="html")], []


def setup(app):
    """Connect the extension to the Sphinx application."""
    app.add_directive("polyhedron-table", PolyhedronTableDirective)
    roles.register_local_role("polyhedron", polyhedron_role)
    app.add_css_file("css/polyhedron_models.css")
    app.add_js_file("js/polyhedron_models.js", type="module", loading_method="defer")
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
