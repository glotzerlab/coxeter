"""This module defines various convenience functions for generating shapes.

The methods here provide routes for generating instances of
:class:`~coxeter.shape_classes.Shape` based on certain pre-specified mappings.
"""

from .shape_classes import (
    Circle,
    ConvexPolygon,
    ConvexPolyhedron,
    ConvexSpheropolygon,
    ConvexSpheropolyhedron,
    Ellipse,
    Ellipsoid,
    Polygon,
    Polyhedron,
    Sphere,
)


def from_gsd_type_shapes(params, dimensions=3):  # noqa: C901
    """Create a :class:`~.Shape` from a dict conforming to the GSD schema.

    See :ref:`here <shapes>` for the specification of the schema. Note that the
    schema does not differentiate between 2D and 3D shapes for spheres (vs
    circles) and ellipsoids (vs ellipses) because in context those can be
    inferred from simulation boxes.  To address this ambiguity, this function
    accepts an dimensions parameter that can be used to disambiguate explicitly
    between these two cases.

    Args:
        params (dict):
            The parameters of the shape to construct.
        dimensions (int):
            The dimensionality of the shape (either 2 or 3). Ignored except
            when the shape is a sphere or an ellipsoid, in which case a value
            of 2 is used to indicate generating a
            :class:`~.shape_classes.Circle` or :class:`~.shape_classes.Ellipse`
            instead of a :class:`~.shape_classes.Sphere` or
            :class:`~.shape_classes.Ellipsoid` (Default value: 3).

    Returns:
        list[:class:`~coxeter.shape_classes.Shape`]:
            The desired shape.
    """
    if "type" not in params:
        raise ValueError(
            "The parameters are malformed, there must be a type "
            "key indicating what type of shape this is."
        )

    if params["type"] == "Sphere":
        if dimensions == 2:
            return Circle(params["diameter"] / 2)
        else:
            return Sphere(params["diameter"] / 2)
    elif params["type"] == "Ellipsoid":
        if dimensions == 2:
            return Ellipse(params["a"], params["b"])
        else:
            return Ellipsoid(params["a"], params["b"], params["c"])
    elif params["type"] == "Polygon":
        if "rounding_radius" in params:
            return ConvexSpheropolygon(params["vertices"], params["rounding_radius"])
        else:
            try:
                return ConvexPolygon(params["vertices"])
            except ValueError:
                # If it's not a convex polygon, return a simple polygon.
                return Polygon(params["vertices"])
    elif params["type"] == "ConvexPolyhedron":
        if "rounding_radius" in params:
            return ConvexSpheropolyhedron(params["vertices"], params["rounding_radius"])
        else:
            return ConvexPolyhedron(params["vertices"])
    elif params["type"] == "Mesh":
        return Polyhedron(params["vertices"], params["faces"])
    else:
        raise ValueError("Unsupported shape type.")
