# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Define the abstract base class for all shape families.

This module defines the :class:`~.ShapeFamily` class, which defines the core
API for shape families. A shape family is simply defined as a functor that
produces instances of :class:`~coxeter.shapes.Shape` when called. This
flexible API can be used in a variety of ways, including both tabulated and
continuously parametrizable shape families.
"""

from abc import ABC, abstractmethod


class ShapeFamily(ABC):
    """A factory for instances of :class:`~coxeter.shapes.Shape`.

    This abstract class represents a simple promise of a `get_shape` method that accepts
    some set of arguments and returns some shape class. The precise behavior is left up
    to specific subclasses, which document the parameters in the class docstring.

    This class is designed to *never be instantiated*. All relevant operations of its
    subclasses should be classmethods, and any data intrinsic to a family should be
    stored within the class. This design avoids creating an antipattern of instantiating
    a stateless class, while also providing a suitable means for using inheritance to
    create meaningful relationships between shape families. It also simplifies user
    APIs, avoiding confusing idioms like ``shape = family()(SHAPE_NAME)``. For instance,
    given a family for generating regular polygons, getting a hexagon should look
    roughly like ``family.get_shape(n=6)``.
    """

    @classmethod
    @abstractmethod
    def get_shape(cls):
        """Generate a shape.

        Subclasses must define this function to accept whatever parameters are
        necessary to define a shape. The method should return an instance of
        some concrete subclass of :class:`~coxeter.shapes.Shape`.
        """
        pass
