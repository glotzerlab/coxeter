"""Define the abstract base class for all shape families.

This module defines the :class:`~.ShapeFamily` class, which defines the core
API for shape families. A shape family is simply defined as a functor that
produces instances of :class:`~coxeter.shape_classes.Shape` when called. This
flexible API can be used in a variety of ways, including both tabulated and
continuously parametrizable shape families.
"""

from abc import ABC, abstractmethod


class _ShapeFamily(ABC):
    """A functor that generates instances of :class:`~coxeter.shape_classes.Shape`.

    This class represents a simple promise of a __call__ method that accepts
    some set of arguments and returns some shape class. The precise behavior is
    left up to specific subclasses, which document their callable parameters in
    the class docstring.
    """

    @abstractmethod
    def __call__(self):
        """Generate a shape.

        Subclasses must define this function to accept whatever parameters are
        necessary to define a shape. The method should return an instance of
        some concrete subclass of :class:`~coxeter.shape_classes.Shape`.
        """
        pass
