from abc import ABC, abstractmethod


class _ShapeFamily(ABC):
    """A functor that generates :class:`~coxeter.shape_classes.Shape`s.

    This class represents a simple promise of a __call__ method that accepts
    some set of arguments and returns some shape class. The precise behavior is
    left up to specific subclasses, which document their callable parameters in
    the class docstring.
    """
    @abstractmethod
    def __call__(self):
        """Generate an instance of :class:`~coxeter.shape_classes.Shape` based
        on the provided parameters."""
        pass
