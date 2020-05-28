from abc import ABC, abstractmethod


class ShapeFamily(ABC):
    """A shape family encapsulates the ability to generate a set of shapes of
    of type :class:`~coxeter.shape_classes.Shape`.
    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """:class:`~coxeter.shape_classes.Shape`: Generate a shape based on the
        provided parameters."""
        pass
