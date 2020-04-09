import numpy as np
from scipy.special import ellipe


class Ellipse(object):
    def __init__(self, a, b):
        """An ellipse with principal axes a and b.

        Args:
            a (float):
                Principal axis a of the ellipse (radius in the x direction).
            b (float):
                Principal axis b of the ellipse (radius in the y direction).
        """
        self._a = a
        self._b = b

    @property
    def a(self):
        """float: Length of principal axis a (radius in the x direction)."""
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def b(self):
        """float: Length of principal axis b (radius in the y direction)."""
        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    @property
    def area(self):
        """float: The area."""
        return np.pi * self.a * self.b

    @property
    def eccentricity(self):
        """float: The eccentricity."""
        # Requires that a >= b, so we sort the principal axes:
        b, a = sorted([self.a, self.b])
        e = np.sqrt(1 - b**2/a**2)
        return e

    @property
    def perimeter(self):
        """float: The perimeter."""
        # Implemented from this example:
        # https://scipython.com/book/chapter-8-scipy/examples/the-circumference-of-an-ellipse/
        # It requires that a >= b, so we sort the principal axes:
        b, a = sorted([self.a, self.b])
        result = 4 * a * ellipe(self.eccentricity**2)
        return result

    @property
    def circumference(self):
        """float: Alias for :meth:`~.perimeter`."""
        return self.perimeter

    @property
    def planar_moments_inertia(self):
        R"""Get the planar moments with respect to the x and y axis as well as
        the product of inertia.

        The `planar moments <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        and the
        `product moment <https://en.wikipedia.org/wiki/Second_moment_of_area#Product_moment_of_area>`__
        are defined by the formulas:

        .. math::
            \begin{align}
                I_x &= {\int \int}_A y^2 dA = \frac{\pi}{4} a b^3 = \frac{Ab^2}{4} \\
                I_y &= {\int \int}_A z^2 dA = \frac{\pi}{4} a^3 b = \frac{Aa^2}{4}\\
                I_{xy} &= {\int \int}_A xy dA = 0 \\
            \end{align}

        These formulas are given `here https://en.wikipedia.org/wiki/List_of_second_moments_of_area`__.
        Note that the product moment is zero by symmetry.
        """  # noqa: E501
        A = self.area
        Ix = A/4 * self.b**2
        Iy = A/4 * self.a**2
        Ixy = 0
        return Ix, Iy, Ixy

    @property
    def polar_moment_inertia(self):
        """The polar moment of inertia.

        The `polar moment of inertia <https://en.wikipedia.org/wiki/Polar_moment_of_inertia>`__
        is always calculated about an axis perpendicular to the ellipse (i.e. the
        normal vector) placed at the centroid of the ellipse.

        The polar moment is computed as the sum of the two planar moments of inertia.
        """  # noqa: E501
        return np.sum(self.planar_moments_inertia[:2])

    @property
    def iq(self):
        """float: The isoperimetric quotient."""
        A = self.area
        P = self.perimeter
        return np.min([4 * np.pi * A / (P**2), 1])
