.. _quickstart:

Quickstart Tutorial
===================

Once you have :ref:`installed <installing>` **coxeter**, most workflows involve creating an instance of a shape, such as a :class:`~coxeter.shapes.Polygon`:

.. code-block:: python

    >>> import coxeter
    >>> square = coxeter.shapes.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])

All shapes may be found in the `coxeter.shapes` subpackage and are created in a similar manner.
For instance, making a sphere requires a radius: ``sphere = coxeter.shapes.Sphere(3)``.
Once you have a shape, you can immediately query it for properties.

.. code-block:: python

    >>> square.vertices
    array([[0., 0., 0.],
           [1., 0., 0.],
           [1., 1., 0.],
           [0., 1., 0.]])
    >>> square.area
    1.0
    >>> square.perimeter
    4.0

The **coxeter** library comes with a range of *shape families*, collections of shapes with standard definitions so that you don't have to parameterize them yourself.
The regular :math:`n`-gon family is such an example, provided as :class:`coxeter.families.RegularNGonFamily`.

.. code-block:: python

    >>> hexagon = coxeter.families.RegularNGonFamily.get_shape(6)
    >>> hexagon.vertices.round(2)
    array([[ 0.62,  0.  ,  0.  ],
           [ 0.31,  0.54,  0.  ],
           [-0.31,  0.54,  0.  ],
           [-0.62,  0.  ,  0.  ],
           [-0.31, -0.54,  0.  ],
           [ 0.31, -0.54,  0.  ]])

Part of what makes **coxeter** so powerful is that all shapes are mutable.
This means that once you have a prototype of a shape, it can be modified to fit a specific need.
For example, the snippet below finds the area of the smallest regular pentagon that contains an equilateral triangle of unit area:

.. code-block:: python

    >>> triangle = coxeter.families.RegularNGonFamily.get_shape(3)
    >>> pentagon = coxeter.families.RegularNGonFamily.get_shape(5)
    >>> pentagon.incircle_radius = triangle.circumcircle.radius
    >>> triangle.area
    0.9999999999999999
    >>> triangle.circumcircle.area
    2.418399152312292
    >>> pentagon.area
    2.796463494144044

This tutorial just scratches the surface of the features **coxeter** offers.
For more complete demonstrations of the package's features, see the :ref:`examples`.
