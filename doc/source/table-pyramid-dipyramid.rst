Pyramids and Dipyramids
=======================

Coxeter's :py:class:`~coxeter.families.UniformPyramidFamily` and :py:class:`~coxeter.families.UniformDipyramidFamily` allow for easy generation of the uniform pyramids :math:`n \in \set{3, 4, 5}` using the associated :code:`Family.get_shape(n)` method.
Some may also be found among the :doc:`Johnson Solids<table-johnson>` from :cite:`Damasceno2012a`.

Pyramids
--------

Any member of the family of uniform pyramids can be generated with :code:`families.UniformPyramidFamily.get_shape(n)`, where ``n`` is the number of vertices of the base polygon.

.. polyhedron-table::
   :family: coxeter.families.UniformPyramidFamily
   :id-header: n

   3-5

Dipyramids
----------

Any member of the family of uniform dipyramids can be generated with :code:`families.UniformDipyramidFamily.get_shape(n)`, where ``n`` is the number of vertices of the base polygon.

.. polyhedron-table::
   :family: coxeter.families.UniformDipyramidFamily
   :id-header: n

   3-5
