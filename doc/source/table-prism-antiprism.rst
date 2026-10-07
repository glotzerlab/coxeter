Prisms and Antiprisms
=====================

Coxeter allows for the generation of any of the infinite family of right uniform prisms and antiprisms :code:`families.UniformPrismFamily.get_shape(n)` and :code:`families.UniformAntiprismFamily.get_shape(n)`, where ``n`` is the number of vertices of the polar faces. Some may also be found among the :doc:`Other Polyhedra<table-other>` from :cite:`Damasceno2012a`.

For implementation details see :py:class:`~coxeter.families.UniformPrismFamily` and :py:class:`~coxeter.families.UniformAntiprismFamily`.
A subset of each family is shown in the following tables.

Prisms
------

Any of the infinite family of right uniform prisms can be generated with :code:`families.UniformPrismFamily.get_shape(n)`, where ``n`` is the number of vertices of the polar faces. A subset of this family is shown below.

.. polyhedron-table::
   :family: coxeter.families.UniformPrismFamily
   :id-header: n

   3-10

Antiprisms
----------

Any of the infinite family of right uniform antiprisms can be generated with :code:`families.UniformAntiprismFamily.get_shape(n)`, where ``n`` is the number of vertices of the polar faces. A subset of this family is shown below.

.. polyhedron-table::
   :family: coxeter.families.UniformAntiprismFamily
   :id-header: n

   3-10
