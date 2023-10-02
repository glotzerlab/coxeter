coxeter
=======

|JOSS|
|ReadTheDocs|
|PyPI|
|conda-forge|

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.03098/status.svg
   :target: https://doi.org/10.21105/joss.03098
.. |ReadTheDocs| image:: https://readthedocs.org/projects/coxeter/badge/?version=latest
   :target: http://coxeter.readthedocs.io/en/latest/?badge=latest
.. |PyPI| image:: https://img.shields.io/pypi/v/coxeter.svg
   :target: https://pypi.org/project/coxeter/
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/coxeter.svg
   :target: https://anaconda.org/conda-forge/coxeter

Welcome to the documentation for **coxeter**!
The **coxeter** Python library provides tools for working with common geometric objects in two and three dimensions.
Named for the `20th century geometer <https://en.wikipedia.org/wiki/Harold_Scott_MacDonald_Coxeter>`__ best known for his work on polytopes, **coxeter** is especially focused on polygons and polyhedra, but it also support various standard curved shapes such as spheres and ellipsoids.

The package emphasizes working with shapes as mutable objects whose geometric attributes may be accessed using property-based APIs.
Since **coxeter** originally arose to support representations of anisotropic nanoparticles, many shapes support calculations of physical properties (such as form factors and inertia tensors) in addition to purely geometric ones.
However, the package is designed with more general audiences in mind as well, and it aims to support precise calculations of a wide range of geometric quantities that are useful in a number of fields.

Some core features of **coxeter** include:

* Libraries of common shapes to support easy construction.
* Mutable shape objects that can be rescaled in a variety of ways to suit a number of needs.
* Immediate access to geometric properties of shapes via Python properties of shape objects.
* Plotting functionality to make it easy to visualize shapes in both two and three dimensions.

More detailed information on **coxeter**'s features and examples of how to use them may be found in the `documentation <https://coxeter.readthedocs.io/>`__.

.. _installing:

Setup
-----

The recommended methods for installing coxeter are using **pip** or **conda**.

Installation via pip
~~~~~~~~~~~~~~~~~~~~

To install the package from PyPI, execute:

.. code:: bash

   pip install coxeter --user

Installation via conda
~~~~~~~~~~~~~~~~~~~~~~

To install the package from conda, first add the **conda-forge** channel:

.. code:: bash

   conda config --add channels conda-forge

After the **conda-forge** channel has been added, you can install coxeter by executing

.. code:: bash

   conda install coxeter

Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

Start by executing the following:

.. code:: bash

   git clone https://github.com/glotzerlab/coxeter.git
   cd coxeter

To install coxeter and other optional dependencies, choose one of the following:

.. code:: bash

   pip install . # Install with no additional dependencies
   pip install .[tests] # RECOMMENDED: Install with dependencies required to run pytests
   pip install .[tests,doc] # Install all dependencies required to develop for coxeter


Requirements
~~~~~~~~~~~~

-  Python >= 3.8
-  NumPy >= 1.19.0
-  SciPy >= 1.0.0
-  rowan >= 1.2.0

Testing
-------

The package is currently tested for Python >= 3.8 on Unix-like systems.
Continuous integrated testing is performed using Github actions on these Python versions.

First, install the packages required to test coxeter (if not already done):

.. code:: bash

   pip install -r tests/requirements.txt

To run the packaged unit tests, execute the following line from the root of the repository:

.. code:: bash

   pytest

To run the packaged unit tests with the coverage module:

.. code:: bash

   pytest --cov=coxeter

Building Documentation
----------------------

Documentation for coxeter is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`__ and compiled using `Sphinx <http://www.sphinx-doc.org/en/master/>`__.
To build the documentation, first install Sphinx and the other required packages:

.. code:: bash

   pip install -r doc/requirements.txt
   conda install -c conda-forge fresnel

.. warning::
   The `fresnel <https://fresnel.readthedocs.io/>`_ package on conda-forge must be used. The PyPI package *fresnel* is different and will not function properly.

You can then use Sphinx to create the actual documentation in either PDF or HTML form by running the following commands:

.. code:: bash

   cd doc
   make html # For html output
   make latexpdf # For a LaTeX compiled PDF file
   open build/html/index.html

Support and Contribution
========================

This package is hosted on `GitHub <https://github.com/glotzerlab/coxeter>`_.
Please report any bugs or problems that you find on the `issue tracker <https://github.com/glotzerlab/coxeter/issues>`_.
All contributions to coxeter are welcomed via pull requests!
