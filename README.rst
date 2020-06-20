coxeter
=======

.. contents::
   :local:

|ReadTheDocs|
|CircleCI|
|PyPI|
|conda-forge|

.. |ReadTheDocs| image:: https://readthedocs.org/projects/coxeter/badge/?version=latest
    :target: http://coxeter.readthedocs.io/en/latest/?badge=latest
.. |CircleCI| image:: https://circleci.com/gh/glotzerlab/coxeter.svg?style=svg
    :target: https://circleci.com/gh/glotzerlab/coxeter
.. |PyPI| image:: https://img.shields.io/pypi/v/coxeter.svg
    :target: https://pypi.org/project/coxeter/
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/coxeter.svg
   :target: https://anaconda.org/conda-forge/coxeter

Welcome to the documentation for **coxeter**!
The **coxeter** Python library provides tools for working with common geometric objects in two and three dimensions.
Named for the `20th century geometer <https://en.wikipedia.org/wiki/Harold_Scott_MacDonald_Coxeter>`__ best known for his work on polytopes, **coxeter** is especially focused on polygons and polyhedra, but it also support various standard curved shapes such as spheres and ellipsoids.

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

To install from source, execute:

.. code:: bash

   git clone https://github.com/glotzerlab/coxeter.git
   cd coxeter
   python setup.py install --user

Requirements
~~~~~~~~~~~~

-  Python >= 3.6
-  NumPy >= 1.15
-  SciPy >= 1.0.0
-  rowan >= 1.2

Testing
-------

The package is currently tested for Python >= 3.6 on Unix-like systems.
Continuous integrated testing is performed using CircleCI on these Python versions.

To run the packaged unit tests, execute the following line from the root of the repository:

.. code:: bash

   pytest

To check test coverage, make sure the coverage module is installed:

.. code:: bash

   pip install coverage

and then run the packaged unit tests with the coverage module:

.. code:: bash

   pytest --cov=coxeter

Documentation
-------------

Documentation for coxeter is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`__ and compiled using `Sphinx <http://www.sphinx-doc.org/en/master/>`__.
To build the documentation, first install Sphinx:

.. code:: bash

   cd doc
   pip install -r requirements.txt

You can then use Sphinx to create the actual documentation in either PDF or HTML form by running the following commands in the coxeter root directory:

.. code:: bash

   make html # For html output
   make latexpdf # For a LaTeX compiled PDF file
   open build/html/index.html

Support and Contribution
========================

This package is hosted on `GitHub <https://github.com/glotzerlab/coxeter>`_.
Please report any bugs or problems that you find on the `issue tracker <https://github.com/glotzerlab/coxeter/issues>`_.
All contributions to coxeter are welcomed via pull requests!
