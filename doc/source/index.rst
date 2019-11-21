==================================
Welcome to euclid's documentation!
==================================

Welcome to the documentation for euclid, a package for working with various geometric shapes.
The package works with both two- and three-dimensional shapes such as polygons and polyhedra.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

    package-euclid


Getting Started
===============

Requirements
------------

The minimum requirements for using euclid are:

* Python >= 3.6
* NumPy
* SciPy

Installation
------------

The package can be installed from source by cloning `the repository <https://github.com/glotzerlab/euclid>`_ and running the setup script:

.. code-block:: bash

    $ git clone https://github.com/glotzerlab/euclid.git
    $ cd euclid
    $ python setup.py install

Running Tests
-------------

The package is currently tested for Python versions 3.6 and 3.7 on Unix-like systems.
Continuous integrated testing is performed using CircleCI on these Python versions.

To run the packaged unit tests, execute the following line from the root of the repository:

.. code-block:: bash

    python -m unittest discover tests

Building Documentation
----------------------

You can also build this documentation from source if you clone the repository.
The documentation is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ and compiled using `Sphinx <http://www.sphinx-doc.org/en/master/>`_.
To build from source, first install Sphinx:

.. code-block:: bash

    pip install sphinx sphinx_rtd_theme

You can then use Sphinx to create the actual documentation in either PDF or HTML form by running the following commands in the euclid root directory:

.. code-block:: bash

    cd doc
    make html # For html output
    make latexpdf # For a LaTeX compiled PDF file
    open build/html/index.html

Support and Contribution
========================

This package is hosted on `GitHub <https://github.com/glotzerlab/euclid>`_.
Please report any bugs or problems that you find on the `issue tracker <https://github.com/glotzerlab/euclid/issues>`_.
All contributions to euclid are welcomed via pull requests!

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
