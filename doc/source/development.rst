.. _development:

=================
Development Guide
=================


All contributions to **coxeter** are welcome!
Developers are invited to contribute to the framework by pull request to the package repository on `GitHub`_, and all users are welcome to provide contributions in the form of **user feedback** and **bug reports**.
We recommend discussing new features in form of a proposal on the issue tracker for the appropriate project prior to development.

.. _github: https://github.com/glotzerlab/coxeter

General Guidelines
==================

All code contributed to **coxeter** must adhere to the following guidelines:

  * Use the OneFlow_ model of development:
    - Both new features and bug fixes should be developed in branches based on ``master``.
    - Hotfixes (critical bugs that need to be released *fast*) should be developed in a branch based on the latest tagged release.
  * Avoid external dependencies wherever possible, and avoid introducing **any** hard dependencies outside the standard Python scientific stack (NumPy, SciPy, etc). Soft dependencies are allowed for specific functionality, but such dependencies cannot impede the installation of **coxeter** or the use of any other features.
  * All code should adhere to the source code conventions and satisfy the documentation and testing requirements discussed below.
  * Preserve backwards-compatibility whenever possible. Make clear if something must change, and notify package maintainers that merging such changes will require a major release.

To provide a reasonable balance between a high level of backwards compatibility and a reasonable maintenance burden, **coxeter** has adopted `NEP 29`_ to limit the Python and NumPy versions that will be supported.


.. tip::

    During continuous integration, the code is checked automatically with `flake8`_, including a number of plugins that validate parts of the style.
    To run this locally, you can install and run flake8 locally:

    .. code-block:: bash

        python -m pip install flake8 flake8-black flake8-bugbear flake8-docstrings flake8-rst-docstrings pep8-naming flake8-isort
        python -m flake8 coxeter tests

    To avoid having commits fail in case you forget to run this, you can set up a git pre-commit hook using `pre-commit`_:

    .. code-block:: bash

        python -m pip install pre-commit
        pre-commit install

.. _OneFlow: https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow
.. _flake8: http://flake8.pycqa.org/en/latest/
.. _pre-commit: https://pre-commit.com/
.. _NEP 29: https://numpy.org/neps/nep-0029-deprecation_policy.html


Style Guidelines
----------------

The **coxeter** package adheres to a relatively strict set of style guidelines.
All code in **coxeter** should be formatted using `black`_.
Imports should be formatted using `isort`_.
For guidance on the style, see `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_, but any ambiguities should be resolved automatically by running black.
All code should of course also follow the principles in `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_.

.. tip::

    Developers should format their code using black and isort locally. Running the pre-commit hooks will take care of this:

    .. code-block:: bash

        pre-commit run

    Alternatively, the tools can be run manually using the commands:

    .. code-block:: bash

        black --exclude "coxeter/[polytri|bentley_ottmann]" coxeter/ tests/
        isort coxeter/ tests/

.. _black: https://black.readthedocs.io/
.. _isort: https://timothycrosley.github.io/isort/


Documentation
-------------

API documentation should be written as part of the docstrings of the package in the `Google style <https://google.github.io/styleguide/pyguide.html#383-functions-and-methods>`__.
There is one notable exception to the guide: class properties should be documented in the getters functions, not as class attributes, to allow for more useful help messages and inheritance of docstrings.
Docstrings may be validated using `pydocstyle <http://www.pydocstyle.org/>`__ (or using the flake8-docstrings plugin as documented above).
The `official documentation <https://coxeter.readthedocs.io/>`_ is generated from the docstrings using `Sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.

In addition to API documentation, inline comments are **highly encouraged**.
Code should be written as transparently as possible, so the primary goal of documentation should be explaining the algorithms or mathematical concepts underlying the code.
Avoid comments that simply restate the nature of lines of code.
For example, the comment "solve the system of equations" is uninformative, since the code itself should make this obvious, *e.g*, ``np.linalg.solve``.
On the other hand, the comment "the solution to this system of equations is the intersection of truncation planes" is instructive.


Unit Tests
----------

All code should include a set of tests which test for correct behavior.
All tests should be placed in the ``tests`` folder at the root of the project.
In general, most parts of coxeter primarily require `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`_, but where appropriate `integration tests <https://en.wikipedia.org/wiki/Integration_testing>`_ are also welcome.
Tests in **coxter** use the `pytest <https://docs.pytest.org/>`__ testing framework.
To run the tests, simply execute ``pytest`` at the root of the repository.


Release Guide
=============

To make a new release of **coxeter**, follow the following steps:

#. Make a new branch off of master based on the expected new version, *e.g.*
   release-2.3.1.
#. Make any final changes as desired on this branch. Push the changes and
   ensure all tests are passing as expected on the new branch.
#. Once the branch is completely finalized, run bumpversion with the
   appropriate type (patch, minor, major) so that the version now matches the
   version number in the branch name.
#. Merge the branch back into master, then push master and push tags. The
   tagged commit will automatically trigger generation of binaries and upload
   to PyPI and conda-forge.
#. Delete the release branch both locally and on the remote.
