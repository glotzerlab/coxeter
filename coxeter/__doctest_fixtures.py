# Copyright (c) 2021 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Make the coxeter package available within doctest tests.

This module serves as a pseudo-conftest file to generate fixtures to use during doctest.
This file is necessary because doctests in coxeter are included as part of the project
source code in Sphinx-style docstrings, so doctest will discover them in the source
tree. The problem is that pytest will not apply fixtures from the conftest in the tests
directory (see https://docs.pytest.org/en/stable/doctest.html#doctest-namespace-fixture)
so we need the fixtures available inside the repository. Rather than creating a separate
conftest file that pollutes the project namespace, this clearly named internal module
is loaded as part of normal pytest runs and injects the appropriate variable into the
pytest doctest namespace.
"""

import pytest

import coxeter


# Allow all doctests to access the parent coxeter namespace.
@pytest.fixture(autouse=True)
def setup_namespace(doctest_namespace):
    """Configure the global doctest_namespace fixture."""
    doctest_namespace["coxeter"] = coxeter
