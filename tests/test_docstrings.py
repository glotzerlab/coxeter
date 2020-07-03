"""Test examples in code docstrings using doctest."""
import doctest
import pkgutil
import unittest

import coxeter


def load_tests(loader, tests, ignore):
    """Create tests from all docstrings by walking the package hierarchy."""
    modules = pkgutil.walk_packages(coxeter.__path__, coxeter.__name__ + ".")
    for _, module_name, _ in modules:
        tests.addTests(doctest.DocTestSuite(module_name, globs={"coxeter": coxeter}))
    return tests


if __name__ == "__main__":
    unittest.main()
