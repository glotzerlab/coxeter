# coxeter

## About

Welcome to the documentation for **coxeter**!
The **coxeter** Python library provides tools for working with common geometric objects in two and three dimensions.
Named for the 20th century geometer best known for his work on polytopes, **coxeter** is especially focused on polygons and polyhedra, but it also support various standard curved shapes such as spheres and ellipsoids.
If you have any questions about how to work with coxeter, please visit the [ReadTheDocs page](http://coxeter.readthedocs.io/en/latest/).

## Authors

*   Vyas Ramasubramani <mailto:vramasub@umich.edu> (Lead developer)

## Setup

The recommended methods for installing coxeter are using **pip** or **conda**.

### Installation via pip

To install the package from PyPI, execute:
```bash
pip install coxeter --user
```

### Installation via conda

To install the package from conda, first add the **conda-forge** channel:
```bash
conda config --add channels conda-forge
```

After the **conda-forge** channel has been added, you can install coxeter by
executing
```bash
conda install coxeter
```

### Installation from source

To install from source, execute:
```bash
git clone https://github.com/glotzerlab/coxeter.git
cd coxeter
python setup.py install --user
```

### Requirements

*   Python >= 3.3
*   NumPy >= 1.10
*   SciPy >= 1.0.0
*   rowan >= 1.2

## Testing

The package is currently tested for Python >= 3.3 on Unix-like systems.
Continuous integrated testing is performed using CircleCI on these Python versions.

To run the packaged unit tests, execute the following line from the root of the repository:

```bash
pytest
```

To check test coverage, make sure the coverage module is installed:

```bash
pip install coverage
```

and then run the packaged unit tests with the coverage module:

```bash
pytest --cov=coxeter
```

## Documentation
Documentation for coxeter is written in [reStructuredText](http://docutils.sourceforge.net/rst.html) and compiled using [Sphinx](http://www.sphinx-doc.org/en/master/).
To build the documentation, first install Sphinx:

```bash
pip install sphinx sphinx_rtd_theme
```

You can then use Sphinx to create the actual documentation in either PDF or HTML form by running the following commands in the coxeter root directory:

```bash
cd doc
make html # For html output
make latexpdf # For a LaTeX compiled PDF file
open build/html/index.html
```
