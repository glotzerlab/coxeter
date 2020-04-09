from setuptools import setup, find_packages
import os

# Read README for PyPI, fallback if it fails.
desc = 'Tools for creating and manipulating shapes.'
try:
    readme_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'README.md')
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = desc

version = '0.2.0'


################################################
# Set up for the various optional dependencies
# that may be installed for additional features.
################################################

test_deps = [
    'pytest',
    'hypothesis[numpy]',
]

bounding_deps = [
    'miniball',
]

extras = {
    'test': test_deps + bounding_deps,
    'bounding_sphere': bounding_deps,
}

setup(
    name='coxeter',
    version=version,
    description=desc,
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/glotzerlab/coxeter',
    author='Vyas Ramasubramani',
    author_email='vramasub@umich.edu',
    packages=find_packages(),
    package_data={'coxeter': ['science.1220869.json']},
    install_requires=[
        'numpy',
        'rowan>=1.2',
        'scipy',
    ],
    tests_require=test_deps,
    extras_require=extras,
    zip_safe=True)
