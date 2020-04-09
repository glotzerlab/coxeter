from setuptools import setup, find_packages

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
    name='coexeter',
    version='0.1.0',
    description='Tools for creating and manipulating shapes.',
    url='https://github.com/glotzerlab/coexeter',
    author='Vyas Ramasubramani',
    author_email='vramasub@umich.edu',
    packages=find_packages(),
    package_data={'coexeter': ['science.1220869.json']},
    install_requires=[
        'numpy',
        'rowan>=1.2',
        'scipy',
    ],
    tests_require=test_deps,
    extras_require=extras,
    zip_safe=False)
