from setuptools import setup, find_packages

test_deps = [
    'pytest',
    'hypothesis[numpy]',
]

extras = {
    'test': test_deps,
}

setup(
    name='euclid',
    version='0.1',
    description='Tools for creating and manipulating shapes.',
    url='https://github.com/glotzerlab/euclid',
    author='Bryan VanSaders',
    author_email='bvansade@umich.edu',
    packages=find_packages(),
    package_data={'euclid': ['science.1220869.json']},
    install_requires=[
        'numpy',
        'rowan>=1.2',
        'scipy'
    ],
    tests_require=test_deps,
    extras_require=extras,
    zip_safe=False)
