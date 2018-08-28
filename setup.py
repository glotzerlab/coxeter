from setuptools import setup

setup(
    name='euclid',
    version='0.1',
    description='Tools for creating and manipulating shapes.',
    url='https://bitbucket.org/glotzer/euclid',
    author='Bryan VanSaders',
    author_email='bvansade@umich.edu',
    packages=['euclid','euclid.FreudShape'],
    install_requires=['numpy', 'scipy'],
    zip_safe=False)
