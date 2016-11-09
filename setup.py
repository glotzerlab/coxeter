from setuptools import setup

setup(name = 'euclid',
        version = '0.1',
        description = 'shape tools',
        url = 'https://bvansade@bitbucket.org/glotzer/euclid.git',
        author = 'Bryan VanSaders',
        author_email = 'bvansade@umich.edu',
        license = 'None',
        packages = ['euclid','euclid.FreudShape'],
        install_requires=['numpy', 'scipy'],
        zip_safe = False)
