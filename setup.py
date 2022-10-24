""" setup.py - Main setup module """
import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(HERE, 'README.md')).read()
VERSION = '0.1'

# Publicly Available Packages (PyPi)
INSTALL_REQUIRES = [
    'black',
    'flake8',
    'isort',
    'mypy',
    'numpy',
    'pre-commit',
    'pytest',
    'pytest-cov',
    'toml',
    'torch',
    'types-requests',
    'types-setuptools'
]

setup(
    name='vprompt',
    version=VERSION,
    description="Codes developed for prompt engineering project at the Vector Institute.",
    long_description=README,
    classifiers=['Programming Language :: Python :: 3.8'],
    keywords="NLP, Machine Learning",
    author="David Emerson, Saeed Najafi",
    author_email="david.emerson@vectorinstitute.ai, saeed.najafi@vectorinstitute.ai",
    license='MIT',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'dev': [
            'ipython',
            'jupyterlab'
        ]
    }
)
