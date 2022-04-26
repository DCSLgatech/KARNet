import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

setup(
    name="cdae",
    version="0.1.0",
    description="Latent Prediction Network",
    packages=find_packages(),
    python_requires=">= 3.8",
)
