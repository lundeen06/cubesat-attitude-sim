# setup.py
from setuptools import setup, find_packages

setup(
    name="cubesat_sim",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
    ],
)