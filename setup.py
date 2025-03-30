#!/usr/bin/env python
"""
Setup file for Pisces-Geometry module.

Written by: Eliza Diggins
Last Updated: 03/29/24
"""
from setuptools import find_packages, setup

# @@ CYTHON UTILITIES @@ #
# All of the cython extensions for the package have to be added
# here to ensure that they are accessible and installed on use.
# Read the readme from /Pisces/README.rst. As long as we are in the
# project directory, this should be accessible directly in path.
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup function
setup(
    name="pisces_geometry",
    version="0.0.1",
    description="Geometric backend for the Pisces project.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Eliza C. Diggins",
    author_email="eliza.diggins@utah.edu",
    url="https://github.com/Pisces-Project/Pisces-Geometry",
    setup_requires=[
        "numpy",
        "cython",
    ],  # Ensure numpy and cython are installed before setup
    download_url="https://github.com/Pisces-Project/pisces-geometry/tarbar/0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
        "scipy",
        "cython",
        "matplotlib",
        "tqdm",
        "ruamel.yaml",
        "h5py",
        "sympy",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,
    python_requires=">=3.6",
)
