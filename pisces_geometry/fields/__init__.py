"""
Data containers for storing information on geometric grids.

The :py:mod:`fields` module provides a variety of "field" classes (subclassed from :py:class:`~fields.base.GenericField`,
which are effectively numpy arrays (:py:class:`numpy.ndarray`) defined on a geometric grid (from :py:mod:`grids`). Because of
this, each field has the ability to self-consistently perform differential operations on itself.

Fields of different types are split into various submodules below:
"""
__all__ = [
    "TensorField",
    "VectorField",
    "CovectorField",
    "ScalarField",
    "ArrayBuffer",
    "HDF5Buffer",
    "UnytArrayBuffer",
    "DirichletBC",
    "NeumannBC",
    "PeriodicBC",
]
from pisces_geometry.fields.boundary_conditions import (
    DirichletBC,
    NeumannBC,
    PeriodicBC,
)
from pisces_geometry.fields.buffers import ArrayBuffer, HDF5Buffer, UnytArrayBuffer
from pisces_geometry.fields.tensor import (
    CovectorField,
    ScalarField,
    TensorField,
    VectorField,
)
