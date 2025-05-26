"""
Physical grid support in PyMetric.

This module provides the core grid abstractions used for defining structured computational domains
in PyMetric. Grids represent the spatial discretization of a coordinate system and define
how data fields are laid out in space.

These grids support the use of any coordinate system in the library as well as many other
advanced extensions like the use of ghost cells, unit management, lazy loading, and chunked operations.

For a detailed, user-friendly introduction to the :mod:`grids` module, see :ref:`grids`.
"""
__all__ = [
    "GenericGrid",
    "UniformGrid",
]
from .core import GenericGrid, UniformGrid
