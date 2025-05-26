"""
Field representation and operations on geometric grids.

The :mod:`fields` module defines core abstractions for representing physical quantities—
such as scalars, vectors, and tensors—on structured geometric grids within the Pymetric
framework. The module supports the following:

- Seamless support for fields in curvilinear and orthogonal coordinate systems
- Rich support for arithmetic operations, broadcasting, and indexing
- Automatic grid validation and component management
- Integration with differential operators and coordinate transformations

This module is designed to be extensible for both dense and sparse field types,
and integrates tightly with the :mod:`grids` and :mod:`coordinates` subsystems.
"""
__all__ = [
    "FieldComponent",
    "DenseField",
    "DenseTensorField",
]

# Import the buffer module first.
from . import buffers
from .buffers import *

__all__ += buffers.__all__

from . import utils
from .utils import *

__all__ += utils.__all__

from .base import DenseField

# Import the FieldComponent class.
from .components import FieldComponent
from .tensors import DenseTensorField
