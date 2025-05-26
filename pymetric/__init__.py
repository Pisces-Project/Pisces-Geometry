"""
Pymetric is a computational geometry library designed to facilitate advanced
geometric and differential operations in a wide range of coordinate systems.
It provides a unified and extensible interface for working with:

- Curvilinear and orthogonal coordinate systems
- Metric tensors and their properties
- Differential operators such as gradient, divergence, and Laplacian
- Tensor transformations and symbolic dependence tracking
- Structured grids for numerical computations
- Buffer abstraction layers for in-memory, unit-aware, and HDF5-backed data

Pymetric is built to serve scientific computing applications involving
non-Cartesian geometries, such as plasma physics, astrophysics, general relativity,
and other fields requiring curvilinear analysis.

This package is part of the Pisces ecosystem and is designed to integrate
cleanly with other Pisces geometry and simulation tools.
"""
__all__ = []

# Import the geometric field's library.
from . import fields
from .fields import *

__all__ += fields.__all__

# Import the grid's library
from . import grids
from .grids import *

__all__ += grids.__all__

# Import the coordinate systems
from . import coordinates
from .coordinates import *

__all__ += coordinates.__all__

# Import the differential geometry.
from . import differential_geometry
from .differential_geometry import *

__all__ += differential_geometry.__all__

# Import the utilities.
from . import utilities
from .utilities import *

__all__ += utilities.__all__
