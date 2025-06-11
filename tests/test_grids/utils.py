"""
Utility functions for testing pymetric grids.
"""
from pymetric.grids import UniformGrid, GenericGrid
import pytest
import numpy as np

# =============================================== #
# Grid Class Factory Generators                   #
# =============================================== #
# Each new grid class needs to have a factory method defined here
# which is then registered in the __grid_factories__ dictionary at
# the bottom of this section of code.
#
# Each factory should have signature `factory(coordinate_system)` and return
# the fully realized grid. These factories should be as extensive in their coverage
# of the underlying class as possible.
__all_grid_classes_params__ = [
    pytest.param(UniformGrid),
    pytest.param(GenericGrid),
]

def __grid_factory_UniformGrid__(coordinate_system):
    """
    Factory for creating a UniformGrid from a coordinate system.

    The uniform grid requires a bounding box and a resolution. For this
    case, we use [0,1] as the bounding box by default (can be overridden in the
    settings) and a resolution of 10 for each dimension.
    """
    # --- Factory Settings --- #
    _settings = {
        "coordinate_system_overrides": {},
        "resolution": 20,
        "center": "cell",
        "chunk_size": 5,
        "ghost_zone": 1,
    }

    # --- Build the bounding box --- #
    ndim = coordinate_system.ndim

    if coordinate_system.__class__.__name__ in _settings["coordinate_system_overrides"]:
        bbox = _settings["coordinate_system_overrides"][coordinate_system.__class__.__name__]
    else:
        bbox = [[0] * ndim,[1] * ndim]  # Default bounding box from 0 to 1 in each dimension


    # --- Create the grid --- #
    grid = UniformGrid(coordinate_system,
                       bbox,
                       [_settings["resolution"]]*ndim,
                       center=_settings["center"],
                       ghost_zones=_settings["ghost_zone"],
                       chunk_size=[_settings["chunk_size"]]*ndim)

    return grid
def __grid_factory_GenericGrid__(coordinate_system):
    """
    Factory for creating a GenericGrid. We use the default coordinates
    to build the x,y,z etc.
    """
    # --- Factory Settings --- #
    _settings = {
        "coordinate_system_overrides": {},
        "resolution": 23,
        "center": "vertex",
        "chunk_size": 5,
        "ghost_zone": 1,
    }

    # --- Build the bounding box --- #
    ndim = coordinate_system.ndim

    if coordinate_system.__class__.__name__ in _settings["coordinate_system_overrides"]:
        # We are overriding the default positions for the grid for this test. To check
        # that this works, we need to extract all of the bounding boxes provided and then
        # create the coordinates.
        ranges = _settings["coordinate_system_overrides"][coordinate_system.__class__.__name__]
        coords = (np.linspace(*r,_settings["resolution"]) for r in ranges)
    else:
        coords = (np.linspace(0,1,_settings["resolution"]),)*ndim


    # --- Create the grid --- #
    grid = GenericGrid(coordinate_system,
                       coords,
                       center=_settings["center"],
                       ghost_zones=_settings["ghost_zone"],
                       chunk_size=[_settings["chunk_size"]]*ndim)

    return grid

__grid_factories__ = {
    UniformGrid: __grid_factory_UniformGrid__,
    GenericGrid: __grid_factory_GenericGrid__,
}

# =============================================== #
# IO Protocols                                    #
# =============================================== #
# These are the protocols and settings for the I/O tests.
__grid_io_protocols__ = [
    pytest.param("HDF5", 'from_hdf5', 'to_hdf5'),
    pytest.param("YAML", 'from_yaml', 'to_yaml'),
    pytest.param("JSON", 'from_json', 'to_json'),
]