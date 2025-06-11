"""
Testing suite to verify that coordinate systems can be written and read correctly
from disk using our various protocols.
"""
import os

import pytest

from .utils import (
    __all_grid_classes_params__,
    __grid_factories__,
    __grid_io_protocols__,
)


# ------------------------------- #
# Configuration                   #
# ------------------------------- #
# This module relies on the `coordinate_io_temp_dir` fixture to
# create a temporary directory for testing coordinate system I/O.
#
# For each coordinate system, we effectively write the coordinate system
# to disk and then read it back, checking that the read coordinate system
# matches the original in terms of its attributes and properties.
@pytest.fixture(scope="module")
def grid_io_temp_dir(tmp_path_factory):
    """
    Fixture to create a temporary directory for coordinate system I/O tests.
    """
    return tmp_path_factory.mktemp("grid_io_temp")


@pytest.mark.parametrize("protocol, from_method, to_method", __grid_io_protocols__)
@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
def test_coordinate_system_io_hdf5(
    cs_flag,
    coordinate_systems,
    grid_class,
    grid_io_temp_dir,
    protocol,
    from_method,
    to_method,
):
    """
    This test verifies that we can write and read coordinate systems
    using the various disk protocols. All we do is write the coordinate system
    to disk, read it back, and check its basic attributes.

    Parameters
    ----------
    cs_flag: The coordinate system flag to test.
    coordinate_systems: The coordinate systems collection from which to fetch our coordinate system.
    protocol: The protocol to use for I/O (e.g., 'HDF5', 'YAML', 'JSON').
    from_method: The method to read the coordinate system from disk.
    to_method: The method to write the coordinate system to disk.
    """
    # Run the grid class factory to create the grid.
    grid = __grid_factories__[grid_class](coordinate_systems[cs_flag])
    cs = grid.coordinate_system

    # Parse the to / from methods from the grid.
    assert hasattr(grid, from_method), "No method to read coordinate system from disk."
    assert hasattr(grid, to_method), "No method to write coordinate system to disk."

    _from_method, _to_method = getattr(grid, from_method), getattr(grid, to_method)

    # Now perform the writing segment of the operation.
    path = os.path.join(grid_io_temp_dir, f"{cs_flag}.{protocol.lower()}")
    _to_method(path, overwrite=True)

    # Now read the coordinate system back from disk.
    _loaded_grid = _from_method(path, cs)
