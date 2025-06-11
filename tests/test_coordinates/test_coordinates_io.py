"""
Testing suite to verify that coordinate systems can be written and read correctly
from disk using our various protocols.
"""
import os

import pytest

from .utils import __coordinate_system_io_protocols__


# ================================= #
# Module Level Fixtures             #
# ================================= #
@pytest.fixture(scope="module")
def coordinate_io_temp_dir(tmp_path_factory):
    """
    Fixture to create a temporary directory for coordinate system I/O tests.
    """
    return tmp_path_factory.mktemp("coordinate_system_io")


# ================================= #
# IO TESTING                        #
# ================================= #
@pytest.mark.parametrize(
    "protocol, from_method, to_method", __coordinate_system_io_protocols__
)
def test_coordinate_system_io(
    cs_flag,
    coordinate_systems,
    coordinate_io_temp_dir,
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
    coordinate_io_temp_dir: The temp dir for the data.
    protocol: The protocol to use for I/O (e.g., 'HDF5', 'YAML', 'JSON').
    from_method: The method to read the coordinate system from disk.
    to_method: The method to write the coordinate system to disk.
    args: Additional positional arguments for the I/O methods.
    kwargs: Additional keyword arguments for the I/O methods.
    """
    # Load in the coordinate system from the collection.
    coordinate_system = coordinate_systems[cs_flag]

    # Parse the to / from methods from the coordinate system.
    assert hasattr(
        coordinate_system, from_method
    ), "No method to read coordinate system from disk."
    assert hasattr(
        coordinate_system, to_method
    ), "No method to write coordinate system to disk."

    _from_method, _to_method = getattr(coordinate_system, from_method), getattr(
        coordinate_system, to_method
    )

    # Now perform the writing segment of the operation.
    path = os.path.join(coordinate_io_temp_dir, f"{cs_flag}.{protocol.lower()}")
    _to_method(path, overwrite=True)

    # Now read the coordinate system back from disk.
    _loaded_cs = _from_method(path)
