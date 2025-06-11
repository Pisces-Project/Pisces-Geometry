"""
Utilities for coordinate system testing in PyMetric.

These utilities are largely dedicated to pytest configuration.
"""
import pytest

# ------------------------------- #
# Configuration                   #
# ------------------------------- #
# This module relies on the `coordinate_io_temp_dir` fixture to
# create a temporary directory for testing coordinate system I/O.
#
# For each coordinate system, we effectively write the coordinate system
# to disk and then read it back, checking that the read coordinate system
# matches the original in terms of its attributes and properties.
__coordinate_system_io_protocols__ = [
    pytest.param("HDF5", "from_hdf5", "to_hdf5"),
    pytest.param("YAML", "from_yaml", "to_yaml"),
    pytest.param("JSON", "from_json", "to_json"),
]
