"""
Utilities for testing across the test_fields testing module.
"""
import pytest
from pymetric.fields import ArrayBuffer, HDF5Buffer
import numpy as np
import os

# ============================= #
# Collections / Settings        #
# ============================= #
# The `all_buffer_classes__` object provides a marked
# set of all of the buffer classes.
#
# These are used in the parameterization logic.
# !NOTE!: Any new ones need new marks in conftest.py
__all_buffer_classes__ = [
    pytest.param(ArrayBuffer, marks=pytest.mark.array),
    pytest.param(HDF5Buffer, marks=pytest.mark.hdf5),
]

# ============================= #
# Fixtures                      #
# ============================= #
@pytest.fixture
def test_array():
    """
    Returns a simple 2x2 NumPy array of ones for use as test data
    in buffer generation and validation tasks.
    """
    return np.ones((2, 2))

# ============================= #
# Utility Functions             #
# ============================= #
def __ArrayBuffer_args_kwargs_factory__(_):
    """Args/kwargs generator for the from_array test."""
    return (),{}

def __HDF5Buffer_args_kwargs_factory__(buffer_test_directory):
    """Args/kwargs generator for the from_array test."""
    return ((os.path.join(buffer_test_directory, "test.h5"),"test"),
            dict(overwrite=True))

__from_array_args_factories__ = {
    ArrayBuffer: __ArrayBuffer_args_kwargs_factory__,
    HDF5Buffer: __HDF5Buffer_args_kwargs_factory__,
}
