"""
Utilities for buffer tests.
"""
import os

import numpy as np
import pytest

from pymetric.fields.buffers import ArrayBuffer, HDF5Buffer


# ============================= #
# Fixtures                      #
# ============================= #
@pytest.fixture
def component_test_directory(tmp_path_factory):
    """
    The buffer test directory.
    """
    return tmp_path_factory.mktemp("component_tests")


# ============================= #
# Utility Functions             #
# ============================= #
def __func_ArrayBuffer_args_kwargs_factory__(_):
    """Args/kwargs generator for the from_array test."""
    return (), {}


def __func_HDF5Buffer_args_kwargs_factory__(component_test_directory):
    """Args/kwargs generator for the from_array test."""
    return (
        (os.path.join(component_test_directory, "test.h5"), "test"),
        dict(overwrite=True),
    )


__from_func_args_factories__ = {
    ArrayBuffer: __func_ArrayBuffer_args_kwargs_factory__,
    HDF5Buffer: __func_HDF5Buffer_args_kwargs_factory__,
}
