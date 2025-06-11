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
def dense_test_directory(tmp_path_factory):
    """
    The buffer test directory.
    """
    return tmp_path_factory.mktemp("dense_field_tests")


# ============================= #
# Utility Functions             #
# ============================= #
def __func_ArrayBuffer_args_kwargs_factory__(_):
    """Args/kwargs generator for the from_array test."""
    return (), {}


def __func_HDF5Buffer_args_kwargs_factory__(dense_test_directory):
    """Args/kwargs generator for the from_array test."""
    return (
        (os.path.join(dense_test_directory, "test.h5"), "test"),
        dict(overwrite=True),
    )


__from_func_args_factories__ = {
    ArrayBuffer: __func_ArrayBuffer_args_kwargs_factory__,
    HDF5Buffer: __func_HDF5Buffer_args_kwargs_factory__,
}

__all_numpy_builtin_methods__ = [
    pytest.param("astype", (), {"dtype": np.float32}),
    pytest.param("conj", (), {}),
    pytest.param("conjugate", (), {}),
    pytest.param("copy", (), {}),
    pytest.param("flatten", (), {"order": "C"}),
    pytest.param("ravel", (), {"order": "C"}),
    pytest.param("reshape", ((4,),), {}),
    pytest.param("resize", ((4,),), {}),
    pytest.param("swapaxes", (), {"axis1": 0, "axis2": 1}),
    pytest.param("transpose", (), {}),
]
