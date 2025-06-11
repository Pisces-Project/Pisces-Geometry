"""
Utilities for buffer tests.
"""
import pytest
from pymetric.fields.buffers import ArrayBuffer, HDF5Buffer
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

# ============================= #
# Fixtures                      #
# ============================= #
@pytest.fixture
def buffer_test_directory(tmp_path_factory):
    """
    The buffer test directory.
    """
    return tmp_path_factory.mktemp("buffer_tests")

