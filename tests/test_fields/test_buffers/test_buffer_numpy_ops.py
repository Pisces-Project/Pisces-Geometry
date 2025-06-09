"""
Unit tests for the core buffer types in the PyMetric library.

This module verifies the correct behavior and NumPy compatibility of
ArrayBuffer and HDF5Buffer classes, including construction, data integrity,
and support for NumPy-like operations and ufuncs.
"""
import os

import numpy as np
import pytest

from pymetric.fields.buffers import ArrayBuffer, HDF5Buffer

# ------------------------------------- #
# Settings                              #
# ------------------------------------- #
# Define a list of buffer classes to be tested.
# This allows for parameterized tests that can easily iterate over
# all supported buffer types (e.g., ArrayBuffer, HDF5Buffer) without
# duplicating test logic. Each entry is wrapped with a pytest mark
# for selective test execution and reporting.
__all_buffer_classes_params__ = [
    pytest.param(ArrayBuffer, marks=pytest.mark.array),
    pytest.param(HDF5Buffer, marks=pytest.mark.hdf5),
]
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
# ------------------------------------- #
# Test Fixtures (Module Level)          #
# ------------------------------------- #
# This section of the testing suite is used for
# building relevant testing fixtures.
@pytest.fixture(scope="module")
def simple_data_array():
    """
    Returns a simple 2x2 NumPy array of ones for use as test data
    in buffer generation and validation tasks.
    """
    return np.ones((2, 2))

# ------------------------------------- #
# Utility Functions (Module Level)      #
# ------------------------------------- #
#
# This section contains helper functions used throughout the test suite,
# such as buffer construction utilities that abstract away differences
# between buffer implementations (e.g., ArrayBuffer vs. HDF5Buffer).
#
def build_buffer(buffer_class, data, tempdir, name="default"):
    """
    Simple buffer generation logic to encapsulate
    logic for .from_array that depends on the buffer class.
    """
    if buffer_class is HDF5Buffer:
        file = os.path.join(tempdir, f"{name}.hdf5")
        return buffer_class.from_array(data, file=file, path="test", create_file=True)
    return buffer_class.from_array(data)

@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
@pytest.mark.parametrize("ufunc", [np.add, np.multiply, np.sqrt, np.negative])
def test_numpy_ufunc_behavior(buffer_class, simple_data_array, tmp_path_factory, ufunc):
    """
    Validate NumPy ufunc behavior on all buffer types.

    Tests each ufunc in two modes:
    - Without `out=` → should return a NumPy array.
    - With `out=buffer` → should modify and return the same buffer.

    Ensures that:
    - Return types follow our `__array_ufunc__` dispatch semantics.
    - Results match numerically in both modes.
    """
    tempdir = tmp_path_factory.mktemp("buffers")
    buffer = build_buffer(
        buffer_class, simple_data_array, str(tempdir), name=f"ufunc_{ufunc.__name__}"
    )

    # Construct the inputs required by the ufunc.
    nin = ufunc.nin
    args = (buffer, simple_data_array) if nin > 1 else (buffer,)

    # Apply ufunc with and without `out=`
    result_np = ufunc(*args)
    result_buf = ufunc(*args, out=buffer)

    # Validate return types
    assert isinstance(result_np, np.ndarray), (
        f"Expected NumPy array when out=None, got {type(result_np)}"
    )
    assert isinstance(result_buf, buffer_class), (
        f"Expected {buffer_class.__name__} when using out=, got {type(result_buf)}"
    )

    # Validate numerical equivalence
    np.testing.assert_allclose(
        result_np, np.asarray(result_buf),
        err_msg=f"{ufunc.__name__} result mismatch between array and buffer"
    )


@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
@pytest.mark.parametrize("method_name,args,kwargs", __all_numpy_builtin_methods__)
def test_numpy_like_methods(
    buffer_class, method_name, args, kwargs, simple_data_array, tmp_path_factory
):
    """
    Verify that NumPy-like transformation methods on buffers behave correctly.

    For each method (e.g. `reshape`, `astype`), verify that:
    - Calling with `numpy=False` returns a new buffer of the same class.
    - Calling with `numpy=True` returns a raw NumPy array.
    - Both results are numerically identical.

    Additional arguments:
    - `bargs`: Passed to `.from_array()` to reconstruct buffers (HDF5 needs file/path).
    """
    tempdir = tmp_path_factory.mktemp("buffer_transforms")
    buffer = build_buffer(
        buffer_class, simple_data_array, str(tempdir), name=method_name
    )

    method = getattr(buffer, method_name)

    # HDF5 requires explicit file/path to rewrap result
    if buffer_class is HDF5Buffer:
        bargs = (buffer.file, f"modified_{method_name}")
    else:
        bargs = ()

    # 1. Return as buffer
    result = method(*args, numpy=False, bargs=bargs, **kwargs)
    assert isinstance(result, buffer_class), f"{method_name} did not return a buffer"

    # 2. Return as raw NumPy array
    result_np = method(*args, numpy=True, **kwargs)
    assert isinstance(result_np, np.ndarray), f"{method_name} did not return ndarray"

    # 3. Ensure result content matches between buffer and array
    np.testing.assert_allclose(
        result_np, result.as_array(),
        err_msg=f"{method_name} produced mismatched result"
    )

