"""
Unit tests for the core buffer types in the PyMetric library.

This module verifies the correct behavior and NumPy compatibility of
ArrayBuffer and HDF5Buffer classes, including construction, data integrity,
and support for NumPy-like operations and ufuncs.
"""
import numpy as np
import pytest

from tests.test_fields.utils import (
    __all_buffer_classes__,
    __from_array_args_factories__,
    test_array,
)

from .utils import __all_numpy_builtin_methods__, buffer_test_directory


@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)
@pytest.mark.parametrize("ufunc", [np.add, np.multiply, np.sqrt, np.negative, np.abs])
def test_numpy_ufunc_behavior(buffer_class, ufunc, buffer_test_directory, test_array):
    """
    Validate NumPy ufunc behavior on all buffer types.

    Tests each ufunc in two modes:
    - Without `out=` → should return a NumPy array.
    - With `out=buffer` → should modify and return the same buffer.

    Ensures that:
    - Return types follow our `__array_ufunc__` dispatch semantics.
    - Results match numerically in both modes.
    """
    # Fetch the generator factories from `utils.py` and
    # fetch out the relevant args and kwargs.
    args, kwargs = __from_array_args_factories__[buffer_class](buffer_test_directory)

    # Create the buffer from array using the provided args and
    # kwargs along with the test array.
    buffer = buffer_class.from_array(test_array, *args, **kwargs)

    # Construct the inputs required by the ufunc.
    nin = ufunc.nin
    uargs = (buffer, test_array) if nin > 1 else (buffer,)

    # Apply ufunc with and without `out=`
    result_np = ufunc(*uargs)
    result_buf = ufunc(*uargs, out=buffer)

    # Validate return types
    assert isinstance(
        result_np, np.ndarray
    ), f"Expected NumPy array when out=None, got {type(result_np)}"
    assert isinstance(
        result_buf, buffer_class
    ), f"Expected {buffer_class.__name__} when using out=, got {type(result_buf)}"

    # Validate numerical equivalence
    np.testing.assert_allclose(
        result_np,
        np.asarray(result_buf),
        err_msg=f"{ufunc.__name__} result mismatch between array and buffer",
    )


@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)
@pytest.mark.parametrize("method_name,args,kwargs", __all_numpy_builtin_methods__)
def test_numpy_like_methods(
    buffer_class, method_name, args, kwargs, buffer_test_directory, test_array
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
    # Fetch the generator factories from `utils.py` and
    # fetch out the relevant args and kwargs.
    bargs, bkwargs = __from_array_args_factories__[buffer_class](buffer_test_directory)

    # Create the buffer from array using the provided args and
    # kwargs along with the test array.
    buffer = buffer_class.from_array(test_array, *bargs, **bkwargs)

    method = getattr(buffer, method_name)

    # 1. Return as buffer
    result = method(*args, numpy=False, bargs=bargs, bkwargs=bkwargs, **kwargs)
    assert isinstance(result, buffer_class), f"{method_name} did not return a buffer"

    # 2. Return as raw NumPy array
    result_np = method(*args, numpy=True, bargs=bargs, bkwargs=bkwargs, **kwargs)
    assert isinstance(result_np, np.ndarray), f"{method_name} did not return ndarray"

    # 3. Ensure result content matches between buffer and array
    np.testing.assert_allclose(
        result_np,
        result.as_array(),
        err_msg=f"{method_name} produced mismatched result",
    )
