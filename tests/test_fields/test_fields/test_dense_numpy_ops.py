"""
Unit tests for the core buffer types in the PyMetric library.

This module verifies the correct behavior and NumPy compatibility of
ArrayBuffer and HDF5Buffer classes, including construction, data integrity,
and support for NumPy-like operations and ufuncs.
"""
import numpy as np
import pytest
from pygments.lexers.configs import DesktopLexer

from pymetric import DenseField
from tests.test_fields.utils import (
    __all_buffer_classes__,
    __from_array_args_factories__,
)

from .utils import __all_numpy_builtin_methods__, dense_test_directory


@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)
@pytest.mark.parametrize("ufunc", [np.add, np.multiply, np.sqrt, np.negative, np.abs])
def test_numpy_ufunc_behavior(buffer_class, ufunc, dense_test_directory, uniform_grids):
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
    args, kwargs = __from_array_args_factories__[buffer_class](dense_test_directory)

    grid = uniform_grids["spherical"]
    # Construct the dense field from the test array.
    field = DenseField.ones(
        grid,
        ["r", "theta"],
        buffer_class=buffer_class,
        buffer_args=args,
        buffer_kwargs=kwargs,
    )
    field2 = DenseField.ones_like(field)

    # Construct the inputs required by the ufunc.
    nin = ufunc.nin
    uargs = (field, field2) if nin > 1 else (field,)

    # Apply ufunc with and without `out=`
    result_np = ufunc(*uargs)
    result_buf = ufunc(*uargs, out=field)

    # Validate return types
    assert isinstance(
        result_np, DenseField
    ), f"Expected NumPy array when out=None, got {type(result_np)}"
    assert isinstance(
        result_buf, DenseField
    ), f"Expected {DenseField.__name__} when using out=, got {type(result_buf)}"

    # Validate numerical equivalence
    np.testing.assert_allclose(
        np.asarray(result_np),
        np.asarray(result_buf),
        err_msg=f"{ufunc.__name__} result mismatch between array and buffer",
    )
