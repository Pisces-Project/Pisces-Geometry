"""
Unit tests for the core buffer types in the PyMetric library.

This module verifies the correct construction behavior of
buffers.
"""
import numpy as np
import pytest

from tests.test_fields.utils import (
    __all_buffer_classes__,
    __from_array_args_factories__,
    test_array,
)

from .utils import buffer_test_directory


@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)
def test_from_array_raw_numpy(buffer_class, buffer_test_directory, test_array):
    """
    Test that each buffer class correctly wraps a raw NumPy array using `.from_array()`.

    This test verifies that the buffer:

    - Initializes without error from a NumPy array.
    - Preserves shape and dtype information.
    - Produces equivalent data when unwrapped via `.as_array()`.
    - Supports standard NumPy-style indexing (e.g., slicing returns a valid ndarray).
    """
    # Fetch the generator factories from `utils.py` and
    # fetch out the relevant args and kwargs.
    args, kwargs = __from_array_args_factories__[buffer_class](buffer_test_directory)

    # Create the buffer from array using the provided args and
    # kwargs along with the test array.
    _ = buffer_class.from_array(test_array, *args, **kwargs)


@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)
@pytest.mark.parametrize("method", ["ones", "zeros", "full", "empty"])
def test_buffer_constructors(buffer_class, method, buffer_test_directory):
    """
    Test that each of the buffer types can correctly instantiate from
    its relevant generator methods (ones, zeros, full, and empty).
    """
    # Fetch the generator factories from `utils.py` and
    # fetch out the relevant args and kwargs.
    args, kwargs = __from_array_args_factories__[buffer_class](buffer_test_directory)

    # Configure the shape, dtype, and create the tempdir if
    # not already existent. We create and fill the kwargs and
    # expected values ahead of time.
    args = ((4, 4),) + args
    kwargs["dtype"] = np.float64

    if method == "full":
        kwargs["fill_value"] = 10.0

    # Fetch the factory from the buffer class.
    factory = getattr(buffer_class, method)

    # Build the buffer.
    _ = factory(*args, **kwargs)
