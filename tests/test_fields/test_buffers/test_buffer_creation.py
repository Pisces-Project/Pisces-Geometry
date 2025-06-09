"""
Unit tests for the core buffer types in the PyMetric library.

This module verifies the correct construction behavior of
buffers.
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


# ===================================== #
# TESTING FUNCTIONS: Buffer Creation    #
# ===================================== #
@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
def test_from_array_raw_numpy(buffer_class, simple_data_array, tmp_path_factory):
    """
    Test that each buffer class correctly wraps a raw NumPy array using `.from_array()`.

    This test verifies that the buffer:

    - Initializes without error from a NumPy array.
    - Preserves shape and dtype information.
    - Produces equivalent data when unwrapped via `.as_array()`.
    - Supports standard NumPy-style indexing (e.g., slicing returns a valid ndarray).
    """
    tempdir = tmp_path_factory.mktemp("buffers")
    buffer = build_buffer(
        buffer_class, simple_data_array, str(tempdir), name="from_numpy"
    )

    # Ensure that shapes are correct and that everything
    # has correct behaviors.
    assert buffer.shape == (2, 2), "Shape mismatch"
    assert buffer.dtype == simple_data_array.dtype, "Dtype mismatch"
    np.testing.assert_allclose(
        buffer.as_array(), simple_data_array, err_msg="Values do not match"
    )

    # Ensure that slicing produces the correct output.
    # NOTE: this might cast to different types so we need
    #       to check that.
    buffer_slice = buffer[:, 0]  # size (2,) array.
    assert buffer_slice.shape == (2,), "Shape mismatch"
    assert isinstance(buffer_slice, np.ndarray), "Wrong type."


@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
@pytest.mark.parametrize("method", ["ones", "zeros", "full", "empty"])
def test_buffer_constructors(buffer_class, method, tmp_path_factory):
    """
    Test that each of the buffer types can correctly instantiate from
    its relevant generator methods (ones, zeros, full, and empty).
    """
    # Configure the shape, dtype, and create the tempdir if
    # not already existent. We create and fill the kwargs and
    # expected values ahead of time.
    shape = (4, 4)
    dtype = np.float64
    tempdir = tmp_path_factory.mktemp("buffers")

    # Set the expected values and alter kwargs.
    factory = getattr(buffer_class, method)
    expected_value = dict(zeros=0.0, ones=1.0, full=3.14, empty=None)[method]
    kwargs = {"dtype": dtype}
    args = []

    if method == "full":
        kwargs["fill_value"] = 3.14

    # Add HDF5-specific args to ensure we are
    # able to build correctly.
    if buffer_class is HDF5Buffer:
        args = [
            os.path.join(tempdir, f"{method}_{buffer_class.__name__}.h5"),
            f"buffer_from_{method}",
        ]
        kwargs.update(
            {
                "create_file": True,
            }
        )

    # START TEST: Begin by loading in the buffer,
    # then check the shape, dtype, etc. Finally we check
    # the value.
    buffer = factory(shape, *args, **kwargs)

    # Basic checks
    assert buffer.shape == shape
    assert buffer.dtype == dtype

    # Check content only if well-defined
    if expected_value is not None:
        arr = buffer.as_array()
        np.testing.assert_allclose(
            arr, expected_value, err_msg=f"{method} failed on {buffer_class.__name__}"
        )