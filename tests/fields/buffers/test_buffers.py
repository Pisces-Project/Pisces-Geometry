"""
Tests for the core PyMetric buffer types.
"""
import os

import numpy as np
import pytest
import unyt

from pymetric.fields.buffers import ArrayBuffer, HDF5Buffer, UnytArrayBuffer

# Create fixture class lists for easier, more readable
# testing semantics / syntax.
__all_buffer_classes_params__ = [
    pytest.param(ArrayBuffer, marks=pytest.mark.array),
    pytest.param(UnytArrayBuffer, marks=pytest.mark.unyt),
    pytest.param(HDF5Buffer, marks=pytest.mark.hdf5),
]
__unit_buffer_classes_params__ = [
    pytest.param(UnytArrayBuffer, marks=pytest.mark.unyt),
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
    Create a simple array of data that can be
    used as the underlying object for buffer generation
    tasks.
    """
    return np.ones((2, 2))


# ------------------------------------- #
# Utility Functions (Module Level)      #
# ------------------------------------- #
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
    Verify that all buffer types correctly wrap plain NumPy arrays.

    This includes:

    - correct shape and dtype
    - content equality
    - instantiation without errors
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
    assert isinstance(
        buffer_slice, buffer_class.__representation_types__
    ), "Wrong type."


@pytest.mark.parametrize("buffer_class", __unit_buffer_classes_params__)
def test_from_array_unyt_array(buffer_class, simple_data_array, tmp_path_factory):
    """
    Verify that unit-aware buffer types accept unyt_array inputs.

    This includes:

    - correct shape and dtype
    - unit preservation
    - value equality
    """
    tempdir = tmp_path_factory.mktemp("buffers")
    unyt_data = unyt.unyt_array(simple_data_array, units="keV")

    buffer = build_buffer(buffer_class, unyt_data, str(tempdir), name="from_unyt")

    assert hasattr(buffer, "units"), "Buffer missing 'units' attribute"
    assert str(buffer.units) == "keV", "Unit mismatch"
    assert buffer.shape == (2, 2), "Shape mismatch"
    np.testing.assert_allclose(
        buffer.as_array(), simple_data_array, err_msg="Numerical data mismatch"
    )

    # Ensure that slicing produces the correct output.
    # NOTE: this might cast to different types so we need
    #       to check that.
    buffer_slice = buffer[:, 0]  # size (2,) array.
    assert buffer_slice.shape == (2,), "Shape mismatch"
    assert isinstance(
        buffer_slice, buffer_class.__representation_types__
    ), "Wrong type."


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

    if method == "full":
        kwargs["fill_value"] = 3.14

    # Add HDF5-specific args to ensure we are
    # able to build correctly.
    if buffer_class is HDF5Buffer:
        kwargs.update(
            {
                "file": os.path.join(tempdir, f"{method}_{buffer_class.__name__}.h5"),
                "name": f"buffer_from_{method}",
                "create_file": True,
            }
        )

    # START TEST: Begin by loading in the buffer,
    # then check the shape, dtype, etc. Finally we check
    # the value.
    buffer = factory(shape, **kwargs)

    # Basic checks
    assert buffer.shape == shape
    assert buffer.dtype == dtype

    # Check content only if well-defined
    if expected_value is not None:
        arr = buffer.as_array()
        np.testing.assert_allclose(
            arr, expected_value, err_msg=f"{method} failed on {buffer_class.__name__}"
        )


@pytest.mark.parametrize("buffer_class", __unit_buffer_classes_params__)
@pytest.mark.parametrize("method", ["ones", "zeros", "full", "empty"])
def test_buffer_constructors_units(buffer_class, method, tmp_path_factory):
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
    u = unyt.Unit("keV")
    expected_value = dict(zeros=0.0 * u, ones=1.0 * u, full=3.14 * u, empty=None)[
        method
    ]
    kwargs = {"dtype": dtype, "units": u}

    if method == "full":
        kwargs["fill_value"] = 3.14

    # Add HDF5-specific args to ensure we are
    # able to build correctly.
    if buffer_class is HDF5Buffer:
        kwargs.update(
            {
                "file": os.path.join(tempdir, f"{method}_{buffer_class.__name__}.h5"),
                "name": f"buffer_from_{method}",
                "create_file": True,
            }
        )

    # START TEST: Begin by loading in the buffer,
    # then check the shape, dtype, etc. Finally we check
    # the value.
    buffer = factory(shape, **kwargs)

    # Basic checks
    assert buffer.shape == shape
    assert buffer.dtype == dtype
    assert buffer.units == u, "Unit mismatch"

    # Check content only if well-defined
    if expected_value is not None:
        arr = buffer[...]
        np.testing.assert_allclose(
            arr, expected_value, err_msg=f"{method} failed on {buffer_class.__name__}"
        )


# ============================================ #
# TESTING FUNCTIONS: Buffer NumPy Semantics    #
# ============================================ #
@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
@pytest.mark.parametrize("ufunc", [np.add, np.multiply, np.sqrt, np.negative])
def test_numpy_ufunc_behavior(buffer_class, simple_data_array, tmp_path_factory, ufunc):
    """
    Test numpy ufunc behavior. For each of these, we will perform the ufunc twice:

    - Once with no out= specified.
    - Once with out= specified.

    We use the simply_data_array as many times as needed to have the correct number of
    inputs, the first input being the buffer.

    The rules for numpy semantics are in the docs; but in summary we unwrap the buffer unless
    out = is specified.
    """
    tempdir = tmp_path_factory.mktemp("buffers")
    buffer = build_buffer(
        buffer_class, simple_data_array, str(tempdir), name=f"ufunc_{ufunc.__name__}"
    )

    # Construct the inputs.
    nin = ufunc.nin
    if nin > 1:
        args = (buffer, simple_data_array)
    else:
        args = (buffer,)

    # TEST START: Compute the relevant ufunc and retrieve the output.
    result = ufunc(*args)
    result_out = ufunc(*args, out=buffer)

    # Based on `out` we anticipate either getting a matching buffer type
    # of we anticipate getting a representation type.
    assert isinstance(
        result_out, buffer_class
    ), f"out= not changing class. ({type(result)},{type(buffer)})"
    assert isinstance(
        result, buffer_class.__representation_types__
    ), f"out=None not unwrapped. ({type(result)},{type(buffer)})"

    # Ensure that the buffer did retain the data in the out case.
    np.testing.assert_allclose(result, result_out.as_array())
