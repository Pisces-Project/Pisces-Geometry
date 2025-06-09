"""
Testing suite for looking at DenseField classes and its
subclasses. This does not test the mathematical operations side
of the field API, but focuses on generation, accessors, and other
general semantics of the DenseField class.
"""
import os
import numpy as np
import pytest

from pymetric import (
    ArrayBuffer,
    FieldComponent,
    DenseField,
    HDF5Buffer,
)

# -------------------------------------- #
# Buffer Class Fixtures and Parameters   #
# -------------------------------------- #
__all_buffer_classes_params__ = [
    pytest.param(ArrayBuffer, marks=pytest.mark.array),
    pytest.param(HDF5Buffer, marks=pytest.mark.hdf5),
]


# -------------------------------------- #
# DenseField Construction and Access     #
# -------------------------------------- #
@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
def test_dense_field_accessors(buffer_class, tmp_path_factory, uniform_grids):
    """
    Test DenseField construction and core properties using a basic scalar component.
    """
    # Create the temporary directory and access the relevant grid.
    tempdir = tmp_path_factory.mktemp("dense_field_access")
    grid = uniform_grids["cartesian3D"]
    axes = grid.axes

    # Create the buffer args and kwargs.
    buffer_args = []
    buffer_kwargs = {"dtype": np.float64}
    if buffer_class is HDF5Buffer:
        buffer_args = [os.path.join(tempdir, "access_test.h5"), "data"]
        buffer_kwargs["create_file"] = True

    # Create the full component and DenseField instance.
    comp = FieldComponent.full(
        grid,
        axes,
        buffer_args=buffer_args,
        fill_value=42.0,
        buffer_class=buffer_class,
        buffer_kwargs=buffer_kwargs,
    )
    field = DenseField(grid, comp)

    # Accessors and properties
    assert field.grid is grid
    assert field.buffer.shape == comp.shape
    assert field.shape == comp.shape
    assert field.spatial_shape == comp.spatial_shape
    assert field.element_shape == comp.element_shape
    assert field.axes == comp.axes
    assert field.rank == comp.element_ndim
    assert field.ndim == comp.ndim
    assert field.dtype == comp.dtype
    assert field.size == comp.size
    assert field.spatial_size == comp.spatial_size
    assert field.element_size == comp.element_size
    assert field.is_scalar

    # Value checks
    np.testing.assert_allclose(field[:], 42.0)
    np.testing.assert_allclose(field.buffer[:], 42.0)


@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
def test_dense_field_array_ufunc_scalar(buffer_class, tmp_path_factory, uniform_grids):
    """
    Test np.add and np.multiply with scalars and verify broadcasting and buffer integrity.
    """
    tempdir = tmp_path_factory.mktemp("dense_field_ufunc")
    grid = uniform_grids["cartesian2D"]
    axes = grid.axes

    buffer_args = []
    buffer_kwargs = {"dtype": np.float64}
    if buffer_class is HDF5Buffer:
        buffer_args = [os.path.join(tempdir, "dense_ufunc.h5"), "data"]
        buffer_kwargs["create_file"] = True

    comp = FieldComponent.ones(
        grid,
        axes,
        buffer_args=buffer_args,
        buffer_class=buffer_class,
        buffer_kwargs=buffer_kwargs,
    )
    field = DenseField(grid, comp)

    result = np.add(field, 1.0)
    np.testing.assert_allclose(result, 2.0)

    result = np.multiply(field, 10.0)
    np.testing.assert_allclose(result, 10.0)


@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
def test_dense_field_array_ufunc_out(buffer_class, tmp_path_factory, uniform_grids):
    """
    Test DenseField out= kwarg handling for in-place operations.
    """
    tempdir = tmp_path_factory.mktemp("dense_field_out")
    grid = uniform_grids["cartesian2D"]
    axes = grid.axes

    args = []
    kwargs = {"dtype": np.float64}
    if buffer_class is HDF5Buffer:
        args = [os.path.join(tempdir, "dense_out1.h5"), "data1"]
        kwargs["create_file"] = True

    comp1 = FieldComponent.ones(
        grid,
        axes,
        buffer_args=args,
        buffer_class=buffer_class,
        buffer_kwargs=kwargs,
    )
    field1 = DenseField(grid, comp1)

    # Second buffer (output)
    args_out = []
    if buffer_class is HDF5Buffer:
        args_out = [os.path.join(tempdir, "dense_out2.h5"), "data2"]
        kwargs["create_file"] = True

    comp2 = FieldComponent.zeros(
        grid,
        axes,
        buffer_args=args_out,
        buffer_class=buffer_class,
        buffer_kwargs=kwargs,
    )
    field2 = DenseField(grid, comp2)

    # In-place multiply
    np.multiply(field1, 3.0, out=field2)
    np.testing.assert_allclose(field2[:], 3.0)


@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
def test_dense_field_array_function(buffer_class, tmp_path_factory, uniform_grids):
    """
    Test that NumPy high-level functions (e.g., np.sum) forward to the buffer.
    """
    tempdir = tmp_path_factory.mktemp("dense_field_function")
    grid = uniform_grids["cartesian2D"]
    axes = grid.axes

    buffer_args = []
    buffer_kwargs = {"dtype": np.float64}
    if buffer_class is HDF5Buffer:
        buffer_args = [os.path.join(tempdir, "dense_func.h5"), "data"]
        buffer_kwargs["create_file"] = True

    comp = FieldComponent.full(
        grid,
        axes,
        fill_value=2.0,
        buffer_args=buffer_args,
        buffer_class=buffer_class,
        buffer_kwargs=buffer_kwargs,
    )
    field = DenseField(grid, comp)

    # Sum over entire field
    result = np.sum(field)
    expected = np.sum(np.full(grid.shape, 2.0))
    assert result == expected

    # Mean over spatial axis
    result2 = np.mean(field, axis=(0, 1))
    assert np.allclose(result2, 2.0)
