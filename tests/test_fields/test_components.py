"""
Testing suite for field components.
"""
import os

import numpy as np
import pytest

from pymetric import (
    ArrayBuffer,
    CartesianCoordinateSystem3D,
    FieldComponent,
    HDF5Buffer,
    SphericalCoordinateSystem,
    UniformGrid,
)

# Create fixture class lists for easier, more readable
# testing semantics / syntax.
__all_buffer_classes_params__ = [
    pytest.param(ArrayBuffer, marks=pytest.mark.array),
    pytest.param(HDF5Buffer, marks=pytest.mark.hdf5),
]
__unit_buffer_classes_params__ = [
    pytest.param(HDF5Buffer, marks=pytest.mark.hdf5),
]

# ------------------------------------- #
# Utility Functions (Module Level)      #
# ------------------------------------- #


# ===================================== #
# TESTING FUNCTIONS: Component Creation #
# ===================================== #
# This tests the ability to construct components
# with each of the relevant backends vis-a-vis all
# of the relevant constructors:
#
# - Direct wrapping,
# - zeros, ones, etc.
# - from_function
@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
@pytest.mark.parametrize("method", ["ones", "zeros", "full", "empty"])
def test_comp_constructors(
    cs_flag, buffer_class, method, uniform_grids, tmp_path_factory
):
    """
    Test that components can be generated using the .zeros, .ones, .full, and .empty constructors
    with all buffer backends and coordinate systems.
    """
    # --- Temp directory setup for HDF5 --- #
    tempdir = tmp_path_factory.mktemp("components_generation")

    # --- Factory and kwargs --- #
    factory = getattr(FieldComponent, method)
    factory_args = []
    factory_kwargs = dict(dtype=np.float64)

    if buffer_class is HDF5Buffer:
        factory_args.extend(
            [os.path.join(tempdir, f"{method}_{cs_flag}.hdf5"), f"field_component"]
        )
        factory_kwargs["create_file"] = True

    if method == "full":
        factory_kwargs["fill_value"] = 3.14

    # --- Grid selection --- #
    grid = uniform_grids[cs_flag]
    axes = grid.axes

    # --- Construct the component --- #
    component = factory(
        grid, axes, *factory_args, buffer_class=buffer_class, **factory_kwargs
    )

    # --- Validations --- #
    assert isinstance(component, FieldComponent), "Output is not a FieldComponent"
    assert component.grid is grid, "Grid mismatch"
    assert component.axes == axes, "Axis mismatch"
    assert component.shape == tuple(grid.shape), "Shape mismatch"
    assert component.buffer.shape == component.shape, "Buffer shape mismatch"
    assert component.buffer.dtype == np.float64, "Dtype mismatch"

    # Buffer and value check
    array = component.as_array()
    if method == "zeros":
        np.testing.assert_allclose(array, 0.0)
    elif method == "ones":
        np.testing.assert_allclose(array, 1.0)
    elif method == "full":
        # Default fill value assumed; modify if needed
        factory_kwargs.setdefault("fill_value", 3.14)
        np.testing.assert_allclose(array, factory_kwargs["fill_value"])
    elif method == "empty":
        assert array.shape == component.shape  # content undefined, just check shape


@pytest.mark.parametrize("buffer_class", __all_buffer_classes_params__)
def test_comp_from_function(buffer_class, tmp_path_factory, uniform_grids):
    """
    Test our ability to build a field from the function x^2 + y^2 + z^2
    on a 3D cartesian grid.
    """
    # --- Setup --- #
    # construct the tempdir for the HDF5 backed case.
    tempdir = tmp_path_factory.mktemp("components_from_function")
    cartesian_grid = uniform_grids["cartesian3D"]

    # Define simple test function: f(x, y) = x + y
    def func(x, y, z):
        return x**2 + y**2 + z**2

    # Set kwargs
    buffer_kwargs = {"dtype": np.float64}
    args = []

    if buffer_class is HDF5Buffer:
        args = [os.path.join(tempdir, "from_function.hdf5"), "data"]
        buffer_kwargs["create_file"] = True

    # --- Run Constructor --- #
    component = FieldComponent.from_function(
        func,
        cartesian_grid,
        ["x", "y", "z"],
        *args,
        buffer_class=buffer_class,
        buffer_kwargs=buffer_kwargs,
    )

    # --- Validations --- #
    # check that the values are correct.
    X, Y, Z = cartesian_grid.compute_domain_mesh(axes=["x", "y", "z"], origin="global")
    Ftrue = func(X, Y, Z)
    F = component.as_array()

    np.testing.assert_allclose(F, Ftrue), "Values not equal."
