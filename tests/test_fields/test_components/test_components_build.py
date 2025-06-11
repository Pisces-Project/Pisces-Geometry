"""
Testing suite for testing the creation of component objects.
"""
import numpy as np
import pytest

from pymetric import FieldComponent
from tests.test_fields.utils import (
    __all_buffer_classes__,
    __from_array_args_factories__,
)

from .utils import __from_func_args_factories__, component_test_directory


@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)
@pytest.mark.parametrize("method", ["ones", "zeros", "full", "empty"])
def test_comp_constructors(
    cs_flag, buffer_class, method, uniform_grids, component_test_directory
):
    """
    Test that components can be generated using the .zeros, .ones, .full, and .empty constructors
    with all buffer backends and coordinate systems.
    """
    # Fetch the generator factories from `utils.py` and
    # fetch out the relevant args and kwargs.
    bargs, bkwargs = __from_array_args_factories__[buffer_class](
        component_test_directory
    )
    args, kwargs = (), {}
    # Configure the shape, dtype, and create the tempdir if
    # not already existent. We create and fill the kwargs and
    # expected values ahead of time.
    bkwargs["dtype"] = np.float64

    if method == "full":
        kwargs["fill_value"] = 10.0

    # Fetch the factory from the buffer class.
    factory = getattr(FieldComponent, method)

    # Extract the coordinate system / grid so that
    # we can use it in the factory and then extract axes.
    grid = uniform_grids[cs_flag]

    # Build the buffer.
    _ = factory(
        grid,
        grid.axes,
        *args,
        buffer_args=bargs,
        buffer_class=buffer_class,
        buffer_kwargs=bkwargs,
        **kwargs,
    )


@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)
def test_comp_from_function(buffer_class, component_test_directory, uniform_grids):
    """
    Test our ability to build a field from the function x^2 + y^2 + z^2
    on a 3D cartesian grid.
    """
    # --- Setup --- #
    # construct the tempdir for the HDF5 backed case.
    cartesian_grid = uniform_grids["cartesian3D"]

    # Define simple test function: f(x, y) = x + y
    def func(x, y, z):
        return x**2 + y**2 + z**2

    # Fetch the generator factories from `utils.py` and
    # fetch out the relevant args and kwargs.
    args, kwargs = __from_func_args_factories__[buffer_class](component_test_directory)

    # --- Run Constructor --- #
    component = FieldComponent.from_function(
        func,
        cartesian_grid,
        ["x", "y", "z"],
        buffer_args=args,
        buffer_class=buffer_class,
        buffer_kwargs=kwargs,
    )

    # --- Validations --- #
    # check that the values are correct.
    X, Y, Z = cartesian_grid.compute_domain_mesh(axes=["x", "y", "z"], origin="global")
    Ftrue = func(X, Y, Z)
    F = component.as_array()

    np.testing.assert_allclose(F, Ftrue), "Values not equal."
