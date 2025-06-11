"""
Testing suite to handle tests of the grid level differential geometry
methods.

This is a **critical** testing suite because the differential geometry at the
grid level is one of the most complicated elements of the code base.

The plan is to perform each of the differential geometry operations on
each of the grid classes for 3 sets of coordinate system:

- cartesian3d
- spherical
- OblateHomoeoidal

By ensuring that each of the 3 produces values, we can ensure that the API
at least calls properly. True answer testing is only performed in the spherical case.
"""
import pytest

from pymetric import DenseTensorField
from tests.test_grids.utils import __all_grid_classes_params__, __grid_factories__

# --- Configuration --- #
__coordinate_systems_run__ = [
    pytest.param("cartesian3D"),
    pytest.param("spherical"),
    pytest.param("oblate_homoeoidal"),
]


@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
@pytest.mark.parametrize("coordinate_system", __coordinate_systems_run__)
def test_gradient_runs(grid_class, coordinate_system, coordinate_systems):
    # Create the grid for this test.
    grid = __grid_factories__[grid_class](coordinate_systems[coordinate_system])

    # Initialize a zeros grid.
    zeros_field = DenseTensorField.zeros(grid, grid.axes, 0)

    # Compute a gradient.
    _ = zeros_field.gradient(output_axes=grid.axes)


@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
@pytest.mark.parametrize("coordinate_system", __coordinate_systems_run__)
def test_divergence_runs(grid_class, coordinate_system, coordinate_systems):
    # Create the grid for this test.
    grid = __grid_factories__[grid_class](coordinate_systems[coordinate_system])

    # Initialize a zeros grid.
    zeros_field = DenseTensorField.zeros(grid, grid.axes, 1)

    # Compute a gradient.
    _ = zeros_field.vector_divergence(output_axes=grid.axes)


@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
@pytest.mark.parametrize("coordinate_system", __coordinate_systems_run__)
def test_laplacian_runs(grid_class, coordinate_system, coordinate_systems):
    # Create the grid for this test.
    grid = __grid_factories__[grid_class](coordinate_systems[coordinate_system])

    # Initialize a zeros grid.
    zeros_field = DenseTensorField.zeros(grid, grid.axes, 0)

    # Compute a gradient.
    _ = zeros_field.scalar_laplacian(output_axes=grid.axes)
