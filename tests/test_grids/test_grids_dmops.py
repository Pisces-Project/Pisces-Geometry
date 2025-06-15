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
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymetric.grids.base import GridBase

from .utils import __all_grid_classes_params__, __grid_factories__

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
    zeros_field = grid.zeros(dtype=float, include_ghosts=True)

    # Compute a gradient.
    _ = grid.dense_gradient(zeros_field, grid.axes)


@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
@pytest.mark.parametrize("coordinate_system", __coordinate_systems_run__)
def test_divergence_runs(grid_class, coordinate_system, coordinate_systems):
    # Create the grid for this test.
    grid = __grid_factories__[grid_class](coordinate_systems[coordinate_system])

    # Initialize a zeros grid.
    zeros_field = grid.zeros(element_shape=(3,), dtype=float, include_ghosts=True)

    # Compute a gradient.
    _ = grid.dense_vector_divergence(zeros_field, grid.axes)


@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
@pytest.mark.parametrize("coordinate_system", __coordinate_systems_run__)
def test_laplacian_runs(grid_class, coordinate_system, coordinate_systems):
    # Create the grid for this test.
    grid = __grid_factories__[grid_class](coordinate_systems[coordinate_system])

    # Initialize a zeros grid.
    zeros_field = grid.zeros(dtype=float, include_ghosts=True)

    # Compute a gradient.
    _ = grid.dense_scalar_laplacian(zeros_field, grid.axes)


@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
def test_gradient_answer(grid_class, coordinate_systems):
    # Create the grid for this test.
    grid: GridBase = __grid_factories__[grid_class](coordinate_systems["spherical"])

    # We'll use f(r) = r * cos(theta) -> grad f = cos(theta), -r sin(theta).
    function = lambda r, theta: r * np.cos(theta)
    field = grid.compute_function_on_grid(function, output_axes=["r", "theta"])

    # Compute the covariant gradient
    cov_grad_field = grid.dense_covariant_gradient(field, ["r", "theta"])[..., :-1]

    # Compute the contravariant gradient
    contra_grad_field = grid.dense_contravariant_gradient(field, ["r", "theta"])[
        ..., :-1
    ]

    # -- Check Shapes -- #
    assert tuple(cov_grad_field.shape) == tuple(grid.gdd[:2]) + (
        2,
    ), "Wrong covariant shape"
    assert tuple(contra_grad_field.shape) == tuple(grid.gdd[:2]) + (
        2,
    ), "Wrong contravariant shape"

    # -- Check answers --- #
    R, THETA = grid.compute_domain_mesh(axes=["r", "theta"], origin="global")
    cov_grad_field_true = np.zeros_like(cov_grad_field)
    cov_grad_field_true[..., 0] = np.cos(THETA)
    cov_grad_field_true[..., 1] = -R * np.sin(THETA)

    assert_allclose(cov_grad_field, cov_grad_field_true, rtol=0.001)

    # -- Check answers --- #
    contra_grad_field_true = np.zeros_like(contra_grad_field)
    contra_grad_field_true[..., 0] = np.cos(THETA)
    contra_grad_field_true[..., 1] = -np.sin(THETA) / R

    assert_allclose(contra_grad_field, contra_grad_field_true, rtol=0.001)


@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
def test_divergence_answer(grid_class, coordinate_systems):
    # Use spherical coordinates
    grid: GridBase = __grid_factories__[grid_class](coordinate_systems["spherical"])

    # v(r) = r^2 in radial direction
    r_func = lambda r, theta: r**2
    field = grid.compute_function_on_grid(r_func, output_axes=["r", "theta"])
    vector_field = grid.zeros(
        element_shape=(3,), axes=["r", "theta"], dtype=float, include_ghosts=True
    )
    vector_field[..., 0] = field  # Radial component only

    divergence_field = grid.dense_vector_divergence(vector_field, ["r", "theta"])

    # Truth: div(r^2 * rhat) = 4r
    R, _ = grid.compute_domain_mesh(axes=["r", "theta"], origin="global")
    expected = 4 * R

    assert_allclose(divergence_field, expected, rtol=1e-3)


def test_laplacian_answer(coordinate_systems):
    """
    For this test, we build a custom grid to get far away from the
    origin in the spherical coordinate system.

    """
    from pymetric import UniformGrid

    cs = coordinate_systems["spherical"]
    bbox = [[1, 0, 0], [2, np.pi, 2 * np.pi]]
    dd = [3000, 100, 100]
    ghost_zones = 2

    grid = UniformGrid(cs, bbox, dd, ghost_zones=ghost_zones, center="cell")

    # f(r) = 1/r
    scalar_func = lambda r, theta: 1 / r
    field = grid.compute_function_on_grid(scalar_func, output_axes=["r", "theta"])

    laplacian_field = grid.dense_scalar_laplacian(field, ["r", "theta"])

    # Truth: Laplacian of 1/r is 0 away from r=0
    expected = np.zeros_like(laplacian_field)

    # Allow a slightly higher tolerance near r=0 if needed
    assert np.amax(np.abs(laplacian_field)) < 1e-1
