"""
Utility functions for working with array-based coordinate grids.

This module provides helper functions used in Pisces-Geometry for manipulating and analyzing
multi-dimensional arrays, particularly in the context of numerical coordinate systems and
differential operations.
"""

from typing import Sequence, Union

import numpy as np


def get_grid_spacing(
    coordinate_grid: np.ndarray, is_uniform: bool = False
) -> Sequence[Union[float, np.ndarray]]:
    """
    Compute the spacing between elements in a coordinate grid.

    This function determines the spacing along each dimension of a multi-dimensional
    coordinate grid. It supports both uniform and non-uniform grids and is typically
    used in gradient, divergence, and Laplacian computations.

    Parameters
    ----------
    coordinate_grid : np.ndarray
        A NumPy array of shape (..., ndim), where the last axis corresponds to the
        coordinate dimensions. For example, a 3D grid over (x, y, z) would have shape
        (Nx, Ny, Nz, 3).
    is_uniform : bool, optional
        If True, assumes the coordinate grid is uniform and computes spacing using
        a simple difference between adjacent grid points. If False, spacing is computed
        using `np.diff` along each axis independently.

    Returns
    -------
    Sequence[Union[float, np.ndarray]]
        A list of spacing values, one per spatial axis. For uniform grids, this will be
        a list of floats. For non-uniform grids, each entry will be an array of shape
        matching the grid (minus one element along the respective axis).

    Raises
    ------
    ValueError
        If the input array is not at least 2D (i.e., does not include a coordinate axis).

    Examples
    --------
    .. code-block:: python

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y, indexing='ij')
        grid = np.stack([X, Y], axis=-1)
        get_grid_spacing(grid, is_uniform=True)
        [0.111..., 0.0526...]

    Notes
    -----
    - For uniform grids, the function assumes that all spacing along a given axis is constant
      and uses the difference between the first two adjacent values.
    - For non-uniform grids, spacing is computed via `np.diff` and may vary across the grid.
    """
    grid_ndim = coordinate_grid.ndim - 1
    if is_uniform:
        spacing = (
            coordinate_grid[*[1] * grid_ndim][:] - coordinate_grid[*[0] * grid_ndim][:]
        )
    else:
        spacing = [np.diff(coordinate_grid, axis=_i) for _i in range(grid_ndim)]

    return spacing
