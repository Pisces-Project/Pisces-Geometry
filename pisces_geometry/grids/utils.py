"""
Utilities for managing grids.
"""
import numpy as np


def _get_coordinate_parameters(bounding_box, cell_size, grid_type: str = "cell"):
    """
    Same as `get_coordinate_parameters`, but ignores all checks.
    """
    # Compute start and end coordinates
    offset = (cell_size / 2) if grid_type == "cell" else 0
    start = bounding_box[0, :] + offset
    end = bounding_box[1, :] + offset

    return tuple(zip(start, end, cell_size))


def get_coordinate_parameters(bounding_box, cell_size, grid_type: str = "cell"):
    """
    Computes coordinate slice parameters for a structured grid.

    Parameters
    ----------
    bounding_box : :py:class:`numpy.ndarray`
        A ``(2, ndim)`` array defining the lower and upper bounds of the grid.
    cell_size : :py:class:`numpy.ndarray` or list
        A ``(ndim,)`` array specifying the cell size along each axis.
    grid_type : str, optional
        The grid type, either ``'cell'`` (default) or ``'edge'``. Determines whether
        the coordinate points are centered in cells or placed at edges.

    Returns
    -------
    tuple of tuples
        A tuple containing coordinate parameters in the form (start, end, step) for each dimension.

    Example
    -------
    >>> bbox = np.array([[0, 0, 0], [1, 1, 1]])
    >>> cell_size = np.array([0.1, 0.1, 0.1])
    >>> get_coordinate_parameters(bbox, cell_size, 'cell')
    ((0.05, 1.05, 0.1), (0.05, 1.05, 0.1), (0.05, 1.05, 0.1))
    """
    #
    bounding_box = np.asarray(bounding_box, dtype=float)
    cell_size = np.asarray(cell_size, dtype=float)

    if bounding_box.shape != (2, len(cell_size)):
        raise ValueError(
            f"Bounding box shape {bounding_box.shape} does not match cell size length {len(cell_size)}."
        )

    if grid_type not in {"cell", "edge"}:
        raise ValueError("grid_type must be either 'cell' or 'edge'.")

    # Compute start and end coordinates
    return _get_coordinate_parameters(bounding_box, cell_size, grid_type)
