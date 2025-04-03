def identify_grid_axes_from_array(array, grid):
    """
    Identify which leading axes of the grid are represented in the array shape.

    This attempts to match the leading dimensions of the array against the grid's
    axes in order, stopping as soon as a mismatch is found. Only contiguous leading
    axes are considered valid.

    Parameters
    ----------
    array : np.ndarray
        The array to check.
    grid : _BaseGrid
        The grid whose axes and shape define the spatial structure.

    Returns
    -------
    list[str]
        A list of grid axis names (in order) that are present in the array's shape.

    Notes
    -----
    - If no axes match, an empty list is returned.
    - Broadcasted dimensions (i.e., shape 1) are *not* considered valid matches.

    Examples
    --------
    If `grid.axes = ['x', 'y', 'z']` and `grid.shape = (10, 20, 30)`:

    - array.shape == (10, 20) → ['x', 'y']
    - array.shape == (10, 20, 30) → ['x', 'y', 'z']
    - array.shape == (5, 20, 30) → [] (first axis doesn't match)
    """
    axes = []
    for i, (ax, expected_size) in enumerate(zip(grid.axes, grid.shape)):
        if i >= array.ndim:
            break
        if array.shape[i] == expected_size:
            axes.append(ax)
        else:
            break
    return axes
