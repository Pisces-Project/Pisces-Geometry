"""
Module for managing fields of data defined on grids in Pisces geometry.
"""
"""
Scalar, vector, and tensor field support over grid domains.
"""
from typing import List

import numpy as np
from pisces_geometry.grids.base import BaseGrid

class _GenericField(np.ndarray):

    # @@ Initialization Methods @@ #
    # These methods form the sequence of procedures during __init__. Alterations
    # can be made in subclasses to ensure custom behavior is obtained without
    # major rebasing.
    def __validate_arguments__(self, field, grid, axes):
        """
        This initialization procedure fixes any issues with the field and / or the grid
        and raises errors if there are any issues. The entire call is wrapped in a try / except
        pattern to ensure that errors are caught and information is passed on to the user.
        """
        # Convert the axes to an axes mask and create the attributes.
        self._axes: List[str] = axes if axes is not None else grid.axes
        self._axes_mask: np.ndarray = grid.coordinate_system.build_axes_mask(self._axes)

        # Extract the grid shape and the number of dimensions so that the field
        # can be checked and the rank determined.
        _grid_shape = grid.ddims
        _grid_dimensions = grid.ndim

        # Check that the field has the right shapes. If the field is a tensor field, it still needs
        # to specify all the axes even if some of them are 0.
        _expected_grid_shape = tuple(_grid_shape[self._axes_mask]) + (_grid_dimensions,)*(field.ndim-_grid_dimensions)
        if field.shape != _expected_grid_shape:
            raise ValueError(f"The shape of the input field is {field.shape}, which does not match the expected shape ({_expected_grid_shape}) "
                             f"of a field over a {_grid_shape} grid.")
        else:
            self._rank: int = field.ndim - _grid_dimensions

    def __init__(self,
                 field: np.ndarray,
                 grid: BaseGrid,
                 axes: List[str] = None,
                 **kwargs):
        """
        Initialize a generic field object given the field data and grid.

        The field ``field`` corresponds to the tensor field being loaded while the ``grid`` provides
        the domain information. Once initialized, the field will automatically resolve things like its
        shape, type, etc.

        Parameters
        ----------
        field: :py:class:`numpy.ndarray`
            The data buffer to load as a field. For a grid with shape ``gshape`` and a field with rank ``rank``,
            the shape of ``field`` should be ``(*ghape, ) + rank*(ndim,)``, where ``ndim`` is the number of dimensions
            in the coordinate system. The rank of the field is determined during the loading process based on
            this shape assumption.
        grid: :py:class:`~pisces_geometry.grids.base.BaseGrid`
            The grid on which to load the field. This should be initialized with the correct number of dimensions
            and a valid coordinate system.
        """
        # Validate the field and the grid via __validate_arguments__. Should supply self._rank, self._axes, and self._axes_mask.
        try:
            self.__validate_arguments__(field, grid, axes)
        except Exception as e:
            raise ValueError(f"Unable to form {self.__class__.__name__} with field {field} and grid {grid} due to errors while"
                             f" validating the arguments: {e}") from e

        # The field has been validated, we now set references to both the grid and the field.
        self._grid: BaseGrid = grid
        self._field_array: np.ndarray = field
