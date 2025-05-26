"""
Specialized types for use with the grids module.
"""
from typing import Sequence, Tuple, Union

import numpy as np

# ============================ #
# Basic Type Aliases           #
# ============================ #
# These aliases are just useful for typing
# hinting larger / complicated input typing scenarios.
ChunkSizeInput = Union[int, Sequence[int], np.ndarray]
HaloOffsetInput = Union[int, Sequence[int], np.ndarray]
GhostZonesInput = Union[int, Sequence[int], np.ndarray]
IndexInput = Union[int, Sequence[int]]
AxesInput = Union[str, Sequence[str]]
_ChunkIndexElement = Union[int, Tuple[int, int], slice]
ChunkIndexInput = Union[_ChunkIndexElement, Sequence[_ChunkIndexElement]]


class BoundingBox(np.ndarray):
    r"""
    A class representing a bounding box, inheriting from :py:class:`numpy.ndarray`.
    Expected to be a 2xNDIM array where the first row represents
    the minimum bounds and the second row represents the maximum bounds.
    """

    def __new__(cls, input_array):
        """
        Create a new BoundingBox instance and validate its shape.

        Parameters
        ----------
        input_array : array-like
            Input data to be cast as a BoundingBox. Must be shape (2, NDIM).

        Returns
        -------
        BoundingBox
            A new BoundingBox instance.
        """
        # Type coercion: must be array cast-able and float64.
        try:
            array = np.asarray(input_array, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to coerce input to BoundingBox: {e}") from e

        # Dimensionality coercion: must be 1D (create a new dim) or 2D.
        if array.ndim == 1:
            array = array[:, np.newaxis]
        elif array.ndim > 2:
            raise ValueError(f"BoundingBox must be 2D (was {array.ndim}D).")

        if array.shape != (2, array.size // 2):
            # The shape isn't correct. We want to try to rework it.
            # We can first check if it needs to be reversed.
            # Enforce shape requirements
            if array.shape == (array.size // 2, 2):
                array = array.T
            else:
                # We need to try to reshape?
                try:
                    array = array.reshape((2, array.size // 2))
                except Exception as e:
                    raise ValueError(
                        f"Could not reshape input array to be a valid BoundingBox: {e}."
                    ) from e

        # Validate the values
        if np.any(array[0, :] > array[1, :]):
            raise ValueError("BoundingBox cannot have LL corner above UR corner.")

        # Create the DomainDimensions instance as a view of the validated array
        obj = array.view(cls)
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array view, ensuring operations result in regular :py:class:`numpy.ndarray`.

        Parameters
        ----------
        obj : ndarray or None
            The original array from which the new object was created.
        """
        if obj is None:
            return  # Called during explicit construction, no additional setup needed
        # Convert any sliced or operated result back to a plain np.ndarray
        if type(self) is not BoundingBox:
            self.__class__ = np.ndarray


class DomainDimensions(np.ndarray):
    """
    Array representation of a grid's domain dimensions.

    This class descents from :py:class:`numpy.ndarray` and is simply a ``(NDIM,)`` array specifying
    the number of grid cells in each dimension of a grid.
    """

    def __new__(cls, input_array):
        """
        Create a new DomainDimensions instance, coercing shape and orientation if needed.

        Parameters
        ----------
        input_array : array-like
            Input data to be cast as DomainDimensions. Should be a 1D array of positive integers
            or a 2D array with shape ``(NDIM, 1)`` or ``(1,NDIM)``. If it is a 1D array, it will be reshaped automatically.

        Returns
        -------
        DomainDimensions
            A new DomainDimensions instance.

        Raises
        ------
        ValueError
            If input cannot be coerced to a 1D array of positive unsigned 32-bit integers.
        """
        # Attempt to coerce input to a uint32 numpy array
        try:
            array = np.asarray(input_array, dtype=int)
        except Exception as e:
            raise ValueError(f"Failed to coerce input to DomainDimensions: {e}") from e

        # Enforce dimensionality
        if array.ndim != 1:
            raise ValueError(f"DomainDimensions must be 1D (was {array.ndim}D).")

        # Enforce shape requirements
        if array.shape != (array.size,):
            # The shape isn't correct. We want to try to rework it.
            try:
                array = array.reshape((array.size,))
            except Exception as e:
                raise ValueError(
                    f"Could not reshape input array to be a valid set of DomainDimensions: {e}."
                ) from e

        # Create the DomainDimensions instance as a view of the validated array
        obj = array.view(cls)
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array view, ensuring operations result in regular :py:class:`numpy.ndarray`.

        Parameters
        ----------
        obj : ndarray or None
            The original array from which the new object was created.
        """
        if obj is None:
            return  # Called during explicit construction, no additional setup needed
        # Convert any sliced or operated result back to a plain np.ndarray
        if type(self) is not DomainDimensions:
            self.__class__ = np.ndarray
