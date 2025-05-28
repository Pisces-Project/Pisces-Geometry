"""
Mixin classes to support chunking in grids.
"""
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from tqdm.auto import tqdm

from pymetric.grids.utils._typing import BoundingBox

if TYPE_CHECKING:
    from scipy.interpolate import RegularGridInterpolator

    from pymetric.grids.mixins._typing import _SupportsGridChunking
    from pymetric.grids.utils._typing import AxesInput, ChunkIndexInput, HaloOffsetInput

# =================================== #
# Type Annotations                    #
# =================================== #
# These type annotations are used for compatibility
# with static type checkers like mypy.
_SupGridChunking = TypeVar("_SupGridChunking", bound="_SupportsGridChunking")


class GridChunkingMixin(Generic[_SupGridChunking]):
    """
    Mixin class supporting chunking in subclasses of
    :py:class:`grids.base.GridBase`.
    """

    # ================================ #
    # Input Coercion Methods           #
    # ================================ #
    # These methods provide support for coercing various
    # common inputs like chunks, axes, halo offsets, etc.
    def _ensure_supports_chunking(self: _SupGridChunking):
        """
        Raise an error if chunking is not enabled on this grid.

        This method is used internally to ensure that chunking-related methods
        are only called on grids that support chunking.

        Raises
        ------
        TypeError
            If chunking is not enabled on the grid.
        """
        if not self.__chunking__:
            raise TypeError(
                f"Instance {self} of {self.__class__.__name__} does not support chunking."
            )

    def _standardize_chunk_indices_type(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        """
        Normalize a flexible chunk specification into a tuple of (start, stop) pairs per axis.

        Parameters
        ----------
        chunks : ChunkIndexInput
            A chunk index specification — either a single value or a list of per-axis
            values. Each element may be:

            - int → interpreted as a single chunk (i, i+1)
            - tuple[int, int] → a (start, stop) chunk range
            - slice → with optional start/stop, defaulting to 0 and cdd

        axes : str or list of str, optional
            Axes to which the chunks apply. If None, applies to all axes.

        Returns
        -------
        tuple of (int, int)
            Normalized chunk index ranges for each axis.
        """
        # Normalize axes and get metadata
        axes = self.standardize_axes(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)
        num_axes = len(axes_indices)
        cdd = self.__cdd__[axes_indices]

        # Normalize scalar inputs
        if isinstance(chunks, (int, slice)):
            chunks = [chunks]
        elif (
            isinstance(chunks, Sequence)
            and isinstance(chunks[0], int)
            and len(chunks) == num_axes
        ):
            pass
        elif (
            isinstance(chunks, Sequence)
            and isinstance(chunks[0], int)
            and len(chunks) == 2
        ):
            chunks = [chunks]

        # Validate number of chunk specs
        if len(chunks) != num_axes:
            raise ValueError(
                f"Expected {num_axes} chunk specs for axes {axes}, got {len(chunks)}."
            )

        # Normalize each axis' chunk spec to (start, stop)
        normalized = []
        for i, item in enumerate(chunks):
            if isinstance(item, int):
                normalized.append((item, item + 1))
            elif isinstance(item, tuple):
                if len(item) != 2:
                    raise TypeError(f"Chunk tuple must be (start, stop), got {item}")
                normalized.append(tuple(item))
            elif isinstance(item, slice):
                start = 0 if item.start is None else item.start
                stop = cdd[i] if item.stop is None else item.stop
                normalized.append((start, stop))
            else:
                raise TypeError(
                    f"Invalid chunk index type at axis {axes[i]}: expected int, tuple, or slice, got {type(item)}"
                )

        return tuple(normalized)

    def _standardize_chunk_indices(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        """
        Normalize and validate chunk indices for the specified axes.

        Parameters
        ----------
        chunks : sequence of int, tuple, or slice

            Chunk index specifications for each axis. Each element can be:
            - int: a single chunk index (e.g., 1)
            - tuple of int: a (start, stop) chunk range (half-open interval)
            - slice: a Python slice object (must have step of 1 or None)

        axes : list of str, optional
            Subset of axes to which the chunk specification applies.
            If None, all axes are considered in order.

        Returns
        -------
        tuple of (int, int)
            A sequence of (start, stop) pairs, one for each selected axis.

        Raises
        ------
        ValueError
            If the number of chunk specs does not match number of axes.
        TypeError
            If any chunk spec is not a supported type or invalid format.
        IndexError
            If chunk indices are out of bounds.
        """
        # Ensure that we have enabled chunking at all.
        self._ensure_supports_chunking()

        # Coerce the typings of the chunk indices to ensure that we
        # have correct specifiers.
        axes = self.standardize_axes(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)
        chunks = self._standardize_chunk_indices_type(chunks, axes)
        cdd = self.__cdd__

        for chunk, axis in zip(chunks, axes_indices):
            if not (0 <= chunk[0] < chunk[1] <= cdd[axes_indices]):
                raise IndexError(
                    f"Chunk ({chunk[0]},{chunk[1]}) out of bounds for axis {self.axes[axis]}."
                )

        return chunks

    # ================================ #
    # Casting                          #
    # ================================ #
    # These methods help with determining shapes
    # of chunks and constructing arrays matching those
    # chunks / chunk stencils.
    def get_chunk_shape(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
    ) -> Tuple[int, ...]:
        """
        Compute the shape of the buffer corresponding to a given chunk,
        including optional ghost zones and halo padding.

        Parameters
        ----------
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.

        Returns
        -------
        tuple of int
            The shape of the selected chunk in global buffer space.
        """
        chunk_slice = self.compute_chunk_slice(
            chunks=chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )
        return tuple(slc.stop - slc.start for slc in chunk_slice)

    def empty_like_chunks(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        /,
        element_shape: Union[int, Sequence[int]] = (),
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
        **kwargs,
    ) -> np.ndarray:
        """
        Return an uninitialized array shaped to match a domain-aligned region of the grid.

        This is useful for quickly allocating grid-aligned buffers (e.g., field data)
        without initializing values.

        Parameters
        ----------
        element_shape : int or sequence of int, default=()
            Additional trailing shape to append to the grid dimensions.
            For scalar fields, use `()`. For vector fields, e.g. `(3,)`.
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.

        **kwargs :
            Additional keyword arguments passed to `np.empty`.

        Returns
        -------
        np.ndarray
            An uninitialized array of shape `(domain_shape, *element_shape)`.
        """
        # Determine the domain shape we want.
        domain_shape = self.get_chunk_shape(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.empty(array_shape, **kwargs)

    def zeros_like_chunks(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        /,
        element_shape: Union[int, Sequence[int]] = (),
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
        **kwargs,
    ) -> np.ndarray:
        """
        Return an array of zeros shaped to match a domain-aligned region of the grid.

        This is useful for creating initialized scalar/vector/tensor buffers over the grid.

        Parameters
        ----------
        element_shape : int or sequence of int, default=()
            Additional trailing shape to append to the grid dimensions.
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.

        **kwargs :
            Additional keyword arguments passed to `np.zeros`.

        Returns
        -------
        np.ndarray
            A zero-initialized array of shape `(domain_shape, *element_shape)`.
        """
        # Determine the domain shape we want.
        domain_shape = self.get_chunk_shape(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.zeros(array_shape, **kwargs)

    def ones_like_chunks(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        /,
        element_shape: Union[int, Sequence[int]] = (),
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
        **kwargs,
    ) -> np.ndarray:
        """
        Return an array of ones shaped to match a domain-aligned region of the grid.

        This is useful for initializing buffers to a uniform value over the grid.

        Parameters
        ----------
        element_shape : int or sequence of int, default=()
            Additional trailing shape to append to the grid dimensions.
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.
        **kwargs :
            Additional keyword arguments passed to `np.ones`.

        Returns
        -------
        np.ndarray
            A one-initialized array of shape `(domain_shape, *element_shape)`.
        """
        # Determine the domain shape we want.
        domain_shape = self.get_chunk_shape(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.ones(array_shape, **kwargs)

    def full_like_chunks(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        fill_value: Any,
        /,
        element_shape: Union[int, Sequence[int]] = (),
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
        **kwargs,
    ) -> np.ndarray:
        """
        Return a constant-valued array shaped to match a domain-aligned region of the grid.

        Parameters
        ----------
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        fill_value : Any
            The value to fill the array with.
        element_shape : int or sequence of int, default=()
            Additional trailing shape to append to the grid dimensions.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.
        **kwargs :
            Additional keyword arguments passed to `np.full`.

        Returns
        -------
        np.ndarray
            A constant-valued array of shape `(domain_shape, *element_shape)`.
        """
        # Determine the domain shape we want.
        domain_shape = self.get_chunk_shape(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.full(array_shape, fill_value=fill_value, **kwargs)

    # ================================ #
    # Chunk Slicing Methods            #
    # ================================ #
    # These methods provide support for various slicing
    # needs when working in chunks.
    def _compute_chunk_slice_fast(
        self: _SupGridChunking,
        start: np.ndarray,
        stop: np.ndarray,
        axes_indices: np.ndarray,
        include_ghosts: bool,
        halo_offsets: np.ndarray,
        oob_behavior: str,
    ) -> Tuple[slice, ...]:
        """
        From a specific set of stopping and starting indices,
        compute the coordinate slices for a specific chunk.

        Notably, this is a slice into the GRID COORDINATES, and therefore
        will depend on the coordinate centering.
        """
        size = self.__chunk_size__[axes_indices]  # Number of CELLS per chunk.
        ghost = self.__ghost_zones__[:, axes_indices]  # Number of ghost CELLS.
        full = self.__ghost_dd__[axes_indices]
        total_chunks = self.__cdd__[axes_indices]

        # Compute raw grid-space indices
        start_idx = start * size
        stop_idx = stop * size

        # Adjust for ghost zones at edges
        if include_ghosts:
            is_left_edge = start == 0
            is_right_edge = stop == total_chunks
            start_idx -= is_left_edge * ghost[0]
            stop_idx += is_right_edge * ghost[1]

        # Add ghost and halo padding (ghost[0] is base offset)
        start_idx += ghost[0] - halo_offsets[0]
        stop_idx += ghost[0] + halo_offsets[1]

        # Adjust for the cell centering of the grid.
        if self.__center__ == "vertex":
            stop_idx += 1

        # Clip or raise on bounds violation
        if oob_behavior == "clip":
            start_idx = np.clip(start_idx, 0, full)
            stop_idx = np.clip(stop_idx, 0, full)
        elif oob_behavior == "raise":
            if np.any(start_idx < 0) or np.any(stop_idx > full):
                raise IndexError(
                    f"Chunk slice out of bounds: start={start_idx}, stop={stop_idx}, bounds={full}"
                )
        else:
            raise ValueError(f"Invalid oob_behavior: {oob_behavior!r}")
        return tuple(slice(int(s), int(e)) for s, e in zip(start_idx, stop_idx))

    def _compute_chunk_slice_fast_scalar(
        self: _SupGridChunking,
        start: int,
        axis_index: int,
        include_ghosts: bool,
        halo_offsets: np.ndarray,
        oob_behavior: str,
    ) -> slice:
        """
        Fast internal core for computing a single-axis chunk slice.

        This is optimized for cases where only one axis changes (e.g., tight inner loop).
        """
        size = int(self.__chunk_size__[axis_index])
        ghost_lo, ghost_hi = self.__ghost_zones__[:, axis_index]
        total_chunks = self.__cdd__[axis_index]
        full = int(self.__ghost_dd__[axis_index])

        start_idx = start * size
        stop_idx = (start + 1) * size

        # Edge-aware ghost padding
        if include_ghosts:
            if start == 0:
                start_idx -= ghost_lo
            if start + 1 == total_chunks:
                stop_idx += ghost_hi

        # Apply offset
        start_idx += ghost_lo - halo_offsets[0]
        stop_idx += ghost_lo + halo_offsets[1]

        if self.__center__ == "vertex":
            stop_idx += 1

        # Bounds logic
        if oob_behavior == "clip":
            start_idx = max(0, min(start_idx, full))
            stop_idx = max(0, min(stop_idx, full))
        elif oob_behavior == "raise":
            if start_idx < 0 or stop_idx > full:
                raise IndexError(
                    f"Chunk slice out of bounds on axis {axis_index}: "
                    f"start={start_idx}, stop={stop_idx}, bounds=[0,{full})"
                )
        else:
            raise ValueError(f"Invalid oob_behavior: {oob_behavior!r}")

        return slice(int(start_idx), int(stop_idx))

    def compute_chunk_slice(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
    ) -> Tuple[slice, ...]:
        """
        Compute a global index-space slice corresponding to one or more chunks
        along selected axes, with optional ghost zones and halo padding.

        This method returns the global index-space slice that selects the desired chunk
        region from the grid buffer. It supports ghost zones, halo extensions, and
        relaxed boundary behavior.

        Parameters
        ----------
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.

        Returns
        -------
        tuple of slice
            One slice per selected axis, expressed **in global index space**.
        """
        axes = self.standardize_axes(axes)
        ndim = len(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)
        halo = self._standardize_halo_offset(halo_offsets, ndim)

        chunks = self._standardize_chunk_indices_type(chunks, axes)
        chunk_indices = np.stack(chunks).astype(int)

        start = chunk_indices[:, 0]
        stop = chunk_indices[:, 1]

        return self._compute_chunk_slice_fast(
            start=start,
            stop=stop,
            axes_indices=axes_indices,
            include_ghosts=include_ghosts,
            halo_offsets=halo,
            oob_behavior=oob_behavior,
        )

    # =============================== #
    # Coordinates                     #
    # =============================== #
    # These methods handle the coordinate generation
    # procedures. The only method in the base class is the
    # `compute_coords_from_slices` abstract method. Everything
    # else utilizes that.
    def compute_chunk_coords(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
    ):
        """
        Compute physical coordinate arrays for a selected chunk region.

        This method returns the 1D coordinate arrays along each axis for a
        given chunk, optionally including ghost zones and additional halo padding.

        Parameters
        ----------
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.
        __validate__ : bool, default=True
            Whether to perform full type/shape validation. Disable in performance-critical paths.

        Returns
        -------
        tuple of np.ndarray
            Physical coordinate arrays, one per selected axis.
        """
        slices = self.compute_chunk_slice(
            chunks, axes, include_ghosts, halo_offsets, oob_behavior
        )
        return self.compute_coords_from_slices(
            slices, axes=axes, origin="global", __validate__=__validate__
        )

    def compute_chunk_mesh(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
        **kwargs,
    ):
        """
        Compute a meshgrid of physical coordinates over a selected chunk region.

        This method returns a set of multidimensional coordinate arrays representing
        a full mesh over the selected chunk. It supports optional ghost zones and halo padding.

        Parameters
        ----------
        chunks : int, tuple, slice, or sequence of those
            Chunk specification per axis. For each axis, you may specify:

            - `int`: selects a single chunk (i, i+1)
            - `(start, stop)`: a range of chunks
            - `slice(start, stop)`: a Python slice (step is ignored)

            These are converted to global-space data index ranges.
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.

        __validate__ : bool, default=True
            Whether to perform full type/shape validation. Disable in performance-critical paths.

        kwargs : dict
            Additional keyword arguments passed to `numpy.meshgrid`.

        Returns
        -------
        tuple of np.ndarray
            A tuple of meshgrid arrays, one for each selected axis.
        """
        coords = self.compute_chunk_coords(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
            __validate__=__validate__,
        )
        return np.meshgrid(*coords, **kwargs)

    def compute_chunk_edges(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
    ) -> List[np.ndarray]:
        """
        Return coordinate edge arrays for a specific chunk along the selected axes.

        If the grid is vertex-centered, the coordinates are already edges and are returned directly.
        If the grid is cell-centered, edges are computed from the center coordinates and the chunk-local bounding box.

        Parameters
        ----------
        chunks : ChunkIndexInput
            The chunk index specification (int, tuple, slice, or sequence of those).
        axes : str or list of str, optional
            Axes to include. If None, all axes are included.
        include_ghosts : bool, default=False
            Whether to include ghost zones when computing chunk extent.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra ghost-cell-padding to apply on top of the standard ghost zones.
        oob_behavior : {"raise", "clip"}, default="raise"
            What to do if the computed chunk region would go out of bounds.
        __validate__ : bool, default=True
            Whether to validate inputs (axes, shapes, etc).

        Returns
        -------
        list of np.ndarray
            One coordinate array per axis, with length = n+1 (edge-aligned).
        """
        axes = self.standardize_axes(axes)

        # Retrieve center coordinates and bounding box for this chunk
        coords = self.compute_chunk_coords(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
            __validate__=__validate__,
        )
        bbox = self.get_chunk_bbox(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        if self.centering == "vertex":
            return coords
        elif self.centering == "cell":
            return self._centers_to_edges(coords, bbox)
        else:
            raise NotImplementedError(
                f"Centering mode '{self.centering}' is not supported."
            )

    def compute_chunk_centers(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
    ) -> List[np.ndarray]:
        """
        Return coordinate center arrays for a specific chunk along the selected axes.

        If the grid is cell-centered, the coordinates are already centers and are returned directly.
        If the grid is vertex-centered, centers are computed from the edge coordinates.

        Parameters
        ----------
        chunks : ChunkIndexInput
            The chunk index specification (int, tuple, slice, or sequence of those).
        axes : str or list of str, optional
            Axes to include. If None, all axes are included.
        include_ghosts : bool, default=False
            Whether to include ghost zones when computing chunk extent.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra ghost-cell-padding to apply on top of the standard ghost zones.
        oob_behavior : {"raise", "clip"}, default="raise"
            What to do if the computed chunk region would go out of bounds.
        __validate__ : bool, default=True
            Whether to validate inputs (axes, shapes, etc).

        Returns
        -------
        list of np.ndarray
            One coordinate array per axis, center-aligned.
        """
        axes = self.standardize_axes(axes)

        coords = self.compute_chunk_coords(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
            __validate__=__validate__,
        )

        if self.centering == "cell":
            return coords
        elif self.centering == "vertex":
            return self._edges_to_centers(coords)
        else:
            raise NotImplementedError(
                f"Centering mode '{self.centering}' is not supported."
            )

    def get_chunk_bbox(
        self: _SupGridChunking,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
    ) -> BoundingBox:
        """
        Return the physical bounding box (lower, upper) of a given chunk.

        This works for both vertex- and cell-centered grids by computing edge coordinates
        over the full domain and slicing into them appropriately.
        """
        axes = self.standardize_axes(axes)

        # Get full domain edges for the relevant axes
        edge_coords = self.compute_domain_edges(
            axes=axes, origin="global", __validate__=True
        )

        # Get chunk slice (in point index space)
        chunk_slice = self.compute_chunk_slice(
            chunks=chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        # Adjust slice endpoints if cell-centered: we need (n+1) points to get n cells' edges
        if self.centering == "cell":
            adjusted_slices = [slice(slc.start, slc.stop + 1) for slc in chunk_slice]
        else:
            adjusted_slices = chunk_slice

        # Build bounding box from edge values
        bbox = []
        for edge, slc in zip(edge_coords, adjusted_slices):
            lower = edge[slc.start]
            upper = edge[slc.stop]
            bbox.append([lower, upper])

        return BoundingBox(bbox)

    # =============================== #
    # Interpolation on chunks         #
    # =============================== #
    def construct_chunk_interpolator(
        self: _SupGridChunking,
        field: np.ndarray,
        field_axes: Sequence[str],
        chunks: "ChunkIndexInput",
        method: str = "linear",
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        bounds_error: bool = False,
        fill_value: Optional[float] = np.nan,
        oob_behavior: str = "raise",
        __validate__: bool = True,
        **kwargs,
    ) -> "RegularGridInterpolator":
        """
        Construct an interpolator for a specific chunk using SciPy's RegularGridInterpolator.

        Parameters
        ----------
        field : np.ndarray
            Field data over the chunk, aligned with `field_axes`.
        field_axes : Sequence[str]
            Axes spanned by the field (e.g., ["x", "y"]).
        chunks : ChunkIndexInput
            Chunk index (e.g., (1, 2) or [1, 2]) for the region over which to interpolate.
        method : {"linear", "nearest"}, default="linear"
            Interpolation method.
        include_ghosts : bool, default=False
            Whether to include ghost zones in the chunk coordinate extraction.
        halo_offsets : int, sequence[int], or array, optional
            Extra padding around the chunk region.
        bounds_error : bool, default=False
            Whether to raise an error for out-of-bound queries.
        fill_value : float or None, default=np.nan
            Fill value for out-of-bound queries if `bounds_error=False`.
        oob_behavior : {"raise", "clip"}, default="raise"
            How to handle index overflow when applying halo or ghost zones.
        __validate__ : bool, default=True
            Whether to perform full input validation.
        **kwargs
            Extra options forwarded to RegularGridInterpolator.

        Returns
        -------
        interpolator : RegularGridInterpolator
            A callable that performs interpolation over the selected chunk.
        """
        from scipy.interpolate import RegularGridInterpolator

        # Extract the physical coordinates over the chunk
        coords = self.compute_chunk_coords(
            chunks=chunks,
            axes=field_axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
            __validate__=__validate__,
        )

        # Construct the interpolator using the chunk-local field
        interpolator = RegularGridInterpolator(
            points=coords,
            values=field,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            **kwargs,
        )

        return interpolator

    # ================================ #
    # Iterables                        #
    # ================================ #
    # These methods allow users to loop through
    # chunks in each grid.
    def iter_chunk_slices(
        self: _SupGridChunking,
        /,
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        pbar: bool = True,
        pbar_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[slice, ...]]:
        """
        Efficiently iterate over chunk-wise slices in global index space.

        This iterator yields a tuple of slices per chunk along selected axes,
        updating only the slices that change. Supports ghost zones, halo padding,
        and progress bar display.

        Parameters
        ----------
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.
        pbar : bool, default=True
            Whether to display a tqdm progress bar.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to `tqdm`.

        Yields
        ------
        tuple of slice
            A tuple of slices selecting the current chunk in global index space.
        """
        # Standardize axes and prepare chunk metadata
        axes = self.standardize_axes(axes)
        num_axes = len(axes)
        axes_indices = np.asarray(self.__cs__.convert_axes_to_indices(axes), dtype=int)
        halo_offsets = self._standardize_halo_offset(halo_offsets, num_axes)

        cdd = self.__cdd__[axes_indices]

        # Initialize first slice using fast multi-axis function
        current_slices = list(
            self._compute_chunk_slice_fast(
                start=np.zeros(num_axes, dtype=int),
                stop=np.ones(num_axes, dtype=int),
                axes_indices=axes_indices,
                include_ghosts=include_ghosts,
                halo_offsets=halo_offsets,
                oob_behavior=oob_behavior,
            )
        )

        # Initialize progress bar
        progress_bar = None
        if pbar:
            pbar_kwargs = pbar_kwargs or {}
            pbar_kwargs.setdefault("desc", f"Iterating over {axes} chunks")
            pbar_kwargs.setdefault("total", int(np.prod(cdd)))
            progress_bar = tqdm(**pbar_kwargs)

        # Iterate over all chunk index tuples (e.g. (0,0), (0,1), ...)
        _prev_chunk_index = (0,) * num_axes
        for chunk_index in np.ndindex(*cdd):
            # Only recompute slices along axes that changed
            for i in range(num_axes):
                if chunk_index[i] != _prev_chunk_index[i]:
                    current_slices[i] = self._compute_chunk_slice_fast_scalar(
                        start=chunk_index[i],
                        axis_index=int(axes_indices[i]),
                        include_ghosts=include_ghosts,
                        halo_offsets=halo_offsets[:, i],
                        oob_behavior=oob_behavior,
                    )

            _prev_chunk_index = chunk_index
            if progress_bar:
                progress_bar.update(1)

            yield tuple(current_slices)

    def iter_chunk_indices(
        self: _SupGridChunking,
        axes: Optional["AxesInput"] = None,
        *,
        pbar: bool = True,
        pbar_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[int, ...]]:
        """
        Iterate over chunk indices for the grid along specified axes.

        This yields the index tuple (e.g., (i, j, k)) for each chunk
        along the selected axes. It does not compute slices or include
        ghost/halo metadata — it is simply the raw chunk index iterator.

        Parameters
        ----------
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.

        pbar : bool, default=True
            Whether to display a tqdm progress bar during iteration.

        pbar_kwargs : dict, optional
            Additional keyword arguments to pass to `tqdm`.

        Yields
        ------
        tuple of int
            The current chunk index along the selected axes.
        """
        # Normalize the axes
        axes = self.standardize_axes(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)

        # Extract shape of chunk layout along those axes
        cdd = self.__cdd__[axes_indices]

        # Configure optional progress bar
        progress_bar = None
        if pbar:
            pbar_kwargs = pbar_kwargs or {}
            pbar_kwargs.setdefault("desc", f"Iterating over {axes} chunk indices")
            pbar_kwargs.setdefault("total", int(np.prod(cdd)))
            progress_bar = tqdm(**pbar_kwargs)

        for chunk_index in np.ndindex(*cdd):
            if progress_bar:
                progress_bar.update(1)
            yield chunk_index

    def iter_chunk_coords(
        self: _SupGridChunking,
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        pbar: bool = True,
        pbar_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """
        Efficiently iterate over physical coordinates for each chunk in the grid.

        This yields 1D coordinate arrays along the selected axes, one set per chunk.
        Only the coordinate arrays for axes that change between chunks are recomputed,
        making this efficient for tight iteration.

        Parameters
        ----------
        axes : str or list of str, optional
            The names of the axes to include in the slice. If None, all axes are used.
            This determines the number of :py:class:`slice` objects returned in
            the output tuple — one slice per selected axis.

            .. note::

                Regardless of the ordering of ``axes``, the axes are reordered to
                match canonical ordering as defined by the coordinate system.
        include_ghosts : bool, optional
            If ``True``, then the slice will include the ghost zones around the boundary of
            the domain. Otherwise (default), the ghost zones are excluded from the resulting slice.
        halo_offsets : int, sequence[int], or np.ndarray, optional
            Extra padding (in ghost-cell units) to apply on top of the active or ghost-augmented domain.

            This allows you to extend the region further outward beyond ghost zones — for example,
            to reserve additional halo space for multi-pass stencils.

            Supported formats:

            - **int**: same padding applied to both sides of all axes
            - **sequence of length N**: symmetric padding per axis (left = right)
            - **array of shape (2, N)**: explicit left/right padding per axis

            .. note::

                This is applied *after* ghost zones (if `include_ghosts=True`). Total padding = ghost zones + halo.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do if the computed slice (after applying ghost zones and halo padding)
            would extend beyond the allocated grid buffer (`__ghost_dd__`).

            Options:
                - **"raise"** : Raise an `IndexError` if any part of the slice is out of bounds.
                - **"clip"**  : Clamp the slice to stay within the valid grid extent.

            Use "clip" if you're performing relaxed operations (e.g., visualization or padding-aware reads),
            and "raise" for strict enforcement during stencil computations or buffer validation.
        pbar : bool, default=True
            Whether to display a tqdm progress bar.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to `tqdm`.

        Yields
        ------
        tuple of np.ndarray
            A tuple of coordinate arrays, one per selected axis, for each chunk.
        """
        # Normalize inputs and initialize metadata
        axes = self.standardize_axes(axes)
        num_axes = len(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)
        halo_offsets = self._standardize_halo_offset(halo_offsets, num_axes)
        halo_offsets[0] *= -1  # pre-negate left side

        cdd = self.__cdd__[axes_indices]

        # Track initial slices and coordinates
        current_slices = list(
            self._compute_chunk_slice_fast(
                start=np.zeros(num_axes, dtype=int),
                stop=np.ones(num_axes, dtype=int),
                axes_indices=axes_indices,
                include_ghosts=include_ghosts,
                halo_offsets=halo_offsets,
                oob_behavior=oob_behavior,
            )
        )
        current_coords = list(
            self.compute_coords_from_slices(
                current_slices, axes=axes, origin="global", __validate__=False
            )
        )

        # Setup progress bar
        progress_bar = None
        if pbar:
            pbar_kwargs = pbar_kwargs or {}
            pbar_kwargs.setdefault("desc", f"Iterating over {axes} chunk coordinates")
            pbar_kwargs.setdefault("total", int(np.prod(cdd)))
            progress_bar = tqdm(**pbar_kwargs)

        # Loop over all chunk indices
        _prev_chunk_index = (0,) * num_axes
        for chunk_index in np.ndindex(*cdd):
            for i in range(num_axes):
                if chunk_index[i] != _prev_chunk_index[i]:
                    current_slices[i] = self._compute_chunk_slice_fast_scalar(
                        start=chunk_index[i],
                        axis_index=axes_indices[i],
                        include_ghosts=include_ghosts,
                        halo_offsets=halo_offsets[:, i],
                        oob_behavior=oob_behavior,
                    )
                    current_coords[i] = self.compute_coords_from_slices(
                        [current_slices[i]],
                        axes=[axes[i]],
                        origin="global",
                        __validate__=False,
                    )[
                        0
                    ]  # unwrap single-axis result

            _prev_chunk_index = chunk_index
            if progress_bar:
                progress_bar.update(1)

            yield tuple(current_coords)
