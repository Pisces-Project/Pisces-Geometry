"""
Core mixins for the PyMetric grid infrastructure.

This module contains mixin classes that are included (by default) in the core :py:class:`~grids.base.GridBase`
class and are therefore structural elements of the base class. These are primarily a method
for ensuring that code structure is coherent and readable on the developer end. Mixins manage

- Coordinate system introspection and axis normalization
- Unit system access and serialization
- Ghost zone and halo management
- Chunking behavior and iteration
- HDF5 input/output operations
- Index validation and vectorized access utilities

Each mixin expects the parent class to implement specific attributes (e.g., ``__cs__``, ``__dd__``,
``__ghost_dd__``, etc.) that are required for it to function correctly. These mixins follow a
minimal and declarative style, avoiding unnecessary side effects or object instantiation.

Each mixin is typed with ``Generic[...]`` bounds on their expected protocols.
"""
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import h5py
import numpy as np

from pymetric.grids.utils._typing import DomainDimensions
from pymetric.utilities.arrays import normalize_index

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

    from pymetric.coordinates.base import _CoordinateSystemBase
    from pymetric.grids.mixins._typing import (
        _SupportsGridChunking,
        _SupportsGridCore,
        _SupportsGridIO,
    )
    from pymetric.grids.utils._typing import (
        AxesInput,
        ChunkSizeInput,
        GhostZonesInput,
        HaloOffsetInput,
        IndexInput,
    )

# =================================== #
# Type Annotations                    #
# =================================== #
# These type annotations are used for compatibility
# with static type checkers like mypy.
_SupGridCore = TypeVar("_SupGridCore", bound="_SupportsGridCore")
_SupGridIO = TypeVar("_SupGridIO", bound="_SupportsGridIO")
_SupGridChunking = TypeVar("_SupGridChunking", bound="_SupportsGridChunking")

# =================================== #
# Mixin Classes                       #
# =================================== #
# These classes form core mixins for the base
# grid types and are used primarily for development
# organization so that the base class does not become
# over cluttered.
class GridUtilsMixin(Generic[_SupGridCore]):
    """
    Grid mixin class for supporting various utilities,
    validators, etc. for use in grid subclasses.
    """

    # ================================ #
    # Input Coercion Methods           #
    # ================================ #
    # These methods provide support for coercing various
    # common inputs like chunks, axes, halo offsets, etc.
    def _standardize_chunk_size(
        self: _SupGridCore, chunk_size: Optional["ChunkSizeInput"]
    ) -> DomainDimensions:
        """
        Normalize and validate a chunk size specification.

        Parameters
        ----------
        chunk_size : int or sequence of int
            Chunk size per axis. If an int, it is broadcasted to all axes.

        Returns
        -------
        DomainDimensions
            A tuple-like object (typically a NumPy array) of length ndim, specifying the chunk size
            along each axis.

        Raises
        ------
        ValueError
            If chunk sizes do not evenly divide the domain shape.
        TypeError
            If input format is invalid.
        """
        ndim = self.ndim
        dd = np.asarray(self.__dd__)  # active grid dimensions (excluding ghost zones)

        # Convert chunk_size to ndarray of shape (ndim,)
        if isinstance(chunk_size, int):
            chunks = np.full(ndim, chunk_size, dtype=int)
        elif isinstance(chunk_size, Sequence):
            chunks = np.asarray(chunk_size, dtype=int)
            if chunks.shape != (ndim,):
                raise ValueError(
                    f"chunk_size must be scalar or sequence of length {ndim}, got shape {chunks.shape}"
                )
        else:
            raise TypeError(
                f"chunk_size must be int or sequence of ints, got {type(chunk_size)}"
            )

        # Check for divisibility
        if np.any(dd % chunks != 0):
            raise ValueError(
                f"chunk_size {chunks.tolist()} does not evenly divide grid shape {dd.tolist()}."
            )

        return DomainDimensions(chunks)

    @staticmethod
    def _standardize_halo_offset(
        halo_offsets: Optional["HaloOffsetInput"], num_axes: int
    ) -> np.ndarray:
        """
        Normalize user input into a halo offset array of shape (2, naxes).

        This utility is used internally by methods like `get_slice_from_chunks` to
        convert various halo specification formats into a consistent `(2, naxes)` array
        of integer offsets.

        Parameters
        ----------
        halo_offsets : int, list[int], or np.ndarray, optional

        - int: symmetric halo on all sides of all axes.
        - 1D array (length = num_axes): symmetric halo per axis.
        - 2D array of shape (2, num_axes): explicit (left, right) per axis.
        - 2D array of shape (num_axes, 2): same, auto-transposed.
        - None: no halo.

        num_axes : int
            Number of axes the halo should apply to.

        Returns
        -------
        np.ndarray
            A (2, naxes) array of integer offsets, with [0] = left, [1] = right.

        Raises
        ------
        ValueError
            If the input shape is invalid or contains non-integers.
        """
        if halo_offsets is None:
            return np.zeros((2, num_axes), dtype=int)

        halo = np.asarray(halo_offsets)

        if halo.ndim == 0:
            # Scalar → symmetric halo for all sides, all axes
            return np.full((2, num_axes), int(halo), dtype=int)

        if halo.ndim == 1:
            if halo.shape[0] != num_axes:
                raise ValueError(
                    f"1D halo must have length {num_axes}, got {halo.shape[0]}."
                )
            return np.tile(halo, (2, 1)).astype(int)

        if halo.ndim == 2:
            if halo.shape == (2, num_axes):
                return halo.astype(int)
            elif halo.shape == (num_axes, 2):
                return halo.T.astype(int)
            else:
                raise ValueError(
                    f"2D halo must have shape (2, {num_axes}) or ({num_axes}, 2), got {halo.shape}."
                )

        raise ValueError(
            f"Invalid halo specification: expected scalar, 1D (len={num_axes}), "
            f"or 2D shape (2, {num_axes}) or ({num_axes}, 2). Got shape {halo.shape}."
        )

    @staticmethod
    def _standardize_ghost_zones(
        ghost_zones: Union["GhostZonesInput", None], num_axes: int
    ) -> np.ndarray:
        """
        Normalize input into a ghost zone array of shape (2, num_axes).

        This method standardizes user input for ghost zone sizes along each axis.
        Valid inputs include:

        - A single integer (symmetric ghost zone on all sides of all axes)
        - A sequence or array of length `num_axes` (symmetric per axis)
        - A 2D array of shape (2, num_axes) (explicit left/right per axis)

        Parameters
        ----------
        ghost_zones : int, list[int], np.ndarray or None
            The ghost zone specification to standardize.

        num_axes : int
            Number of grid axes the ghost zone specification should apply to.

        Returns
        -------
        np.ndarray
            A (2, num_axes) array where row 0 is the number of ghost cells on the lower
            (left) side of each axis, and row 1 is for the upper (right) side.

        Raises
        ------
        ValueError
            If the input shape is invalid or contains the wrong number of elements.
        """
        if ghost_zones is None:
            return np.zeros((2, num_axes), dtype=int)

        gz = np.asarray(ghost_zones, dtype=int)

        if gz.ndim == 0:
            # Scalar: same for all sides and axes
            return np.full((2, num_axes), int(gz), dtype=int)
        if gz.ndim == 1:
            if gz.shape[0] != num_axes:
                raise ValueError(
                    f"1D ghost_zones must have length {num_axes}, got {gz.shape[0]}."
                )
            return np.tile(gz, (2, 1)).astype(int)

        if gz.ndim == 2 and gz.shape == (2, num_axes):
            return gz.astype(int)

        raise ValueError(
            f"Invalid ghost_zones specification: expected scalar, "
            f"1D array of length {num_axes}, or shape (2, {num_axes}). Got shape {gz.shape}."
        )

    def standardize_axes(
        self: _SupGridCore, axes: Optional["AxesInput"] = None
    ) -> List[str]:
        """
        Standardize the axes provided.

        This will ensure that all axes are valid axes and that the axes
        are ordered canonically.

        Parameters
        ----------
        axes: list of str
            The axes specification to standardize.

        Returns
        -------
        list of str
            The standardized axes.
        """
        # Fill axes if it is None.
        if axes is None:
            axes = self.axes

        # Reorder the axes to match the canonical
        # ordering.
        axes = self.__cs__.order_axes_canonical(axes)
        return list(axes)

    def _standardize_index(
        self: _SupGridCore,
        index: "IndexInput",
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
    ) -> Tuple[int, ...]:
        """
        Normalize and validate a grid index for a subset of axes.

        Parameters
        ----------
        index : int, tuple[int, ...], or array-like
            Index to validate. May be flat, tuple-form, or array-like.
        axes : sequence of str or str, optional
            Subset of axes being accessed. If None, all axes are assumed.
        origin : {"active", "global"}, default="active"
            Whether the index is relative to the active domain or the global domain.

        Returns
        -------
        tuple of int
            A tuple-form index in global coordinates.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the index is malformed.
        """
        # Manage the axes and the number of axes. These
        # determine the number of expected indices.
        axes = self.standardize_axes(axes)
        axis_indices = self.coordinate_system.convert_axes_to_indices(axes)
        num_axes = len(axis_indices)

        # Perform the type coercion on the index
        # to ensure that it is the correct type casting.
        if isinstance(index, int):
            index = (index,)
        elif isinstance(index, Sequence):
            index = tuple(index)
        else:
            raise TypeError(
                f"Index must be int, tuple[int,...], or array-like, got {type(index)}"
            )

        # Extract the shape depending on the origin parameter
        # and then start performing checks.
        if origin == "active":
            shape = self.__dd__[axis_indices]
        elif origin == "global":
            shape = self.__ghost_dd__[axis_indices]
        else:
            raise ValueError(f"Unknown origin {origin!r}")

        # Check the length of the array to ensure that
        # it matches.
        if len(index) != num_axes:
            raise IndexError(
                f"Expected {num_axes}D index for axes {axes}, got {len(index)}"
            )

        # Now normalize the axes and then check that they are all reasonable.
        for _i, _v in enumerate(index):
            _shape_ = int(shape[_i])
            _new_index_ = normalize_index(_v, _shape_)

            if not (0 <= _new_index_ < _shape_):
                raise IndexError(
                    f"Index {_v} (normalized={_new_index_}) out of bounds for axis '{self.axes[axis_indices[_i]]}' (size {_shape_})."
                )

        # Convert to global space
        if origin == "active":
            index = tuple(
                index[i] + self.__ghost_zones__[0, axis_indices[i]]
                for i in range(num_axes)
            )

        return index

    def check_field_shape(
        self: _SupGridCore,
        array_shape: Sequence[int],
        axes: Optional["AxesInput"] = None,
        field_shape: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Validate that a field array shape is compatible with the grid over the specified axes.

        Parameters
        ----------
        array_shape : Sequence[int]
            The shape of the input array to check.
        axes : str or list of str, optional
            The axes that the field is defined over. If None, all axes are used.
        field_shape : Sequence[int], optional
            Expected trailing dimensions (e.g., for vector or tensor fields). If None, no trailing dims are checked.

        Raises
        ------
        ValueError
            If the array's shape does not match the expected grid shape or combined (grid + field) shape.
        """
        # Standardize axis names and compute expected grid shape
        axes = self.standardize_axes(axes)
        num_axes, mask = self._count_and_mask_axes(axes)
        grid_shape = tuple(self.__ghost_dd__[mask])

        # Check the leading dimensions match grid shape
        actual_grid_shape = tuple(array_shape[:num_axes])
        if actual_grid_shape != grid_shape:
            raise ValueError(
                f"Incompatible grid shape for field over axes {axes}.\n"
                f"  Expected grid shape: {grid_shape}\n"
                f"  Found leading shape : {actual_grid_shape}"
            )

        # Check trailing field dimensions if provided
        if field_shape is not None:
            expected_shape = grid_shape + tuple(field_shape)
            if tuple(array_shape) != expected_shape:
                raise ValueError(
                    f"Incompatible full field shape.\n"
                    f"  Expected: {expected_shape}\n"
                    f"  Found   : {tuple(array_shape)}"
                )

    # ================================ #
    # Minimal Utilities                #
    # ================================ #
    # These are little methods that get used a lot in
    # various other methods.
    def _count_and_mask_axes(
        self: _SupGridCore, axes: Optional["AxesInput"] = None
    ) -> Tuple[int, np.ndarray]:
        # Start by standardizing the axes that we have before passing
        # them down to create the mask.
        axes = self.standardize_axes(axes)
        mask = self.__cs__.build_axes_mask(axes)
        return len(axes), mask

    def _adjust_slices_for_origin(
        self: _SupGridCore,
        slices: Union[slice, Sequence[slice]],
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
    ) -> List[slice]:
        """
        Adjust a list of slices from active-space to global-space index coordinates.

        Parameters
        ----------
        slices : slice or sequence of slice
            The slices to adjust. Each one corresponds to an axis in `axes`.
        origin : {"active", "global"}, default="active"
            The origin of the slices. If "active", ghost zone offsets will be applied.
            If "global", the slices are returned unchanged.
        axes : str or list of str, optional
            The axes the slices apply to. If None, all axes are assumed.

        Returns
        -------
        list of slice
            Slices adjusted for ghost zone offsets, with `None` boundaries resolved
            to valid global index-space extents.
        """
        # Normalize inputs
        axes = self.standardize_axes(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)

        slices = [slices] if isinstance(slices, slice) else list(slices)
        if len(slices) != len(axes):
            raise ValueError(
                f"Expected {len(axes)} slices for axes {axes}, got {len(slices)}"
            )

        adjusted = []

        for slc, ax in zip(slices, axes_indices):
            gz_lo, gz_hi = self.__ghost_zones__[:, ax]
            gdd = self.__ghost_dd__[ax]

            if origin == "active":
                # Resolve `start`
                if slc.start is None:
                    start = gz_lo
                else:
                    start = gz_lo + slc.start

                # Resolve `stop`
                if slc.stop is None:
                    stop = gdd - gz_hi
                else:
                    stop = gz_lo + slc.stop

            elif origin == "global":
                # Pass through `None` or direct values
                start = slc.start
                stop = slc.stop
            else:
                raise ValueError(f"Invalid origin: {origin!r}")
            adjusted.append(slice(start, stop, slc.step))

        return adjusted

    @staticmethod
    def _centers_to_edges(
        coordinates: Sequence[np.ndarray],
        bbox: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Compute coordinate edge arrays from center coordinates and a given bounding box.

        For each axis, computes (n+1) edge values from (n) center values using
        midpoint averaging, and applies the provided bounding box to fill
        the outer edge values.

        Parameters
        ----------
        coordinates : Sequence[np.ndarray]
            Coordinate arrays representing cell centers (shape (n,)).
        bbox : np.ndarray
            A (2, ndim) array specifying [lower, upper] bounds for each axis.
            Used to fill in the boundary edges.

        Returns
        -------
        list of np.ndarray
            One edge array per axis with shape (n + 1,).
        """
        edge_arrays = []
        for i, coords in enumerate(coordinates):
            # Compute interior edge midpoints
            centers = 0.5 * (coords[1:] + coords[:-1])
            # Add bounding box values as exterior edges
            lower_edge = bbox[0, i]
            upper_edge = bbox[1, i]
            edges = np.concatenate([[lower_edge], centers, [upper_edge]])
            edge_arrays.append(edges)

        return edge_arrays

    @staticmethod
    def _edges_to_centers(
        coordinates: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Compute coordinate center arrays from edge coordinates.

        For each axis, computes n center values from n+1 edge values.

        Parameters
        ----------
        coordinates : Sequence[np.ndarray]
            Coordinate arrays representing edge positions (shape (n + 1,)).

        Returns
        -------
        list of np.ndarray
            One array per axis with shape (n,) containing center positions.
        """
        return [0.5 * (coords[1:] + coords[:-1]) for coords in coordinates]

    # =============================== #
    # Coordinates                     #
    # =============================== #
    # These methods handle the coordinate generation
    # procedures. The only method in the base class is the
    # `compute_coords_from_slices` abstract method. Everything
    # else utilizes that.
    def compute_coords_from_slices(
        self: _SupGridCore,
        slices: Union[slice, Sequence[slice]],
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        """
        Compute coordinate arrays over a region of the grid specified by slice objects.

        Parameters
        ----------
        slices : slice or list of slice
            Slice(s) specifying the region to extract along the selected axes.
        axes : str or list of str, optional
            Axes to apply the slices to. If None, all axes are used.
        origin : {"active", "global"}, default="active"
            Whether the slices are relative to the active domain (excluding ghost zones)
            or the full ghost-augmented domain.
        __validate__ : bool, default=True
            Enable or disable all validation procedures. This should only be done in
            tight loop scenarios where performance is critical and typing is well controlled.

        Returns
        -------
        tuple of np.ndarray
            Physical coordinate arrays corresponding to the sliced region, one per axis.
        """
        # Standardize axes and their indices
        if __validate__:
            axes = self.standardize_axes(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)

        # Normalize slice input
        if __validate__:
            slices = [slices] if isinstance(slices, slice) else list(slices)
            if len(slices) != len(axes):
                raise ValueError(
                    f"Expected {len(axes)} slice(s) for axes {axes}, but got {len(slices)}."
                )

        # Adjust slice indices if origin is 'active'
        slices = self._adjust_slices_for_origin(slices, axes=axes, origin=origin)

        # Dispatch to low-level coordinate computation
        return self._compute_coordinates_on_slices(
            np.asarray(axes_indices, dtype=int), slices
        )

    def compute_mesh_from_slices(
        self: _SupGridCore,
        slices: Union[slice, Sequence[slice]],
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
        **kwargs,
    ):
        """
        Compute coordinate arrays over a region of the grid specified by slice objects.

        Parameters
        ----------
        slices : slice or list of slice
            Slice(s) specifying the region to extract along the selected axes.
        axes : str or list of str, optional
            Axes to apply the slices to. If None, all axes are used.
        origin : {"active", "global"}, default="active"
            Whether the slices are relative to the active domain (excluding ghost zones)
            or the full ghost-augmented domain.
        __validate__ : bool, default=True
            Enable or disable all validation procedures. This should only be done in
            tight loop scenarios where performance is critical and typing is well controlled.
        kwargs:
            Additional keyword arguments to pass to :func:`numpy.meshgrid`.

        Returns
        -------
        tuple of np.ndarray
            Physical coordinate arrays corresponding to the sliced region, one per axis.
        """
        coords = self.compute_coords_from_slices(
            slices, axes=axes, origin=origin, __validate__=__validate__
        )
        return np.meshgrid(*coords, **kwargs)

    def compute_coords_from_index(
        self: _SupGridCore,
        index: "IndexInput",
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ) -> Tuple[float, ...]:
        """
        Compute the physical coordinates at a specific grid index.

        Parameters
        ----------
        index : int or sequence of int
            The grid index to evaluate. If `axes` is provided, must match its length.
        axes : str or list of str, optional
            The axes to extract coordinates along. If None, all axes are used.
        origin : {"active", "global"}, default="active"
            Whether the index is specified relative to the active domain or the global grid.
        __validate__ : bool, default=True
            If True, checks index bounds and normalizes it to global coordinates.

        Returns
        -------
        tuple of float
            The physical coordinate values at the given index, one per selected axis.
        """
        # Standardize and validate axes
        if __validate__:
            axes = self.standardize_axes(axes)

        axes_indices = np.asarray(self.__cs__.convert_axes_to_indices(axes), dtype=int)

        # Standardize and validate index if requested
        if __validate__:
            index = self._standardize_index(index, axes, origin)

        # Turn index into a slice of width 1 for each axis
        slices = [slice(i, i + 1) for i in index]

        # Compute coordinate arrays for the 1-point slice and extract scalars
        coords = self._compute_coordinates_on_slices(
            axis_indices=axes_indices, slices=slices
        )
        return tuple(float(c[0]) for c in coords)

    def compute_domain_coords(
        self: _SupGridCore,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        """
        Compute the 1D coordinate arrays for the full extent of the grid domain
        along the selected axes.

        This is a convenience wrapper around `compute_coords_from_slices(...)` that returns
        coordinates spanning the entire active or ghost-augmented domain.

        Parameters
        ----------
        axes : str or list of str, optional
            The axes for which to compute coordinates. If None, all axes are used.
        origin : {"active", "global"}, default="active"
            Whether to generate coordinates for the active domain or the global
            (ghost-augmented) domain.
        __validate__ : bool, default=True
            Whether to validate axis input and index bounds. Can be disabled in tight loops.

        Returns
        -------
        tuple of np.ndarray
            A tuple of 1D arrays — one per axis — representing physical coordinates along each axis.
        """
        axes = self.standardize_axes(axes)
        slices = [slice(None)] * len(axes)
        return self.compute_coords_from_slices(
            slices, axes=axes, origin=origin, __validate__=__validate__
        )

    def compute_domain_mesh(
        self: _SupGridCore,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
        **kwargs,
    ):
        """
        Compute a meshgrid of coordinate arrays spanning the full domain of the grid.

        This constructs a broadcasted meshgrid over the selected axes, using either the
        active domain or the full ghost-augmented domain, as controlled by `origin`.

        Parameters
        ----------
        axes : str or list of str, optional
            The axes to include in the meshgrid. If None, all axes are used.
        origin : {"active", "global"}, default="active"
            Whether to base the meshgrid on the active domain or full global domain.
        __validate__ : bool, default=True
            Whether to validate inputs. Can be disabled for performance.
        **kwargs :
            Additional keyword arguments passed to `np.meshgrid`, such as `indexing`.

        Returns
        -------
        tuple of np.ndarray
            A tuple of N coordinate arrays, each of shape matching the domain,
            suitable for use in vectorized field evaluation.
        """
        coords = self.compute_domain_coords(
            axes=axes, origin=origin, __validate__=__validate__
        )
        return np.meshgrid(*coords, indexing="ij", **kwargs)

    def compute_domain_edges(
        self: _SupGridCore,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        """
        Return coordinate edge arrays along each axis.

        If the grid is vertex-centered, returns coordinate arrays directly.
        If the grid is cell-centered, computes edges from centers and bounding box.

        Parameters
        ----------
        axes : str or list of str, optional
            Axes to include. Defaults to all.
        origin : {"active", "global"}, default="active"
            Whether to include ghost zones.
        __validate__ : bool, default=True
            Whether to validate axis inputs.

        Returns
        -------
        list of np.ndarray
            Coordinate edge arrays per axis.
        """
        axes = self.standardize_axes(axes)
        axes_indices = self.__cs__.convert_axes_to_indices(axes)
        coords = self.compute_domain_coords(
            axes=axes, origin=origin, __validate__=__validate__
        )

        if origin == "active":
            bbox = self.bbox[:, axes_indices]
        else:
            bbox = self.gbbox[:, axes_indices]

        if self.centering == "vertex":
            return coords
        elif self.centering == "cell":
            return self._centers_to_edges(coords, bbox)
        else:
            raise NotImplementedError(
                f"Centering mode '{self.centering}' not supported."
            )

    def compute_domain_centers(
        self: _SupGridCore,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        """
        Return coordinate center arrays along each axis.

        If the grid is vertex-centered, computes centers from edges.
        If the grid is cell-centered, returns coordinate arrays directly.

        Parameters
        ----------
        axes : str or list of str, optional
            Axes to include. Defaults to all.
        origin : {"active", "global"}, default="active"
            Whether to include ghost zones.
        __validate__ : bool, default=True
            Whether to validate axis inputs.

        Returns
        -------
        list of np.ndarray
            Coordinate center arrays per axis.
        """
        coords = self.compute_domain_coords(
            axes=axes, origin=origin, __validate__=__validate__
        )

        if self.centering == "vertex":
            return self._edges_to_centers(coords)
        elif self.centering == "cell":
            return coords
        else:
            raise NotImplementedError(
                f"Centering mode '{self.centering}' not supported."
            )

    # =============================== #
    # Slicing                         #
    # =============================== #
    # These methods each handle various slicing procedures for
    # the grid. These methods do not include methods which
    # center chunking semantics as that is in its own mixin class.
    def compute_domain_slice(
        self: _SupGridCore,
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
    ) -> Tuple[slice, ...]:
        """
        Compute a slice representing the entire grid domain with some set
        of axes, halo offsets, and ghost zone behavior.

        .. hint::

            This method is useful when you need to extract the entire domain
            but want to exclude ghost zones or want to include some number of
            additional cells around the boundary.

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

        Returns
        -------
        tuple of slice
            One slice per selected axis, expressed **in global index space**.
        """
        # Standardize / verify the inputs.
        num_axes, axes_mask = self._count_and_mask_axes(axes)
        axes_indices = np.asarray(self.__cs__.convert_axes_to_indices(axes), dtype=int)
        halo_offsets = self._standardize_halo_offset(halo_offsets, num_axes)

        # Modify the halo offsets so that they can
        # be used in a vectorized context later.
        halo_offsets[0, :] *= -1

        # Determine base start/stop slices
        if include_ghosts:
            start = np.zeros(num_axes, dtype=int)
            stop = self.__ghost_dd__[axes_indices]
        else:
            start = self.__ghost_zones__[0, axes_indices].copy()
            stop = start + self.__dd__[axes_indices]

        # Apply halo padding (already flipped left side above)
        start += halo_offsets[0, :]
        stop += halo_offsets[1, :]

        # Bounds checking
        full_extent = self.__ghost_dd__[axes_indices]
        if oob_behavior == "raise":
            oob_left = start < 0
            oob_right = stop > full_extent
            if np.any(oob_left | oob_right):
                for i, (s, e, maxlen) in enumerate(zip(start, stop, full_extent)):
                    if s < 0 or e > maxlen:
                        ax = self.axes[axes_indices[i]]
                        raise IndexError(
                            f"Domain slice out of bounds on axis '{ax}' (axis {axes_indices[i]}):\n"
                            f"  Computed slice     : slice({s}, {e})\n"
                            f"  Valid domain extent: [0, {maxlen})"
                        )
        elif oob_behavior == "clip":
            start = np.maximum(start, 0)
            stop = np.minimum(stop, full_extent)
        else:
            raise ValueError(
                f"Invalid oob_behavior: {oob_behavior!r}. Choose 'raise' or 'clip'."
            )

        # Return Python slice objects
        return tuple(slice(s, e) for s, e in zip(start, stop))

    def determine_domain_shape(
        self: _SupGridCore,
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
    ) -> Tuple[int, ...]:
        """
        Compute the shape of the domain along specified axes, including ghost zones and
        optional halo padding.

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

        Returns
        -------
        tuple of int
            The full shape of the region over the selected axes, including ghosts and halo.
        """
        # Standardize axes and get dimensionality
        axes = self.standardize_axes(axes)
        num_axes = len(axes)
        axes_indices = np.asarray(self.__cs__.convert_axes_to_indices(axes), dtype=int)

        # Standardize halo specification
        halo_offsets = self._standardize_halo_offset(halo_offsets, num_axes)
        total_halo = np.sum(halo_offsets, axis=0)

        # Base domain shape
        max_shape = self.__ghost_dd__[axes_indices]
        base_shape = (
            self.__ghost_dd__[axes_indices]
            if include_ghosts
            else self.__dd__[axes_indices]
        )
        shape = base_shape + total_halo

        # Handle out-of-bounds behavior
        overrun = shape > max_shape
        if np.any(overrun):
            if oob_behavior == "raise":
                raise ValueError(
                    f"Halo padding exceeds allocated domain on axes: "
                    f"{[self.axes[ai] for ai, flag in zip(axes_indices, overrun) if flag]}"
                )
            elif oob_behavior == "clip":
                shape = np.minimum(shape, max_shape)
            else:
                raise ValueError(
                    f"Invalid oob_behavior: {oob_behavior!r}. Must be 'raise' or 'clip'."
                )

        return tuple(int(s) for s in shape)

    # =============================== #
    # Interpolation on Domains        #
    # =============================== #
    def construct_domain_interpolator(
        self: _SupGridCore,
        field: np.ndarray,
        field_axes: Sequence[str],
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: Optional[float] = np.nan,
        **kwargs,
    ):
        """
        Construct an interpolator object over the entire grid domain on
        a particular subset of axes.

        This method utilizes SciPy's :class:`~scipy.interpolate.RegularGridInterpolator`
        to leverage its C-level interpolation methods.

        Parameters
        ----------
        field : np.ndarray
            The data to interpolate, shaped according to `field_axes`.
        field_axes : Sequence[str]
            Logical axes spanned by the field (e.g., ["x", "y"]).
        method : {"linear", "nearest"}, default="linear"
            Interpolation method.
        bounds_error : bool, default=False
            Whether to raise an error when queried points fall outside the domain.
        fill_value : float or None, default=np.nan
            Value used to fill in for out-of-bound points if `bounds_error=False`.
        **kwargs
            Additional keyword arguments passed to the interpolator constructor.

        Returns
        -------
        interpolator : RegularGridInterpolator
            A callable that takes physical coordinates and returns interpolated values.
        """
        from scipy.interpolate import RegularGridInterpolator

        # Standardize the field axes and then obtain the relevant coordinates.
        field_axes = self.standardize_axes(field_axes)
        coords = self.compute_domain_coords(axes=field_axes, origin="global")

        # Create the interpolator.
        self.check_field_shape(field.shape, axes=field_axes)
        interpolator = RegularGridInterpolator(
            points=coords,
            values=field,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            **kwargs,
        )

        return interpolator

    # =============================== #
    # Casting                         #
    # =============================== #
    def broadcast_shape_to_axes(
        self: _SupGridCore,
        shape: Union[int, Sequence[int]],
        axes_in: "AxesInput",
        axes_out: "AxesInput",
    ) -> Tuple[int, ...]:
        """
        Expand an input shape aligned to `axes_in` so it is broadcastable against `axes_out`.

        This inserts singleton dimensions (1s) in the positions corresponding to any axes
        present in `axes_out` but not in `axes_in`, while preserving trailing non-grid dimensions.

        Parameters
        ----------
        shape : int or sequence of int
            Shape aligned with `axes_in`, possibly followed by extra non-grid dimensions
            (e.g., for vector or tensor fields). If a scalar, it is treated as `(shape,)`.

        axes_in : str or list of str
            The axes corresponding to the leading dimensions of `shape`.

        axes_out : str or list of str
            The target axes into which the shape should be broadcastable.

        Returns
        -------
        tuple of int
            A shape tuple broadcastable over the axes in `axes_out`, preserving any
            trailing non-grid dimensions.

        Raises
        ------
        ValueError
            If `axes_in` is not a subset of `axes_out`, or if the shape does not match
            the number of input axes.
        """
        # Standardize axes
        axes_in = self.standardize_axes(axes_in)
        axes_out = self.standardize_axes(axes_out)

        # Convert shape to tuple
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)

        n_in, _ = len(axes_in), len(axes_out)

        if not set(axes_in).issubset(set(axes_out)):
            raise ValueError(
                f"axes_in {axes_in} must be a subset of axes_out {axes_out}"
            )

        if len(shape) < n_in:
            raise ValueError(
                f"Shape {shape} does not have enough dimensions to match axes_in {axes_in}"
            )

        trailing = shape[n_in:]  # preserve any non-grid dims (e.g. for vector fields)
        shape_map = dict(zip(axes_in, shape[:n_in]))
        broadcast_shape = tuple(shape_map.get(ax, 1) for ax in axes_out) + trailing

        return broadcast_shape

    def broadcast_shapes_to_axes(
        self,
        shapes: Sequence[Union[int, Sequence[int]]],
        axes_in: Sequence["AxesInput"],
        axes_out: "AxesInput",
    ) -> Sequence[Tuple[int, ...]]:
        """
        Broadcast multiple input shapes aligned to `axes_in` so they are compatible with `axes_out`.

        This is a pluralized version of :meth:`broadcast_shape_to_axes` that applies the same
        logic to a list of shapes and axis specifications.

        Parameters
        ----------
        shapes : sequence of shape-like
            A list of shapes, each of which corresponds to an input array. Each shape must be aligned
            with the axes provided in `axes_in`, possibly followed by additional non-grid dimensions
            (e.g., for vector or tensor components).
        axes_in : sequence of AxesInput
            A list of axis specifications. Each entry describes the grid axes that the corresponding
            shape aligns to.
        axes_out : AxesInput
            The target axes into which all shapes should be broadcast-compatible. Axes not present
            in an input are inserted as singleton dimensions (1).

        Returns
        -------
        sequence of tuple[int, ...]
            The broadcasted shapes, each of which is compatible with the `axes_out` layout.

        Raises
        ------
        ValueError
            If the number of input shapes and axis specs do not match, or if any axes_in are not
            a subset of axes_out.
        """
        return tuple(
            self.broadcast_shape_to_axes(s, ai, axes_out)
            for s, ai in zip(shapes, axes_in)
        )

    def broadcast_array_to_axes(
        self: _SupGridCore,
        array: np.ndarray,
        axes_in: "AxesInput",
        axes_out: "AxesInput",
        **kwargs,
    ) -> np.ndarray:
        """
        Reshape and broadcast an array aligned with `axes_in` so it is compatible with `axes_out`.

        This utility is used to make field-like arrays broadcastable over a grid whose axes
        are specified by `axes_out`. It inserts singleton dimensions (1s) for any axes
        present in `axes_out` but not in `axes_in`, while preserving any trailing (non-grid)
        dimensions such as vector or tensor components.

        Parameters
        ----------
        array : np.ndarray
            Input array. Its leading dimensions must correspond to the axes in `axes_in`.
            Any remaining dimensions (e.g., component indices) are preserved.
        axes_in : str or list of str
            Axes that the leading dimensions of the array are aligned to.
        axes_out : str or list of str
            Target axes to broadcast against. The returned array will have dimensions
            aligned with these axes.
        **kwargs:
            Any kwargs to pass to the array's `.reshape` method.

        Returns
        -------
        np.ndarray
            A view or broadcasted copy of the input array, shaped to match `axes_out`
            and ready for operations over the target grid.

        Raises
        ------
        ValueError
            If `axes_in` is not a subset of `axes_out`, or if the array shape is incompatible
            with the specified input axes.
        """
        # Determine the broadcast shape.
        new_shape = self.broadcast_shape_to_axes(array.shape, axes_in, axes_out)
        return array.reshape(new_shape, **kwargs)

    def broadcast_arrays_to_axes(
        self,
        arrays: Sequence[np.ndarray],
        axes_in: Sequence["AxesInput"],
        axes_out: "AxesInput",
    ) -> Sequence[np.ndarray]:
        """
        Broadcast multiple arrays aligned to `axes_in` so they are compatible with `axes_out`.

        This pluralized version of :meth:`broadcast_array_to_axes` takes a sequence of arrays
        and corresponding axis specifications and returns arrays reshaped to be broadcastable
        over a common target axis layout.

        Parameters
        ----------
        arrays : sequence of np.ndarray
            A list of arrays to reshape and broadcast. Each array's leading dimensions must align
            with the corresponding entry in `axes_in`. Trailing dimensions are preserved.
        axes_in : sequence of AxesInput
            A list of axis specifications for the input arrays.
        axes_out : AxesInput
            The target axes into which all arrays will be broadcast-compatible.

        Returns
        -------
        sequence of np.ndarray
            The broadcast-compatible arrays, each reshaped according to the `axes_out` layout.

        Raises
        ------
        ValueError
            If the input shapes and axes are incompatible or misaligned.
        """
        return tuple(
            self.broadcast_array_to_axes(a, ai, axes_out)
            for a, ai in zip(arrays, axes_in)
        )

    def tile_array_to_axes(
        self: _SupGridCore,
        array: np.ndarray,
        axes_in: "AxesInput",
        axes_out: "AxesInput",
        /,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        **kwargs,
    ) -> np.ndarray:
        """
        Tile a lower-dimensional array across the full domain of the grid defined by `axes_out`.

        This function reshapes the input array to match the leading axes in `axes_out`,
        inserts singleton dimensions for any missing axes, and tiles (repeats) the data
        over the selected region of the grid, which may include ghost zones and halos.

        Parameters
        ----------
        array : np.ndarray
            The input array to tile. Its leading dimensions must match the axes in `axes_in`.
        axes_in : str or list of str
            Axes corresponding to the array's grid-aligned dimensions.
        axes_out : str or list of str
            Target axes over which to tile. This defines the full spatial shape.
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
            Additional keyword arguments passed to `np.broadcast_to`.

        Returns
        -------
        np.ndarray
            A tiled array broadcasted over the full region selected by `axes_out`.
            The trailing dimensions (not in `axes_out`) are preserved.

        Raises
        ------
        ValueError
            If the array is not compatible with the grid or specified axes.
        """
        # Standardize the input shape and axes
        shape_in = array.shape
        axes_in = self.standardize_axes(axes_in)
        axes_out = self.standardize_axes(axes_out)

        # Get the shape of the region we're tiling over
        region_slice = self.compute_domain_slice(
            axes=axes_out,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )
        region_shape = tuple(s.stop - s.start for s in region_slice)

        # Determine the full target shape (grid + trailing dims)
        trailing_shape = shape_in[len(axes_in) :]
        target_shape = region_shape + trailing_shape

        # Broadcast the array to the target shape
        broadcast_shape = self.broadcast_shape_to_axes(shape_in, axes_in, axes_out)
        reshaped = array.reshape(broadcast_shape + trailing_shape)
        return np.broadcast_to(reshaped, target_shape, **kwargs)

    def empty(
        self: _SupGridCore,
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
        domain_shape = self.determine_domain_shape(
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.empty(array_shape, **kwargs)

    def zeros(
        self: _SupGridCore,
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
        domain_shape = self.determine_domain_shape(
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.zeros(array_shape, **kwargs)

    def ones(
        self: _SupGridCore,
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
        domain_shape = self.determine_domain_shape(
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.ones(array_shape, **kwargs)

    def full(
        self: _SupGridCore,
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
        domain_shape = self.determine_domain_shape(
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        array_shape = domain_shape + tuple(element_shape)
        return np.full(array_shape, fill_value=fill_value, **kwargs)

    # =============================== #
    # Summary Methods                 #
    # =============================== #
    def summary(self: _SupGridCore) -> None:
        """
        Print a summary of the grid structure, including domain shape, chunking,
        coordinate system, and ghost zone information.
        """
        print(f"\nGrid Summary — {self.__class__.__name__}")
        print("-" * 60)

        # Coordinate system and axes
        cs = self.coordinate_system
        print(f"\tCoordinate System : {cs.__class__.__name__}")
        print(f"\tCoordinate Axes   : {', '.join(self.axes)}")

        # Bounding boxes
        print("\nBounding Box (active):")
        print(f"\tLower : {self.bbox[0]}")
        print(f"\tUpper : {self.bbox[1]}")

        print("Bounding Box (global):")
        print(f"\tLower : {self.gbbox[0]}")
        print(f"\tUpper : {self.gbbox[1]}")

        # Grid shapes
        print(
            "\nShape: ",
            np.array(self.shape),
            " (active). ",
            np.array(self.gdd),
            " (global).",
        )

        # Chunking info
        if self.chunking:
            print("\nChunking Enabled : TRUE")
            print("\tChunk Size        : ", np.array(self.chunk_size))
            print("\tNumber of Chunks  : ", np.array(self.cdd))
        else:
            print("\nChunking Enabled : FALSE")

        # Ghost zones
        print("\nGhost Zones:")
        for i, ax in enumerate(self.axes):
            left, right = self.ghost_zones[:, i]
            print(f"  - {ax:<5s} : left={left}, right={right}")

        print("-" * 60)

    def show(self: _SupGridCore) -> None:
        """
        Alias for `summary()` to allow interactive/quick display.
        """
        return self.summary()

class GridIOMixin(Generic[_SupGridIO]):
    """
    Grid IO mixin class for structured coordinate grids.
    """
    def to_json(self: _SupGridIO,
                filepath: Union[str, Path],
                overwrite: bool = False):
        """
        Save the grid metadata to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Destination path for the JSON file.
        overwrite : bool, default=False
            Whether to overwrite the file if it already exists.
        """
        import json

        # Coerce the filepath to a Path object and ensure
        # that we correctly handle the overwrite parameters.
        filepath = Path(filepath)
        if filepath.exists():
            if overwrite:
                filepath.unlink()
            else:
                raise FileExistsError(f"File '{filepath}' already exists. Use overwrite=True to replace it.")

        # Create the metadata dictionary from the grid properties.
        # We encapsulate this so that we can standardize the user facing error
        # message.
        try:
            metadata = self.to_metadata_dict()
        except Exception as e:
            raise RuntimeError(
                f"Failed to write grid to JSON due to serialization error in subclass: {e}"
            ) from e

        # Now open and dump the json data.
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=4)

    def to_yaml(self: _SupGridIO, filepath: Union[str, Path], overwrite: bool = False):
        """
        Save the grid metadata to a YAML file.

        Parameters
        ----------
        filepath : str or Path
            Destination path for the YAML file.
        overwrite : bool, default=False
            Whether to overwrite the file if it already exists.
        """
        import yaml

        # Coerce the filepath to a Path object and ensure
        # that we correctly handle the overwrite parameters.
        filepath = Path(filepath)
        if filepath.exists():
            if overwrite:
                filepath.unlink()
            else:
                raise FileExistsError(f"File '{filepath}' already exists. Use overwrite=True to replace it.")

        # Create the metadata dictionary from the grid properties.
        # We encapsulate this so that we can standardize the user facing error
        # message.
        try:
            metadata = self.to_metadata_dict()
        except Exception as e:
            raise RuntimeError(
                f"Failed to write grid to JSON due to serialization error in subclass: {e}"
            ) from e

        with open(filepath, "w") as f:
            yaml.safe_dump(metadata, f)

    def to_hdf5(self: _SupGridIO,
                filepath: Union[str, Path],
                group_name: Optional[str] = None,
                overwrite: bool = False):
        """
        Save the grid metadata to an HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to the HDF5 file.
        group_name : str, optional
            Optional group name under which to store metadata.
        overwrite : bool, default=False
            Whether to overwrite existing file or group.
        """
        import h5py
        import json

        filepath = Path(filepath)
        if filepath.exists() and group_name is None:
            if overwrite:
                filepath.unlink()
            else:
                raise FileExistsError(f"File '{filepath}' exists. Use `overwrite=True` or set `group_name`.")

        metadata = self.to_metadata_dict()

        # Ensure the file exists before appending
        with h5py.File(filepath, "a") as f:
            if group_name is None:
                group = f
            else:
                group =  f.require_group(group_name)

                if group_name in f and overwrite:
                    del f[group_name]
                    group = f.create_group(group_name)

            # Save key-value pairs as JSON-compatible attributes
            for key, val in metadata.items():
                try:
                    if isinstance(val, (int, float, str)):
                        group.attrs[key] = val
                    else:
                        group.attrs[key] = json.dumps(val)
                except Exception as e:
                    raise TypeError(f"Cannot serialize key '{key}' to HDF5: {e}")

    @classmethod
    def from_json(cls: _SupGridIO,
                  filepath: Union[str, Path],
                  coordinate_system: "_CoordinateSystemBase") -> _SupGridIO:
        """
        Load grid metadata from a JSON file and reconstruct the grid.

        Parameters
        ----------
        filepath : str or Path
            Path to the input JSON file.
        coordinate_system : ~coordinates.core.CurvilinearCoordinateSystem
            A coordinate system instance to associate with the grid.

        Returns
        -------
        GridBase
            An instance of the grid reconstructed from metadata.
        """
        import json

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' does not exist.")

        try:
            with open(filepath, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON metadata: {e}") from e

        try:
            return cls.from_metadata_dict(coordinate_system, metadata)
        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct grid from JSON metadata: {e}") from e

    @classmethod
    def from_yaml(cls: _SupGridIO,
                  filepath: Union[str, Path],
                  coordinate_system: "_CoordinateSystemBase") -> _SupGridIO:
        """
        Load grid metadata from a YAML file and reconstruct the grid.

        Parameters
        ----------
        filepath : str or Path
            Path to the input YAML file.
        coordinate_system : ~coordinates.core.CurvilinearCoordinateSystem
            A coordinate system instance to associate with the grid.

        Returns
        -------
        GridBase
            An instance of the grid reconstructed from metadata.
        """
        import yaml

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' does not exist.")

        try:
            with open(filepath, "r") as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read YAML metadata: {e}") from e

        try:
            return cls.from_metadata_dict(coordinate_system, metadata)
        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct grid from YAML metadata: {e}") from e

    @classmethod
    def from_hdf5(cls: _SupGridIO,
                  filepath: Union[str, Path],
                  coordinate_system: "_CoordinateSystemBase",
                  group_name: Optional[str] = None) -> _SupGridIO:
        """
        Load grid metadata from an HDF5 file and reconstruct the grid.

        Parameters
        ----------
        filepath : str or Path
            Path to the HDF5 file.
        coordinate_system : ~coordinates.core.CurvilinearCoordinateSystem
            A coordinate system instance to associate with the grid.
        group_name : str, optional
            Group name in the file under which metadata is stored. If None, reads from the root.

        Returns
        -------
        GridBase
            An instance of the grid reconstructed from metadata.
        """
        import h5py
        import json

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' does not exist.")

        try:
            with h5py.File(filepath, "r") as f:
                group = f if group_name is None else f[group_name]
                metadata = {}
                for key, val in group.attrs.items():
                    try:
                        metadata[key] = json.loads(val)
                    except (TypeError, json.JSONDecodeError):
                        metadata[key] = val
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata from HDF5: {e}") from e

        try:
            return cls.from_metadata_dict(coordinate_system, metadata)
        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct grid from HDF5 metadata: {e}") from e

class GridPlotMixin(Generic[_SupGridChunking]):
    """
    Mixin class for plotting 2D projections of structured coordinate grids.

    Provides utility methods to visualize:

    - Grid lines aligned with coordinate axes
    - Chunk boundaries (if chunking is enabled)

    Useful for debugging domain layout, visualizing ghost zones, and
    showing subdomain tiling in structured finite difference computations.
    """

    def plot_grid_lines(
        self: _SupGridChunking,
        grid_axes: Optional[Sequence[str]] = None,
        ax: Optional["Axes"] = None,
        include_ghosts: bool = False,
        **kwargs,
    ):
        """
        Plot coordinate grid lines along the specified 2D slice of the domain.

        Parameters
        ----------
        grid_axes : list of str, optional
            The two coordinate axes to display in the plot. If not specified,
            the first two axes of the coordinate system are selected.
        ax : ~matplotlib.axes.Axes, optional
            Matplotlib axis object to draw into. If None, a new figure is created.
        include_ghosts : bool, default=False
            Whether to include ghost zones in the plotting extent and coordinates.
        **kwargs
            Additional keyword arguments passed to :func:`matplotlib.pyplot.hlines` and
            :func:`matplotlib.pyplot.vlines` (e.g., linewidth, linestyle, etc).

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis object used for plotting.
        """
        # Ensure matplotlib is actually imported but we don't
        # want to waste time importing it from the top of the module.
        import matplotlib.pyplot as plt

        # Validate the axes and ensure everything is set up.
        if self.ndim < 2:
            raise ValueError("Cannot show grid lines for coordinate system in 1D.")
        grid_axes = (
            self.standardize_axes(grid_axes) if grid_axes is not None else self.axes[:2]
        )
        grid_axes_indices = self.__cs__.convert_axes_to_indices(grid_axes)
        if len(grid_axes) != 2:
            raise ValueError(f"Exactly two axes must be provided. Got {grid_axes}.")

        # Extract the X and Y coordinates from the grid. This requires resolving the
        # desired origin as well.
        origin: Literal["global", "active"] = "global" if include_ghosts else "active"
        x_coords, y_coords = self.compute_domain_coords(grid_axes, origin=origin)

        # Determine the boundaries for the x and y coordinates. This will depend
        # specifically on the relevant bouding box.
        # Get axis indices and bounding box
        bbox = (
            self.gbbox[:, grid_axes_indices]
            if include_ghosts
            else self.bbox[:, grid_axes_indices]
        )

        # -- PLOTTING SEQUENCE -- #
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # Draw the vertical lines at each of the x positions.
        # Draw the horizontal lines at each of the y positions.
        kwargs["color"] = kwargs.get("color", "k")

        ax.vlines(x_coords, bbox[0, 1], bbox[1, 1], **kwargs)
        ax.hlines(y_coords, bbox[0, 0], bbox[1, 0], **kwargs)

        return ax

    def plot_chunk_lines(
        self: _SupGridChunking,
        grid_axes: Optional[Sequence[str]] = None,
        ax: Optional["Axes"] = None,
        include_ghosts: bool = False,
        **kwargs,
    ):
        """
        Plot chunk boundaries along the specified 2D projection of the domain.

        Parameters
        ----------
        grid_axes : list of str, optional
            The two coordinate axes to display in the plot. If not specified,
            the first two axes of the coordinate system are selected.
        ax : ~matplotlib.axes.Axes, optional
            Matplotlib axis object to draw into. If None, a new figure is created.
        include_ghosts : bool, default=False
            Whether to include ghost zones in the plotting extent and chunk structure.
        **kwargs
            Additional keyword arguments passed to :func:`matplotlib.pyplot.hlines` and
            :func:`matplotlib.pyplot.vlines` (e.g., linewidth, linestyle, etc).

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis object used for plotting.

        Raises
        ------
        ValueError
            If the coordinate system is 1D or if an invalid number of axes is specified.
        RuntimeError
            If chunking is not supported by this grid.
        """
        # Ensure matplotlib is actually imported but we don't
        # want to waste time importing it from the top of the module.
        import matplotlib.pyplot as plt

        # Validate the axes and ensure everything is set up.
        self._ensure_supports_chunking()

        if self.ndim < 2:
            raise ValueError("Cannot show chunk lines for coordinate system in 1D.")
        grid_axes = (
            self.standardize_axes(grid_axes) if grid_axes is not None else self.axes[:2]
        )
        grid_axes_indices = self.__cs__.convert_axes_to_indices(grid_axes)
        if len(grid_axes) != 2:
            raise ValueError(f"Exactly two axes must be provided. Got {grid_axes}.")

        # Extract the chunk coordinates.
        cx, cy = [], []
        for ccx, ccy in self.iter_chunk_coords(
            axes=grid_axes, include_ghosts=include_ghosts
        ):
            cx.append(ccx[0])
            cy.append(ccy[0])

        # Determine the boundaries for the x and y coordinates. This will depend
        # specifically on the relevant bouding box.
        # Get axis indices and bounding box
        bbox = (
            self.gbbox[:, grid_axes_indices]
            if include_ghosts
            else self.bbox[:, grid_axes_indices]
        )
        cx.append(bbox[1, 0])
        cy.append(bbox[1, 1])

        # -- PLOTTING SEQUENCE -- #
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # Draw the vertical lines at each of the x positions.
        # Draw the horizontal lines at each of the y positions.
        kwargs["color"] = kwargs.get("color", "k")
        ax.vlines(cx, bbox[0, 1], bbox[1, 1], **kwargs)
        ax.hlines(cy, bbox[0, 0], bbox[1, 0], **kwargs)

        return ax

    def plot_ghost_zone_shading(
        self: _SupGridChunking,
        grid_axes: Optional[Sequence[str]] = None,
        ax: Optional["Axes"] = None,
        facecolor: str = "gray",
        alpha: float = 0.3,
    ):
        """
        Shade the ghost zones in a 2D grid projection.

        Parameters
        ----------
        grid_axes : list of str, optional
            Axes to visualize (must be 2D). Defaults to the first two axes.
        ax : matplotlib.axes.Axes, optional
            Axis to plot into. A new one is created if None.
        facecolor : str
            Color to fill the ghost region.
        alpha : float
            Transparency level for shading.

        Returns
        -------
        matplotlib.axes.Axes
            Axis object containing the shaded plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if self.ndim < 2:
            raise ValueError("Cannot shade ghost zones for coordinate system in 1D.")
        grid_axes = (
            self.standardize_axes(grid_axes) if grid_axes is not None else self.axes[:2]
        )
        if len(grid_axes) != 2:
            raise ValueError("Shading requires exactly two axes.")
        grid_axes_indices = self.__cs__.convert_axes_to_indices(grid_axes)

        bbox = self.bbox[:, grid_axes_indices]
        gbbox = self.gbbox[:, grid_axes_indices]

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # Add ghost rectangles along each axis
        ax.add_patch(
            Rectangle(
                (float(gbbox[0, 0]), float(gbbox[0, 1])),
                float(bbox[0, 0] - gbbox[0, 0]),
                float(gbbox[1, 1] - gbbox[0, 1]),
                facecolor=facecolor,
                alpha=alpha,
            )
        )
        ax.add_patch(
            Rectangle(
                (float(bbox[1, 0]), float(gbbox[0, 1])),
                float(gbbox[1, 0] - bbox[1, 0]),
                float(gbbox[1, 1] - gbbox[0, 1]),
                facecolor=facecolor,
                alpha=alpha,
            )
        )
        ax.add_patch(
            Rectangle(
                (float(gbbox[0, 0]), float(gbbox[0, 1])),
                float(gbbox[1, 0] - gbbox[0, 0]),
                float(bbox[0, 1] - gbbox[0, 1]),
                facecolor=facecolor,
                alpha=alpha,
            )
        )
        ax.add_patch(
            Rectangle(
                (float(gbbox[0, 0]), float(bbox[1, 1])),
                float(gbbox[1, 0] - gbbox[0, 0]),
                float(gbbox[1, 1] - bbox[1, 1]),
                facecolor=facecolor,
                alpha=alpha,
            )
        )

        return ax
