"""
Core grid structures for Pisces-Geometry. All user facing grid classes are descended
from parent classes in this module.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from coordinates.base import _CoordinateSystemBase
from tqdm.auto import tqdm

from pisces_geometry.grids._exceptions import GridInitializationError
from pisces_geometry.grids._typing import BoundingBox, DomainDimensions
from pisces_geometry.utilities.logging import pg_log


# noinspection PyTypeChecker
class _BaseGrid(ABC):
    """
    Generic coordinate grid base class from which all Pisces-Geometry grid subclasses are descended.

    This class serves as the foundational abstraction for all grid types used in Pisces Geometry.
    It handles setup and storage of coordinate systems, domain dimensions, bounding boxes, boundary
    conditions, and ghost zones. Subclasses are responsible for implementing actual coordinate logic,
    spacing behavior, and field interactions.

    Subclasses should override the initialization methods to define specific behavior for:

    - Setting up the coordinate system
    - Defining the domain and shape of the grid
    - Configuring boundaries and ghost cells

    Notes
    -----
    This class does not compute or store coordinates directly. It exists to manage the metadata
    and structure of the computational domain and should be extended to support concrete behavior.
    """

    # @@ Initialization Procedures @@ #
    # These initialization procedures may be overwritten in subclasses
    # to specialize the behavior of the grid. Ensure that all the REQUIRED
    # attributes are set in the initialization methods, otherwise unintended behavior
    # may arise.
    def __set_coordinate_system__(
        self, coordinate_system: "_CoordinateSystemBase", *args, **kwargs
    ):
        """
        Assign the coordinate system to the grid.

        Subclasses may override this method to enforce validation logic, such as checking whether
        the coordinate system is orthogonal, curvilinear, etc.

        **Required Parameters**:

        1. ``self.__cs__`` the ``CoordinateSystemBase`` class for the grid.

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system associated with this grid.

        Raises
        ------
        GridInitializationError
            If the coordinate system is invalid or incompatible.
        """
        # Default implementation is to simply assign self.__cs__ to the input
        # coordinate system.
        self.__cs__: "_CoordinateSystemBase" = coordinate_system

    def __set_grid_domain__(self, *args, **kwargs):
        """
        Configure the shape and physical bounding box of the domain.

        This method is responsible for defining:

        - ``self.__bbox__``: The physical bounding box of the domain (without ghost cells). This should be a
           valid ``BoundingBox`` instance with shape (2,NDIM) defining first the bottom left corner of the domain
           and then the top right corner of the domain.
        - ``self.__dd__``: The DomainDimensions object defining the grid shape (without ghost cells). This should be
           a valid ``DomainDimensions`` instance with shape ``(NDIM,)``.
        - ``self.__chunking__``: boolean flag indicating if chunking is allowed.
        - ``self.__chunk_size__``: The DomainDimensions for a single chunk of the grid.
        - ``self.__cdd__``: The number of chunks in each dimension of the domain.

        If ``__chunking__`` is false, then the various chunk-related variables can
        remain as ``None``.

        Should be overridden in subclasses to support specific grid shape logic.
        """
        # Standard domain variables (must set always).
        self.__bbox__: BoundingBox = None
        self.__dd__: DomainDimensions = None

        # Chunking domain variables. (relevant if chunking is turned on).
        self.__chunking__: bool = False
        self.__chunk_size__: DomainDimensions = None
        self.__cdd__: DomainDimensions = None

    def __set_grid_boundary__(self, *args, **kwargs):
        """
        Configure boundary-related attributes for the grid.

        This includes:

        - ``self.__ghost_zones__``: Number of ghost cells on each side (2, ndim).
        - ``self.__ghost_bbox__``: Bounding box including ghost regions.
        - ``self.__ghost_dd__``: DomainDimensions object including ghost cells.

        This is where boundary conditions (periodic, Dirichlet, etc.) and ghost cell layout
        should be resolved.

        Should be overridden in subclasses to implement behavior.
        """
        self.__ghost_zones__ = None
        self.__ghost_bbox__ = None
        self.__ghost_dd__ = None

    def __init__(self, coordinate_system: "_CoordinateSystemBase", *args, **kwargs):
        """
        Initialize the _BaseGrid object with a coordinate system and optional parameters.

        The initialization sequence involves:
        1. Setting the coordinate system
        2. Defining the physical bounding box and grid shape
        3. Setting up boundary conditions and ghost zones

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system in which this grid is defined.
        args : tuple
            Additional positional arguments forwarded to initialization hooks.
        kwargs : dict
            Additional keyword arguments forwarded to initialization hooks.

        Raises
        ------
        GridInitializationError
            If any step of the initialization process fails.
        """
        try:
            self.__set_coordinate_system__(coordinate_system, *args, **kwargs)
        except Exception as e:
            raise GridInitializationError(
                f"Failed to set up coordinate system for grid: {e}"
            ) from e

        try:
            self.__set_grid_domain__(*args, **kwargs)
        except Exception as e:
            raise GridInitializationError(
                f"Failed to set up domain for grid: {e}"
            ) from e

        try:
            self.__set_grid_boundary__(*args, **kwargs)
        except Exception as e:
            raise GridInitializationError(
                f"Failed to set up boundary for grid: {e}"
            ) from e

        # Run the __post_init__ method afterward.
        self.__post_init__()
        pg_log.debug("Initialized grid %s on coordinate system %s.", self, self.__cs__)

    def __post_init__(self):
        """
        __post_init__ can be used to configure any additional aspects of the subclass after
        the rest of the __initialization__ procedure has been performed.
        """
        pass

    # @@ Grid Properties @@ #
    # Subclasses can (and often should) add additional properties; however,
    # existing properties should be consistent in behavior (return type, meaning, etc.) with
    # superclasses and sibling classes to ensure that the use experiences
    # are conserved.
    @property
    def coordinate_system(self) -> "_CoordinateSystemBase":
        """
        The coordinate system (e.g. a subclass of :py:class:`~coordinates.core.OrthogonalCoordinateSystem`) which
        underlies this grid.

        The coordinate system determines which axes are available in the grid (:py:attr:`axes`) and also determines
        how various differential procedures are performed in this grid structure.
        """
        return self.__cs__

    @property
    def bbox(self) -> BoundingBox:
        """
        The physical bounding box of the grid (excluding ghost zones).

        This defines the actual spatial extent of the active computational domain
        using the physical coordinates derived from the coordinate arrays.

        Returns
        -------
        numpy.ndarray
            A ``(2, ndim)`` array representing ``[lower_corner, upper_corner]`` in physical space.
        """
        return self.__bbox__

    @property
    def dd(self) -> DomainDimensions:
        """
        The shape of the active grid (excluding ghost cells), expressed in grid points.

        This defines the number of grid points along each axis, not counting ghost zones.

        Returns
        -------
         numpy.ndarray
            A tuple-like object specifying the number of grid points per axis.
        """
        return self.__dd__

    @property
    def ndim(self) -> int:
        """
        The number of spatial dimensions in the grid.

        This is inferred from the associated coordinate system's number of dimensions.

        Returns
        -------
        int
            The number of dimensions in the grid.
        """
        return self.__cs__.ndim

    @property
    def axes(self) -> List[str]:
        """
        The names of the coordinate axes in this grid.

        These are inherited from the coordinate system and may include labels like
        ``["x", "y", "z"]`` or curvilinear variants like ``["r", "theta", "phi"]``.

        Returns
        -------
        list of str
            A list of axis names.
        """
        return self.__cs__.axes

    @property
    def shape(self) -> Sequence[int]:
        """
        The shape of the grid (excluding ghost cells), as a tuple of point counts.

        This is an alias for :attr:`dd` and provides compatibility with numpy-like APIs.

        Returns
        -------
        tuple of int
            The number of grid points along each axis.
        """
        return self.dd

    @property
    def gbbox(self) -> BoundingBox:
        """
        The full bounding box of the grid, including ghost regions.

        This includes additional layers of ghost cells on each boundary
        as specified by the grid’s ghost zone configuration.

        Returns
        -------
        BoundingBox
            A (2, ndim) array specifying [lower_corner, upper_corner] with ghost zones included.
        """
        return self.__ghost_bbox__

    @property
    def gdd(self) -> DomainDimensions:
        """
        The full grid dimensions, including ghost cells.

        This represents the shape of the full buffer or storage array needed
        to hold all values including stencil padding.

        Returns
        -------
        DomainDimensions
            Grid dimensions including ghost zones.
        """
        return self.__ghost_dd__

    @property
    def chunk_size(self) -> DomainDimensions:
        """
        The size of each chunk along every axis, if chunking is enabled.

        Chunking divides the grid into smaller subdomains (chunks) for
        more efficient memory management or parallelization. Each chunk has
        this shape (excluding ghost cells).

        Returns
        -------
        DomainDimensions
            Size of a single chunk along each axis.

        Raises
        ------
        GridInitializationError
            If chunking is not enabled for this grid.
        """
        if self.__chunking__:
            return self.__chunk_size__
        else:
            raise GridInitializationError("Chunking is not enabled on this grid.")

    @property
    def chunking(self) -> bool:
        """
        Whether chunking is enabled for this grid.

        When enabled, the domain is partitioned into regularly sized blocks (chunks),
        each potentially processable in isolation.

        Returns
        -------
        bool
            True if chunking is active; False otherwise.
        """
        return self.__chunking__

    @property
    def ghost_zones(self) -> np.ndarray:
        """
        Number of ghost cells on either side of each axis.

        Ghost zones are extra layers of points added beyond the physical domain
        to facilitate finite-difference stencils or boundary conditions.

        Returns
        -------
        np.ndarray
            A (2, ndim) array where the first row is the number of ghost cells
            on the "left" (lower) side of each axis, and the second row is for the "right" (upper) side.
        """
        return self.__ghost_zones__

    # @@ Dunder Methods @@ #
    # Subclasses should NOT ALTER these methods in order to ensure that
    # all grid subclasses behave (more or less) the same way. Alterations can
    # be made in instances where it is necessary to do so in the interest of preserving
    # the inherited behavior correctly; however, this should be done very cautiously.
    def __repr__(self) -> str:
        """
        Unambiguous string representation of the grid object.
        """
        return (
            f"<{self.__class__.__name__} | "
            f"ndim={self.ndim}, shape={self.shape}, bbox={self.bbox}>"
        )

    def __str__(self) -> str:
        """
        Human-readable summary of the grid object.
        """
        return f"<{self.__class__.__name__} | shape={self.shape}>"

    def __len__(self) -> int:
        """
        Return the total number of grid points (excluding ghost zones).
        """
        return int(np.prod(self.shape))

    def __getitem__(self, index: Tuple[int, ...]):
        """
        Return the coordinates at a given index in the grid.

        Parameters
        ----------
        index : tuple of int
            Index tuple into the grid. Must match the dimensionality of the grid.

        Returns
        -------
        sequence of float
            Coordinate values corresponding to the given index.
        """
        return self.get_coordinates(index)

    def __call__(
        self,
        index: Union[int, Tuple[int, ...]],
        axes: Sequence[str] = None,
        include_ghosts: bool = False,
    ):
        """
        Alias for :py:meth:`__getitem__`. Enables `grid(index)` syntax.
        """
        return self.get_coordinates(index, axes=axes, include_ghosts=include_ghosts)

    def __contains__(self, item: Sequence[float]) -> bool:
        """
        Check whether a physical point lies within the grid bounding box.

        Parameters
        ----------
        item : sequence of float
            A point in physical space.

        Returns
        -------
        bool
            True if the point lies within the physical bounding box of the grid.
        """
        item = np.asarray(item)
        return bool(np.all(self.bbox[0] <= item) and np.all(item <= self.bbox[1]))

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """
        Abstract equality operator. Subclasses should implement
        a thorough attribute-wise comparison.
        """
        pass

    def __iter__(self):
        """
        Iterate over all grid indices (excluding ghost zones).

        Yields
        ------
        tuple of int
            Index tuple into the grid.
        """
        return iter(np.ndindex(*self.shape))

    # @@ Utility Functions @@ #
    # Utility functions which are included in the grid base class in order to
    # improve readability. These can be added to and modified as necessary to
    # produce the desired behavior.
    def __check_chunking__(self):
        """
        Raises an error if chunking is not enabled on this grid.

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

    def __get_axes_count_and_mask__(
        self, axes: List[str] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Return the number of selected axes and a boolean mask array indicating which axes are selected.

        Parameters
        ----------
        axes : list of str, optional
            List of axis names. If None, all axes are selected.

        Returns
        -------
        int
            Number of selected axes.
        np.ndarray
            Boolean mask indicating which axes are selected.
        """
        if axes is None:
            axes = self.axes
        return len(axes), self.coordinate_system.build_axes_mask(axes)

    def validate_chunks(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        axes: Optional[List[str]] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        """
        Validate and normalize a chunk specification along given axes.

        Parameters
        ----------
        chunks : sequence of int, tuple, or slice
            Chunk index specifications for each axis. Each element can be:
            - int: a single chunk index (e.g., 1)
            - tuple: a slice-like triple (start, stop, step)
            - slice: a Python slice object
        axes : list of str, optional
            If provided, limits the specification to the subset of axes listed.
            If None, all axes are considered.

        Returns
        -------
        tuple of tuples
            Normalized chunk specification of form (start, stop, step) for each axis.

        Raises
        ------
        ValueError
            If the number of chunks specified does not match the number of axes.
        TypeError
            If any chunk specifier is not an int, tuple, or slice.
        IndexError
            If a chunk range is out of bounds, negative, or invalid in structure.
        """
        # Check that we even permit chunking in this case.
        self.__check_chunking__()

        # Create the axes mask and count the number of axes in the axes.
        # This is used later to ensure we cut different domain properties correctly.
        naxes, axes_mask = self.__get_axes_count_and_mask__(axes=axes)
        if len(chunks) != naxes:
            raise ValueError(
                f"Chunk specifier {chunks} contains {len(chunks)} axes, "
                f"but {naxes} were expected."
            )

        # Iterate through each of the chunks and perform the validation checks
        # and the retyping.
        _validated_chunks = []
        for chunk, axis_index in zip(chunks, np.arange(self.ndim)[axes_mask]):
            # Normalize to (start, stop, step)
            if isinstance(chunk, int):
                cstart, cstop = chunk, chunk + 1
            elif isinstance(chunk, tuple):
                if len(chunk) != 2:
                    raise TypeError(
                        f"Chunk tuple {chunk} must be of length 2 (start, stop)."
                    )
                cstart, cstop = chunk
            elif isinstance(chunk, slice):
                if slice.step not in [None, 1]:
                    raise TypeError(f"Chunk slice {chunk} must be of step 1.")

                cstart = chunk.start if chunk.start is not None else 0
                cstop = (
                    chunk.stop if chunk.stop is not None else self.__cdd__[axis_index]
                )
            else:
                raise TypeError(
                    f"Invalid chunk type for axis {axis_index}: {type(chunk)}"
                )

            # Validate bounds and structure
            if cstart < 0:
                raise IndexError(
                    f"Chunk start ({cstart}) cannot be negative for axis {axis_index}."
                )
            if cstop < cstart:
                raise IndexError(
                    f"Chunk stop ({cstop}) cannot be less than start ({cstart}) for axis {axis_index}."
                )
            if cstop > self.__cdd__[axis_index]:
                raise IndexError(
                    f"Chunk stop ({cstop}) exceeds available number of chunks "
                    f"({self.__cdd__[axis_index]}) on axis {axis_index}."
                )

            _validated_chunks.append((cstart, cstop))

        return tuple(_validated_chunks)

    def iter_chunks(
        self,
        axes: Optional[List[str]] = None,
        use_tqdm: bool = False,
        **tqdm_kwargs,
    ) -> Iterator[Tuple[int, ...]]:
        """
        Create an iterator over all the chunk index tuples in the grid.

        Parameters
        ----------
        axes : list of str, optional
            List of axes to iterate over. If None, all axes are used.
        use_tqdm : bool, default=False
            Whether to wrap the iterator in a tqdm progress bar.
        **tqdm_kwargs : dict
            Additional keyword arguments passed to tqdm (e.g., desc="Sweeping chunks").

        Yields
        ------
        tuple of int
            Chunk index along the selected axes. For example, (0, 1, 2) for a 3D chunked grid.

        Raises
        ------
        RuntimeError
            If chunking is not enabled on the grid.
        """
        if not self.chunking:
            raise RuntimeError("Chunking is not enabled for this grid.")

        axes = axes if axes is not None else self.axes
        axes_indices = [self.coordinate_system.axes_string_to_index(ax) for ax in axes]
        chunk_counts = [self.__cdd__[i] for i in axes_indices]

        iterator = np.ndindex(*chunk_counts)
        if use_tqdm:
            total = np.prod(chunk_counts)
            iterator = tqdm(iterator, total=total, **tqdm_kwargs)

        yield from iterator

    # @@ Core Functionality @@ #
    # These methods are required to be implemented in all subclasses
    # and should maintain the same core functionality across.
    def get_slice_from_chunks(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        axes: Optional[List[str]] = None,
        include_ghosts: bool = False,
    ) -> Tuple[slice, ...]:
        """
        Convert a chunk specification to a tuple of slices in data space.

        This method is used to extract a portion of the data array corresponding to
        specific chunks along each axis, optionally including ghost cells.

        Parameters
        ----------
        chunks : sequence of int, tuple, or slice
            Specification of chunks for each axis. Each entry can be:

            - int: selects a single chunk
            - tuple: (start, stop) chunk indices
            - slice: standard slice object (start, stop)

        axes : list of str, optional
            List of axis names (e.g., ``['x', 'y']``) to restrict the chunking to.
            If ``None``, applies to all axes in order.
        include_ghosts : bool, default=``False``
            If ``True``, extends the resulting slices to include ghost zones around
            the data chunks (if not on the domain boundary).

        Returns
        -------
        tuple of slice
            Slices corresponding to the data region in ghost-augmented storage.

        Raises
        ------
        ValueError, IndexError
            If the chunk specification is invalid.
        """
        naxes, axes_mask = self.__get_axes_count_and_mask__(axes=axes)
        chunks = self.validate_chunks(chunks, axes=axes)

        slices = []
        for (chunk_start, chunk_stop), axis_id in zip(
            chunks, np.arange(self.ndim)[axes_mask]
        ):
            # Get the size of each chunk in this axis
            chunk_size = self.__chunk_size__[axis_id]

            # Convert chunk indices to data space
            start_idx = chunk_start * chunk_size
            stop_idx = chunk_stop * chunk_size

            if include_ghosts:
                # Extend the slice based on ghost zones — but only if we're not on a boundary
                ghost_start = (
                    self.__ghost_zones__[0, axis_id] if chunk_start != 0 else 0
                )
                ghost_end = (
                    self.__ghost_zones__[1, axis_id]
                    if chunk_stop != self.__cdd__[axis_id]
                    else 0
                )
                start_idx += ghost_start
                stop_idx += ghost_start + ghost_end
            else:
                # Shift forward by left ghost zone offset to align to data
                start_idx += self.__ghost_zones__[0, axis_id]
                stop_idx += self.__ghost_zones__[0, axis_id]

            slices.append(slice(start_idx, stop_idx))

        return tuple(slices)

    def get_coordinate_grid(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]] = None,
        axes: Sequence[str] = None,
        include_ghosts: bool = False,
    ) -> List[np.ndarray]:
        """
        Return a full coordinate grid (meshgrid-style) over the specified chunk region and axes.

        This method computes a broadcasted meshgrid of physical coordinates across the domain
        or a subset of the domain, suitable for vectorized evaluation of scalar or tensor fields.

        Parameters
        ----------
        chunks : tuple of int, (int, int), or slice, optional
            Specification of which chunks to retrieve along each axis. Can be:

            - int: selects a single chunk
            - tuple: (start, stop) chunk index range
            - slice: a Python slice object

            If None, the full domain is returned (or all chunks, if chunking is enabled).

        axes : sequence of str, optional
            The axes for which to construct the coordinate grid. If None, all axes are used.

        include_ghosts : bool, default=False
            Whether to include ghost zones in the returned coordinate arrays.

        Returns
        -------
        tuple of np.ndarray
            A tuple of coordinate arrays broadcasted to a common shape via np.meshgrid(..., indexing="ij").
            Each array corresponds to one axis in `axes`.

        Raises
        ------
        ValueError
            If the chunk specification is invalid or does not match the axes provided.

        Notes
        -----
        This is really just an alias for :py:func:`get_coordinates` which are then pushed through :py:func:`numpy.meshgrid`.
        """
        return np.meshgrid(
            *self.get_coordinate_arrays(
                chunks=chunks, axes=axes, include_ghosts=include_ghosts
            ),
            indexing="ij",
        )

    def evalf(
        self,
        func: Callable,
        /,
        axes: Sequence[str] = None,
        fixed_axes: Dict[str, float] = None,
        include_ghosts: bool = False,
        chunks: Sequence[Union[int, Tuple[int, int], slice]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Evaluate a function over the grid’s coordinate mesh.

        This method evaluates a scalar- or tensor-valued function over the specified region
        of the grid and returns the resulting array.

        Parameters
        ----------
        func : callable
            A function accepting coordinate arrays (e.g., ``func(x, y)`` or ``func(r, theta, phi)``),
            broadcasted via :py:func:`numpy.meshgrid`, and returning a scalar or array value. By default,
            all of the axes are passed to ``func``. To evaluate ``func`` over only specific axes, specify the
            ``axes`` argument.

            .. note::

                By default, if ``axes`` has (for example) 2 axes, then the function should take 2 arguments. If it
                takes more arguments (corresponding to the other axes not in ``axes``), the values for those arguments
                should be specified in ``fixed_axes``.

            The function must take the axes arrays in order.

        axes : list of str, optional
            The axes over which to evaluate the function. If None, all axes are used.
        fixed_axes : dict of str to float, optional
            If `axes` is provided, axes listed in `fixed_axes` are passed to the function as scalars.
            These axes must not overlap with `axes`.
        include_ghosts : bool, default=False
            Whether to include ghost zones in the coordinate arrays.
        chunks : sequence of int, tuple, or slice, optional
            If provided, restricts evaluation to the specified chunked region.
        **kwargs : dict
            Additional keyword arguments passed to the function being evaluated.

        Returns
        -------
        np.ndarray
            The evaluated result of `func` broadcast across the grid's mesh.
        """
        all_axes = self.axes
        fixed_axes = fixed_axes or {}

        if axes is None:
            if fixed_axes:
                raise ValueError(
                    "Cannot specify fixed_axes when evaluating over all axes."
                )
            axes = all_axes
        else:
            invalid_axes = [ax for ax in axes if ax not in all_axes]
            if invalid_axes:
                raise ValueError(
                    f"Invalid axes: {invalid_axes}. Valid axes: {all_axes}"
                )

            invalid_fixed = [ax for ax in fixed_axes if ax not in all_axes]
            if invalid_fixed:
                raise ValueError(
                    f"Invalid fixed_axes: {invalid_fixed}. Valid axes: {all_axes}"
                )

            overlapping = set(axes) & set(fixed_axes)
            if overlapping:
                raise ValueError(
                    f"Axes {overlapping} cannot appear in both `axes` and `fixed_axes`."
                )

        # Get coordinate mesh for selected axes
        mesh = self.get_coordinate_grid(
            chunks=chunks,
            axes=axes,
            include_ghosts=include_ghosts,
        )

        # Build function arguments in the order of all_axes
        coord_args = []
        mesh_index = 0
        for ax in all_axes:
            if ax in axes:
                coord_args.append(mesh[mesh_index])
                mesh_index += 1
            elif ax in fixed_axes:
                coord_args.append(fixed_axes[ax])

        # Evaluate and return
        return func(*coord_args, **kwargs)

    # @@ Abstract Functionality @@ #
    # These abstract methods MUST be implemented in all
    # subclasses.
    @abstractmethod
    def get_coordinates(
        self,
        index: Union[int, Tuple[int, ...]],
        axes: Sequence[str] = None,
        include_ghosts: bool = False,
    ) -> Sequence[float]:
        """
        Return the physical coordinates at a specific index in the grid.

        This method retrieves the coordinate values corresponding to a particular
        grid index (or flat index) along one or more axes.

        Parameters
        ----------
        index : int or tuple of int
            The index of the grid point. Can be:
            - int: flattened (1D) index into the full grid
            - tuple of int: multi-dimensional index (i.e., one index per axis)

        axes : sequence of str, optional
            List of axis names for which to return coordinates. If None, returns coordinates
            for all axes.

        include_ghosts : bool, default=False
            Whether to treat the index as including ghost zones. If True, the index is interpreted
            in the full grid including ghost cells.

        Returns
        -------
        sequence of float
            Coordinate values for the specified grid point along the selected axes.

        Raises
        ------
        IndexError
            If the index is out of bounds or not valid for the grid shape.
        ValueError
            If axes are invalid or inconsistent with the index dimensionality.
        """
        pass

    @abstractmethod
    def get_coordinate_arrays(
        self,
        chunks: Tuple[Union[int, Tuple[int, int], slice]] = None,
        axes: Sequence[str] = None,
        include_ghosts: bool = False,
    ) -> Tuple[np.ndarray]:
        """
        Return the individual coordinate arrays for the grid over the specified chunks and axes.

        This method computes or retrieves the physical coordinates associated with the
        grid points over a specified chunked region and subset of axes.

        Parameters
        ----------
        chunks : tuple of int, (int, int), or slice, optional
            Specification of which chunks to retrieve along each axis. Can be:

            - int: selects a single chunk
            - tuple: (start, stop) chunk index range
            - slice: a Python slice object

            If None, the full domain is returned (or all chunks, if chunking is enabled).

        axes : sequence of str, optional
            Names of axes for which coordinates should be returned. If None, coordinates
            for all axes are returned.

        include_ghosts : bool, default=False
            Whether to include ghost zones in the returned coordinate arrays. This is useful
            for stencil operations where field values near the boundary are required.

        Returns
        -------
        tuple of np.ndarray
            A tuple of coordinate arrays, one for each requested axis, with shape matching the
            spatial dimensions (and ghost cells if `include_ghosts=True`).

        Raises
        ------
        ValueError
            If the chunk specification is incompatible with the grid dimensions or axes.
        """
        pass

    @abstractmethod
    def extract_subgrid(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        axes: Optional[Sequence[str]] = None,
        include_ghosts: bool = True,
    ) -> "GenericGrid":
        """
        Extract a subgrid (subdomain) from this grid based on chunk specification.

        Parameters
        ----------
        chunks : sequence of int, tuple, or slice
            Specification of the region to extract in chunk units. Each element can be:
            - int: selects a single chunk
            - tuple: (start, stop) range of chunks
            - slice: standard slice object

        axes : list of str, optional
            The axes to which the chunk specification applies. If None, applies to all axes.

        include_ghosts : bool, default=True
            Whether to include ghost cells in the extracted subgrid.

        Returns
        -------
        GenericGrid
            A new subgrid over the specified chunked region.
        """
        pass

    # @@ IO Methods @@ #
    @abstractmethod
    def to_hdf5(self, filename: str, group_name: Optional[str] = None):
        """
        Save the grid to an HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the output HDF5 file.
        group_name : str, optional
            The name of the group in which to store the grid data. If None, stores at the root.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def from_hdf5(self, filename: str, group_name: Optional[str] = None):
        """
        Load the grid from an HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file.
        group_name : str, optional
            The name of the group from which to load the grid. If None, loads from the root.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        pass


class GenericGrid(_BaseGrid):
    r"""
    Generic coordinate grid with arbitrary, non-uniform spacing.

    The :py:class:`GenericGrid` class represents a general-purpose structured grid in which
    the coordinate values along each axis are explicitly specified via user-provided
    1D arrays. This allows for arbitrarily non-uniform grids, such as those used in
    mapped or curvilinear coordinate systems.

    When initializing the class, the ``coordinates`` argument takes a set of :math:`N` arrays of increasing
    coordinate values :math:`x_i^k`. The grid is then constructed as the Cartesian product of these :math:`N` arrays:

    .. math::

        G_{ijk...} = (x_1^i,x_2^j,\ldots)

    .. important::

        The :py:class:`GenericGrid` is not the most memory efficient grid structure, but still performs much better than
        storing an entire grid in memory. For :math:`N` axes with :math:`n_i` coordinates in each axis, the :py:class:`GenericGrid`
        stores :math:`N` 1D arrays with each of the coordinates, for a total of

        .. math::

            S = \sum_i n_i

        total floats. The grid itself has

        .. math::

            S_{\rm grid} = N \prod_i n_i.

    Notes
    -----

    - The coordinate arrays provided during initialization must be strictly increasing
      and one-dimensional.
    - Chunking is optional but can be useful for parallel processing or out-of-core computations.
    - This class supports serialization to and from HDF5 using :py:meth:`to_hdf5` and  :py:meth:`from_hdf5`.
    """

    # @@ Initialization Procedures @@ #
    # These initialization procedures may be overwritten in subclasses
    # to specialize the behavior of the grid. Ensure that all the REQUIRED
    # attributes are set in the initialization methods, otherwise unintended behavior
    # may arise.
    def __set_grid_domain__(self, *args, **kwargs):
        """
        Configure the shape and physical bounding box of the domain.

        This method is responsible for defining:

        - ``self.__bbox__``: The physical bounding box of the domain (without ghost cells). This should be a
           valid ``BoundingBox`` instance with shape (2,NDIM) defining first the bottom left corner of the domain
           and then the top right corner of the domain.
        - ``self.__dd__``: The DomainDimensions object defining the grid shape (without ghost cells). This should be
           a valid ``DomainDimensions`` instance with shape ``(NDIM,)``.
        - ``self.__chunking__``: boolean flag indicating if chunking is allowed.
        - ``self.__chunk_size__``: The DomainDimensions for a single chunk of the grid.

        Should be overridden in subclasses to support specific grid shape logic.
        """
        # Validate and define the arrays for the grid. They need to match the
        # dimensions of the coordinate system and they need to be increasing.
        _coordinates_ = args[0]
        if len(_coordinates_) != self.__cs__.ndim:
            raise GridInitializationError(
                f"Coordinate system {self.__cs__} has {self.__cs__.ndim} dimensions but only {len(_coordinates_)} were "
                "provided."
            )
        self.__coordinate_arrays__ = tuple(
            _coordinates_
        )  # Ensure each array is 1D and strictly increasing
        for i, arr in enumerate(_coordinates_):
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise GridInitializationError(
                    f"Coordinate array for axis {i} must be 1-dimensional."
                )
            if not np.all(np.diff(arr) > 0):
                raise GridInitializationError(
                    f"Coordinate array for axis {i} must be strictly increasing."
                )

        self.__coordinate_arrays__ = tuple(np.asarray(arr) for arr in _coordinates_)

        # Now use the coordinate arrays to compute the bounding box. This requires calling out
        # to the ghost_zones a little bit early and validating them. The domain dimensions are computed
        # from the length of each of the coordinate arrays.
        _ghost_zones = kwargs.get("ghost_zones", None)
        _ghost_zones = (
            np.array(_ghost_zones, dtype=int)
            if _ghost_zones is not None
            else np.zeros((2, self.ndim), dtype=int)
        )
        if _ghost_zones.shape == (self.ndim, 2):
            _ghost_zones = np.moveaxis(_ghost_zones, 0, -1)
            self.__ghost_zones__ = _ghost_zones
        if _ghost_zones.shape == (2, self.ndim):
            self.__ghost_zones__ = _ghost_zones
        else:
            raise ValueError(
                f"`ghost_zones` is not a valid shape. Expected (2,{self.ndim}), got {_ghost_zones.shape}."
            )

        # With the ghost zones set up, we are now in a position to correctly manage the
        # bounding box and the domain dimensions.
        _ghost_zones_per_axis = np.sum(self.__ghost_zones__, axis=0)
        self.__bbox__ = BoundingBox(
            [
                [
                    self.__coordinate_arrays__[_idim][_ghost_zones[0, _idim]],
                    self.__coordinate_arrays__[_idim][-(_ghost_zones[1, _idim] + 1)],
                ]
                for _idim in range(self.ndim)
            ]
        )
        self.__dd__ = DomainDimensions(
            [
                self.__coordinate_arrays__[_idim].size - _ghost_zones_per_axis[_idim]
                for _idim in range(self.ndim)
            ]
        )

        # Manage chunking behaviors. This needs to ensure that the chunk size is set,
        # figure out if chunking is even enabled, and then additionally determine if the
        # chunks equally divide the shape of the domain (after ghost zones!).
        _chunk_size_ = kwargs.get("chunk_size", None)
        if _chunk_size_ is None:
            self.__chunking__ = False
        else:
            # Validate the chunking.
            _chunk_size_ = np.asarray(_chunk_size_).ravel()
            if len(_chunk_size_) != self.ndim:
                raise ValueError(
                    f"'chunk_size' had {len(_chunk_size_)} dimensions but grid was {self.ndim} dimensions."
                )

            elif ~np.all(self.__dd__ % _chunk_size_ == 0):
                raise ValueError(
                    f"'chunk_size' ({_chunk_size_}) must equally divide the grid (shape = {self.dd})."
                )

            self.__chunking__: bool = True

        if self.__chunking__:
            self.__chunk_size__: Optional[DomainDimensions] = DomainDimensions(
                _chunk_size_
            )
            self.__cdd__: Optional[DomainDimensions] = self.dd // self.__chunk_size__

    def __set_grid_boundary__(self, *args, **kwargs):
        """
        Configure boundary-related attributes for the grid.

        This includes:

        - ``self.__ghost_zones__``: Number of ghost cells on each side (2, ndim).
        - ``self.__ghost_bbox__``: Bounding box including ghost regions.
        - ``self.__ghost_dd__``: DomainDimensions object including ghost cells.

        This is where boundary conditions (periodic, Dirichlet, etc.) and ghost cell layout
        should be resolved.

        Should be overridden in subclasses to implement behavior.
        """
        # Ghost zones is already set, so that simplifies things a little bit. We now need to
        # simply set the __ghost_dd__ and the __ghost_bbox__. These are actually the "natural" bbox and
        # ddims given how the grid was specified.
        self.__ghost_bbox__ = BoundingBox(
            [
                [
                    self.__coordinate_arrays__[_idim][0],
                    self.__coordinate_arrays__[_idim][-1],
                ]
                for _idim in range(self.ndim)
            ]
        )
        self.__ghost_dd__ = DomainDimensions(
            [self.__coordinate_arrays__[_idim].size for _idim in range(self.ndim)]
        )

    def __init__(
        self,
        coordinate_system: "_CoordinateSystemBase",
        coordinates: Sequence[np.ndarray],
        /,
        ghost_zones: Optional[Sequence[Sequence[float]]] = None,
        chunk_size: Optional[Sequence[int]] = None,
        *args,
        **kwargs,
    ):
        args = [coordinates, *args]
        kwargs = {"ghost_zones": ghost_zones, "chunk_size": chunk_size, **kwargs}
        super().__init__(coordinate_system, *args, **kwargs)

    # @@ Dunder Methods @@ #
    # These should not be altered in any subclasses to
    # ensure behavioral consistency.
    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another _GenericGrid instance.

        Two grids are considered equal if:
        - They are instances of the same class
        - They use the same coordinate system
        - Their coordinate arrays are equal
        - Their ghost zones are the same
        - Their chunking flags and chunk sizes (if enabled) match

        Parameters
        ----------
        other : Any
            The object to compare with.

        Returns
        -------
        bool
            True if the grids are equivalent, False otherwise.
        """
        if not isinstance(other, GenericGrid):
            return NotImplemented

        if self.coordinate_system != other.coordinate_system:
            return False

        if len(self.__coordinate_arrays__) != len(other.__coordinate_arrays__):
            return False

        for a1, a2 in zip(self.__coordinate_arrays__, other.__coordinate_arrays__):
            if not np.array_equal(a1, a2):
                return False

        if not np.array_equal(self.ghost_zones, other.ghost_zones):
            return False

        if self.chunking != other.chunking:
            return False

        if self.chunking and (self.chunk_size != other.chunk_size):
            return False

        return True

    # @@ Core Functionality @@ #
    # These methods are required to be implemented in all subclasses
    # and should maintain the same core functionality across.
    def get_coordinates(
        self,
        index: Union[int, Tuple[int, ...]],
        axes: Sequence[str] = None,
        include_ghosts: bool = False,
    ) -> Sequence[float]:
        """
        Return the physical coordinates at a specific index in the grid.

        This method retrieves the coordinate values corresponding to a particular
        grid index (or flat index) along one or more axes.

        Parameters
        ----------
        index : int or tuple of int
            The index of the grid point. Can be:
            - int: flattened (1D) index into the full grid
            - tuple of int: multi-dimensional index (i.e., one index per axis)

        axes : sequence of str, optional
            List of axis names for which to return coordinates. If None, returns coordinates
            for all axes.

        include_ghosts : bool, default=False
            Whether to treat the index as including ghost zones. If True, the index is interpreted
            in the full grid including ghost cells.

        Returns
        -------
        sequence of float
            Coordinate values for the specified grid point along the selected axes.

        Raises
        ------
        IndexError
            If the index is out of bounds or not valid for the grid shape.
        ValueError
            If axes are invalid or inconsistent with the index dimensionality.
        """
        # Coordinate the axes input so that we know what slice of the
        # grid is relevant.
        axes = axes if axes is not None else self.axes
        axes_indices = [self.coordinate_system.axes_string_to_index(ax) for ax in axes]

        # check that the number of indices actually matches the number of axes we have
        # present. If not, we need to raise an error.
        if not hasattr(index, "__len__"):
            index = (index,)
        if len(axes) != len(index):
            raise ValueError(
                f"Index has length {len(index)}, but only {len(axes)} axes were specified."
            )

        # Convert the indices as needed to accommodate the ghost zones.
        if not include_ghosts:
            index = [
                idx + self.__ghost_zones__[0, axi]
                for idx, axi in zip(index, axes_indices)
            ]

        try:
            return [
                self.__coordinate_arrays__[axi][idx]
                for axi, idx in zip(axes_indices, index)
            ]
        except IndexError:
            raise ValueError(f"Index out of range: {index}")

    def get_coordinate_arrays(
        self,
        chunks: Tuple[Union[int, Tuple[int, int], slice], ...] = None,
        axes: Sequence[str] = None,
        include_ghosts: bool = False,
    ) -> Tuple[Any, ...]:
        """
        Return the individual coordinate arrays for the grid over the specified chunks and axes.

        This method computes or retrieves the physical coordinates associated with the
        grid points over a specified chunked region and subset of axes.

        Parameters
        ----------
        chunks : tuple of int, (int, int), or slice, optional
            Specification of which chunks to retrieve along each axis. Can be:

            - int: selects a single chunk
            - tuple: (start, stop) chunk index range
            - slice: a Python slice object

            If None, the full domain is returned (or all chunks, if chunking is enabled).

        axes : sequence of str, optional
            Names of axes for which coordinates should be returned. If None, coordinates
            for all axes are returned.

        include_ghosts : bool, default=False
            Whether to include ghost zones in the returned coordinate arrays. This is useful
            for stencil operations where field values near the boundary are required.

        Returns
        -------
        tuple of np.ndarray
            A tuple of coordinate arrays, one for each requested axis, with shape matching the
            spatial dimensions (and ghost cells if `include_ghosts=True`).

        Raises
        ------
        ValueError
            If the chunk specification is incompatible with the grid dimensions or axes.
        """
        # Enforce validation standards on the axes and construct the indices so that
        # we can retrieve the correct set of coordinate arrays.
        axes = axes if axes is not None else self.axes
        axes_indices = [self.coordinate_system.axes_string_to_index(ax) for ax in axes]

        # Get the chunk slice for the chunks that are specified if chunks are asked for.
        # If the chunks aren't asked for, we can simply apply the ghost behavior.
        if chunks is None:
            # We aren't using chunks. If include_ghosts is true, then we still need to
            # shift things correctly.
            if include_ghosts:
                slices = [slice(0, self.dd[i] + 1) for i in axes_indices]
            else:
                slices = [
                    slice(
                        self.ghost_zones[0, i], self.dd[i] + 1 - self.ghost_zones[1, i]
                    )
                    for i in axes_indices
                ]
        else:
            slices = self.get_slice_from_chunks(
                chunks, axes=axes, include_ghosts=include_ghosts
            )

        # Collect the slices and return the arrays.
        return tuple(
            [self.__coordinate_arrays__[i][_s] for i, _s in zip(axes_indices, slices)]
        )

    def extract_subgrid(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        **kwargs,
    ) -> "GenericGrid":
        """
        Extract a subgrid (subdomain) from this grid based on chunk specification.

        Parameters
        ----------
        chunks : sequence of int, tuple, or slice
            Specification of the region to extract in chunk units. Each element can be:
            - int: selects a single chunk
            - tuple: (start, stop) range of chunks
            - slice: standard slice object

        Returns
        -------
        GenericGrid
            A new subgrid over the specified chunked region.
        """
        # Get slice ranges for data extraction
        slices = self.get_slice_from_chunks(chunks)
        new_coords = tuple(
            arr[sl] for arr, sl in zip(self.__coordinate_arrays__, slices)
        )

        # Construct the new grid
        return GenericGrid(self.coordinate_system, new_coords, **kwargs)

    # @@ IO Methods @@ #
    def to_hdf5(
        self, filename: str, group_name: Optional[str] = None, overwrite: bool = False
    ):
        """
        Save the grid to an HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the output HDF5 file.
        group_name : str, optional
            The name of the group in which to store the grid data. If None, data is stored at the root level.
        overwrite : bool, default=False
            Whether to overwrite existing data. If False, raises an error when attempting to overwrite.
        """
        import h5py

        # Ensure that the filename is a Path object and then check for existence and overwrite violations.
        # These are only relevant at this stage if a particular group has not yet been specified.
        filename = Path(filename)
        if filename.exists():
            # Check if there are overwrite issues and then delete it if it is
            # relevant to do so.
            if (not overwrite) and (group_name is None):
                # We can't overwrite and there is data. Raise an error.
                raise IOError(
                    f"File '{filename}' already exists and overwrite=False. "
                    "To store data in a specific group, provide `group_name`."
                )
            elif overwrite and group_name is None:
                # We are writing to the core dir and overwrite is true.
                # delete the entire file and rebuild it.
                filename.unlink()
                with h5py.File(filename, "w"):
                    pass
        else:
            # The file didn't already exist, we simply create it and then
            # let it close again so that we can reopen it in the next phase.
            with h5py.File(filename, "w"):
                pass

        # Now that the file has been opened at least once and looks clean, we can
        # proceed with the actual write process. This will involve first checking
        # if there are overwrite violations when ``group_name`` is actually specified. Then
        # we can proceed with actually writing the data.
        with h5py.File(filename, "a") as f:
            # Start checking for overwrite violations and the group information.
            if group_name is None:
                group = f
            else:
                # If the group exists, handle overwrite flag
                if group_name in f:
                    if overwrite:
                        del f[group_name]
                        group = f.create_group(group_name)
                    else:
                        raise IOError(
                            f"Group '{group_name}' already exists in '{filename}' and overwrite=False."
                        )
                else:
                    group = f.create_group(group_name)

            # The group has been identified and the overwrite status has
            # been handled. We now store the grid's metadata in the hdf5 file.
            # To load again, we need the coordinate arrays, the chunk size, and the
            # ghost zones. We write a bit more information that this in case anyone needs
            # to manually inspect the files and wants information.
            #
            # Write the basic metadata for the axes, chunking, chunk size, etc.
            group.attrs["axes"] = list(self.axes)
            group.attrs["chunking"] = self.chunking
            group.attrs["grid_shape"] = list(self.shape)
            group.attrs["bbox"] = np.asarray(self.bbox, dtype=float)
            group.attrs["coordinate_system"] = self.coordinate_system.__class__.__name__

            if self.chunking:
                group.attrs["chunk_size"] = self.chunk_size

            # Write out the number of ghost zones in the
            # grid structure.
            group.attrs["ghost_zones"] = np.asarray(self.ghost_zones, dtype=int)

            # Save the coordinate axes to the hdf5 file in datasets.
            try:
                coord_group = group.require_group("coordinates")
                for axis_name, arr in zip(self.axes, self.__coordinate_arrays__):
                    coord_group.create_dataset(axis_name, data=arr)
            except Exception as e:
                raise IOError(f"Failed to write the coordinate arrays to disk: {e}")

        # Finally, the coordinate system itself must now be saved to disk. We leave
        # the context manager to ensure that everything flushes correctly.
        if group_name is None:
            cgroup_name = "coord_systm"
        else:
            cgroup_name = group_name + "/coord_systm"
        self.coordinate_system.to_hdf5(filename, cgroup_name, overwrite=overwrite)

    @classmethod
    def from_hdf5(cls, filename: str, group_name: Optional[str] = None):
        import h5py

        # Ensure existence of the hdf5 file before attempting to
        # load data from it.
        filename = Path(filename)
        if not filename.exists():
            raise IOError(f"HDF5 file '{filename}' does not exist.")

        # Open the hdf5 file in read mode and start parsing the
        # data from it.
        with h5py.File(filename, "r") as f:
            # Navigate to the appropriate group
            group = f[group_name] if group_name else f

            # Load metadata attributes
            chunk_size = group.attrs.get("chunk_size", None)
            ghost_zones = group.attrs.get("ghost_zones", None)

            # Check that coordinate datasets are present
            if "coordinates" not in group:
                raise IOError(f"No 'coordinates' group found in '{group.name}'.")

            coord_group = group["coordinates"]
            coord_arrays = {key: coord_group[key][...] for key in coord_group.keys()}

            # Load the coordinate system object
            if group_name is None:
                cgroup_name = "coord_systm"
            else:
                cgroup_name = group_name + "/coord_systm"
            coordinate_system = _CoordinateSystemBase.from_hdf5(
                filename, group_name=cgroup_name
            )

            # Ensure that coordinates are ordered to match the coordinate system axes
            try:
                ordered_coord_arrays = [
                    coord_arrays[axis][...] for axis in coordinate_system.axes
                ]
            except KeyError as e:
                missing = e.args[0]
                raise IOError(
                    f"Missing coordinate array for axis '{missing}' required by coordinate system '{coordinate_system}'."
                ) from e

            print(ordered_coord_arrays)

        return cls(
            coordinate_system,
            ordered_coord_arrays,
            chunk_size=chunk_size,
            ghost_zones=ghost_zones,
        )
