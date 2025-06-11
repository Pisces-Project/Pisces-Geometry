"""
Base classes for PyMetric grid API.

This module contains the abstract base class :class:`GridBase`, which
provides to core API for grid manufacturing and interaction.
"""
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike

from pymetric.utilities.logging import pg_log

from .mixins import (
    DenseMathOpsMixin,
    GridChunkingMixin,
    GridIOMixin,
    GridPlotMixin,
    GridUtilsMixin,
)
from .utils import BoundingBox, DomainDimensions, GridInitializationError

if TYPE_CHECKING:
    from pymetric.coordinates.base import _CoordinateSystemBase


class GridBase(
    GridUtilsMixin,
    GridIOMixin,
    GridChunkingMixin,
    GridPlotMixin,
    DenseMathOpsMixin,
    ABC,
):
    """
    Generic coordinate grid base class from which all PyMetric grid subclasses are descended.

    This class serves as the foundational abstraction for all grid types used in PyMetric.
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

    # -------------------------------------- #
    # Class Initialization                   #
    # -------------------------------------- #
    # These initialization procedures may be overwritten in subclasses
    # to specialize the behavior of the grid. Ensure that all the REQUIRED
    # attributes are set in the initialization methods, otherwise unintended behavior
    # may arise.
    # noinspection PyTypeChecker
    def __configure_coordinate_system__(
        self, coordinate_system: "_CoordinateSystemBase", *_, **__
    ):
        """
        Assign the coordinate system to the grid.

        This method is called during grid initialization to associate a coordinate system
        with the grid instance. Subclasses may override this to enforce validation logic,
        such as checking whether the coordinate system is orthogonal, curvilinear, or matches
        expected dimensionality.

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system to assign to the grid. This must be an instance of a subclass
            of `CoordinateSystemBase`, such as `OrthogonalCoordinateSystem` or
            `CurvilinearCoordinateSystem`.

        *args, **kwargs
            Additional arguments are accepted for interface compatibility and may be used by subclasses.

        Raises
        ------
        GridInitializationError
            If the coordinate system is invalid or incompatible with the grid type.
        """
        self.__cs__: "_CoordinateSystemBase" = coordinate_system

    # noinspection PyTypeChecker
    def __configure_domain__(self, *args, **kwargs):
        """
        Configure the shape, resolution, and physical extent of the grid domain.

        This method is called during initialization to define the core geometric properties
        of the grid. Subclasses must implement this method to populate the following attributes:

        - ``self.__center__``: Whether the grid is cell centered or grid centered.
        - ``self.__bbox__``: A `(2, ndim)` array specifying the physical bounding box
          of the active domain (excluding ghost zones).
        - ``self.__dd__``: A `DomainDimensions` instance specifying the number of grid points
          along each axis (excluding ghost zones).
        - ``self.__chunking__``: A boolean flag indicating whether chunking is enabled.
        - ``self.__chunk_size__``: A `DomainDimensions` object specifying the size of each chunk.
        - ``self.__cdd__``: A `DomainDimensions` object giving the number of chunks per axis.

        If chunking is not enabled, the chunking-related attributes should be set to `False` or `None`.

        Parameters
        ----------
        *args, **kwargs
            Subclass-specific arguments such as coordinate arrays, shape, resolution, bounding box, etc.

        Raises
        ------
        GridInitializationError
            If the domain cannot be configured due to invalid shapes, resolution mismatch,
            or chunking inconsistencies.
        """
        self.__center__: Literal["vertex", "cell"] = None
        self.__bbox__: BoundingBox = None
        self.__dd__: DomainDimensions = None
        self.__chunking__: bool = False
        self.__chunk_size__: DomainDimensions = None
        self.__cdd__: DomainDimensions = None

    # noinspection PyTypeChecker
    def __configure_boundary__(self, *args, **kwargs):
        """
        Configure ghost zones and boundary-related metadata for the grid.

        This method is called during initialization to set up boundary padding for
        stencil operations, boundary conditions, and ghost cell management.

        Subclasses must populate the following attributes:

        - ``self.__ghost_zones__``: A `(2, ndim)` array specifying the number of ghost cells
          on the lower and upper sides of each axis.
        - ``self.__ghost_bbox__``: A `(2, ndim)` array defining the bounding box that includes
          ghost regions.
        - ``self.__ghost_dd__``: A `DomainDimensions` instance representing the shape of the grid
          including ghost cells.

        Parameters
        ----------
        *args, **kwargs
            Subclass-specific arguments used to configure boundary padding (e.g., ghost zone sizes).

        Notes
        -----
        This method is typically called after the domain is configured, so that ghost cells
        can be appended to a valid domain geometry.

        Raises
        ------
        GridInitializationError
            If ghost zone configuration fails due to shape mismatch or invalid layout.
        """
        self.__ghost_zones__ = None
        self.__ghost_bbox__ = None
        self.__ghost_dd__ = None

    def __init__(
        self,
        coordinate_system: "_CoordinateSystemBase",
        *args,
        **kwargs,
    ):
        """
        Initialize a :py:class:`GridBase` instance.

        This constructor sets up the core infrastructure for a computational grid by
        configuring the coordinate system, domain geometry, and boundary/ghost zone settings.
        All user-facing grid types in Pisces Geometry (e.g., :py:class:`GenericGrid`)
        should subclass :py:class:`GridBase` and implement the required logic in the appropriate
        initialization hooks.

        Parameters
        ----------
        coordinate_system : ~coordinates.core.CurvilinearCoordinateSystem
            A coordinate system instance (from :py:mod:`coordinates`)
            which defines the dimensionality, axis names, and differential geometry used
            by the grid. The coordinate system is assigned to the grid and used to validate
            all subsequent logic and structure.

        *args:
            Positional arguments passed to subclass initialization hooks
            (e.g., ``__set_grid_domain__``, ``__set_grid_boundary__``). These may include
            axis-specific metadata like coordinate arrays, resolutions, or chunking information.

        **kwargs:
            Keyword arguments passed to subclass initialization hooks. May include configuration
            such as ghost zone specifications, chunk size, domain bounds, and others.

        Raises
        ------
        GridInitializationError
            Raised if any stage of the setup process fails, including coordinate system assignment,
            domain geometry, or boundary condition setup.

        Notes
        -----
        This method drives a three-stage initialization sequence by dispatching to the following hooks,
        each of which may be overridden by subclasses:

        1. ``__configure_coordinate_system__``:
            Assigns the grid's coordinate system. Subclasses may impose constraints
            (e.g., requiring orthogonality or specific dimensionality).

        2. ``__configure_domain__``:
            Defines the physical domain shape, bounding box, and optionally enables chunking.
            Subclasses must populate ``self.__bbox__``, ``self.__dd__``, and if chunking is enabled,
            ``self.__chunking__``, ``self.__chunk_size__``, and ``self.__cdd__``.

        3. ``__configure_boundary__``:
            Configures ghost cells and extended domain geometry. Subclasses must set
            ``self.__ghost_zones__``, ``self.__ghost_dd__``, and ``self.__ghost_bbox__``.

        After these steps, ``__post_init__`` is called, which may be optionally overridden by subclasses
        to finalize internal state.

        This class is abstract and cannot be instantiated directly. Concrete subclasses (e.g.,
        :py:class:`~pymetric.grids.GenericGrid`) must implement coordinate handling and
        domain-specific logic.
        """
        # Begin the initialization process by dispatching to
        # self.__set_coordinate_system__, which needs to configure the coordinate
        # system attribute self.__cs__.
        try:
            self.__configure_coordinate_system__(coordinate_system, *args, **kwargs)
        except Exception as e:
            raise GridInitializationError(
                f"Failed to set up coordinate system for grid: {e}"
            ) from e

        # Now configure the grid domain and the boundary via the
        # other initialization dispatches.
        try:
            self.__configure_domain__(*args, **kwargs)
        except Exception as e:
            raise GridInitializationError(
                f"Failed to set up domain for grid: {e}"
            ) from e

        try:
            self.__configure_boundary__(*args, **kwargs)
        except Exception as e:
            raise GridInitializationError(
                f"Failed to set up boundary for grid: {e}"
            ) from e

        # Setup the fill values.
        self.__fill_values__: Dict[str, float] = {
            axis: float(self.__bbox__[0, _i])
            for _i, axis in enumerate(self.__cs__.axes)
        }

        # Pass off to __post_init__ for any remaining initialization.
        self.__post_init__(*args, **kwargs)
        pg_log.debug("Initialized grid %s on coordinate system %s.", self, self.__cs__)

    def __post_init__(self, *args, **kwargs):
        """
        __post_init__ can be used to configure any additional aspects of the subclass after
        the rest of the __initialization__ procedure has been performed.
        """
        pass

    # -------------------------------------- #
    # Properties                             #
    # -------------------------------------- #
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
    def centering(self) -> Literal["cell", "vertex"]:
        """Returns whether the grid is cell- or vertex-centered."""
        return self.__center__

    @property
    def ncells(self) -> DomainDimensions:
        """
        Number of computational cells (excluding ghost zones).

        - For cell-centered grids: same as :attr:`dd`.
        - For vertex-centered grids: one fewer than the number of vertices.

        Returns
        -------
        DomainDimensions
            The number of active computational cells along each axis.
        """
        if self.__center__ == "cell":
            return self.dd
        else:
            return DomainDimensions([n - 1 for n in self.dd])

    @property
    def nvertices(self) -> DomainDimensions:
        """
        Number of grid vertices (excluding ghost zones).

        - For vertex-centered grids: same as :attr:`dd`.
        - For cell-centered grids: one more than the number of cells.

        Returns
        -------
        DomainDimensions
            The number of grid point vertices along each axis.
        """
        if self.__center__ == "vertex":
            return self.dd
        else:
            return DomainDimensions([n + 1 for n in self.dd])

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
        self._ensure_supports_chunking()
        return self.__chunk_size__

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
    def cdd(self) -> DomainDimensions:
        """
        The shape of a single chunk, expressed in grid points.

        Returns
        -------
         numpy.ndarray
            A tuple-like object specifying the number of grid points per axis.
        """
        self._ensure_supports_chunking()
        return self.__cdd__

    @property
    def fill_values(self) -> Dict[str, float]:
        """
        Dictionary of default fill values for each axis.

        The `fill_values` are used when evaluating coordinate-dependent functions
        with only a subset of axes specified. In such cases, any axes not explicitly
        provided are implicitly filled using the values defined in this dictionary.
        This allows consistent evaluation of expressions or coordinate mappings in
        reduced-dimensional contexts.

        For example, when slicing or projecting a higher-dimensional field along a
        subset of axes, `fill_values` provide default values for the omitted dimensions.

        By default, the ``fill_values`` are the lower corner of the bounding box.

        Returns
        -------
        dict of str, float
            A copy of the internal fill values dictionary, where each key is an axis name
            and each value is the corresponding default fill value used in partial evaluations.
        """
        return self.__fill_values__.copy()

    @fill_values.setter
    def fill_values(self, values: Dict[str, float]) -> None:
        """
        Set the fill values for each axis.

        Parameters
        ----------
        values : Dict[str, float]
            A dictionary mapping axis names to their fill values.

        Raises
        ------
        ValueError
            If any axis in `self.axes` is missing from the provided values.
        """
        missing_axes = [ax for ax in self.axes if ax not in values]
        if missing_axes:
            raise ValueError(f"Missing fill values for axes: {missing_axes}")

        self.__fill_values__ = values

    # -------------------------------------- #
    # Dunder Methods                         #
    # -------------------------------------- #
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
        return self.compute_coords_from_index(index)

    def __call__(
        self,
        chunks: Optional[Sequence[Union[int, Tuple[int, int], slice]]] = None,
        axes: Optional[Sequence[str]] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional[Union[int, ArrayLike]] = None,
        oob_behavior: Literal["raise", "clip"] = "raise",
    ) -> Tuple[np.ndarray, ...]:
        """
        Call shorthand for :meth:`get_coordinate_arrays`.

        Enables syntax like `grid()` or `grid(chunks=...)` to retrieve coordinate arrays.

        Parameters
        ----------
        chunks : sequence of int, tuple, or slice, optional
            Specification of which chunks to retrieve along each axis. If None, returns full domain.
        axes : sequence of str, optional
            Names of the axes to return. If None, all axes are returned.
        include_ghosts : bool, default=False
            Whether to include ghost zones in the returned coordinates.
        halo_offsets : int, sequence of int, or np.ndarray, optional
            Extra halo padding to apply, in ghost-cell units.
        oob_behavior : {"raise", "clip"}, default="raise"
            What to do if the requested region exceeds grid bounds.

        Returns
        -------
        tuple of np.ndarray
            One array per axis in `axes`, shaped appropriately for the selected region.
        """
        return self.compute_chunk_coords(
            chunks=chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

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

    def __eq__(self, other: Any) -> bool:
        """
        Abstract equality operator. Subclasses should implement
        a thorough attribute-wise comparison.
        """
        return self is other

    def __iter__(self):
        """
        Iterate over all grid indices (excluding ghost zones).

        Yields
        ------
        tuple of int
            Index tuple into the grid.
        """
        return iter(np.ndindex(*self.shape))

    # -------------------------------------- #
    # Coordinate Management                  #
    # -------------------------------------- #
    # These methods handle the construction / obtaining
    # of coordinates from different specifications. Some of
    # these methods will be abstract, others will be declarative.
    @abstractmethod
    def _compute_coordinates_on_slices(
        self, axis_indices: np.ndarray, slices: Sequence[slice]
    ) -> Tuple[np.ndarray, ...]:
        """
        Compute 1D coordinate arrays for the specified axis indices and slices.

        This method should return one 1D coordinate array per axis, corresponding
        to the physical coordinate values along the specified slices.

        Parameters
        ----------
        axis_indices : np.ndarray of int
            Indices of the grid axes to compute coordinates for (e.g., [0, 2]).
        slices : list of slice
            Python slice objects selecting the subset of the grid along each axis.

        Returns
        -------
        tuple of np.ndarray
            One 1D array per axis in axis_indices, giving physical coordinates
            along that axis over the specified slice.

        Notes
        -----
        This method does not need to handle meshgrid-style broadcasting.
        It is expected to return *1D arrays*, one per axis.
        """
        pass

    @abstractmethod
    def extract_subgrid(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        axes: Optional[Union[str, Sequence[str]]] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional[Union[int, ArrayLike]] = None,
        oob_behavior: str = "raise",
        **kwargs,
    ) -> "GridBase":
        """
        Extract a subgrid (i.e., subdomain) from the current grid using a chunk specification.

        This method returns a new grid object representing a subregion of the original domain,
        optionally including ghost zones and halo padding. The subgrid inherits coordinate
        properties and geometry from the parent grid.

        Parameters
        ----------
        chunks : sequence of int, tuple, or slice
            Specification of which chunks to extract along each axis. Each element can be:

            - ``int``: selects a single chunk
            - ``tuple of int``: ``(start, stop)`` range of chunk indices
            - ``slice``: standard Python ``slice`` object for chunk indexing

            The number of entries must match the number of axes selected via ``axes``,
            or the full grid dimensionality if ``axes`` is ``None``.

        axes : list of str, optional
            List of axis names to which the chunk specification applies. If ``None``,
            the chunk specification is applied to all axes in order.

        include_ghosts : bool, default=True
            Whether to include ghost zones at the edges of the domain. If ``True``,
            ghost cells will be added to the subgrid *only* on boundaries where the
            selected chunk touches the edge of the full grid.

        halo_offsets : int, sequence of int, or np.ndarray of shape ``(2, naxes)``, optional
            Additional padding to include around the extracted region, applied in ghost-index space:

            - If a scalar is provided, it applies symmetrically to all axes and sides.
            - If a 1D list or array is provided, it applies symmetrically to each axis.
            - If a 2D array of shape ``(2, naxes)`` is provided, it gives explicit
              left and right padding per axis.

            This padding is applied *in addition to* any ghost zones included via ``include_ghosts``.

        oob_behavior : {"raise", "clip"}, default="raise"
            Determines what to do when the requested region (including ghost zones and halos)
            exceeds the grid's physical extent:

            - ``"raise"``: Raises an :py:exc:`IndexError` if the resulting slice would go out of bounds.
            - ``"clip"``: Truncates the region to stay within the valid bounds of the ghost-augmented domain.

            .. note::

               The ``"ignore"`` option is not supported — subgrid slices must remain within the
               grid’s valid ghost-augmented space.

        kwargs : dict
            Additional keyword arguments passed through to the grid subclass’s
            :py:meth:`~GridBase.__init__` method when constructing the subgrid.

        Returns
        -------
        GridBase
            A new subgrid instance representing the selected region. This will be an instance
            of the same class as the original grid (e.g., :py:class:`~pymetric.grids.GenericGrid`).

        Raises
        ------
        IndexError
            If the region exceeds grid bounds and ``oob_behavior="raise"``.

        ValueError
            If inputs are invalid, malformed, or inconsistent with the grid axes.
        """
        pass

    # -------------------------------------- #
    # Grid IO Support                        #
    # -------------------------------------- #
    # support for grid IO is provided by the more generic
    # from_metadata and to_metadata methods implemented here for
    # each of the grid classes. These simply store the data necessary to
    # reconstruct the grid with the exception of the coordinate system.
    #
    # All of the IO support methods then simply process the metadata
    # to / from the respective file formats.
    # We also allow the underlying coordinate system to either be loaded
    # with the grid or separately, depending on the use case.
    @abstractmethod
    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Serialize the grid configuration to a metadata dictionary.

        This method captures all the static configuration information necessary
        to reconstruct the grid, including its shape, extents, spacing, axes, and any
        subclass-specific attributes. The coordinate system is *not* included and should
        be serialized separately, if needed.

        This metadata dictionary is format-agnostic and can be used for saving the grid
        to disk (e.g., as JSON/YAML) or for interprocess communication.

        Returns
        -------
        dict
            A dictionary containing all metadata required to reconstruct the grid using
            :meth:`from_metadata_dict`.

        Notes
        -----
        - This method should be **minimal** in the sense that it only serializes the necessary data
          to unambiguously reconstruct the grid.

        """
        pass

    @classmethod
    @abstractmethod
    def from_metadata_dict(
        cls, coordinate_system: "_CoordinateSystemBase", metadata_dict: Dict[str, Any]
    ) -> "GridBase":
        """
        Create a grid instance from a metadata dictionary.

        This class method reconstructs a grid object from the provided metadata,
        which should match the format produced by :meth:`to_metadata_dict`.

        Parameters
        ----------
        coordinate_system : ~coordinates.core.CurvilinearCoordinateSystem
            A coordinate system instance (from :py:mod:`coordinates`)
            which defines the dimensionality, axis names, and differential geometry used
            by the grid. The coordinate system is assigned to the grid and used to validate
            all subsequent logic and structure.
        metadata_dict : dict
            A dictionary containing grid metadata.

        Returns
        -------
        GridBase
            An instance of the grid class reconstructed from the metadata.
        """
        pass
