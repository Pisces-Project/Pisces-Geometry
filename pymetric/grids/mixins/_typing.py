"""
Mixin typing protocols for supporting mypy on grids.
"""
# mypy: ignore-errors
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike

from pymetric.grids.utils._typing import (
    AxesInput,
    BoundingBox,
    ChunkIndexInput,
    ChunkSizeInput,
    DomainDimensions,
    GhostZonesInput,
    HaloOffsetInput,
    IndexInput,
)

if TYPE_CHECKING:
    from pymetric.coordinates.base import _CoordinateSystemBase
    from pymetric.grids.base import GridBase


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsGridCore(Protocol):
    # @@ Grid Properties @@ #
    # Subclasses can (and often should) add additional properties; however,
    # existing properties should be consistent in behavior (return type, meaning, etc.) with
    # superclasses and sibling classes to ensure that the use experiences
    # are conserved.
    __cs__: "_CoordinateSystemBase"
    __bbox__: BoundingBox
    __dd__: DomainDimensions
    __chunking__: bool
    __chunk_size__: DomainDimensions
    __cdd__: DomainDimensions
    __ghost_zones__: np.ndarray
    __ghost_bbox__: BoundingBox
    __ghost_dd__: DomainDimensions
    __center__: Literal["vertex", "cell"]
    coordinate_system: "_CoordinateSystemBase"
    ndim: int
    axes: List[str]
    bbox: BoundingBox
    dd: DomainDimensions
    ncells: DomainDimensions
    nvertices: DomainDimensions
    centering: Literal["vertex", "cell"]
    shape: Sequence[int]
    gbbox: BoundingBox
    gdd: DomainDimensions
    ghost_zones: np.ndarray
    chunk_size: DomainDimensions
    chunking: bool
    cdd: DomainDimensions
    fill_values: Dict[str, float]

    # @@ Coordinate Management @@ #
    # These methods handle the construction / obtaining
    # of coordinates from different specifications. Some of
    # these methods will be abstract, others will be declarative.
    def _compute_coordinates_on_slices(
        self, axis_indices: np.ndarray, slices: Sequence[slice]
    ) -> Tuple[np.ndarray, ...]:
        ...

    def extract_subgrid(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        axes: Optional[Union[str, Sequence[str]]] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional[Union[int, ArrayLike]] = None,
        oob_behavior: str = "raise",
        **kwargs,
    ) -> "GridBase":
        ...

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
    def to_metadata_dict(self) -> Dict[str, Any]:
        ...

    @classmethod
    def from_metadata_dict(
        cls, coordinate_system: "_CoordinateSystemBase", metadata_dict: Dict[str, Any]
    ) -> "GridBase":
        ...

    # ================================ #
    # Input Coercion Methods           #
    # ================================ #
    # These methods provide support for coercing various
    # common inputs like chunks, axes, halo offsets, etc.
    def _standardize_chunk_size(
        self, chunk_size: Optional["ChunkSizeInput"]
    ) -> DomainDimensions:
        ...

    @staticmethod
    def _standardize_halo_offset(
        halo_offsets: Optional["HaloOffsetInput"], num_axes: int
    ) -> np.ndarray:
        ...

    @staticmethod
    def _standardize_ghost_zones(
        ghost_zones: Union["GhostZonesInput"], num_axes: int
    ) -> np.ndarray:
        ...

    def standardize_axes(self, axes: Optional["AxesInput"] = None) -> List[str]:
        ...

    def _standardize_index(
        self,
        index: "IndexInput",
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
    ) -> Tuple[int, ...]:
        ...

    def check_field_shape(
        self,
        array_shape: Sequence[int],
        axes: Optional["AxesInput"] = None,
        field_shape: Optional[Sequence[int]] = None,
    ) -> None:
        ...

    # ================================ #
    # Minimal Utilities                #
    # ================================ #
    # These are little methods that get used a lot in
    # various other methods.
    def _count_and_mask_axes(
        self, axes: Optional["AxesInput"] = None
    ) -> Tuple[int, np.ndarray]:
        ...

    def _adjust_slices_for_origin(
        self,
        slices: Union[slice, Sequence[slice]],
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
    ) -> List[slice]:
        ...

    @staticmethod
    def _centers_to_edges(
        coordinates: Sequence[np.ndarray],
        bbox: np.ndarray,
    ) -> List[np.ndarray]:
        ...

    @staticmethod
    def _edges_to_centers(
        coordinates: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        ...

    # =============================== #
    # Coordinates                     #
    # =============================== #
    # These methods handle the coordinate generation
    # procedures. The only method in the base class is the
    # `compute_coords_from_slices` abstract method. Everything
    # else utilizes that.
    def compute_coords_from_slices(
        self,
        slices: Union[slice, Sequence[slice]],
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        ...

    def compute_mesh_from_slices(
        self,
        slices: Union[slice, Sequence[slice]],
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
        **kwargs,
    ):
        ...

    def compute_coords_from_index(
        self,
        index: "IndexInput",
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ) -> Tuple[float, ...]:
        ...

    def compute_domain_coords(
        self,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        ...

    def compute_domain_mesh(
        self,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
        **kwargs,
    ):
        ...

    def compute_domain_edges(
        self,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        ...

    def compute_domain_centers(
        self,
        /,
        axes: Optional["AxesInput"] = None,
        origin: Literal["active", "global"] = "active",
        __validate__: bool = True,
    ):
        ...

    # =============================== #
    # Slicing                         #
    # =============================== #
    # These methods each handle various slicing procedures for
    # the grid. These methods do not include methods which
    # center chunking semantics as that is in its own mixin class.
    def compute_domain_slice(
        self,
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
    ) -> Tuple[slice, ...]:
        ...

    def determine_domain_shape(
        self,
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
    ) -> Tuple[int, ...]:
        ...

    def construct_domain_interpolator(
        self,
        field: np.ndarray,
        field_axes: Sequence[str],
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: Optional[float] = np.nan,
        **kwargs,
    ):
        ...

    # =============================== #
    # Casting                         #
    # =============================== #
    def broadcast_shape_to_axes(
        self,
        shape: Union[int, Sequence[int]],
        axes_in: "AxesInput",
        axes_out: "AxesInput",
    ) -> Tuple[int, ...]:
        ...

    def broadcast_array_to_axes(
        self, array: np.ndarray, axes_in: "AxesInput", axes_out: "AxesInput", **kwargs
    ) -> np.ndarray:
        ...

    def tile_array_to_axes(
        self,
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
        ...

    def empty(
        self,
        /,
        element_shape: Union[int, Sequence[int]] = (),
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
        **kwargs,
    ) -> np.ndarray:
        ...

    def zeros(
        self,
        /,
        element_shape: Union[int, Sequence[int]] = (),
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
        **kwargs,
    ) -> np.ndarray:
        ...

    def ones(
        self,
        /,
        element_shape: Union[int, Sequence[int]] = (),
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
        **kwargs,
    ) -> np.ndarray:
        ...

    def full(
        self,
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
        ...

    # =============================== #
    # Summary Methods                 #
    # =============================== #
    def summary(self) -> None:
        ...

    def show(self) -> None:
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsGridIO(_SupportsGridCore):
    def to_json(self, filepath: Union[str, Path], overwrite: bool = False):
        ...

    def to_yaml(self, filepath: Union[str, Path], overwrite: bool = False):
        ...

    def to_hdf5(
        self,
        filepath: Union[str, Path],
        group_name: Optional[str] = None,
        overwrite: bool = False,
    ):
        ...

    @classmethod
    def from_json(
        cls,
        filepath: Union[str, Path],
        coordinate_system: "_CoordinateSystemBase",
    ) -> "_SupportsGridIO":
        ...

    @classmethod
    def from_yaml(
        cls,
        filepath: Union[str, Path],
        coordinate_system: "_CoordinateSystemBase",
    ) -> "_SupportsGridIO":
        ...

    @classmethod
    def from_hdf5(
        cls,
        filepath: Union[str, Path],
        coordinate_system: "_CoordinateSystemBase",
        group_name: Optional[str] = None,
    ) -> "_SupportsGridIO":
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsGridChunking(_SupportsGridCore):
    # ================================ #
    # Input Coercion Methods           #
    # ================================ #
    # These methods provide support for coercing various
    # common inputs like chunks, axes, halo offsets, etc.
    def _ensure_supports_chunking(self):
        ...

    def _standardize_chunk_indices_type(
        self, chunks: "ChunkIndexInput", axes: Optional["AxesInput"] = None
    ) -> Tuple[Tuple[int, int], ...]:
        ...

    def _standardize_chunk_indices(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        ...

    # ================================ #
    # Casting                          #
    # ================================ #
    # These methods help with determining shapes
    # of chunks and constructing arrays matching those
    # chunks / chunk stencils.
    def get_chunk_shape(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: Literal["clip", "raise"] = "raise",
    ) -> Tuple[int, ...]:
        ...

    def empty_like_chunks(
        self,
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
        ...

    def zeros_like_chunks(
        self,
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
        ...

    def ones_like_chunks(
        self,
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
        ...

    def full_like_chunks(
        self,
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
        ...

    # ================================ #
    # Chunk Slicing Methods            #
    # ================================ #
    # These methods provide support for various slicing
    # needs when working in chunks.
    def _compute_chunk_slice_fast(
        self,
        start: np.ndarray,
        stop: np.ndarray,
        axes_indices: np.ndarray,
        include_ghosts: bool,
        halo_offsets: np.ndarray,
        oob_behavior: str,
    ) -> Tuple[slice, ...]:
        ...

    def _compute_chunk_slice_fast_scalar(
        self,
        start: int,
        axis_index: int,
        include_ghosts: bool,
        halo_offsets: np.ndarray,
        oob_behavior: str,
    ) -> slice:
        ...

    def compute_chunk_slice(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
    ) -> Tuple[slice, ...]:
        ...

    # =============================== #
    # Coordinates                     #
    # =============================== #
    # These methods handle the coordinate generation
    # procedures. The only method in the base class is the
    # `compute_coords_from_slices` abstract method. Everything
    # else utilizes that.
    def compute_chunk_coords(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
    ):
        ...

    def compute_chunk_mesh(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
        **kwargs,
    ):
        ...

    def compute_chunk_edges(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
    ) -> List[np.ndarray]:
        ...

    def compute_chunk_centers(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        __validate__: bool = True,
    ) -> List[np.ndarray]:
        ...

    def get_chunk_bbox(
        self,
        chunks: "ChunkIndexInput",
        axes: Optional["AxesInput"] = None,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
    ) -> BoundingBox:
        ...

    # ================================ #
    # Iterables                        #
    # ================================ #
    # These methods allow users to loop through
    # chunks in each grid.
    def iter_chunk_slices(
        self,
        /,
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        pbar: bool = True,
        pbar_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[slice, ...]]:
        ...

    def iter_chunk_indices(
        self,
        axes: Optional["AxesInput"] = None,
        *,
        pbar: bool = True,
        pbar_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[int, ...]]:
        ...

    def iter_chunk_coords(
        self,
        axes: Optional["AxesInput"] = None,
        *,
        include_ghosts: bool = False,
        halo_offsets: Optional["HaloOffsetInput"] = None,
        oob_behavior: str = "raise",
        pbar: bool = True,
        pbar_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsDenseGridMathOps(_SupportsGridChunking):
    """
    Mixin classes for supporting mathematical operations on grids.

    These wrap lower-level methods in the :mod:`differential_geometry` module and
    in other places to provide easy access at the level of grids with automated
    coordinate computations and chunking.
    """

    # ======================================= #
    # Utility Functions                       #
    # ======================================= #
    # These are utility methods that can be called during the
    # execution sequence for operational clarity.
    def _prepare_output_buffer(
        self,
        axes: Sequence[str],
        *,
        out: Optional[np.ndarray] = None,
        output_element_shape: Optional[Tuple[int, ...]] = (),
        **kwargs,
    ) -> ArrayLike:
        ...

    def _set_input_output_axes(self, field_axes, output_axes=None):
        ...

    def _compute_fixed_axes_and_values(
        self, free_axes: Sequence[str]
    ) -> Tuple[Sequence[str], dict]:
        """
        Compute the fixed axes and corresponding fill values for a set of free axes.

        This utility method identifies all axes not included in `free_axes` (i.e., the complement
        of the coordinate system's axes) and returns a dictionary mapping those fixed axes
        to their default fill values. This is useful when computing expressions that depend on
        a subset of the coordinate system's axes (e.g., chunked computations with broadcasting).

        Parameters
        ----------
        free_axes : Sequence[str]
            The list of logical axes over which the operation is being performed.

        Returns
        -------
        fixed_axes : Sequence[str]
            Axes not in `free_axes`. These are assumed to be fixed during computation.
        fixed_values : dict
            Dictionary mapping each fixed axis to its default value as specified in
            `self.fill_values`.

        Examples
        --------
        >>> fixed_axes, fixed_values = self.compute_fixed_axes_and_values(["x", "y"])
        >>> print(fixed_axes)      # ['z'] (for example)
        >>> print(fixed_values)    # {'z': 0.0}
        """
        fixed_axes = self.__cs__.axes_complement(free_axes)
        fixed_values = {k: v for k, v in self.fill_values.items() if k in fixed_axes}
        return fixed_axes, fixed_values

    # noinspection PyDefaultArgument
    def _make_expression_chunk_fetcher(
        self, expr_name: str, fixed_axes: dict, value: Optional[np.ndarray] = None
    ):
        ...

    # ======================================= #
    # Generic Methods                         #
    # ======================================= #
    # These are generic methods for various mathematics
    # computations.
    def dense_element_wise_partial_derivatives(
        self,
        field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...

    def compute_function_on_grid(
        self: "_SupportsDenseGridMathOps",
        func: Callable,
        /,
        result_shape: Optional[Sequence[int]] = None,
        out: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...

    # ======================================= #
    # Gradient Methods                        #
    # ======================================= #
    # These methods are used for computing the gradient
    # on grids.
    def dense_covariant_gradient(
        self,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...

    def dense_contravariant_gradient(
        self,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...

    def dense_gradient(
        self,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        basis: Optional[Literal["contravariant", "covariant"]] = "covariant",
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...

    # ======================================= #
    # Divergence Methods                      #
    # ======================================= #
    # These methods are used to compute the divergence of
    # a field.
    def dense_vector_divergence_contravariant(
        self,
        vector_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        Dterm_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...

    def dense_vector_divergence_covariant(
        self,
        vector_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        Dterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...

    # ======================================= #
    # Laplacian Methods                       #
    # ======================================= #
    # These methods are used to compute the divergence of
    # a field.
    def dense_scalar_laplacian(
        self,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        Lterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        second_derivative_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        ...
