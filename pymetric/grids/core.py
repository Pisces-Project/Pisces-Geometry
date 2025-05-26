"""
Structured grid classes for use in physical simulations and geometrical modeling.

This module provides the primary grid abstractions used throughout the Pisces Geometry library.
These grids define the structure of the computational domain and are used to evaluate differential
operators, store field data, and perform numerical computations.


Features
--------
- Ghost zone support for stencil-based operations and boundary condition handling.
- Chunking interface for domain decomposition and out-of-core or parallel computations.
- Support for serialization to/from HDF5, including coordinate system metadata.
- Interoperable with Pisces Geometry coordinate systems and field abstractions.

Notes
-----
This module is a core component of the Pisces Geometry ecosystem and is intended
to be subclassed or extended for use in PDE solvers, mesh-based numerical methods,
and physical field representations.
"""

from pathlib import Path
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
import unyt

from pymetric.grids.base import GridBase
from pymetric.grids.utils._exceptions import GridInitializationError
from pymetric.grids.utils._typing import BoundingBox, DomainDimensions

if TYPE_CHECKING:
    from pymetric.coordinates.base import _CoordinateSystemBase


class GenericGrid(GridBase):
    r"""
    Generic coordinate grid with arbitrary, non-uniform spacing.

    The :py:class:`GenericGrid` class represents a general-purpose structured grid in which
    the coordinate values along each axis are explicitly specified via user-provided
    1D arrays. This allows for arbitrarily non-uniform grids, such as those used in
    mapped or curvilinear coordinate systems.

    When initializing the class, the ``coordinates`` argument takes a set of
    :math:`N` arrays of increasing coordinate values :math:`x_i^k`. The grid is
    then constructed as the Cartesian product of these :math:`N` arrays:

    .. math::

        G_{ijk...} = (x_1^i,x_2^j,x_3^k,\ldots).


    Notes
    -----

    The provided ``coordinates`` are interpreted as either vertices or cell
    centers depending on the ``center=`` argument to ``__init__`.

    Examples
    --------

    Construct a uniformly spaced grid in Cartesian coordinates:

    .. code-block:: python

        >>> from pymetric.coordinates import CartesianCoordinateSystem3D
        >>> import numpy as np
        >>>
        >>> # Create the coordinate system.
        >>>
        >>> u = CartesianCoordinateSystem3D()
        >>>
        >>> # Construct the coordinates using
        >>> # NumPy arrays.
        >>> coords = [np.linspace(0, 1, 10), np.linspace(0, 1, 20), np.linspace(0, 1, 30)]
        >>>
        >>> # Now build the grid.
        >>> # In this case, 2 of the cells are assigned to ghost regions
        >>> # on each axis.
        >>> grid = GenericGrid(u, coords, ghost_zones=[[1, 1, 1],[1, 1, 1]])
        >>>
        >>> # Inspect the bounding box and the ghost bounding box.
        >>> print(grid.bbox)
        [[0.11111111 0.05263158 0.03448276]
         [0.88888889 0.94736842 0.96551724]]
        >>> print(grid.gbbox)
        [[0. 0. 0.]
         [1. 1. 1.]]
        >>>
        >>> # Inspect the number of cells in the
        >>> # grid (ghost and no ghost).
        >>> print(grid.dd)
        [ 8 18 28]
        >>> print(grid.gdd)
        [10 20 30]

    See Also
    --------
    :py:class:`UniformGrid` : Grid with uniform spacing and analytically defined coordinate structure.
    :py:class:`~fields.base.GenericField` : Field defined on a grid.
    """

    # @@ Initialization Procedures @@ #
    # These initialization procedures may be overwritten in subclasses
    # to specialize the behavior of the grid. Ensure that all the REQUIRED
    # attributes are set in the initialization methods, otherwise unintended behavior
    # may arise.
    def __configure_domain__(self, *args, **kwargs):
        """
        Configure the domain geometry of the grid.

        This method sets the grid’s internal shape, bounding box, and chunking structure.
        It validates the user-provided coordinate arrays and computes:

        - ``self.__bbox__``: Bounding box (without ghost zones)
        - ``self.__dd__``: Domain shape (excluding ghost zones)
        - ``self.__chunking__``: Whether chunking is enabled
        - ``self.__chunk_size__``: Size of each chunk, if chunking is enabled
        - ``self.__cdd__``: Number of chunks per axis, if chunking is enabled

        The standard behavior here is to extract the coordinate arrays provided by
        the user and the "center=" and "bbox=" kwargs.

        - If center='vertex', then we store the coordinates immediately in __coordinate_arrays__
          and use __coordinate_arrays__ to set the bounding box unless it is given to use.
        - If center='cell', then we use the bounding box (now required) to supplement the edges
          and then dump them to the coordinate arrays.
        """
        # Extract and validate the coordinate arrays from the arguments.
        # These are always the first element in args.
        coordinate_arrays = [np.asarray(_a) for _a in args[0]]

        if len(coordinate_arrays) != self.ndim:
            raise GridInitializationError(
                f"Expected {self.ndim} coordinate arrays for {self.ndim}-D system, "
                f"got {len(coordinate_arrays)}."
            )
        for i, arr in enumerate(coordinate_arrays):
            if arr.ndim != 1:
                raise GridInitializationError(f"Axis {i} coordinate array must be 1D.")
            if not np.all(np.diff(arr) > 0):
                raise GridInitializationError(
                    f"Axis {i} coordinate array must be strictly increasing."
                )

        # Configure the ghost zones based on the kwargs that have been passed
        # through __init__.
        self.__ghost_zones__ = self._standardize_ghost_zones(
            kwargs.get("ghost_zones", None), self.__cs__.ndim
        )
        total_ghosts = np.sum(self.__ghost_zones__, axis=0)

        # The coordinates have been validated. We need to now use
        # the cell centering convention and the bounding box to
        # build the cell edges.
        bbox, center = kwargs.pop("bbox", None), kwargs.pop("center", "cell")
        self.__center__: Literal["vertex", "cell"] = center
        self.__coordinate_arrays__ = tuple(np.asarray(arr) for arr in coordinate_arrays)

        # Configure the ghost bounding box (the global bounding box) by
        # encapsulating the centering scheme and the (possible) user provided bbox.
        if center == "vertex":
            # The ghost bbox get's set directly from the vertex edges
            # because that's how a vertex centered grid works.
            if bbox is not None:
                raise RuntimeWarning("Vertex centered GenericGrids cannot use `bbox=`.")

            self.__ghost_bbox__ = BoundingBox(
                [
                    [
                        self.__coordinate_arrays__[i][0],
                        self.__coordinate_arrays__[i][-1],
                    ]
                    for i in range(self.ndim)
                ]
            )
            self.__bbox__ = BoundingBox(
                [
                    [
                        self.__coordinate_arrays__[i][self.__ghost_zones__[0, i]],
                        self.__coordinate_arrays__[i][
                            -(self.__ghost_zones__[1, i] + 1)
                        ],
                    ]
                    for i in range(self.ndim)
                ]
            )

        elif center == "cell":
            # In this case, we require a custom bbox to have been provided.
            if bbox is None:
                raise ValueError(
                    "Cell centered GenericGrids require `bbox=` to be specified."
                )

            self.__ghost_bbox__ = BoundingBox(bbox)

            # To compute the physical bbox, we need to briefly handle
            # the edges.
            _coordinate_edges = [
                np.concatenate(
                    [
                        [self.__ghost_bbox__[0, i]],
                        0.5 * (ca[1:] + ca[:-1]),
                        [self.__ghost_bbox__[1, i]],
                    ]
                )
                for i, ca in enumerate(self.__coordinate_arrays__)
            ]
            self.__bbox__ = BoundingBox(
                np.asarray(
                    [
                        [
                            _coordinate_edges[i][self.__ghost_zones__[0, i]],
                            _coordinate_edges[i][-(self.__ghost_zones__[1, i] + 1)],
                        ]
                        for i in range(self.ndim)
                    ]
                ).T
            )
        else:
            raise ValueError(f"Invalid `center` argument value: {center}.")

        # Compute domain shape (excluding ghost zones)
        self.__ghost_dd__ = DomainDimensions(
            [arr.size for arr in self.__coordinate_arrays__]
        )
        self.__dd__ = DomainDimensions(
            [
                arr.size - total_ghosts[i]
                for i, arr in enumerate(self.__coordinate_arrays__)
            ]
        )

        # Chunking setup
        chunk_size = kwargs.get("chunk_size", None)
        if chunk_size is None:
            self.__chunking__ = False
        else:
            chunk_size = np.asarray(chunk_size).ravel()
            if len(chunk_size) != self.ndim:
                raise ValueError(
                    f"'chunk_size' must have {self.ndim} elements, got {len(chunk_size)}."
                )
            if not np.all(self.ncells % chunk_size == 0):
                raise ValueError(
                    f"'chunk_size' {chunk_size} must evenly divide the number of cells: {self.ncells}."
                )

            self.__chunking__ = True
            self.__chunk_size__ = DomainDimensions(chunk_size)
            self.__cdd__ = self.ncells // self.__chunk_size__

    def __configure_boundary__(self, *args, **kwargs):
        """
        Configure the ghost-region geometry of the grid.

        This is already completed at the __configure_domain__ level.
        """
        pass

    def __init__(
        self,
        coordinate_system: "_CoordinateSystemBase",
        coordinates: Sequence[np.ndarray],
        /,
        ghost_zones: Optional[Sequence[Sequence[float]]] = None,
        chunk_size: Optional[Sequence[int]] = None,
        *args,
        units: Optional[Dict[str, Union[str, unyt.Unit]]] = None,
        center: Literal["vertex", "cell"] = "vertex",
        bbox: Optional[BoundingBox] = None,
        **kwargs,
    ):
        r"""
        Construct a generic grid on a particular coordinate system.

        This method uses the `coordinates` and `coordinate_system` arguments
        to construct the coordinate grid.

        Parameters
        ----------
        coordinate_system : ~coordinates.core.CurvilinearCoordinateSystem
            An instance of a coordinate system subclass (e.g.,
            :py:class:`~coordinates.coordinate_systems.CartesianCoordinateSystem3D`,
            :py:class:`~coordinates.coordinate_systems.CylindricalCoordinateSystem`, or
            :py:class:`~coordinates.coordinate_systems.SphericalCoordinateSystem`),
            which defines the geometry and dimensionality of the grid. This object determines:

            - The number of spatial dimensions spanned by the :py:class:`GenericGrid` (:py:attr:`ndim`)
            - The axis names (:py:attr:`axes`. e.g., ``["x", "y", "z"]`` or ``["r", "theta", "phi"]``)
            - The behavior of differential operations such as gradient and divergence

            .. note::

                The number of coordinate arrays provided in ``coordinates`` must match the dimensionality
                defined by this coordinate system.

        coordinates : list of numpy.ndarray
            The coordinate positions of each of the cells along each coordinate axis. If
            ``center='vertex'``, then these are interpreted as the vertex coordinates and there
            will be `N-1` cells along each axis. If ``center='cell'``, then the coordinates will
            be interpreted as the cell coordinates.

        ghost_zones : numpy.ndarray, optional
            The number of ghost cells to include on each side of every axis.
            Should be a 2D array-like object of shape ``(2, ndim)``, where:

            - The first row specifies the number of ghost cells on the lower side of each axis.
            - The second row specifies the number on the upper side.

            If not provided, defaults to no ghost cells (i.e., zeros on all boundaries).

            .. hint::

                Including ``ghost_zones`` is generally good practice because most numerical
                schemes for computing differential operations can obtain :math:`\mathcal{O}(\delta x^2)`
                precision in the center of the grid but may lose accuracy at edges. Thus, ghost zones
                can be configured to ensure that these low accuracy areas are outside of the desired
                physical domain.
        chunk_size : list of int, optional
            A sequence of integers specifying the number of grid points per chunk along each axis.
            If specified, enables chunking support for domain decomposition, which can be useful
            for parallel processing or memory-constrained workflows.
            Each chunk size must **evenly divide** the non-ghost extent of the corresponding axis.
            If not specified, chunking is disabled.
        args:
            Positional arguments passed through to initialization hooks.
            These are forwarded to ``__configure_domain__`` and ``__configure_boundary__``.
        units : dict of str or :py:class:`~unyt.units.Unit`, optional
            A dictionary mapping base physical dimensions (e.g., "length", "time", "mass", "angle")
            to their corresponding units, either as strings (e.g., "cm", "s") or as :py:class:`~unyt.units.Unit` instances.
            If not provided, the grid defaults to the CGS (centimeter–gram–second) unit system.
            This unit system is used for interpreting coordinate values and for scaling
            differential geometry operations.
        kwargs:
            Additional keyword arguments forwarded to subclass initialization routines,
            such as boundary condition configuration or field metadata.

        Raises
        ------
        GridInitializationError
            If the coordinate arrays are invalid (e.g., wrong shape, not increasing),
            or if chunking/ghost zone settings are inconsistent with the domain.

        Notes
        -----
        This constructor calls the base class :py:meth:`grids.base.GridBase.__init__`, which delegates setup to:

        - ``__configure_coordinate_system__``
        - ``__configure_unit_system__``
        - ``__configure_domain__``
        - ``__configure_boundary__``

        All coordinate arrays are stored internally and used for geometric evaluation,
        boundary layout, and interpolation logic.
        """
        # Parse down the args and kwargs so that we can pass them nicely
        # down to the super initialization layer.
        args = [coordinates, *args]
        kwargs = {
            "ghost_zones": ghost_zones,
            "chunk_size": chunk_size,
            "center": center,
            "bbox": bbox,
            **kwargs,
        }

        # Enter the standard init sequence defined in GridBase.
        super().__init__(coordinate_system, *args, units=units, **kwargs)

    # @@ Coordinate Management @@ #
    # These methods handle the construction / obtaining
    # of coordinates from different specifications. Some of
    # these methods will be abstract, others will be declarative.
    def _compute_coordinates_on_slices(
        self, axis_indices: np.ndarray, slices: List[slice]
    ) -> Tuple[np.ndarray, ...]:
        return tuple(
            [self.__coordinate_arrays__[i][_s] for i, _s in zip(axis_indices, slices)]
        )

    def extract_subgrid(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        axes: Optional[Sequence[str]] = None,
        include_ghosts: bool = True,
        halo_offsets=None,
        oob_behavior=None,
        **kwargs,
    ) -> "GenericGrid":
        # Ensure that OOB is valid.
        if oob_behavior not in {"raise", "clip"}:
            raise ValueError("oob_behavior must be 'raise' or 'clip'")

        # Get slice ranges for data extraction
        slices = self.compute_chunk_slice(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )
        new_coords = tuple(
            arr[sl] for arr, sl in zip(self.__coordinate_arrays__, slices)
        )

        # Construct the new grid
        return GenericGrid(self.coordinate_system, new_coords, **kwargs)

    # @@ IO Methods @@ #
    # These methods are used for reading / writing grids
    # to / from disk. Some of these methods are abstract and
    # must be implemented in subclasses. Others are helper functions.
    def to_hdf5(
        self,
        filename: str,
        group_name: Optional[str] = None,
        overwrite: bool = False,
        **_,
    ):
        """
        Save this grid object as / in an HDF5 file.

        This method will save all the grid metadata (:attr:`bbox`, :attr:`dd`, etc.) under ``group_name/attrs`` and
        dump the coordinate arrays into individual datasets inside of ``group_name``, each named in accordance
        with the coordinate axis it represents. See the notes below for a more comprehensive explanation of
        the resulting file structure.

        Parameters
        ----------
        filename : str
            Path to the target HDF5 file. The file will be created if it does not exist.
        group_name : str, optional
            Name of the HDF5 group to store the grid under. If None, stores directly at the file root.
        overwrite : bool, default=False
            If True, overwrites the existing group or file if it already exists.
            If False, raises an error if the target exists.

        Notes
        -----

        The structure of the resulting HDF5 file will look something like the following:

        .. code-block::

            ├── [attrs]
            │   ├── axes                → list of str           # Axis names (e.g. ["x", "y", "z"])
            │   ├── chunking            → bool                  # Whether chunking is enabled
            │   ├── chunk_size          → list[int]             # Size of each chunk (if enabled)
            │   ├── grid_shape          → list[int]             # Grid shape (excluding ghost zones)
            │   ├── bbox                → (2, ndim) array       # Physical bounding box
            │   ├── ghost_zones         → (2, ndim) array       # Ghost zones on each axis
            │   └── coordinate_system   → str                   # Class name of the coordinate system
            │
            │
            ├── coordinates/            → group                 # Note: names vary with coordinate system.
            │   ├── x                   → 1D array              # Coordinate values along 'x' axis
            │   ├── y                   → 1D array              # Coordinate values along 'y' axis
            │   └── z                   → 1D array              # Coordinate values along 'z' axis (if 3D)
            │
            ├── units/                  → group
            │   ├── length              → str (e.g. "cm")       # Unit string for 'length'
            │   ├── time                → str (e.g. "s")        # Unit string for 'time'
            │   ├── mass                → str (e.g. "g")        # Unit string for 'mass'
            │   ├── angle               → str (e.g. "rad")      # Unit string for 'angle'
            │   └── ...                                        # Any other base dimensions
            │
            └── coord_systm/            → group
                └── ...                 → Defined by coordinate system class
                                         (symbolic metric, Jacobians, etc.)

        See Also
        --------
        from_hdf5 : Method that reads the file structure described above.
        """
        import h5py

        # Construct the HDF5 stub (ensuring that the file exists) and that
        # if the group name was not specified, all of the data gets overwritten.
        self._build_hdf5_stub(filename, group_name=group_name, overwrite=overwrite)

        # We now proceed with opening the file and writing the data.
        with h5py.File(filename, "a") as f:
            # Start by identifying the storage location for the data.
            # if `group_name` is None, then it is stored in the main directory.
            group = self._parse_hdf5_group(
                f, group_name=group_name, overwrite=overwrite
            )

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
            group.attrs["ghost_zones"] = np.asarray(self.ghost_zones, dtype=int)
            group.attrs["coordinate_system"] = self.coordinate_system.__class__.__name__

            if self.chunking:
                group.attrs["chunk_size"] = self.chunk_size

            self._serialize_unit_system(group.require_group("units"))

            # Save the coordinate axes to the hdf5 file in datasets.
            try:
                coord_group = group.require_group("coordinates")
                for axis_name, arr in zip(self.axes, self.__coordinate_arrays__):
                    coord_group.create_dataset(axis_name, data=arr)
            except Exception as e:
                raise IOError(f"Failed to write the coordinate arrays to disk: {e}")

        # Finally, the coordinate system itself must now be saved to disk. We leave
        # the context manager to ensure that everything flushes correctly.
        self._save_coordinate_system(
            filename, group_name=group_name, overwrite=overwrite
        )

    @classmethod
    def from_hdf5(cls, filename: str, group_name: Optional[str] = None, **kwargs):
        """
        Load a grid instance from an HDF5 file.

        This method reconstructs a fully initialized grid object from a file written by
        :meth:`to_hdf5`. It restores all structural metadata, coordinate arrays, unit
        definitions, and the associated coordinate system.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to load from.
        group_name : str, optional
            Name of the HDF5 group in which the grid was saved. If None, loads from the file root.
        **kwargs
            Additional keyword arguments forwarded to the grid constructor.

        Returns
        -------
        GenericGrid
            Reconstructed grid instance.

        Notes
        -----

        The structure of the input HDF5 file will look something like the following:

        .. code-block::

            ├── [attrs]
            │   ├── axes                → list of str           # Axis names (e.g. ["x", "y", "z"])
            │   ├── chunking            → bool                  # Whether chunking is enabled
            │   ├── chunk_size          → list[int]             # Size of each chunk (if enabled)
            │   ├── grid_shape          → list[int]             # Grid shape (excluding ghost zones)
            │   ├── bbox                → (2, ndim) array       # Physical bounding box
            │   ├── ghost_zones         → (2, ndim) array       # Ghost zones on each axis
            │   └── coordinate_system   → str                   # Class name of the coordinate system
            │
            │
            ├── coordinates/            → group                 # Note: names vary with coordinate system.
            │   ├── x                   → 1D array              # Coordinate values along 'x' axis
            │   ├── y                   → 1D array              # Coordinate values along 'y' axis
            │   └── z                   → 1D array              # Coordinate values along 'z' axis (if 3D)
            │
            ├── units/                  → group
            │   ├── length              → str (e.g. "cm")       # Unit string for 'length'
            │   ├── time                → str (e.g. "s")        # Unit string for 'time'
            │   ├── mass                → str (e.g. "g")        # Unit string for 'mass'
            │   ├── angle               → str (e.g. "rad")      # Unit string for 'angle'
            │   └── ...                                        # Any other base dimensions
            │
            └── coord_systm/            → group
                └── ...                 → Defined by coordinate system class
                                         (symbolic metric, Jacobians, etc.)
        See Also
        --------
        to_hdf5 : Method that produces the file structure described above.
        """
        # Ensure existence of the hdf5 file before attempting to
        # load data from it.
        import h5py

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

            # Load the units.
            if "units" in group.keys():
                units = cls._deserialize_unit_system(group["units"])
            else:
                units = None

            # Check that coordinate datasets are present
            if "coordinates" not in group:
                raise IOError(f"No 'coordinates' group found in '{group.name}'.")

            coord_group = group["coordinates"]
            coord_arrays = {key: coord_group[key][...] for key in coord_group.keys()}

        # Load the coordinate system object
        coordinate_system = cls._load_coordinate_system(filename, group_name=group_name)

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

        return cls(
            coordinate_system,
            ordered_coord_arrays,
            chunk_size=chunk_size,
            ghost_zones=ghost_zones,
            units=units,
            **kwargs,
        )


class UniformGrid(GridBase):
    r"""
    Structured grid with uniform spacing along each axis.

    The :py:class:`UniformGrid` represents a logically Cartesian grid defined by a bounding box
    and a set of domain dimensions. Grid points are distributed uniformly along each axis,
    and the grid supports both vertex-centered and cell-centered layouts.

    This class is commonly used in structured mesh simulations, finite difference methods,
    and regular field sampling scenarios where grid spacing is uniform. Compared to
    :py:class:`GenericGrid`, it does not store coordinate arrays explicitly—coordinates are
    computed analytically from the bounding box, spacing, and resolution.

    Features
    --------
    - Vertex-centered or cell-centered coordinate layout
    - Analytical calculation of grid coordinates and spacing
    - Ghost zone support for boundary conditions and stencil operations
    - Chunking support for domain decomposition and parallelism
    - HDF5 serialization with metadata and unit system integration

    Examples
    --------
    Construct a 2D vertex-centered grid over the domain ``[0, 1] x [0, 2]``:

    .. code-block:: python

        >>> from pymetric.coordinates import CartesianCoordinateSystem2D
        >>> import numpy as np
        >>>
        >>> coord_sys = CartesianCoordinateSystem2D()
        >>> bbox = np.array([[0.0, 0.0], [1.0, 2.0]])
        >>> shape = [11, 21]  # 11 x 21 grid points
        >>> grid = UniformGrid(coord_sys, bbox, shape)
        >>> grid.get_coordinates((0, 0))
        [0.0, 0.0]

    Construct a cell-centered grid with ghost zones:

    .. code-block:: python

        >>> ghost = [[1, 1], [1, 1]]
        >>> grid = UniformGrid(coord_sys, bbox, shape, ghost_zones=ghost, cell_centered=True)
        >>> grid.get_coordinate_arrays(include_ghosts=True)[0][:3]
        array([-0.04545455,  0.04545455,  0.13636364])

    Notes
    -----
    - The number of grid points per axis (`domain_dimensions`) excludes ghost zones.
    - The spacing between grid points is uniform and inferred from the bounding box.
    - Cell-centered grids shift all coordinates by half a cell inward.
    - Chunk sizes (if used) must divide domain dimensions exactly.
    - This class interoperates with coordinate systems and field representations in Pisces Geometry.

    See Also
    --------
    :py:class:`GenericGrid` : A structured grid with arbitrary, non-uniform spacing.
    :py:class:`~coordinates.core.CurvilinearCoordinateSystem` : Abstract base for coordinate systems.
    """

    # @@ Initialization Procedures @@ #
    # These initialization procedures may be overwritten in subclasses
    # to specialize the behavior of the grid. Ensure that all the REQUIRED
    # attributes are set in the initialization methods, otherwise unintended behavior
    # may arise.
    def __configure_domain__(self, *args, **kwargs):
        """
        Configure the domain geometry of the grid.

        This method sets the grid’s internal shape, bounding box, and chunking structure.
        It validates the user-provided coordinate arrays and computes:

        - ``self.__center__``: The position of the points relative to cells.
        - ``self.__bbox__``: Bounding box (without ghost zones)
        - ``self.__dd__``: Domain shape (excluding ghost zones)
        - ``self.__chunking__``: Whether chunking is enabled
        - ``self.__chunk_size__``: Size of each chunk, if chunking is enabled
        - ``self.__cdd__``: Number of chunks per axis, if chunking is enabled
        """
        # Coerce the bounding box and the domain dimensions to
        # valid bbox and dd types. These are going to be the physical
        # domain dimensions and bounding box.
        self.__bbox__ = BoundingBox(args[0])
        self.__dd__ = DomainDimensions(args[1])

        # Given the bounding box and the domain dimensions, also
        # construct the cell spacing, and cell volume.
        self.__center__: Literal["vertex", "cell"] = kwargs.get("center", "vertex")
        if self.__center__ == "vertex":
            # Ensure that the number of coordinates is sufficiently specified.
            if not np.all(self.__dd__ > 1):
                raise ValueError(
                    "Vertex-centered grid requires at least two points along each axis.\n"
                    "Found axis with only one point, which would result in undefined spacing "
                    "(division by zero).\nEither increase the number of points or switch to a "
                    "cell-centered grid (cell_centered=True)."
                )
        elif self.__center__ == "cell":
            pass
        else:
            raise ValueError(f"Invalid `center` argument value: {self.__center__}.")

        self.__cell_spacing__ = (self.bbox[1, :] - self.bbox[0, :]) / self.ncells
        self.__cell_volume__ = np.prod(self.__cell_spacing__)

        # Configure optional chunking. If the chunks aren't specified,
        # they can be filled in with blanks. Otherwise, we need to
        # actually populate the attributes.
        chunk_size = kwargs.get("chunk_size", None)
        if chunk_size is None:
            self.__chunking__ = False
            self.__chunk_size__ = None
            self.__cdd__ = None
            self.__chunk_spacing__ = None
            self.__chunk_volume__ = None
        else:
            chunk_size = np.asarray(chunk_size).ravel()
            if len(chunk_size) != self.ndim:
                raise GridInitializationError(
                    f"`chunk_size` must have {self.ndim} elements, got {len(chunk_size)}."
                )
            if not np.all(self.ncells % chunk_size == 0):
                raise GridInitializationError(
                    f"`chunk_size` {chunk_size} must divide the number of (active) cells {self.ncells} exactly."
                )

            self.__chunking__ = True
            self.__chunk_size__ = DomainDimensions(chunk_size)
            self.__cdd__ = self.ncells // self.__chunk_size__

            # In addition to the standard parameters, we can also
            # get the chunk spacing and volumes.
            self.__chunk_spacing__ = (self.bbox[1, :] - self.bbox[0, :]) / self.__cdd__
            self.__chunk_volume__ = np.prod(self.__chunk_spacing__)

    def __configure_boundary__(self, *args, **kwargs):
        """
        Configure the ghost-region geometry of the grid.

        This method computes:

        - ``self.__ghost_bbox__``: Full physical bounding box including ghost zones
        - ``self.__ghost_dd__``: Full domain shape including ghost zones
        """
        # Now use the coordinate arrays to compute the bounding box. This requires calling out
        # to the ghost_zones a little bit early and validating them. The domain dimensions are computed
        # from the length of each of the coordinate arrays.
        self.__ghost_zones__ = self._standardize_ghost_zones(
            kwargs.get("ghost_zones", None), self.__cs__.ndim
        )

        # To construct the ghost bbox, we need to take the cell spacing and back things off either
        # edge of the domain by the number of ghost zones.
        self.__ghost_bbox__ = BoundingBox(
            np.asarray(
                [
                    [
                        self.__bbox__[0, _i]
                        - self.__cell_spacing__[_i] * self.__ghost_zones__[0, _i],
                        self.__bbox__[1, _i]
                        + self.__cell_spacing__[_i] * self.__ghost_zones__[1, _i],
                    ]
                    for _i in range(self.ndim)
                ]
            ).T
        )

        self.__ghost_dd__ = DomainDimensions(
            [
                _dd_ + _nghost_
                for _dd_, _nghost_ in zip(
                    self.__dd__, np.sum(self.__ghost_zones__, axis=0)
                )
            ]
        )

    def __init__(
        self,
        coordinate_system: "_CoordinateSystemBase",
        bounding_box: Any,
        domain_dimensions: Any,
        /,
        ghost_zones: Optional[Sequence[Sequence[float]]] = None,
        chunk_size: Optional[Sequence[int]] = None,
        *args,
        units=None,
        center: Literal["vertex", "cell"] = "vertex",
        **kwargs,
    ):
        r"""
        Initialize a uniformly spaced, structured grid.

        This constructor sets up a logically Cartesian grid
        using a bounding box and number of grid points per axis.
        Optionally supports ghost zones and domain chunking.

        Parameters
        ----------
        coordinate_system : ~coordinates.core.CurvilinearCoordinateSystem
            An instance of a coordinate system subclass (e.g.,
            :py:class:`~coordinates.coordinate_systems.CartesianCoordinateSystem3D`,
            :py:class:`~coordinates.coordinate_systems.CylindricalCoordinateSystem`, or
            :py:class:`~coordinates.coordinate_systems.SphericalCoordinateSystem`),
            which defines the geometry and dimensionality of the grid.

            .. note::

                The number of coordinate arrays provided in ``coordinates`` must match the dimensionality
                defined by this coordinate system.
        bounding_box : numpy.ndarray
            The physical extent of the domain (*excluding ghost zones*).

            The `bounding_box` must be an array-like object with shape ``(2,ndim)`` in
            which the ``bounding_box[0,:]`` corresponds to the lower left corner of
            the domain and ``bounding_box[1,:]`` corresponds to the upper right corner.

            This bounding box defines the continuous coordinate domain that the grid spans. It is used
            to calculate uniform cell spacing along each axis.
        domain_dimensions : list of int
            The number of grid points per axis, *excluding ghost zones*. This argument determines
            the effective resolution of the grid and (in conjunction with ``bounding_box``) determines
            the grid spacing.

            .. note::

                If the grid is using vertex-centered points (``center='vertex'``), then each
                axis must have **at least 2** points otherwise it would lead to a degenerate grid along
                those dimensions.
        ghost_zones : numpy.ndarray, optional
            The number of inactive (ghost) cells to place adjacent to the edge of the grid along
            each axis.

            `ghost_zones` may be scalar, corresponding to a fixed number of cells on each edge. It may
            be ``(ndim,)`` in shape, corresponding to symmetric regions on each axis, or it may be
            ``(2,ndim)``, corresponding to asymmetric regions on each axis.

        chunk_size : list of int, optional
            The number of grid **cells** per chunk along each axis.

            To be valid, `chunk_size` must evenly divide the number of cells. If ``center='cell'``,
            then this is equivalent to `domain_dimensions`, if ``center='vertex'``, then `chunk_size`
            must divide `domain_dimensions + 1` along each axis.
        units : dict of str or :py:class:`~unyt.units.Unit`, optional
            A dictionary mapping base physical dimensions (e.g., "length", "time", "mass", "angle")
            to their corresponding units, either as strings (e.g., "cm", "s") or as :py:class:`~unyt.units.Unit` instances.
            If not provided, the grid defaults to the CGS (centimeter–gram–second) unit system.
            This unit system is used for interpreting coordinate values and for scaling
            differential geometry operations.
        center: {'vertex', 'cell'}, optional
            The position of each point in the grid lattice. If `center` is ``"vertex"``, then
            the grid points are placed at the corners of each cell and there are `domain_dimensions-1`
            cells and `domain_dimensions` edges along each axis.

            If ``"cell"``, then the positions of the coordinates are placed at the centers
            of each cell. There are then `domain_dimensions` cells along each axis and
            `domain_dimensions + 1` edges.
        *args,**kwargs:
            Additional keyword arguments forwarded to subclass initialization routines,
            such as boundary condition configuration or field metadata. These are currently
            ignored.
        """
        # Parse the args and the kwargs so that they can be passed
        # into the super() __init__ method.
        args = [bounding_box, domain_dimensions, *args]
        kwargs = {
            "ghost_zones": ghost_zones,
            "chunk_size": chunk_size,
            "center": center,
            **kwargs,
        }

        # Begin the __init__ process.
        super().__init__(coordinate_system, *args, units=units, **kwargs)

    def __post_init__(self, *args, **kwargs):
        """
        Finalize grid initialization by setting centering and coordinate offset.

        This method configures whether the grid is cell-centered or vertex-centered,
        and sets the offset to be applied when computing physical coordinates.
        """
        if self.__center__ == "cell":
            self.__start_offset__ = self.__cell_spacing__ / 2
        else:
            self.__start_offset__ = np.zeros_like(self.__dd__, dtype=float)

    # @@ Coordinate Management @@ #
    # These methods handle the construction / obtaining
    # of coordinates from different specifications. Some of
    # these methods will be abstract, others will be declarative.
    def _compute_coordinates_on_slices(
        self, axis_indices: np.ndarray, slices: List[slice]
    ) -> Tuple[np.ndarray, ...]:
        origin = self.gbbox[0, axis_indices]
        offset = self.__start_offset__[axis_indices]
        spacing = self.__cell_spacing__[axis_indices]
        dd = self.__ghost_dd__[axis_indices]

        coords = tuple(
            origin[i]
            + offset[i]
            + spacing[i] * np.arange(s.start or 0, s.stop or dd[i])
            for i, s in enumerate(slices)
        )

        return coords

    def extract_subgrid(
        self,
        chunks: Sequence[Union[int, Tuple[int, int], slice]],
        axes: Optional[Sequence[str]] = None,
        include_ghosts: bool = True,
        halo_offsets=None,
        oob_behavior=None,
        **kwargs,
    ) -> "UniformGrid":
        # Ensure that OOB is valid.
        if oob_behavior not in {"raise", "clip"}:
            raise ValueError("oob_behavior must be 'raise' or 'clip'")

        # Build an axes mask
        axes_mask = self.coordinate_system.build_axes_mask(axes)

        # Determine the origin of each selected axis
        origin = self.gbbox[0, axes_mask]
        offset = self.__start_offset__[axes_mask]
        spacing = self.__cell_spacing__[axes_mask]

        # Get slice ranges for data extraction
        slices = self.compute_chunk_slice(
            chunks,
            axes=axes,
            include_ghosts=include_ghosts,
            halo_offsets=halo_offsets,
            oob_behavior=oob_behavior,
        )

        # Construct the coordinate corners.
        _start, _stop = np.asarray([s.start for s in slices]), np.asarray(
            [s.stop for s in slices]
        )
        _dd_ = _stop - _start
        lowerleft = origin + offset + (_start * spacing)
        upperright = origin + offset + (_stop * spacing)

        bbox = BoundingBox(np.stack([lowerleft, upperright], axis=0))

        # Construct the new grid
        return self.__class__(self.coordinate_system, bbox, _dd_, **kwargs)

    # @@ IO Methods @@ #
    # These methods are used for reading / writing grids
    # to / from disk. Some of these methods are abstract and
    # must be implemented in subclasses. Others are helper functions.
    def to_hdf5(
        self,
        filename: str,
        group_name: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Save this grid object to an HDF5 file.

        This method serializes the grid metadata, physical domain information, unit definitions,
        and coordinate system to disk. Unlike :py:class:`GenericGrid`, the coordinates are not explicitly saved,
        since they are analytically defined from the bounding box and resolution.

        Parameters
        ----------
        filename : str
            Path to the target HDF5 file. The file will be created if it does not exist.
        group_name : str, optional
            Name of the HDF5 group to store the grid under. If None, stores directly at the file root.
        overwrite : bool, default=False
            If True, overwrites the existing group or file if it already exists.
            If False, raises an error if the target exists.
        **kwargs
            Additional keyword arguments for forward compatibility (ignored by default).

        Notes
        -----
        The resulting HDF5 structure resembles:

        .. code-block::

            ├── [attrs]
            │   ├── axes                → list of str           # Axis names (e.g. ["x", "y", "z"])
            │   ├── chunking            → bool                  # Whether chunking is enabled
            │   ├── chunk_size          → list[int]             # Size of each chunk (if enabled)
            │   ├── grid_shape          → list[int]             # Grid shape (excluding ghost zones)
            │   ├── bbox                → (2, ndim) array       # Physical bounding box (without ghosts)
            │   ├── ghost_zones         → (2, ndim) array       # Number of ghost cells per axis
            │   ├── coordinate_system   → str                   # Name of the coordinate system class
            │   └── centered            → bool                  # Whether grid is cell-centered
            │
            ├── units/                  → group
            │   └── ...                 → Unit definitions for each base dimension
            │
            └── coord_systm/            → group
                └── ...                 → Coordinate system-specific symbolic definitions

        See Also
        --------
        from_hdf5 : Method to load the grid back from an HDF5 file.
        """
        import h5py

        # Construct the HDF5 stub (ensuring that the file exists) and that
        # if the group name was not specified, all of the data gets overwritten.
        self._build_hdf5_stub(filename, group_name=group_name, overwrite=overwrite)

        # We now proceed with opening the file and writing the data.
        with h5py.File(filename, "a") as f:
            # Start by identifying the storage location for the data.
            # if `group_name` is None, then it is stored in the main directory.
            group = self._parse_hdf5_group(
                f, group_name=group_name, overwrite=overwrite
            )

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
            group.attrs["ghost_zones"] = np.asarray(self.ghost_zones, dtype=int)
            group.attrs["coordinate_system"] = self.coordinate_system.__class__.__name__
            group.attrs["centered"] = self.__center__
            if self.chunking:
                group.attrs["chunk_size"] = self.chunk_size

            # Finish off by saving the unit
            # system (using the _serialize_unit_system method)
            # and the coordinate system (_save_coordinate_system).
            self._serialize_unit_system(group.require_group("units"))

        # Finally, the coordinate system itself must now be saved to disk. We leave
        # the context manager to ensure that everything flushes correctly.
        self._save_coordinate_system(
            filename, group_name=group_name, overwrite=overwrite
        )

    @classmethod
    def from_hdf5(cls, filename: str, group_name: Optional[str] = None, **kwargs):
        """
        Load a :py:class:`UniformGrid` from an HDF5 file.

        This method reconstructs a grid previously saved using :meth:`to_hdf5`. It restores
        the domain shape, bounding box, ghost zones, chunking structure, coordinate system,
        and unit metadata.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to load from.
        group_name : str, optional
            Name of the HDF5 group in which the grid was saved. If None, loads from the file root.
        **kwargs
            Additional keyword arguments passed to the grid constructor.

        Returns
        -------
        UniformGrid
            Fully reconstructed grid instance.

        Raises
        ------
        IOError
            If the file or group does not exist, or if required metadata is missing.

        Notes
        -----
        The method expects a file structure produced by :meth:`to_hdf5`, including the necessary
        attributes and subgroups. For example:

        .. code-block::

            ├── [attrs]
            │   ├── axes                → list of str
            │   ├── chunking            → bool
            │   ├── chunk_size          → list[int]
            │   ├── grid_shape          → list[int]
            │   ├── bbox                → (2, ndim) array
            │   ├── ghost_zones         → (2, ndim) array
            │   ├── coordinate_system   → str
            │   └── centered            → bool
            │
            ├── units/                  → group of base unit strings
            └── coord_systm/            → symbolic data for coordinate system reconstruction

        The coordinate arrays are **not** saved to disk, because they are deterministically defined
        from the bounding box and resolution.
        """
        # Ensure existence of the hdf5 file before attempting to
        # load data from it.
        import h5py

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
            bounding_box = group.attrs.get("bbox", None)
            domain_dimensions = group.attrs.get("grid_shape", None)
            ghost_zones = group.attrs.get("ghost_zones", None)
            centered = group.attrs.get("centered", False)

            # Load the units.
            if "units" in group.keys():
                units = cls._deserialize_unit_system(group["units"])
            else:
                units = None

        # Load the coordinate system object
        coordinate_system = cls._load_coordinate_system(filename, group_name=group_name)

        return cls(
            coordinate_system,
            bounding_box,
            domain_dimensions,
            chunk_size=chunk_size,
            ghost_zones=ghost_zones,
            cell_centered=centered,
            units=units,
            **kwargs,
        )
