.. _grids_building:

====================================
Geometric Grids: Custom Grid Classes
====================================

Grids in PyMetric define the structure of computational domains, encapsulating coordinate systems,
domain resolution, ghost zones, chunking behavior, and coordinate retrieval.
Grid classes are central to all spatial discretizations but **do not** store field values themselves.
Instead, they provide structural context and metadata to higher-level components.

PyMetric includes a flexible and extensible grid framework based on the :py:class:`~grids.base.GridBase` class. This guide
describes how to create new grid subclasses by implementing a small number of well-defined hooks.

Structure of Grid Classes
-------------------------

All concrete grids must inherit from :py:class:`~grids.base.GridBase`, an abstract base class defining the public API for all grids.

This base class is responsible for:

- Binding to a coordinate system (:py:class:`coordinates.base._CoordinateSystemBase`)
- Tracking domain geometry, ghost zones, and chunking
- Providing a uniform interface for grid operations (e.g., coordinate extraction, subgrids)
- Supporting file serialization and deserialization via IO mixins

Initialization Methods
^^^^^^^^^^^^^^^^^^^^^^^^

To define a custom grid class, users override three (optionally 4) core setup routines:

1. :meth:`__configure_coordinate_system__`
2. :meth:`__configure_domain__`
3. :meth:`__configure_boundary__`
4. :meth:`__post_init__` (optional)

Each method is responsible for configuring a specific aspect of the grid.


.. code-block:: python

    def __configure_coordinate_system__(self, coordinate_system, *args, **kwargs): ...
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

These methods are called in order by the :meth:`GridBase.__init__` constructor and should assign specific internal attributes as
described in the corresponding docstrings.

.. hint::

    These exact methods are written for :py:class:`~grids.core.GenericGrid` and :py:class:`~grids.core.UniformGrid`, which
    can provide a good starting point for implementing custom grids.

Any additional setup necessary may be done during the ``.__post_init__`` method.


IO Support Methods
^^^^^^^^^^^^^^^^^^^

Two required methods (:meth:`~grids.base.GridBase.to_metadata_dict` and :meth:`~grids.base.GridBase.from_metadata_dict`)
define the serialization protocol used to save and restore grid configurations.

These methods form the *core metadata exchange format* for all PyMetric grids and are used as the internal backbone
for JSON, YAML, and HDF5 persistence via the :class:`~grids.mixins.GridIOMixin`.

1. :meth:`~grids.base.GridBase.to_metadata_dict`

   Returns a JSON-compatible dictionary containing the grid’s configuration metadata. This dictionary must include
   all the information needed to reconstruct the grid **except** for the coordinate system, which is supplied
   externally when loading.

   Required fields typically include:

   - ``bbox``: A list of two lists representing the lower and upper corners of the grid bounding box.
   - ``dd``: The number of grid points per axis (excluding ghost zones).
   - ``ghost_zones``: A 2 × ndim array giving ghost cell counts on lower and upper edges of each axis.
   - ``center``: Either ``"cell"`` or ``"vertex"``, indicating grid centering.
   - ``chunking``: Boolean flag indicating whether chunking is enabled.
   - ``chunk_size``: (optional) List of per-axis chunk sizes.
   - ``cdd``: (optional) Number of chunks per axis.

   This method must raise an exception if serialization fails due to missing internal state or invalid configuration.

2. :meth:`~grids.base.GridBase.from_metadata_dict`

   Reconstructs a new grid instance from metadata and a coordinate system.

   This method:

   - Must validate all required fields in the metadata dictionary.
   - Must not infer or reconstruct the coordinate system — it must be passed as an argument.
   - May accept additional optional metadata fields for subclass-specific behavior.

   Example:

   .. code-block:: python

      cs = OrthogonalCoordinateSystem(...)
      with open("grid.json") as f:
          metadata = json.load(f)

      grid = MyGrid.from_metadata_dict(cs, metadata)

Mixin Classes
-------------

The :py:class:`~grids.base.GridBase` class in PyMetric is built using a composable, mixin-driven architecture.
This design enables separation of concerns and allows different behaviors (I/O, plotting, chunking, etc.)
to be modular, testable, and easily overridden or extended.

Mixin classes are grouped by functionality and live in submodules of the :mod:`grids` package. They are inherited by
all concrete grid classes through :class:`GridBase`, providing a unified API without bloating the base logic.

Each mixin class defines a narrow set of related capabilities and can be found in one of the following submodules:

.. code-block:: text

    grids/
    ├── base.py              ← defines GridBase
    ├── core.py              ← concrete grid implementations (GenericGrid, UniformGrid)
    ├── mixins/
    │   ├── _typing.py        ← type annotations for mixins
    │   ├── core.py          ← core mixins (GridUtilsMixin, GridIOMixin)
    |   │   ├── GridUtilsMixin  ← general-purpose utilities for grids
    │   │   ├── GridIOMixin     ← file I/O support for grids
    │   |   ├── GridPlotMixin    ← plotting support for grids
    │   |
    │   ├── chunking.py      ← chunking mixins (GridChunkingMixin)
    │   |   ├── GridChunkingMixin  ← chunking and partitioning behavior
    │   |
    │   ├── mathops.py       ← dense math operations (DenseMathOpsMixin)
    │   |   ├── DenseMathOpsMixin       ← dense coordinate-space math operations

Each mixin class is responsible for a well-scoped set of behaviors. Below is a summary of their
responsibilities, references to their API documentation, and guidance on how and when to extend or override them.

Core Mixins
^^^^^^^^^^^

- :class:`~grids.mixins.core.GridUtilsMixin`
  (Defined in ``grids/mixins/core.py``)

  Provides utility methods that support axis name-index resolution, coordinate shape validation, index generation,
  and dimensional consistency checks. This is the primary toolbox for grid structure introspection.

  .. hint::

      Do include: logic for resolving axes, handling slices, reshaping results, etc.
      Do not include: any I/O, math operations, or plotting code.

- :class:`~grids.mixins.core.GridIOMixin`
  (Defined in ``grids/mixins/core.py``)

  Implements file I/O methods using the standardized metadata dictionary produced
  by :meth:`~grids.base.GridBase.to_metadata_dict`.

  .. hint::

        Do include: logic to write/read metadata files and validate external serialization.
        Do not include: logic specific to in-memory representations or full grid reconstruction; that belongs in
        :meth:`~grids.base.GridBase.from_metadata_dict`.

- :class:`~grids.mixins.core.GridPlotMixin`
  (Defined in ``grids/mixins/core.py``)

  Offers diagnostic plotting methods for visualizing grid layout, ghost zones, chunk partitions,
  and spacing irregularities. Uses Matplotlib by default.

  .. hint::

        Do include: methods for plotting grid structure, ghost zones, and chunk boundaries.
        Do not include: field-specific visualizations or domain-specific overlays — those should be handled in
        field visualization utilities.

Chunking Mixin
^^^^^^^^^^^^^^

- :class:`~grids.mixins.chunking.GridChunkingMixin`
  (Defined in ``grids/mixins/chunking.py``)

  Handles logic for breaking up grids into regular subdomains (chunks). Exposes chunk-aware subgrid extraction,
  validation of chunk shapes, and halo/ghost zone expansion utilities.

  .. hint::

        Do include: methods for chunk validation, extraction of subgrids, and chunk metadata.
        Do not include: ghost zone configuration or boundary conditions — those are handled in
        :meth:`~grids.base.GridBase.__configure_boundary__`.


Math Operations Mixin
^^^^^^^^^^^^^^^^^^^^^

- :class:`~grids.mixins.mathops.DenseMathOpsMixin`
  (Defined in ``grids/mixins/mathops.py``)

  Implements NumPy-based differential geometry operations (e.g., gradient, divergence, Laplacian) using coordinate-space
  finite difference stencils. Interfaces with dense coordinate arrays.

  .. hint::

      Do include: methods that perform math over grid-aligned data.
      Do not include: symbolic geometry or coordinate system math — that belongs in the coordinate system or symbolic geometry modules.

