.. _grids_developer:

========================
Developing Custom Grids
========================

Grids (managed in the :py:mod:`grids` module) are the fundamental objects in Pisces Geometry that define the layout of a
computational domain. They encapsulate not just the numerical structure of the domain—such as its shape, resolution,
and physical bounds—but also the coordinate system, ghost zone definitions, chunking behavior, and mechanisms
to retrieve physical coordinates.

Grids do not typically store the actual field data directly, but instead provide the interface by which coordinates
and grid shape are made available to other parts of the framework. There are many types of grids available in Pisces-Geometry by
default; however, extensibility is a critical aspect of the design approach for Pisces-Geometry (and Pisces in general) and
therefore, it is quite easy to define custom grid implementations.

In this guide, we'll introduce the basics of generating new grid classes.

Structuring Subclasses
----------------------

All grid types in Pisces Geometry are subclasses of the abstract base class ``pisces_geometry.grids.base._BaseGrid``, which defines the common
interface and behaviors expected from all grids. This base class should not be used directly, but serves as a foundation
for building concrete grid types like :py:class:`grids.base.GenericGrid`.

The ``_BaseGrid`` class includes detailed documentation within each method to guide the subclass implementation. Developers
can refer to these docstrings for the specific attributes that must be initialized and the responsibilities of each method.

Subclasses may optionally inherit from intermediate classes if available, or directly from ``_BaseGrid`` when implementing novel grid types.

The Initialization Methods
''''''''''''''''''''''''''

Grid initialization is structured into a set of overridable setup routines that configure the various components of the grid.
Subclasses should override these methods where necessary to define their unique behavior.

By default, the ``__init__`` method in ``_BaseGrid`` looks like the following:

.. code-block:: python

    def __init__(self, coordinate_system: "_CoordinateSystemBase", *args, **kwargs):
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

Each of the methods called is responsible for a separate aspect of the ``__init__`` flow and may be overwritten in
any given subclass.

.. note::

    No matter what, all grid subclasses must take ``coordinate_system`` as an argument. Any other arguments may
    be inferred from ``*args`` or ``**kwargs`` and are then concretely specified in the subclasses.

The initialization methods currently implemented as part of the ``__init__`` flow are the following:

- ``__set_coordinate_system__``
  This method assigns the coordinate system (see e.g. :py:mod:`coordinates`) object to the grid. Subclasses may override
  it to enforce validation—e.g., requiring that the coordinate system be orthogonal or curvilinear.

  .. important::

    The following are required attributes to be set in ``__set_coordinate_system__``:

    - ``self.__cs__``: An instance of ``_CoordinateSystemBase`` or its subclass which acts as the
      coordinate system for the grid.

- ``__set_grid_domain__``
  This method controls the domain of the grid that is being generated. Whereas ``__set_coordinate_system__`` deals which what
  coordinate system is connected to the grid, this method figures out the coordinate boundaries (``bbox``), the grid shape (``dd``),
  and various other aspects of the grid's structure.

  .. important::

    The following are required attributes to be set in ``__set_grid_domain``:

    - ``self.__bbox__``: A ``BoundingBox`` specifying the physical limits (without ghost zones). This should be a
      ``(2,ndim)`` array with the bottom left and top right corners of the coordinate domain specified.
    - ``self.__dd__``: A ``DomainDimensions`` object specifying the number of points per axis (excluding ghost cells).
    - ``self.__chunking__``: Boolean flag for whether chunking is active.
    - ``self.__chunk_size__``: The shape of a single chunk, if chunking is enabled.
    - ``self.__cdd__``: The number of chunks in each dimension.

  One of the critical aspects of this method is dealing with the **ghost zones**. These are excess cells to place outside
  of the main grid in order to ensure that boundary effects / conditions are handled correctly. In this method, these cells
  should be explicitly excluded from things like the ``bbox`` and the ``dd``.

- ``__set_grid_boundary__``
  Responsible for defining the ghost zone regions and the full domain including ghosts.

  .. important::

    The following are required attributes to be set in ``__set_grid_domain``:

    - ``self.__ghost_zones__``: A shape (2, ndim) array of ghost cells per side.
    - ``self.__ghost_bbox__``: The full bounding box including ghost cells.
    - ``self.__ghost_dd__``: The full domain dimensions including ghost cells.

- ``__post_init__``
  Optional method used to perform any custom configuration after the rest of the grid initialization is complete.
  Subclasses can use this as a hook to set up derived quantities or verify internal consistency.

These methods are all invoked in sequence by the ``_BaseGrid.__init__`` method, and errors in any stage are wrapped in a ``GridInitializationError`` for clarity.

Subclass authors should ensure that all required internal attributes are correctly set. Failure to do so may result in runtime errors or incorrect behavior in downstream geometry computations.

.. dropdown:: Example

    In this example, we'll show the initialization code for the :py:class:`grids.base.GenericGrid`, which takes arrays
    of coordinates along each axis to generate the grid.

    .. code-block:: python

        class GenericGrid(_BaseGrid):
            # @@ Initialization Procedures @@ #
            # These initialization procedures may be overwritten in subclasses
            # to specialize the behavior of the grid.
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
                _ghost_zones = np.array(_ghost_zones,dtype=int) if _ghost_zones is not None else np.zeros((2, self.ndim),dtype=int)
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
