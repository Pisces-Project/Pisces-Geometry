.. _grids_dev:

========================================
Geometric Grids: Developer Documentation
========================================

This guide is intended for contributors and maintainers working on the **grid infrastructure** in PyMetric.
Grids provide the structural foundation for all computational domains, including support for chunking,
ghost zones, coordinate mapping, and spatial metadata needed by fields and solvers.

This document provides:

- An overview of the grid class hierarchy and mixin architecture
- Guidelines for writing and extending custom grid classes
- Documentation on mixins for I/O, plotting, utilities, and dense math operations
- Best practices for testing, contribution workflow, and architectural boundaries

It is recommended reading for anyone contributing new grid types or modifying existing logic.

.. important::

    If you're interested in contributing, please submit pull requests or open issues via our GitHub repository:

    `Pisces-Project/PyMetric <https://github.com/Pisces-Project/PyMetric>`_

    Contributions are welcome in the form of:

    - Bug reports and stability improvements
    - New grid types (e.g., adaptive meshes, hybrid grids)
    - Enhancements to chunking, ghost zones, or IO pipelines
    - Improvements to documentation and test coverage

    For instructions on setting up your development environment, building the documentation,
    and running the test suite, please refer to the :ref:`developer_quickstart`.

.. note::

    All contributions must follow the internal API and type annotations established in
    :mod:`grids.mixins._typing`, which defines the formal interfaces for grid classes and mixins.

Grids: Overview
----------------------------

Grids: Overview
---------------

Grids in PyMetric define the **computational geometry** used in simulations and numerical analysis.
They provide the discretized domain over which fields are defined, and act as the interface between
coordinate systems, data buffers, and numerical operators.

All grids in PyMetric subclass the abstract base class :py:class:`~grids.base.GridBase`, which defines the common API
for accessing spatial structure, metadata, and derived properties such as chunk partitions or ghost extents.

The core responsibilities of a grid include:

- Binding to a :py:class:`~coordinates.base._CoordinateSystemBase` instance that defines the geometry
- Tracking domain shape, bounding box, and centering (cell or vertex)
- Managing ghost zones and boundary metadata for stencil operations
- Providing access to full and partial coordinate arrays
- Supporting chunked subdomain logic for distributed or block-wise evaluation
- Serializing and deserializing metadata for persistent I/O

PyMetric currently supports two built-in grid classes:

- :py:class:`~grids.core.UniformGrid`: A regular, axis-aligned grid with uniform spacing.
- :py:class:`~grids.core.GenericGrid`: A general-purpose grid that supports irregular spacing or precomputed coordinates.

Developers can define **custom grid types** by subclassing :class:`GridBase` and implementing a small number of configuration methods:

- :meth:`GridBase.__configure_coordinate_system__` to bind a coordinate system
- :meth:`GridBase.__configure_domain__` to define the physical layout
- :meth:`GridBase.__configure_boundary__` to set ghost zones and boundary padding
- :meth:`GridBase.__post_init__` for optional post-processing after initialization

All other features — including plotting, IO, chunking, and dense math operations — are provided via modular mixins.
This composable architecture ensures a clean separation of concerns and allows each capability to evolve independently.

Grids are **structure-only objects**. They do not hold field values or simulation state themselves,
but instead define the metadata required to instantiate and interpret physical fields.


Mixin Classes
^^^^^^^^^^^^^

Grid classes in PyMetric are composed using a modular **mixin stack**. Each mixin encapsulates a distinct set of related behaviors —
such as plotting, I/O, chunking, or dense math — and can be extended or overridden independently of the core grid logic.

All mixins live under the :mod:`grids.mixins` package and are grouped into purpose-specific modules:

.. code-block:: text

    grids/
    ├── mixins/
    │   ├── core.py            ← core mixins (I/O, plotting, utilities)
    │   ├── chunking.py        ← chunking behavior and subgrid extraction
    │   ├── mathops.py         ← dense math operations (finite differences)
    │   ├── _typing.py         ← protocols for type hints

The following mixin classes are currently available:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mixin Class
     - Description

   * - :class:`~grids.mixins.core.GridUtilsMixin`
     - Provides general-purpose utilities used across all grids. Includes methods for
       axis indexing, shape validation, slice conversion, and dimensional metadata.

   * - :class:`~grids.mixins.core.GridIOMixin`
     - Adds support for serializing and deserializing grid metadata. Implements
       :meth:`.to_metadata_dict`, :meth:`.from_metadata_dict`, and helpers for saving
       to YAML, JSON, or HDF5.

   * - :class:`~grids.mixins.core.GridPlotMixin`
     - Implements diagnostic plotting functions using Matplotlib. Allows visualization of
       grid geometry, ghost zones, and chunk layouts.

   * - :class:`~grids.mixins.chunking.GridChunkingMixin`
     - Adds logic for chunk-aware grids. Enables block-wise decomposition, chunk validation,
       overlap calculations, and extraction of subgrids from larger domains.

   * - :class:`~grids.mixins.mathops.DenseMathOpsMixin`
     - Supports dense numerical differential operators (e.g., gradient, divergence, Laplacian)
       using finite differences in coordinate space. Designed for grid-aligned field evaluation.

Mixin Type Protocols
++++++++++++++++++++

For robust type checking and interface enforcement, PyMetric defines abstract `Protocol` classes
in :mod:`grids.mixins._typing`. These protocols describe the expected method signatures for each mixin's capabilities
and should be used in type hints or when writing logic that dispatches on functionality.

Example:

.. code-block:: python

   from grids.mixins._typing import SupportsChunking

   def overlap_mask(grid: SupportsChunking) -> np.ndarray:
       return grid.get_chunk_overlap_mask(...)

This typing structure ensures that all grids and mixins remain composable and interoperable across the PyMetric framework.

Extending Grid Functionality
-----------------------------

PyMetric’s mixin-based architecture allows developers to add new capabilities to grid
classes with minimal friction. Whether you're extending a single method, writing a new mixin,
or modifying an existing one, the following conventions ensure clarity, maintainability, and cross-grid compatibility.

Adding New Methods to Mixin Stacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When adding a new capability to all grid types, you should:

- Add the method to an appropriate mixin:

  - General-purpose logic → :class:`~grids.mixins.core.GridUtilsMixin`
  - IO-related logic → :class:`~grids.mixins.core.GridIOMixin`
  - Math logic → :class:`~grids.mixins.mathops.DenseMathOpsMixin`
  - Chunking logic → :class:`~grids.mixins.chunking.GridChunkingMixin`

- Update the relevant mixin's protocol interface:

  - Edit ``grids/mixins/_typing.py`` to include the new method in the appropriate Protocol subclass. This enables
    full type-checker and IDE support.

Example:

.. code-block:: python

    class SupportsChunking(Protocol):

        def get_chunk_overlap_mask(self, ...) -> np.ndarray: ...


Ensure that the method is compatible with all grid types:

- Do not hardcode assumptions specific to :class:GenericGrid or :class:UniformGrid.
- Use abstracted attributes (e.g., :attr:ndim, :attr:axes, :attr:bbox) wherever possible.
- Document the new method clearly in both the mixin source file and the main developer API.

Overwriting Mixin Stack Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mixin methods can be overridden in subclasses in two ways:

- Direct override in the grid class:

  .. code-block:: python

        class MyGrid(UniformGrid):
            def to_yaml(self, path: str):
                print("Custom YAML output")
                super().to_yaml(path)

  This works well when the change is specific to a single grid class and doesn’t affect the rest of the stack.

- Custom subclass of an existing mixin:

  .. code-block:: python

        class MyIOMixin(GridIOMixin):
            def to_json(self, path: str):
                print("Overridden JSON write")
                super().to_json(path)

        class MyGrid(MyIOMixin, UniformGrid):
        ...

.. warning::

    Be aware of Python’s method resolution order (MRO). The leftmost base class in the inheritance
    list will take precedence when multiple base classes define the same method. Always place custom mixins
    before :class:`~grids.base.GridBase` to ensure your overrides are respected.

Writing a New Mixin Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

New mixins should be created when a capability:

- Is logically orthogonal to existing behaviors (e.g., caching, time-indexing, visualization).
- Requires optional or grid-independent logic.
- May not be needed for all grids or users.

To create a new mixin class:

1. **Choose the correct mixin module**:

   - General-purpose → ``grids/mixins/core.py``
   - Math-related → ``grids/mixins/mathops.py``
   - IO → ``grids/mixins/core.py``
   - Chunking → ``grids/mixins/chunking.py``

2. **Create a minimal mixin class**:

   .. code-block:: python

      class GridCachingMixin:
          def clear_cache(self):
              ...

3. **Register it in your custom grid class**:

   .. code-block:: python

      class CachingGrid(GridCachingMixin, UniformGrid):
          ...

4. **Add an optional `Protocol` for type hinting support**:

   Edit the ``grids/mixins/_typing.py`` file and define a matching protocol:

   .. code-block:: python

      from typing import Protocol

      class SupportsCaching(Protocol):
          def clear_cache(self): ...

5. **Avoid placing new logic directly in `GridBase`**:

   The ``GridBase`` class should only define abstract interfaces and core attribute assignments.
   All optional or behavioral logic should be implemented through mixins.

Following these conventions ensures that new features remain:

- **Modular**: Mixins isolate capabilities so they can be reused across different grid types.
- **Testable**: Unit tests can be written specifically for mixin logic in isolation.
- **Compositional**: Users can include only the functionality they need by subclassing selectively.

This approach helps maintain a clean and extensible architecture in the PyMetric grid ecosystem.

Testing
-------

All grid classes, mixins, and core logic in PyMetric must be accompanied by a comprehensive test suite.
This ensures long-term stability and correctness of the grid infrastructure — especially in areas like chunking,
I/O metadata serialization, and ghost zone handling, where silent errors can easily propagate.

Tests are located in the ``/tests/test_grids`` directory. Each module or behavior (e.g., plotting, I/O, chunking) typically has
its own dedicated test file.

Test Guidelines
^^^^^^^^^^^^^^^

When contributing to or modifying grid infrastructure:

- **Every public method** should be tested, including inherited mixin functionality.
- Tests should validate both **interface correctness** (e.g., expected inputs/outputs) and **behavioral guarantees** (e.g., shape preservation, idempotence).
- Chunking and ghost zone tests should include edge cases like:
  - Minimal domain sizes
  - Chunk overlap at boundaries
  - Full-domain and subgrid equivalence
- I/O tests should confirm that:
  - Metadata is round-trip serializable via YAML, JSON, and HDF5
  - Reconstructed grids match original properties (e.g., shape, spacing, ghost extents)
- Plotting tests should use Matplotlib's testing utilities or image comparison if possible
- Utility mixins (e.g., `GridUtilsMixin`) should have pure unit tests and not depend on full grid construction

Recommended Test Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

The directory is organized roughly as follows:

.. code-block:: text

    tests/
    ├── test_grids/
    │   ├── test_grid_creation.py     ← tests for grid instantiation and configuration
    │   ├── test_grid_io.py          ← tests for grid I/O (YAML, JSON, HDF5)
    │   ├── utils.py                ← utility functions for test setup


Fixtures and Utilities
^^^^^^^^^^^^^^^^^^^^^^

Common fixtures (e.g., sample UniformGrid/GenericGrid instances) are defined in ``conftest.py`` and can be reused
across modules. This improves test isolation and helps maintain consistent setup for parametrized testing.

.. tip::

   When writing tests for new grid types:

   - Use `pytest.mark.parametrize` to validate against different shapes, spacings, and ghost zone sizes.
   - Always validate that the `repr`, `metadata`, and coordinate arrays reflect expected physical structure.
   - Avoid hardcoding assumptions about coordinate systems — instead use mock coordinate system instances if necessary.

.. warning::

   Do not test logic inside `GridBase` directly unless you're writing integration tests. The base class should remain abstract,
   and its features should be tested through concrete subclasses like `UniformGrid` or `GenericGrid`.

To run the grid tests independently:

.. code-block:: bash

   pytest tests/test_grids/

For full suite coverage including symbolic and field logic:

.. code-block:: bash

   pytest tests/

Following these practices helps ensure that PyMetric's grid infrastructure remains robust, extensible, and production-ready.
