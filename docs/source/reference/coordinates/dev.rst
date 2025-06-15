.. _coordinates-dev:

=======================================================
Coordinate Systems: Developer Documentation
=======================================================

This guide is intended for contributors and maintainers working on the coordinate system
infrastructure in **PyMetric**. Coordinate systems form the geometric backbone of all spatial
operations in the library, including grid generation, symbolic differential geometry, and
coordinate-space transformations.

This document provides:

- An overview of the coordinate system class hierarchy and mixin architecture
- Guidelines for writing and extending coordinate system classes
- Documentation on mixins for transformation logic, symbolic math, and I/O
- Best practices for testing, contribution, and implementation boundaries

It is recommended reading for anyone contributing new coordinate systems or modifying existing ones.

.. important::

    If you're interested in contributing, please submit pull requests or open issues via our GitHub repository:

    `Pisces-Project/PyMetric <https://github.com/Pisces-Project/PyMetric>`_

    Contributions are welcome in the form of:

    - Bug reports and fixes
    - New coordinate systems (e.g., oblate spheroidal, log-polar)
    - Enhancements to symbolic geometry or transformation logic
    - Improvements to documentation and test coverage

    For instructions on setting up your development environment, building the documentation,
    and running the test suite, please refer to the :ref:`quickstart`.

.. note::

    All contributions must follow the internal API and type annotations established in
    :mod:`coordinates.mixins._typing`, which defines the formal interfaces for all coordinate system components.


Coordinate Systems: Overview
----------------------------

Coordinate systems in PyMetric define the mathematical and geometric structure used by grids,
differential geometry routines, and physical field representations. All coordinate systems must
inherit from the abstract protocol defined in :class:`~coordinates.base._CoordinateSystemBase`.

This base class defines the public API and required behaviors for all coordinate system implementations,
including:

- Axis metadata (names, labels, units)
- Coordinate transformation interfaces
- Support for symbolic operations
- Compatibility with structured grids and field operators

.. note::

    Coordinate systems do **not** manage data arrays, units, or grid values directly.
    Their role is strictly geometric — they define the *shape* of space, not the contents of it.

Coordinate System Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyMetric supports two main categories of coordinate systems:

- :class:`~coordinates.core.CurvilinearCoordinateSystem`
- :class:`~coordinates.core.OrthogonalCoordinateSystem`

The distinction is as follows:

- **CurvilinearCoordinateSystem**: Allows general non-orthogonal metrics, including shearing and angular coupling.
  This class supports arbitrary Riemannian metrics and is the base for symbolic tensor operations.
- **OrthogonalCoordinateSystem**: Specializes the curvilinear case by assuming a diagonal metric tensor.
  These systems are more efficient for symbolic operations and finite difference stencils.

All concrete coordinate systems (e.g., Cartesian, Cylindrical, Spherical) are implemented as subclasses of one of these two.
They live in the :mod:`coordinates.coordinate_systems` module.

Each subclass must define:

- Axis names and labels
- Dimension (`ndim`)
- Coordinate transformation methods
- Metric components (either symbolic or numerical)

Mixin Classes
^^^^^^^^^^^^^

Coordinate system classes are assembled using a modular mixin system. This enables independent development
of transformation logic, symbolic operations, axis metadata, and I/O support without polluting the core base class.

All mixins reside in the :mod:`coordinates.mixins` package and are grouped by purpose.

Available mixins include:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mixin Class
     - Description

   * - :class:`~coordinates.mixins.coords.CoordinateOperationsMixin`
     - Implements coordinate transformation logic, including cartesian-to-native and native-to-cartesian conversions.
       Defines the public `.to_cartesian()` and `.from_cartesian()` methods used by grids and fields.

   * - :class:`~coordinates.mixins.core.CoordinateSystemCoreMixin`
     - Provides common metadata interfaces and structural features such as `ndim`, `axes`, and axis name resolution.
       This is the default source of axis name/label properties and `__repr__`.

   * - :class:`~coordinates.mixins.core.CoordinateSystemIOMixin`
     - Adds metadata serialization logic (e.g., `.to_metadata_dict()` and `.from_metadata_dict()`), used for saving/loading
       coordinate system configurations.

   * - :class:`~coordinates.mixins.core.CoordinateSystemAxesMixin`
     - Supplies helper methods for indexing axes by name, constructing masks over selected coordinates,
       and validating shape compatibility for transformations.

   * - :class:`~coordinates.mixins.mathops.CoordinateSystemMathMixin`
     - Defines symbolic tensor and differential geometry operations (e.g., computing gradients, divergence,
       Christoffel symbols, and curvature) based on the coordinate system’s metric structure.
       Required for all systems supporting symbolic operations.

Mixin Type Protocols
++++++++++++++++++++

To enable clean type-checking and cross-compatibility, all mixin interfaces are formalized in the
:mod:`coordinates.mixins._typing` module.

If you are extending or using mixin-dependent logic, you should reference these `Protocol` classes
rather than the mixins themselves for proper type inference. For example:

.. code-block:: python

   from coordinates.mixins._typing import SupportsCoordinateOperations

   def uses_transform(cs: SupportsCoordinateOperations):
       cart = cs.to_cartesian(...)
       ...

These protocols enable your own coordinate system implementations or mixin extensions to remain type-safe
and compatible with downstream tools.


Custom Coordinate Systems
--------------------------

If you want to define a new coordinate system in PyMetric, begin by reading the user guide at
:ref:`coordinates_building`. This guide outlines how symbolic and numerical tools are used together
to construct a new coordinate space.

At a high level, all custom coordinate systems must subclass from one of the two public base classes in
:mod:`coordinates.core`:

- :class:`~coordinates.core.OrthogonalCoordinateSystem` — for diagonal metric tensors
- :class:`~coordinates.core.CurvilinearCoordinateSystem` — for arbitrary curvilinear metrics

These subclasses inherit from the internal protocol :class:`~coordinates.base._CoordinateSystemBase`, which
defines the full symbolic and numeric interface for all coordinate systems.

Coordinate systems are declared using a symbolic-first model:

- You define symbolic axes, parameters, and a symbolic metric tensor.
- The system computes all required differential geometry from these definitions.
- You optionally provide Cartesian transformation logic for I/O or geometry compatibility.

Coordinate classes are constructed modularly using mixins, and may define additional methods or override base behavior.
Coordinate systems do **not** handle field values, units, or simulation-specific logic — their scope is limited to geometry.

To create a new coordinate system:

1. Choose a base class depending on whether your metric tensor is diagonal or not.
2. Subclass from it and define the required class attributes:

   - ``__AXES__``: coordinate names
   - ``__PARAMETERS__``: any symbolic parameters
   - ``__construct_metric_tensor_symbol__`` (and optionally its inverse)

3. Optionally define conversion methods to and from Cartesian coordinates:

   - :meth:`~coordinates.base._CoordinateSystemBase._convert_native_to_cartesian`
   - :meth:`~coordinates.base._CoordinateSystemBase._convert_cartesian_to_native`

4. Optionally define symbolic helper expressions using :func:`~coordinates.base.class_expression`.

The coordinate system will automatically expose all metric-dependent properties,
such as gradients, Laplacians, and basis vector representations via the symbolic geometry infrastructure.

.. note::

    For a detailed walkthrough of the required attributes and symbolic construction process, see:
    :ref:`coordinates_building`.


Expanding Coordinate System Functionality
------------------------------------------

Coordinate System Scope
^^^^^^^^^^^^^^^^^^^^^^^^^

Coordinate systems in PyMetric are designed to encode **geometric structure**, not simulation or numerical behavior.
To preserve modularity and clarity, extensions to coordinate systems should remain within their proper domain.

**Coordinate systems should...**

- Implement methods for transforming between native and Cartesian coordinates.
- Define or expose symbolic properties of the space (e.g., scale factors, Jacobians, parameterized tensors).
- Provide interfaces to symbolic differential geometry (via the metric tensor).
- Register reusable symbolic expressions (via class expressions).

**Coordinate systems should NOT...**

- Handle numerical field data or discretization behavior — that belongs in grids or field classes.
- Handle units — unit tracking is performed by buffer and field layers.
- Perform low-level numerical math — elementwise math and vector calculus operations are dispatched from higher layers.

In short, coordinate systems provide **symbolic structure**. Other components in the PyMetric
stack handle **numeric evaluation** and **domain-specific logic**.

Where to Put New Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

When adding new behavior to the coordinate system module, consider scope and reuse. Use the following guidelines:

- **In a subclass**, if the behavior is highly specific or only relevant for one coordinate system.

  Example: A method computing the magnetic field geometry of a tokamak should live in the custom subclass for that geometry.

- **In a mixin**, if the method is general-purpose and could be used by multiple coordinate systems.

  Example: Methods for vectorized coordinate warping, symbolic axis combinations, or caching symbolic derivatives
  should be defined in one of the existing mixins (see Mixin Classes).

- **In a core class** (e.g., :class:`~coordinates.core.OrthogonalCoordinateSystem`) if the method is applicable to
  *all* coordinate systems of that class type and is integral to the way they are defined.

  These methods may also support internal behaviors expected by other parts of the library (e.g., symbolic
  simplification hooks or validation methods).

- **In the protocol base class** (:class:`~coordinates.base._CoordinateSystemBase`) only if the method is architectural — i.e.,
  if it defines a part of the interface contract for all coordinate systems.

  These methods define abstract hooks or system-wide expectations (e.g., symbolic setup behavior, transformation
  interface contracts) and should remain stable.

.. important::

   Never add logic to coordinate systems that duplicates functionality from symbolic geometry modules,
   grids, or field classes. Use delegation and dependency instead.

Testing
--------

All coordinate system classes and related functionality must be accompanied by unit tests.
These tests live in the ``/tests/test_coordinates`` directory and are essential for ensuring correctness,
maintainability, and stability as the symbolic infrastructure evolves.

Each coordinate system should have a dedicated test module or class that validates its behavior, including:

- Metric tensor correctness (symbolic and numerical forms)
- Inverse metric validation
- Coordinate transformation accuracy
- Parameter substitution and expression generation
- Class expressions (e.g., Jacobians, basis vectors)
- Edge cases in evaluation (e.g., zero radius, pole singularities)

.. important::

    All methods — especially those involving symbolic logic or numerical evaluation — must have corresponding tests.

    Coordinate systems must pass tests **both at the class level** (symbolic structure) and
    **at the instance level** (numerical behavior).

To get started, see the README in ``tests/test_coordinates/`` for organizational guidance and available testing utilities.
Most test modules use `pytest <https://docs.pytest.org>`_ and may rely on fixtures from ``conftest.py`` for reusability.

.. tip::

    When adding a new coordinate system:

    - Add symbolic validation tests in a file like ``test_my_coords_symbolic.py``.
    - Add numerical evaluation tests using standard NumPy arrays in ``test_my_coords_numerical.py``.
    - Include regression tests for any specialized logic (e.g., anisotropy, constraints).

Following these practices ensures that your contributions remain robust and compatible with PyMetric’s growing coordinate infrastructure.
