.. _coordinates_user:

================================
Coordinate Systems: General Info
================================

PyMetric provides a flexible and extensible framework for defining and working with coordinate systems,
especially curvilinear coordinate systems used in scientific and engineering applications. The coordinate
systems are built on top of a symbolic foundation, allowing for advanced operations in both symbolic and
numerical form.

Coordinate systems in PyMetric are defined in the :py:mod:`coordinates` module and all
inherit from a common base, ensuring a consistent interface across systems. Each coordinate system represents a
curvilinear coordinate space (e.g., spherical, cylindrical, prolate spheroidal) and includes full support for:

1. Coordinate transformations
2. Metric and inverse metric tensor evaluations,
3. Symbolic and numeric differential operators (e.g., gradient, divergence, Laplacian),
4. Custom parameters and extensibility via subclassing.

These systems can be used in both analytical and simulation contexts, making them ideal for finite-difference,
spectral, or tensor calculus applications in custom geometries.

.. note::

    The underlying design goal of PyMetric is to ensure that

    1. Coordinate systems are easy to use.
    2. Coordinate systems are easily extensible to allow custom geometries.
    3. Coordinate systems are accurate in computation.

    In many cases, we have pursued **as efficient an implementation as possible**; however, efficiency is not
    the highest priority in this module. Therefore, coordinate systems and their differential operations are **not
    suitable for use in (for example) high resolution time dependent PDEs**, where calls to differential geometry
    functions would occur many 1000's of times. Instead, PyMetric is ideal for instances where a PDE needs
    to be solved on the order of 1 time in order to perform a necessary task.


.. contents::
   :local:
   :depth: 2

Overview
--------

Coordinate systems in PyMetric represent curvilinear geometries such as spherical, cylindrical,
and prolate spheroidal spaces. These coordinate systems provide a symbolic and numerical interface
to geometric quantities and operations, including:

- Coordinate transformations
- Metric and inverse metric tensors
- Jacobians and metric densities
- Symbolic and numerical differential operators
- Parameterized geometries

Coordinate systems are very useful in their own right; however, they are most commonly used in PyMetric
as part of the construction of a grid (see :ref:`grids`) or a field (see :ref:`fields`).

Each coordinate system class inherits from a common abstract base and defines:

1. **Axes**: Named coordinate directions (e.g., ``["r", "theta", "phi"]``).
2. **Symbolic infrastructure**: Metric tensors, derivatives, and other expressions are computed symbolically
   using `SymPy <https://www.sympy.org/>`__.
3. **Numerical evaluation**: Expressions are compiled into NumPy-compatible functions for high-performance evaluation
   on structured grids or unstructured inputs.
4. **Parameter support**: Some systems are parameterized (e.g., ellipsoidal focus distance ``a``),
   allowing for flexible instantiation of geometric families.
5. **Differential operators**: Each system defines methods for computing gradients, divergences,
   Laplacians, and other tensor calculus expressions in its own basis.

Coordinate systems in PyMetric are suitable for use in:

- Finite difference and finite volume solvers
- Symbolic exploration of curvilinear geometry
- Tensor calculus in custom geometries
- Evaluation of scalar or vector fields in native coordinates

.. hint::

    Coordinate systems are **not intended** for high-throughput time-stepping applications
    where millions of derivative evaluations are required per second. Instead, they are
    optimized for flexibility, clarity, and correctness in symbolic and semi-analytic workflows.

.. important::

    All coordinate systems in PyMetric share a consistent interface, and expose symbolic and
    numerical methods for working with geometry. This makes it easy to switch between
    geometries or extend the framework with custom systems.

Coordinate systems are defined in the :mod:`~coordinates` module, and typically subclass
either:

- :class:`~coordinates.core.OrthogonalCoordinateSystem` (for diagonal metric tensors)
- :class:`~coordinates.core.CurvilinearCoordinateSystem` (for full curvilinear geometries)

Constructing Coordinate Systems
-------------------------------

Coordinate systems in PyMetric are available in the :py:mod:`~coordinates` module. Each class represents
a specific curvilinear coordinate system, such as

- :class:`~coordinates.coordinate_systems.SphericalCoordinateSystem`: Spherical coordinates.
- :class:`~coordinates.coordinate_systems.CartesianCoordinateSystem2D`: 2D cartesian coordinates.
- :class:`~coordinates.coordinate_systems.CylindricalCoordinateSystem`: Cylindrical coordinates.

These classes provide symbolic and numerical support for differential geometry and coordinate
transformations, and can be directly instantiated as needed.

To create a coordinate system, import the desired class and instantiate it:

.. code-block:: python

    from pymetric.coordinates import (
        SphericalCoordinateSystem,
        ProlateSpheroidalCoordinateSystem
    )

    # Create a standard spherical coordinate system
    spherical = SphericalCoordinateSystem()

    # Create a prolate spheroidal system with a custom focal length
    prolate = ProlateSpheroidalCoordinateSystem(a=1.5)

Coordinate system instances are lightweight and behave like symbolic geometry containers. Once
created, they provide access to axes, symbolic tensors, and geometry-aware operations such as gradient
or Laplacian computations.

.. hint::

    PyMetric coordinate systems support both symbolic inspection and NumPy-compatible numerical evaluation.

Required Parameters
^^^^^^^^^^^^^^^^^^^

Some coordinate systems require parameters to define their shape or scaling. For example, the
:py:class:`~coordinates.coordinate_systems.ProlateSpheroidalCoordinateSystem` requires the focal
distance ``a`` as a parameter, which defines the spacing between the foci of the ellipsoids.

If parameters are not provided, default values are used:

.. code-block:: python

    cs1 = ProlateSpheroidalCoordinateSystem()        # uses a = 1.0 by default
    cs2 = ProlateSpheroidalCoordinateSystem(a=2.0)   # custom focal parameter

    print(cs1.parameters)
    {'a': 1.0}

    print(cs2.parameters)
    {'a': 2.0}

To inspect the current parameters of a coordinate system, use the
:py:attr:`~coordinates.core.CurvilinearCoordinateSystem.parameters` attribute.

Accessing Coordinate System Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each coordinate system exposes useful metadata via attributes:

- :py:attr:`~coordinates.core.CurvilinearCoordinateSystem.axes`:
  The logical axis names (e.g., ``["r", "theta", "phi"]``).
- :py:attr:`~coordinates.core.CurvilinearCoordinateSystem.ndim`:
  The dimensionality of the coordinate system.
- :py:attr:`~coordinates.core.CurvilinearCoordinateSystem.parameters`:
  Dictionary of any shape or transformation parameters.

This metadata is used throughout PyMetric to ensure consistency between coordinate systems,
grids, and differential operations.

.. code-block:: python

    cs = SphericalCoordinateSystem()
    print(cs.axes)         # ['r', 'theta', 'phi']
    print(cs.ndim)         # 3
    print(cs.parameters)   # {}


.. note::

    Some coordinate systems (especially those with nontrivial geometry) may emit logging messages
    during initialization. These messages provide information about expression parsing, symbolic
    expression caching, or internal warnings.

    You can configure or disable this output using the PyMetric logging tools via
    :py:mod:`~utilities.logging`.

Converting Between Coordinate Systems
-------------------------------------

PyMetric provides a unified and extensible API for converting coordinates between different coordinate systems.
All conversions are performed using Cartesian space as an intermediate representation:

.. code-block::

    native (source) → Cartesian → native (target)

This ensures generality and allows conversion between any pair of coordinate systems with matching dimensionality.

.. important::

    Coordinate systems must have the same number of dimensions to be convertible.

Basic Conversion
^^^^^^^^^^^^^^^^

Use the :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.convert_to` method to perform a one-shot conversion
between coordinate systems:

.. code-block:: python

    from pymetric.coordinates import SphericalCoordinateSystem, CylindricalCoordinateSystem

    sph = SphericalCoordinateSystem()
    cyl = CylindricalCoordinateSystem()

    # Convert from spherical to cylindrical coordinates
    r, theta, phi = 1.0, 3.14 / 2, 0.0
    rho, phi_cyl, z = sph.convert_to(cyl, r, theta, phi)

This method returns the native coordinates of the `target` system by first converting to Cartesian and then
to the destination system’s basis.

Creating Reusable Converters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To avoid repeating transformation logic, you can construct a reusable conversion function using
:py:meth:`~coordinates.core.CurvilinearCoordinateSystem.get_conversion_transform`:

.. code-block:: python

    transform = sph.get_conversion_transform(cyl)
    rho, phi_cyl, z = transform(1.0, 3.14 / 2, 0.0)

This is especially useful when you need to convert many points across different contexts, or
embed conversion logic into higher-level functions.

Conversion to/from Cartesian Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each coordinate system provides direct access to Cartesian conversion:

- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.to_cartesian` converts from native coordinates to Cartesian.
- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.from_cartesian` converts from Cartesian to native coordinates.

.. code-block:: python

    x, y, z = sph.to_cartesian(r, theta, phi)
    r2, theta2, phi2 = sph.from_cartesian(x, y, z)

These methods work with both scalar and array inputs, and are automatically vectorized using NumPy broadcasting.

Symbolic Manipulations
----------------------

Coordinate systems in PyMetric utilize a mixed design in which symbolic (CAS) based manipulations are favored for deriving
analytical quantities in the coordinate system (metrics, Christoffel Symbols, etc.) but then provides numerical access to
these quantities via efficient numpy conversion. The symbolic side of PyMetric coordinate systems is handled by
`SymPy <https://docs.sympy.org/latest/index.html>`__.

These symbolic representations form the foundation for both analytical exploration and numerical computations,
allowing you to derive differential operators like gradients or divergences while respecting the geometry
of the coordinate system.

Coordinate System Symbols
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a coordinate system class is created, its axes and parameters are converted into symbolic attributes which
are stored in the :py:attr:`~coordinates.core.CurvilinearCoordinateSystem.axes_symbols` and
:py:attr:`~coordinates.core.CurvilinearCoordinateSystem.parameter_symbols` attributes respectively.

.. code-block:: python

    cs = SphericalCoordinateSystem()
    print(cs.axes_symbols)
    [r, theta, phi]

These symbols are then fed into the class's methods in order to construct critical symbolic infrastructure
like the metric tensor, the inverse metric, etc.

The Metric Tensor
^^^^^^^^^^^^^^^^^

There are a number of symbolic attributes derived as part of class definition; however, the most important
is the metric tensor. The metric tensor is essential for performing a variety of differential operations and
is therefore present in every class. You can access the symbolic version of the attribute using
:py:attr:`~coordinates.core.CurvilinearCoordinateSystem.metric_tensor_symbol`

.. code-block:: python

    cs = SphericalCoordinateSystem()
    print(cs.metric_tensor_symbol)
    [1, r**2, r**2*sin(theta)**2]

.. note::

    Many of the coordinate systems defined in PyMetric are not only curvilinear, but are also
    orthogonal. In this case, the metric is **diagonal** and is therefore represented internally as a vector
    instead of a tensor. For classes like :py:class:`~coordinates.coordinate_systems.OblateHomoeoidalCoordinateSystem`,
    which are fully curvilinear, the output here is a true matrix.

The metric tensor is also available as a **numpy-like** numerical function:

.. code-block:: python

    cs = SphericalCoordinateSystem()
    cs.metric_tensor(1,np.pi/2,0)
    array([1., 1., 1.])

You can call the metric tensor function by simply passing arrays for each coordinate into the function.

Creating / Retrieving Derived Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyMetric supports derived expressions beyond the metric, such as:

1. Christoffel terms (for custom systems)
2. Coordinate Jacobians
3. System-specific auxiliary expressions

along with a few symbols which are of critical importance internally for differential
geometry operations (like the metric determinant). Regardless of which symbolic attribute
is of interest, it is **always possible** to access the attribute symbolically and numerically.

Attributes which are not implemented by default are called **derived attributes** and a list of
them can be accessed with

.. code-block:: python

    cs = OblateHomoeoidalCoordinateSystem(ecc=0.3)
    print(cs.list_expressions())
    ['Lterm', 'Dterm', 'metric_tensor', 'metric_density', 'inverse_metric_tensor']

If you want to retrieve a particular symbolic attribute, you can simply
use the :py:meth:`~coordinates.coordinate_systems.CurvilinearCoordinateSystem.get_expression` method.

.. code-block:: python

    cs = OblateHomoeoidalCoordinateSystem(ecc=0.3)
    print(cs.get_expression('metric_density'))
    sqrt(-xi**4*sin(theta)**2/(0.000729*sin(theta)**6 - 0.0243*sin(theta)**4 ^ 0.27*sin(theta)**2 - 1.0))

    cs = OblateHomoeoidalCoordinateSystem(ecc=0.0)
    print(cs.get_expression('metric_density'))
    sqrt(xi**4*sin(theta)**2)


Accessing Numerical Versions of Symbolic Expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All symbolic expressions can be turned into callable NumPy functions using:

.. code-block:: python

    fn = cs.get_numeric_expression("metric_density")
    val = fn(r=1.0, theta=np.pi/2, phi=0.0)

This process uses :py:func:`sympy.lambdify` under the hood, and allows fast evaluation over grids or datasets.

Class Level Expressions
^^^^^^^^^^^^^^^^^^^^^^^

Some expressions—like the metric tensor—are computed at the class level and
shared across all instances (symbolically). You can inspect or retrieve these
without instantiating the coordinate system:

.. code-block:: python

    from pymetric.coordinates.coordinate_systems import CylindricalCoordinateSystem

    g = CylindricalCoordinateSystem.get_class_expression("metric_tensor")
    print(g)

This is useful for inspecting or manipulating symbolic expressions analytically
before plugging in parameter values.

Coordinate System IO
---------------------

Coordinate systems in PyMetric can be serialized to and from disk using multiple formats. This enables
persistent storage of geometric configurations and facilitates reuse across workflows or between simulation contexts.

PyMetric supports serialization in the following formats:

- **HDF5** (via :meth:`~coordinates.core.CurvilinearCoordinateSystem.to_hdf5`, :meth:`~coordinates.core.CurvilinearCoordinateSystem.from_hdf5`)
- **JSON** (via :meth:`~coordinates.core.CurvilinearCoordinateSystem.to_json`, :meth:`~coordinates.core.CurvilinearCoordinateSystem.from_json`)
- **YAML** (via :meth:`~coordinates.core.CurvilinearCoordinateSystem.to_yaml`, :meth:`~coordinates.core.CurvilinearCoordinateSystem.from_yaml`)

Each serialization method stores only the minimal state needed to reconstruct the coordinate system:

1. The class name (used to locate the appropriate constructor)
2. The parameter dictionary used at instantiation (see :attr:`~coordinates.core.CurvilinearCoordinateSystem.parameters`)

Deserialization uses a **registry** to map class names to constructors.

HDF5 Serialization
^^^^^^^^^^^^^^^^^^

Coordinate systems can be saved to HDF5 with:

.. code-block:: python

    cs.to_hdf5("path/to/file.h5", group_name="geometry", overwrite=True)

This stores data either at the root level or in the named group of the HDF5 file.

To restore:

.. code-block:: python

    from pymetric.coordinates import CurvilinearCoordinateSystem
    cs = CurvilinearCoordinateSystem.from_hdf5("path/to/file.h5", group_name="geometry")

The following methods are available:

- :meth:`~coordinates.core.CurvilinearCoordinateSystem.to_hdf5`
- :meth:`~coordinates.core.CurvilinearCoordinateSystem.from_hdf5`

JSON and YAML Serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

JSON and YAML provide readable text-based representations. They are ideal for configuration files, version control,
and interlanguage workflows.

Save to JSON or YAML:

.. code-block:: python

    cs.to_json("coordsys.json")
    cs.to_yaml("coordsys.yaml")

Load from these files:

.. code-block:: python

    cs = CurvilinearCoordinateSystem.from_json("coordsys.json")
    cs = CurvilinearCoordinateSystem.from_yaml("coordsys.yaml")

These formats use:

- :meth:`~coordinates.core.CurvilinearCoordinateSystem.to_json`
- :meth:`~coordinates.core.CurvilinearCoordinateSystem.from_json`
- :meth:`~coordinates.core.CurvilinearCoordinateSystem.to_yaml`
- :meth:`~coordinates.core.CurvilinearCoordinateSystem.from_yaml`

.. note::

    Only parameters explicitly listed in :attr:`~coordinates.core.CurvilinearCoordinateSystem.__PARAMETERS__` are serialized.
    Derived symbolic expressions (e.g., metric tensor, density) are automatically recomputed upon loading.

.. hint::

    PyMetric automatically converts NumPy types (e.g., ``np.float64``, ``np.ndarray``) to native Python types
    before serialization to JSON or YAML to ensure compatibility.

Registry Handling
^^^^^^^^^^^^^^^^^

Deserialization requires a mapping of class names to types. By default, each coordinate system uses its
own registry (see :attr:`~coordinates.core.CurvilinearCoordinateSystem.__DEFAULT_REGISTRY__`), but you can provide your own:

.. code-block:: python

    registry = {"CustomSystem": CustomSystem}
    cs = CurvilinearCoordinateSystem.from_json("coordsys.json", registry=registry)

This enables loading of user-defined systems or systems not yet imported into the session.
