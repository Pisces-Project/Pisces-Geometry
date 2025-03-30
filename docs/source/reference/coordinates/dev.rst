.. _coordinates_dev:

===================================
Building Custom Coordinate Systems
===================================

Pisces-Geometry provides a flexible framework for defining and working with custom coordinate systems using symbolic and
numerical tools. Whether you're modeling simple orthogonal coordinates or building more complex curvilinear systems, the
coordinate system module enables precise geometric and differential computations. This guide is intended for developers
who want to implement their own coordinate systems by subclassing the appropriate base classes.

Pisces-Geometry is designed around **curvilinear, Euclidean coordinate systems**. It supports:

- Arbitrary symbolic metric tensors.
- Symbolic differentiation, simplification, and substitution.
- Raising/lowering indices, computing gradients, divergence, Laplacians, and Jacobians.
- Conversion to and from Cartesian coordinates (you provide the mapping).

At the heart of the Pisces-Geometry coordinate handling infrastructure is the :py:class:`~pisces_geometry.coordinates.base.CoordinateSystemBase`,
which corresponds to a **generic, curvilinear coordinate system**. There are also a few subclasses of :py:class:`~pisces_geometry.coordinates.base.CoordinateSystemBase`
already built into the infrastructure corresponding to various simplifying cases:

1. :py:class:`~pisces_geometry.coordinates.base._OrthogonalCoordinateSystemBase`: provides support for orthogonal coordinates
   which allow for a simplification of the logic necessary for curvilinear coordinates.

**Limitations**:

- **No support for non-Euclidean spaces**: The system does not currently handle Riemannian or pseudo-Riemannian geometries (e.g., general relativity).
- **Curvilinear only**: The coordinate systems must be defined using smooth, differentiable transformations. Discrete or piecewise geometries are not supported.
- **No automatic embedding**: The system does not auto-detect whether your coordinates live in 2D, 3D, or higher-dimensional Cartesian space — you must define these mappings explicitly.

Coordinate System Structural Principles
---------------------------------------

Before discussing the practicalities of subclassing coordinate system classes, it's important to understand some of the
core design principles that underlie the pisces_geometry.coordinates module. Working with curvilinear coordinates —
particularly when performing differential operations like gradients, divergences, or Laplacians — introduces substantial
complexity. These operations must take into account the geometry of the coordinate system itself, which makes it easy for
implementations to become tightly coupled to specific coordinate systems.

Pisces-Geometry takes a different approach. Instead of building differential logic directly into each coordinate system,
it separates coordinate structure from coordinate behavior. This leads to a unified, extensible, and highly symbolic
framework for working with general curvilinear geometries. The goals of this are as follows:

1. Create a framework for physical modeling which allows for developers to easily utilize any coordinate system relevant
   to their problem without having to work out the details of differential operations.
2. To create a robust system for identifying symmetry breakage during differential operations.

Why a Unified System?
'''''''''''''''''''''

Most traditional implementations hard-code differential operators for each coordinate system (e.g., hardwiring the cylindrical gradient).
This is error-prone, difficult to extend, and virtually impossible to scale to arbitrary coordinate systems.

Instead, Pisces-Geometry treats the metric tensor as the fundamental source of geometric information. Once the metric is
known, all differential operations — gradients, divergence, Laplacians, index manipulation — can be derived from it automatically.

This means:

- You only need to define the metric tensor (and optionally coordinate transforms).
- The system handles the rest — differential operators, basis adjustments, and symbolic simplification.
- The same code works identically in any coordinate system.

This design allows all downstream geometry-aware code to remain completely coordinate system agnostic.
A gradient in spherical coordinates is computed using the same logic as one in elliptical coordinates —
because the underlying differential geometry is derived from the metric.

The Symbolic / Numerical Duality
''''''''''''''''''''''''''''''''

A key part of this system is its reliance on symbolic mathematics through `sympy <docs.sympy.org>`_, which allows Pisces-Geometry to:

- Build symbolic expressions for differential operators, tensors, and coordinate transformations.
- Substitute parameter values and simplify expressions at runtime.
- Convert symbolic expressions to fast numerical functions via automatic lambdification.

Every coordinate system has two parallel representations:

- A **symbolic form**, used for introspection, algebraic manipulation, expression building, and caching.
- A **numerical form**, used for evaluation on grids, fields, and during simulation.

This dual representation ensures that the system is both flexible (symbolic) and efficient (numerical). Once you've defined
the symbolic structure of a coordinate system, Pisces-Geometry takes care of efficiently executing computations without
sacrificing generality.

Subclass Structure
------------------

When defining a new coordinate system in Pisces-Geometry, the first and most important decision is which base class
to inherit from. Pisces provides two primary options:

1. :py:class:`~pisces_geometry.coordinates.base.CoordinateSystemBase`
   This class should be used when building general curvilinear coordinate systems. If your metric tensor includes off-diagonal
   elements (i.e., the system is not orthogonal), this is the appropriate choice. It provides full flexibility and
   requires you to define both the metric and inverse metric explicitly.
2. :py:class:`~pisces_geometry.coordinates.base._OrthogonalCoordinateSystemBase`
   This class is a specialized version of CoordinateSystemBase that simplifies implementation for orthogonal coordinate
   systems — those with diagonal metric tensors. When using this base class, you only need to define the scale factors (diagonal elements of the metric tensor),
   the inverse metric is computed automatically, and tensor algebra operations are more efficient due to diagonal simplifications.

**Which One Should I Use?**

Use the following table as a guide:

+----------------------------------------+------------------------------------------------------------------------------+
| Your Coordinate System Is...           |                              Subclass From...                                |
+========================================+==============================================================================+
| Orthogonal (e.g., cylindrical, polar)  |:py:class:`~pisces_geometry.coordinates.base._OrthogonalCoordinateSystemBase` |
+----------------------------------------+------------------------------------------------------------------------------+
|Has off-diagonal metric terms           |:py:class:`~pisces_geometry.coordinates.base.CoordinateSystemBase`            |
+----------------------------------------+------------------------------------------------------------------------------+
| Requires full control over tensors     |:py:class:`~pisces_geometry.coordinates.base.CoordinateSystemBase`            |
+----------------------------------------+------------------------------------------------------------------------------+
| Has a diagonal metric and is 2D/3D     |:py:class:`~pisces_geometry.coordinates.base._OrthogonalCoordinateSystemBase` |
+----------------------------------------+------------------------------------------------------------------------------+

In either case, your subclass will need to define the symbolic metric tensor and the coordinate transformation logic.
If the coordinate system is orthogonal, the orthogonal base class will take care of several tedious details (like raising/lowering tensor indices efficiently).

In the following sections, we'll walk through how to set up a subclass using either approach.

Setting the Class Parameters
++++++++++++++++++++++++++++

When creating a new coordinate system subclass, you must define a set of class-level attributes that specify how the
coordinate system behaves. These attributes control initialization, dimensionality, parameter handling, and symbolic expression generation.

Required Class Attributes
'''''''''''''''''''''''''

- ``__is_abstract__``: ``bool``
  Indicates whether the class is abstract. Set this to False for any subclass intended to be instantiated.
  If True, the metaclass will skip validation and symbolic setup.

- ``__AXES__`` : ``List[str]``
  A list of axis names representing the coordinate variables (e.g., ``['r', 'theta', 'z']``).
  This defines the dimension of the coordinate system and is used to generate symbolic axis variables.

- ``__PARAMETERS__`` : ``Dict[str, Any]``
  A dictionary mapping parameter names to their default values (e.g., ``{'scale': 1.0}``).
  These are user-provided keyword arguments when instantiating the coordinate system and will be substituted into symbolic expressions.

- ``__setup_point__`` : ``'init' | 'import'``
  Specifies when the symbolic expressions (e.g., metric tensor, class expressions) are computed:

  - ``'init'`` (default): Wait until the class is instantiated.
  - ``'import'``: Build expressions at module import time. This can slow down import but may reduce startup time in some applications.

As an example, the following is the first few lines of the :py:class:`~pisces_geometry.coordinates.coordinate_systems.SphericalCoordinateSystem`
implementation:

.. code-block:: python

    class SphericalCoordinateSystem(_OrthogonalCoordinateSystemBase):
        __is_abstract__ = False
        __setup_point__ = 'init'
        __AXES__ = ['r','theta','phi']
        __PARAMETERS__ = {}

.. note::

    **Development Standard**: If you are developing a new coordinate system for use in the Pisces-Geometry core code,
    it should use ``_setup_point__ = 'init'`` in almost any case (unless there is specific justification). By allowing
    all of the built-in coordinate systems to setup on import, there is a large computation overhead which delays import
    speed.

Behind the Scenes
'''''''''''''''''

When a subclass is instantiated, the following steps occur:

1. The **metaclass** verifies that required attributes are present and ensures that the structure of all of the
   coordinate systems in the package are valid. If there is something wrong in this step, an error will be raised on
   import.
2. The class remains **partially initialized** until the user **instantiates it for the first time**.
3. The system generates symbolic axis symbols and parameter symbols using :py:class:`sympy.core.symbol.Symbol`.
4. The metric tensor and inverse metric tensor are constructed using user-defined logic.
5. Any registered class expressions (see :ref:Class Expressions) are discovered and stored for lazy evaluation.
6. Parameter values passed during instantiation (or taken from defaults) are substituted into symbolic expressions to create instance-level expressions and callables.


Setting up Conversion Standards
+++++++++++++++++++++++++++++++

All coordinate system classes in Pisces-Geometry must define how to convert between the native coordinate system and
standard Cartesian coordinates. This is especially important for visualization, interoperation with external tools, and validating geometric behavior numerically.

To support this, your subclass must implement two methods:

1. :py:meth:`~pisces_geometry.coordinate_systems.base.CoordinateSystemBase._convert_native_to_cartesian`
   Converts native coordinate variables (e.g., r, theta, z) to Cartesian coordinates (x, y, z).
2. :py:meth:`~pisces_geometry.coordinate_systems.base.CoordinateSystemBase._convert_cartesian_to_native`
   Converts from Cartesian coordinates back to your system’s native coordinates.

Each of these functions should take ``self`` and ``x_1,x_2,x_3,...`` where each ``x`` corresponds to a coordinate
of the coordinate system. It should return a tuple of values ``z_1,z_2,...`` corresponding to the converted values.

.. warning::

    It is important to ensure that your computations behave naturally for vectorized inputs. Thus, if ``x,y,z`` are each
    the same size, so to should be the output ``u,v,w``.

**Example**:

.. code-block:: python

    class SphericalCoordinateSystem(_OrthogonalCoordinateSystemBase):
        __is_abstract__ = False
        __setup_point__ = 'init'
        __AXES__ = ['r','theta','phi']
        __PARAMETERS__ = {}

        def _convert_cartesian_to_native(self, x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

            return r,theta,phi


        def _convert_native_to_cartesian(self, r, theta, phi):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            return x,y,z

Setting up The Metric
+++++++++++++++++++++

Every coordinate system in Pisces-Geometry must define a metric tensor, which encodes how distances and derivatives are
computed. The metric defines the inner product structure of the space, and is central to computing gradients, divergence,
Laplacians, and performing index manipulations.

Pisces supports both general curvilinear and orthogonal coordinate systems. The structure of the metric depends on which type you are building.

If you are subclassing from :py:class:~pisces_geometry.coordinates.base.CoordinateSystemBase, you must implement both of the following:

- ``@staticmethod def __construct_metric_tensor_symbol__(*args, **kwargs) -> sp.Matrix``
- ``@staticmethod def __construct_inverse_metric_tensor_symbol__(*args, **kwargs) -> sp.Matrix``

If you subclass from :py:class:`~pisces_geometry.coordinates.base._OrthogonalCoordinateSystemBase`, you only need to define
the diagonal elements of the metric tensor — that is, the scale factors squared. The inverse metric will be computed automatically as ``1 / g[i]``.

You must implement:

- ``@staticmethod def __construct_metric_tensor_symbol__(*args, **kwargs) -> sp.Array``

These methods receive:

- ``*args``: positional arguments representing the symbolic axis variables (e.g., ``r, theta, z``).
- ``**kwargs``: keyword arguments representing symbolic parameters (e.g., ``scale=Symbol('scale')``).

They should return a full SymPy matrix representing the metric tensor (or its inverse).

.. note::

    **What's happening internally?**

    During class setup (either at import time or instantiation time, depending on ``__setup_point__``), the metric tensor is:

    - Constructed symbolically using the method(s) above.
    - Stored as ``cls.__class_metric_tensor__`` and ``cls.__class_inverse_metric_tensor__``.

    Once a user instantiates the class, the ``__class_metric_tensor__`` has its parameters substituted for the true
    values of the parameters to create the ``__metric_tensor_expression__`` attribute. This is then converted to a numerical
    function which is accessible by :py:attr:`~pisces_geometry.coordinates.base.CoordinateSystemBase.metric_tensor`.

**Example**:

.. code-block:: python

    class SphericalCoordinateSystem(_OrthogonalCoordinateSystemBase):
        __is_abstract__ = False
        __setup_point__ = 'init'
        __AXES__ = ['r','theta','phi']
        __PARAMETERS__ = {}

        @staticmethod
        def __construct_metric_tensor_symbol__(r,theta,phi,**kwargs):
            return sp.Array([1,r**2,(r*sp.sin(theta))**2])

        def _convert_cartesian_to_native(self, x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

            return r,theta,phi


        def _convert_native_to_cartesian(self, r, theta, phi):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            return x,y,z


Extending Functionality
-----------------------
While defining a new coordinate system typically involves specifying just the metric, parameters, and coordinate transforms,
Pisces-Geometry provides many extension points for more advanced functionality.

You might consider extending the coordinate system class if:

- You want to define custom expressions (e.g., Jacobians, scale factors, special transformation operators).
- You want to implement custom tensor operators unique to your coordinate system.
- You want to add parameterized behaviors (e.g., boundary-aware metrics, scaling functions, anisotropy).
- You want to define alternate coordinate bases or embed coordinate systems into higher-dimensional manifolds.

Pisces-Geometry is designed to be modular and override-friendly. Any method defined on a subclass can be overridden in
your coordinate system, as long as the expected structure is maintained.

Additionally, you can extend coordinate systems to interact with external systems (e.g., visualization tools, mesh generators, simulation frameworks) by exposing additional utility methods.

Class Expressions
+++++++++++++++++

In many coordinate systems, it's helpful to define reusable symbolic expressions — like Jacobians, divergence terms,
or scale factors. Pisces-Geometry provides a decorator-based mechanism to define these as **class expressions**.

To define a class expression, decorate a ``classmethod`` using ``@class_expression`` (:py:func:`~pisces_geometry.coordinates.base.class_expression`).

**Example**:

.. code-block:: python

    class MyCoordinateSystem(CoordinateSystemBase):
        __AXES__ = ['x', 'y']
        __PARAMETERS__ = {}

        @staticmethod
        def __construct_metric_tensor_symbol__(x, y):
            return sp.Matrix([[1, 0], [0, x**2 + y**2]])

        @staticmethod
        def __construct_inverse_metric_tensor_symbol__(x, y):
            return sp.Matrix([[1, 0], [0, 1 / (x**2 + y**2)]])

        @class_expression(name='jacobian')
        @classmethod
        def _jacobian(cls, x, y):
            return sp.sqrt(sp.det(cls.__class_metric_tensor__))

Adding Methods
++++++++++++++

Coordinate systems in Pisces-Geometry are fully extensible Python classes. You are free to define any instance or class methods that help support your use case. This might include:

- Utility functions for working with particular axes.
- Projection or slicing routines.
- Analytical identities or symmetries.
- Shape checks or domain constraints.
- Integrations with other Pisces modules.

These methods will have full access to the symbolic structure of the system — including parameters, symbolic axes, metric tensors, and coordinate transformations.
