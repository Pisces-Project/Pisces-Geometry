.. _coordinates_user:

=============================
Using Coordinate Systems
=============================

Pisces-Geometry provides a flexible and extensible framework for defining and working with coordinate systems,
especially curvilinear coordinate systems used in scientific and engineering applications. The coordinate
systems are built on top of a symbolic foundation, allowing for advanced operations in both symbolic and
numerical form.

Coordinate systems in Pisces are defined in the :py:mod:`coordinates` module and all
inherit from a common base, ensuring a consistent interface across systems. Each coordinate system represents a
curvilinear coordinate space (e.g., spherical, cylindrical, prolate spheroidal) and includes full support for:

1. Coordinate transformations
2. Metric and inverse metric tensor evaluations,
3. Symbolic and numeric differential operators (e.g., gradient, divergence, Laplacian),
4. Custom parameters and extensibility via subclassing.

These systems can be used in both analytical and simulation contexts, making them ideal for finite-difference,
spectral, or tensor calculus applications in custom geometries.

.. note::

    The underlying design goal of Pisces-Geometry is to ensure that

    1. Coordinate systems are easy to use.
    2. Coordinate systems are easily extensible to allow custom geometries.
    3. Coordinate systems are accurate in computation.

    In many cases, we have pursued **as efficient an implementation as possible**; however, efficiency is not
    the highest priority in this module. Therefore, coordinate systems and their differential operations are **not
    suitable for use in (for example) high resolution time dependent PDEs**, where calls to differential geometry
    functions would occur many 1000's of times. Instead, Pisces-Geometry is ideal for instances where a PDE needs
    to be solved on the order of 1 time in order to perform a necessary task.

Coordinate System Basics
--------------------------------

Coordinate systems in Pisces-Geometry are available in the :py:mod:`coordinates` module. Each class represents a
specific curvilinear coordinate system, such as spherical or cylindrical coordinates. These coordinate systems
support symbolic geometry, conversion functions, and numerical differential operators.

To import and create a coordinate system, simply import and instantiate it from the module:

.. code-block:: python

    from pisces_geometry.coordinates import (SphericalCoordinateSystem,
     ProlateSpheroidalCoordinateSystem)

    # The spherical coordinate system doesn't need any parameters to initialize.
    # We can simply call __init__.
    coordinate_system = SphericalCoordinateSystem()

    # Prolate needs the a= parameter.
    ellipsoidal_coordinate_system = ProlateSpheroidalCoordinateSystem(a=1)

Depending on the coordinate system you've initialized, you may see some logging information; we'll explain all that in
the next few sections.

.. hint::

    You can interact directly with the logger at :py:mod:`utilities.logging`.

Coordinate systems behave like lightweight geometry containers. Once instantiated,
they allow access to various properties like axes, symbolic metric tensors, and differential operations
(e.g., gradient or Laplacian) that reflect the system’s underlying geometry.

Parameters
++++++++++

Some coordinate systems require or support parameters during initialization.
These parameters control aspects of the geometry, such as scaling factors or shape constants.

For example, a `prolate spheroidal coordinate system <https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates>`_
might require a focus parameter :math:`a` which determines how eccentric the ellipsoids are.

These parameters can be fed into the coordinate system when you create it. If you don't give a coordinate system
the parameters it requires, the default values will be filled in.

.. code-block:: python

    cs1 = ProlateSpheroidalCoordinateSystem()
    cs2 = ProlateSpheroidalCoordinateSystem(a=2.0)
    print(cs1.parameters,cs2.parameters)
    {'a': 1.0}, {'a': 2.0}

You can always access the parameters for your coordinate system using the
:py:attr:`~coordinates.core.CurvilinearCoordinateSystem.parameters` attribute.

Converting To / From Cartesian Coordinates
+++++++++++++++++++++++++++++++++++++++++++

All coordinate systems provide methods to convert between native (curvilinear) and Cartesian coordinates:

- ``_convert_native_to_cartesian(*coords)`` converts native coordinates → Cartesian.
- ``_convert_cartesian_to_native(*coords)`` converts Cartesian → native coordinates.

.. code-block:: python

    from pisces_geometry.coordinates.coordinate_systems import CylindricalCoordinateSystem
    import numpy as np

    cs = CylindricalCoordinateSystem()
    r, theta, z = 1.0, np.pi / 2, 2.0
    x, y, z_cart = cs._convert_native_to_cartesian(r, theta, z)

    # Convert back
    r2, theta2, z2 = cs._convert_cartesian_to_native(x, y, z_cart)

These methods can be used on scalars, arrays, or grids. They work seamlessly with NumPy arrays for vectorized
transformations.

Converting To Other Coordinates
++++++++++++++++++++++++++++++++

To convert between two curvilinear coordinate systems (e.g., spherical → cylindrical), use Cartesian space
as an intermediary:

.. code-block:: python

    sph = SphericalCoordinateSystem()
    cyl = CylindricalCoordinateSystem()

    r, theta, phi = 1.0, np.pi / 2, 0.0
    x, y, z = sph._convert_native_to_cartesian(r, theta, phi)
    rho, phi_cyl, z_cyl = cyl._convert_cartesian_to_native(x, y, z)

.. note::

    In future releases, this transformation logic will be made more explicit.

Symbolic Manipulations
----------------------

Coordinate systems in Pisces utilize a mixed design in which symbolic (CAS) based manipulations are favored for deriving
analytical quantities in the coordinate system (metrics, Christoffel Symbols, etc.) but then provides numerical access to
these quantities via efficient numpy conversion. The symbolic side of Pisces-Geometry coordinate systems is handled by
`SymPy <https://docs.sympy.org/latest/index.html>`_.

These symbolic representations form the foundation for both analytical exploration and numerical computations,
allowing you to derive differential operators like gradients or divergences while respecting the geometry
of the coordinate system.

Coordinate System Symbols
+++++++++++++++++++++++++

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
+++++++++++++++++

There are a number of symbolic attributes derived as part of class definition; however, the most important
is the metric tensor. The metric tensor is essential for performing a variety of differential operations and
is therefore present in every class. You can access the symbolic version of the attribute using
:py:attr:`~coordinates.core.CurvilinearCoordinateSystem.metric_tensor_symbol`

.. code-block:: python

    cs = SphericalCoordinateSystem()
    print(cs.metric_tensor_symbol)
    [1, r**2, r**2*sin(theta)**2]

.. note::

    Many of the coordinate systems defined in Pisces-Geometry are not only curvilinear, but are also
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
++++++++++++++++++++++++++++++++++++++++

Pisces supports derived expressions beyond the metric, such as:

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
use the :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.get_expression` method.

.. code-block:: python

    cs = OblateHomoeoidalCoordinateSystem(ecc=0.3)
    print(cs.get_expression('metric_density'))
    sqrt(-xi**4*sin(theta)**2/(0.000729*sin(theta)**6 - 0.0243*sin(theta)**4 + 0.27*sin(theta)**2 - 1.0))

    cs = OblateHomoeoidalCoordinateSystem(ecc=0.0)
    print(cs.get_expression('metric_density'))
    sqrt(xi**4*sin(theta)**2)


Accessing Numerical Versions of Symbolic Expressions
+++++++++++++++++++++++++++++++++++++++++++++++++++++

All symbolic expressions can be turned into callable NumPy functions using:

.. code-block:: python

    fn = cs.get_numeric_expression("metric_density")
    val = fn(r=1.0, theta=np.pi/2, phi=0.0)

This process uses :py:func:`sympy.lambdify` under the hood, and allows fast evaluation over grids or datasets.

Class Level Expressions
+++++++++++++++++++++++

Some expressions—like the metric tensor—are computed at the class level and
shared across all instances (symbolically). You can inspect or retrieve these
without instantiating the coordinate system:

.. code-block:: python

    from pisces_geometry.coordinates.coordinate_systems import CylindricalCoordinateSystem

    g = CylindricalCoordinateSystem.get_class_expression("metric_tensor")
    print(g)

This is useful for inspecting or manipulating symbolic expressions analytically
before plugging in parameter values.

Differential Geometry
---------------------

Pisces coordinate systems support both symbolic and numerical differential geometry operations such as gradients,
divergences, and Laplacians. These operations are implemented in a basis-aware manner and respect the
metric structure of the coordinate system.

Symbolic Operations
+++++++++++++++++++

Every coordinate system includes symbolic expressions for key geometric quantities such as:

- Metric tensor :math:`g_{ij}`
- Inverse metric tensor :math:`g^{ij}`
- Metric density :math:`\rho = \sqrt{\det(g_{ij})}`
- D-term :math:`D_\mu = \frac{1}{\rho} \partial_\mu \rho` (used in divergence)
- L-term :math:`L^\mu = \frac{1}{\rho} \partial_\mu (\rho g^{\mu\nu})` (used in Laplacian)

These expressions are computed on demand and can be accessed with:

.. code-block:: python

    cs.get_expression("Dterm")
    cs.get_expression("Lterm")

Each symbolic expression is built using SymPy and can be substituted with parameters or lambdified
into numeric functions.

Numerical Operations
++++++++++++++++++++

Pisces provides instance methods for evaluating differential operators on discrete data:

- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.compute_gradient`
- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.compute_divergence`
- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.compute_laplacian`
- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.raise_index` /
  :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.lower_index`
- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.adjust_tensor_signature`

Each of these functions has a slightly different call signature

Dependence
++++++++++

Pisces-Geometry offers symbolic tools to analyze how differential operations like gradients, divergences,
and Laplacians depend on input variables. This is especially helpful when determining which coordinate directions
influence the result of an operator, and when optimizing or simplifying computations.

This analysis is done symbolically using SymPy and works with any coordinate system supported by Pisces.

You can check dependencies with the following instance methods:

- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.get_gradient_dependence`
- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.get_divergence_dependence`
- :py:meth:`~coordinates.core.CurvilinearCoordinateSystem.get_laplacian_dependence`

These methods take in a list of symbolic variables and return either a set of dependent variables, or ``0`` if the operation is identically zero.
