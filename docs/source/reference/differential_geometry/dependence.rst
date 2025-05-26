.. _tensor_dependence:

===============================
Symbolic Tensor Dependence API
===============================

A tricky aspect of performing grid-based differential operations is the possibility of
**symmetry breakdown**, which occurs when the geometric behavior of the coordinate system
induces spatial dependence during an operation which was not present in the original function.

As an example, consider a scalar field :math:`\phi(x^1)` over a coordinate system :math:`(x^1,x^2,x^3)`.
The covariant gradient of :math:`\phi` is

.. math::

    \nabla_\mu \phi = \partial_\mu \phi,

and *maintains the dependence of the field*. In contrast,

.. math::

    \nabla^\mu \phi = g^{\mu\nu}(x^1,x^2,x^3) \partial_\nu \phi

will, in the general case depend on additional coordinates.

To contend with this, :class:`~fields.components.FieldComponent` objects and fields themselves pay
attention to the dependence of the operations they perform so that results are **automatically cast
to the correct axes**. Supporting this capability is the :mod:`differential_geometry.dependence` module,
which uses `SymPy <https://docs.sympy.org/latest/index.html>`_ to determine operational dependencies.


.. contents::
   :local:
   :depth: 2

Overview
--------

At the core of this module is the :class:`~differential_geometry.dependence.DependenceObject`, which provides a base class
for representing symbolic dependence of a particular field on coordinate axes.
Subclasses such as :class:`~differential_geometry.dependence.DenseDependenceObject` and :class:`~differential_geometry.dependence.DenseTensorDependence` allow for
dense symbolic modeling of scalars, vectors, and higher-rank tensors over a given
coordinate system.

Key features include:

- Construction of symbolic proxies for tensor fields
- Symbolic gradient, divergence, and Laplacian operations
- Index manipulation: raising/lowering via the metric
- Dependence inspection and shape-aware operations

.. note::

    This module will support eventual sparse fields and components.

What Is a Dependence Object?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **dependence object** is a symbolic representation of a field’s dependence on coordinates
within a given coordinate system. It encapsulates three core components:

- A reference to a **coordinate system**, which provides the geometric context.
- A **symbolic proxy**, constructed using `SymPy <https://www.sympy.org>`_, that mimics the behavior
  of the actual field. For scalars, this is a SymPy function; for tensors, it is a multidimensional
  array of symbolic functions.
- A set of **dependent axes** that describe which coordinates the field depends on.

This structure allows PyMetric to analyze, manipulate, and differentiate fields **without explicit data**.
The symbolic proxy acts as a stand-in for a real field — making it possible to evaluate symbolic gradients,
Laplace–Beltrami operators, or perform index manipulations while tracking which variables influence the result.

This mechanism is particularly powerful when simulating operations that **alter** coordinate dependence, such as:

- Switching from covariant to contravariant derivatives (which may introduce new dependence),
- Applying divergence (which contracts coordinate dependence),
- Raising or lowering tensor indices using the metric tensor.

How Does a Dependence Object Track Dependence?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dependence objects track symbolic dependence via the **free symbols** present in the symbolic proxy expression.

Each symbolic proxy is a scalar or tensor-valued expression that depends on one or more coordinate variables.
When a symbolic operation is applied (e.g., gradient or Laplacian), new coordinate symbols may enter the expression
— for example, through coordinate-dependent metric components — and alter its dependency.

PyMetric detects these changes automatically by:

1. Constructing a new symbolic expression through the desired operation.
2. Extracting the resulting **free symbols**.
3. Comparing those symbols to known coordinate axes to determine updated axis dependence.

This is why all symbolic operations in this module return a new dependence object or symbolic proxy:
so that downstream consumers (like :class:`~fields.components.FieldComponent`) can realign the field
with the correct axes of dependence.

This system provides a rigorous way to **propagate symbolic dependence metadata**, ensuring that grid-level
differential geometry operations remain consistent with the mathematical structure of the coordinate system.

Dense Dependence Objects
------------------------

Dense dependence objects are the most common symbolic models used in PyMetric. These classes assume
that **all components of a tensor field depend on the same subset of coordinate axes**, which enables
uniform symbolic operations.

There are two concrete classes provided:

- :class:`~pymetric.differential_geometry.dependence.DenseDependenceObject` —
  A generic symbolic model for scalars and fixed-shape tensors.
  It supports symbolic operations like gradient and Laplacian component-wise.

- :class:`~pymetric.differential_geometry.dependence.DenseTensorDependence` —
  A tensor-aware subclass that adds support for index manipulation, divergence, and rank-aware operations.

The difference between the two is operational: if you need to raise/lower indices, compute divergence,
or work with explicit tensor ranks, use :class:`~pymetric.differential_geometry.dependence.DenseTensorDependence`. For general symbolic modeling,
:class:`~pymetric.differential_geometry.dependence.DenseDependenceObject` is sufficient.


Dense Proxy Generation
^^^^^^^^^^^^^^^^^^^^^^^

When you create a dense dependence object, a **symbolic proxy** is lazily generated to represent
the field or tensor as a SymPy expression. This is used internally for all symbolic operations.

The proxy takes the form:

- For scalar fields:

  .. code-block:: python

     f = DenseDependenceObject(cs, (), dependent_axes=["r"])
     f.symbolic_proxy  # → T(r)

- For vector fields:

  .. code-block:: python

     v = DenseDependenceObject(cs, (3,), dependent_axes=["r", "theta"])
     v.symbolic_proxy  # → [T_r(r, theta), T_theta(r, theta), T_phi(r, theta)]

- For rank-2 tensors:

  .. code-block:: python

     T = DenseDependenceObject(cs, (3, 3), dependent_axes=["r"])
     T.symbolic_proxy  # → 3×3 SymPy array of component functions

These symbolic proxies are what drive downstream computations like derivatives or contractions.

Dependence Operations
^^^^^^^^^^^^^^^^^^^^^^

The following symbolic operations are supported on dense dependence objects:

- **Arithmetic**: `+`, `-`, `*`, `/` work between compatible dependence objects or scalars.
- **Differential operations**:

  - :meth:`~differential_geometry.dependence.DenseDependenceObject.element_wise_gradient` — Returns the symbolic gradient of each component.
  - :meth:`~differential_geometry.dependence.DenseDependenceObject.element_wise_laplacian` — Computes the Laplace–Beltrami operator on each component.


- **Tensor operations**:

  - :meth:`~differential_geometry.dependence.DenseTensorDependence.raise_index` — Raises an index using the inverse metric.
  - :meth:`~differential_geometry.dependence.DenseTensorDependence.lower_index` — Lowers an index using the metric.
  - :meth:`~differential_geometry.dependence.DenseTensorDependence.adjust_tensor_signature` — Adjusts full index variance signature.
  - :meth:`~differential_geometry.dependence.DenseTensorDependence.gradient` — Compute the gradient.


All operations produce either a new symbolic proxy or a wrapped dependence object that reflects the
updated structure. These operations ensure consistency with the coordinate system’s metric and geometry.

Sparse Dependence Objects
-------------------------

.. important::

    This will be implemented in a coming release of the library.

    Sparse dependence models will allow each tensor component to depend on a distinct set of
    coordinates — enabling more memory-efficient and structurally faithful representations
    for complex symbolic fields.

    They will support selective differentiation, sparse contraction, and hybrid symbolic forms.
