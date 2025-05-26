.. _differential_geometry:

================================
Differential Geometry: Overview
================================

The :mod:`differential_geometry` module provides core functionality for performing tensor calculus
and differential geometry operations. These operations form the mathematical foundation for many
high-level features in the library, including gradients, divergences,
and Laplacians in curved spaces.

This module is designed to be low-level, flexible, and composable—targeted at users who need fine-grained
control over coordinate system behavior, metric tensors, and tensorial representations. It supports both
symbolic workflows (e.g., with `SymPy <https://docs.sympy.org/latest/index.html>`_) and numerical workflows
using dense array operations.

Key features of the module include:

- Differential operators such as **gradient**, **divergence**, and **Laplacian**, compatible with arbitrary
  orthogonal coordinate systems.
- Support for **tensor index manipulation**, including **index raising and lowering**, **signature adjustments**,
  and **metric contractions**.
- Compatibility with both symbolic (e.g., for analytical derivations or code generation) and numerical (e.g., for
  evaluation on structured grids) contexts.

This module is not tied to methods in higher level modules like :mod:`grids` and :mod:`fields`.

.. hint::

    Generally speaking, methods in :mod:`grids` and :mod:`fields` are better places to perform
    operations than with the functions in this module unless you need very specific capabilities.


Module Layout
-------------

The :mod:`differential_geometry` module is organized into several subcomponents, each focused on
a different aspect of differential geometry and tensor operations. These submodules provide both
symbolic and numerical utilities for manipulating tensor fields, enabling a wide range of workflows
from low-level grid computation to symbolic code generation.

Differences Between Dense and Sparse Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Dense representation**:
  Stores all tensor components in a single NumPy or array-like object, with trailing dimensions indexing
  each component. For example, a rank-2 tensor over a 3D grid would have shape ``(nx, ny, nz, 3, 3)``.

  - ✅ Efficient for full-field computations
  - ✅ Compatible with numerical backends
  - ✅ Best for vector/tensor fields with all (or most) components populated

  Notably, dense representations **MUST** have ``(ndim,)`` elements in each tensor index, even if those
  elements are zero.

- **Sparse representation** (planned):
  Uses a dictionary to map multi-indices (e.g., ``(0, 1)``, ``(1, 2)``) to arrays holding scalar component values.
  This format is well-suited for:

  - ✅ Symbolic manipulation
  - ✅ Fields with many zero components
  - ✅ Compact storage of sparse tensors

Module Components
^^^^^^^^^^^^^^^^^

The core subcomponents are:

- **Dense operations** (:mod:`~differential_geometry.dense_utils` and :mod:`~differential_geometry.dense_ops`):
  Implements numerical tensor calculus using dense arrays. Dense operations are optimized for
  structured grids and provide fast evaluation of tensor operations such as:

  - Tensor contraction and trace
  - Volume element computation
  - Index raising and lowering
  - Tensor permutation and signature adjustment

- **Sparse operations (planned)** (:mod:`~differential_geometry.sparse_utils` and :mod:`~differential_geometry.sparse_ops`:
  Sparse tensor operations (not yet fully implemented) would allow manipulating tensor fields in
  a dictionary-of-components format, useful for high-rank or highly sparse tensors in symbolic contexts.

- **General operations** (:mod:`~differential_geometry.general_ops`):
  General mathematical operations which apply numerically to both dense and sparse representations.

- **Symbolic operations** (:mod:`~differential_geometry.symbolic`):
  Symbolic implementations of differential operators using symbolic coordinate systems (e.g., via SymPy).
  These are useful for:

  - Analytical derivation
  - Code generation
  - Simplification of coordinate-dependent expressions

  Includes symbolic versions of gradient, divergence, Laplacian, etc.

- **Dependence modeling**: (:mod:`~differential_geometry.dependence`):
  Symbolic tensor wrappers that allow expressing and
  manipulating the dependence structure of tensors. These classes are used to:

  - Track coordinate dependence
  - Generate symbolic proxies and operation rules
  - Predict the shape and axes of outputs in differential computations

Symbolic Operations
-------------------

The :mod:`differential_geometry.symbolic` submodule provides symbolic utilities for computing core geometric
quantities in curvilinear coordinate systems using SymPy. These operations are central to performing
tensor calculus analytically or in workflows that involve symbolic manipulation, automatic code generation,
or coordinate-dependent derivations.

The symbolic interface supports:

- **Metric manipulation**:

  - :func:`~differential_geometry.symbolic.invert_metric`: Computes the inverse of a metric tensor, whether full or diagonal.
  - :func:`~differential_geometry.symbolic.compute_metric_density`: Computes the metric density :math:`\rho = \sqrt{\det(g)}`.

- **Geometric operators**:

  - :func:`~differential_geometry.symbolic.compute_Dterm`: Computes the :math:`D_\mu` terms used in divergence expressions.
  - :func:`~differential_geometry.symbolic.compute_Lterm`: Computes the :math:`L^\nu` terms used in Laplacian expressions.

- **Tensor index operations**:

  - :func:`~differential_geometry.symbolic.raise_index`: Raises an index using the inverse metric.
  - :func:`~differential_geometry.symbolic.lower_index`: Lowers an index using the metric.
  - :func:`~differential_geometry.symbolic.adjust_tensor_signature`: Transforms a tensor from one index signature (covariant/contravariant) to another.

- **Differential operators**:

  - :func:`~differential_geometry.symbolic.compute_gradient`: Computes the gradient of a scalar field in covariant or contravariant form.
  - :func:`~differential_geometry.symbolic.compute_tensor_gradient`: Computes the gradient of a general tensor field.
  - :func:`~differential_geometry.symbolic.compute_divergence`: Computes the divergence of a vector field in curved coordinates.
  - :func:`~differential_geometry.symbolic.compute_laplacian`: Computes the scalar Laplacian in general or orthogonal coordinates.
  - :func:`~differential_geometry.symbolic.compute_tensor_laplacian`: Applies the Laplacian operator element-wise to a tensor field.


Diagonal vs Full Metrics
^^^^^^^^^^^^^^^^^^^^^^^^

A key feature of the symbolic operations module is its dual support for both **full metric tensors**
and **orthogonal diagonal metrics**. This design enables efficient symbolic modeling without sacrificing generality.

- **Full metrics** are represented as :math:`n \times n` SymPy matrices. They allow for arbitrary coordinate
  systems, including non-orthogonal bases and those with off-diagonal metric terms.

- **Orthogonal (diagonal) metrics** are represented as 1D SymPy arrays containing only the diagonal elements
  :math:`g_{\mu\mu}`. This simplified format is valid for any orthogonal coordinate system—such as spherical,
  cylindrical, or Cartesian—and avoids unnecessary complexity when off-diagonal components are zero.

This distinction is especially important in symbolic workflows, for several reasons:

- **Clarity**: In orthogonal systems, working directly with the diagonal metric makes expressions easier
  to read, understand, and verify.
- **Performance**: Many symbolic operations—such as computing the determinant, inverse, or L-terms—become
  significantly faster when using a diagonal representation.
- **Flexibility**: The ability to accept both formats allows you to use the same symbolic APIs across a wide
  range of coordinate systems. For example, switching between spherical and Cartesian coordinates requires
  only a change in how the metric is passed to the function.
- **Reduced Overhead**: For applications like code generation or educational notebooks, the diagonal format
  avoids unnecessary bloat while maintaining exactness and consistency with the general theory.

For example, both of the following inputs are valid:

.. code-block:: python

    # Full (3x3) matrix representation
    g_full = sp.Matrix([
        [1,     0,                    0],
        [0,   r**2,                   0],
        [0,     0, (r*sp.sin(theta))**2]
    ])

    # Diagonal-only (1D) array representation
    g_diag = sp.Array([1, r**2, (r*sp.sin(theta))**2])

This dual-format support provides a powerful and user-friendly approach to symbolic differential geometry:
you get full generality when needed, and streamlined expressiveness when possible.

Dense Numerical Operations
--------------------------

The :mod:`pymetric.differential_geometry.dense_ops` module provides low-level utilities
for computing differential geometry operations directly on **dense NumPy arrays**. These routines form the
numerical backend for many field-level operations and allow fine-grained control over
coordinate-aware derivatives in general and orthogonal curvilinear systems.

These functions operate on fields stored as contiguous arrays, where the final axes represent tensor
components (ranked structure), and the spatial axes precede them. They are designed to be efficient,
broadcast-aware, and explicitly compatible with coordinate systems via metrics and volume element terms.

Supported operations include:

- **Gradient** (covariant and contravariant)
- **Divergence** (covariant and contravariant)
- **Laplacian** (scalar Laplace-Beltrami operator)
- **Index manipulation** (via metric contraction)

Each operation supports:

- **Covariant and contravariant basis selection**
- **Diagonal or full inverse metrics** (automatically inferred)
- **Precomputed derivatives** (first and second) for advanced workflows
- **Flexible differentiation axes** via `field_axes` and `derivative_axes`

These functions are optimized for use in high-performance workflows, including:

- Broadcasting over multidimensional grid shapes
- Avoiding unnecessary allocations via `out=...` buffers
- Use with scalar, vector, or tensor fields of arbitrary rank

.. important::

    These are **backend functions**. They perform minimal validation and assume consistency in shapes,
    axis mappings, and coordinate contexts. For safety, use high-level field methods unless you require
    fine-grained control or performance.

**Example Use Case**:

Consider a scalar field :math:`\phi(r, \theta) = r^2 \cos(\theta)` on a 2D spherical grid.
Using dense numerical operations, one can compute its Laplacian with:

.. code-block:: python

    from pymetric.differential_geometry.dense_ops import compute_laplacian

    lap = compute_laplacian(
        tensor_field=phi,
        Fterm_field=Fterm,
        inverse_metric_field=inverse_metric,
        rank=0,
        field_axes=[0, 1],
        edge_order=2,
        *grid_coordinates
    )

This example highlights one of the central principles of the dense ops API: **the differential structure is explicit**,
and the user is in control of metrics, axes, and buffers.

See also:

- :func:`~differential_geometry.dense_ops.compute_gradient`
- :func:`~differential_geometry.dense_ops.compute_divergence`
- :func:`~differential_geometry.dense_ops.compute_laplacian`


Sparse Numerical Operations
---------------------------

.. important::

    These features are not yet implemented in current releases of PyMetric.

Dependence Tracking
--------------------

A tricky aspect of performing grid-based differential operations is the possibility of
**symmetry breakdown**, which occurs when the geometric behavior of the coordinate system
induces spatial dependence during an operation which was not present in the original function.

To assist in tracking these changes, PyMetric provides the :mod:`differential_geometry.dependence` module. You
can read more about it at :ref:`tensor_dependence`.
