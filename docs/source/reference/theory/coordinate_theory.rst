.. _theory:
===============================
Curvilinear Coordinate Systems
===============================

In the `Pisces Project <https://www.github.com/Pisces-Project/Pisces>`__, every physical model you can generate is backed
up by a specific coordinate system defined here in PyMetric. These coordinate systems play a critical role in determining
the behavior of various operations and are a necessary step towards doing physics in these exotic coordinate systems. In this
guide, we'll introduce the theory of coordinate systems in a manner akin to that seen in the study
of `differential geometry <https://en.wikipedia.org/wiki/Differential_geometry>`__.

What is a Curvilinear Coordinate System?
----------------------------------------

A **curvilinear coordinate system** is a system of coordinates in which the coordinate lines may be curved rather than
straight. These systems generalize Cartesian coordinates to accommodate more complex geometries and symmetries,
making them especially useful in physics, engineering, and geometry.

Unlike Cartesian coordinates where each basis direction is constant and orthonormal, in a curvilinear system:

- The basis vectors **change direction and magnitude** as you move through space.
- The **coordinate curves** (the paths traced out by holding all but one coordinate constant) are generally curved.
- The **metric tensor** varies spatially and encodes the local geometry of the space.

Mathematically, we describe a curvilinear system by a **coordinate map**:

.. math::

   \mathbf{x} = \mathbf{x}(q^1, q^2, \dots, q^n)

This map transforms from curvilinear coordinates :math:`(q^1, q^2, \dots, q^n)` to Cartesian space :math:`\mathbf{x} \in \mathbb{R}^n`.
The coordinate curves are traced by holding all but one :math:`q^i` constant and letting :math:`q^i` vary.

The **tangent vectors** to these curves form the **coordinate basis**:

.. math::

   \mathbf{e}_i = \frac{\partial \mathbf{x}}{\partial q^i}

These basis vectors vary across space and are generally **not unit vectors** and **not orthogonal**.
Their inner products define the components of the **metric tensor**:

.. math::

    g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j

This tensor captures how distances, angles, and volumes behave locally in the curvilinear space.


Defining a Coordinate System
----------------------------

A coordinate system in PyMetric is defined by a smooth, invertible mapping from a set of curvilinear coordinates to Cartesian space:

.. math::

   \mathbf{x} = \mathbf{x}(q^1, q^2, \dots, q^n)

This **coordinate map** takes a point in the curvilinear domain, expressed in coordinates :math:`(q^1, q^2, \dots, q^n)`,
and assigns it a position vector :math:`\mathbf{x} \in \mathbb{R}^n`.

From this mapping, we define the **coordinate basis vectors** (also called the **tangent basis**) by taking partial derivatives
of :math:`\mathbf{x}` with respect to each coordinate:

.. math::

   \mathbf{e}_i = \frac{\partial \mathbf{x}}{\partial q^i}

These vectors span the **tangent space** at each point and vary smoothly across the domain. They are generally neither orthogonal nor normalized.

.. note::

    More formally, we state that for any point :math:`p \in \mathbb{R}^N`, there is a tangent space :math:`T_p \mathbb{R}^N` which
    is a vector space composed of all of the tangent vectors to all of the curves passing through :math:`p`. This can be made more
    rigorous in the context of differentiable manifolds (see `Tangent Spaces <https://en.wikipedia.org/wiki/Tangent_space>`__) and leads
    to the notion of the `Tangent Bundle <https://en.wikipedia.org/wiki/Tangent_bundle>`__.

As is the case for **all vector spaces**, the space of all **linear maps** :math:`f: T_p \mathbb{R}^N \to \mathbb{R}` also forms
a vector space called the `dual space <https://en.wikipedia.org/wiki/Dual_space>`__ denoted :math:`T^\star_p \mathbb{R}^N`. It is a
special result that for Euclidean space, the **dual space** is equivalent to the Euclidean space itself (seen as a vector space). We therefore
inherit two Euclidean vector spaces at each point in space:

1. The **tangent space** (:math:`T_p\mathbb{R}^N`) which contains **contravariant vectors** :math:`V \in T_p M` which are
   expressed in terms of a contravariant basis:

   .. math::

        \forall V \in T_p \mathbb{R}^N, \exists V^\mu \; \text{s.t.}\; V = V^\mu {\bf e}_\mu.

2. The **cotangent space** (:math:`T_p^\star \mathbb{R}^N`) which contains **covariant vectors** :math:`V \in T_p^\star M` which
   are expressed in terms of a covariant basis:

   .. math::

        \forall V \in T^\star_p \mathbb{R}^N, \exists V_\mu \; \text{s.t.}\; V = V_\mu {\bf e}^\mu.

   where :math:`{\bf e}^\mu` are the **induced dual basis** such that :math:`{\bf e}^\mu ({\bf e}_\nu) = \delta_\nu^\mu`.

To relate the tangent and cotangent spaces, we define the **metric tensor**: a symmetric, bilinear form that provides an
inner product on the tangent space. At each point :math:`p \in \mathbb{R}^N`, the metric is a map:

.. math::

   g_p : T_p \mathbb{R}^N \times T_p \mathbb{R}^N \to \mathbb{R}

which satisfies:

- Symmetry: :math:`g_p(\mathbf{u}, \mathbf{v}) = g_p(\mathbf{v}, \mathbf{u})`
- Bilinearity: linear in each argument
- Positive-definiteness (in Euclidean space): :math:`g_p(\mathbf{v}, \mathbf{v}) > 0` for all non-zero :math:`\mathbf{v}`

In a coordinate basis :math:`\{ \mathbf{e}_\mu \}`, the metric components are given by:

.. math::

   g_{\mu\nu} = g(\mathbf{e}_\mu, \mathbf{e}_\nu) = \mathbf{e}_\mu \cdot \mathbf{e}_\nu

These components form the **metric tensor** :math:`g_{\mu\nu}`, which plays a central role in geometry and analysis.

The metric allows us to map vectors to covectors (and vice versa), effectively bridging the tangent and cotangent spaces.
This process is known as **raising and lowering indices**.

Given a contravariant vector :math:`V^\mu`, we define its covariant form as:

.. math::

   V_\nu = g_{\nu\mu} V^\mu

Similarly, given a covariant vector :math:`\omega_\mu`, its contravariant form is:

.. math::

   \omega^\mu = g^{\mu\nu} \omega_\nu

where :math:`g^{\mu\nu}` is the **inverse metric tensor**, satisfying:

.. math::

   g^{\mu\alpha} g_{\alpha\nu} = \delta^\mu_\nu

These operations allow for seamless transformation between the vector and dual-vector representations and are central to
defining geometric operations like gradients, divergences, and Laplacians in curvilinear coordinates.

.. note::

   In PyMetric, the metric is represented as a tensor field defined by the coordinate system. This enables differential
   operators and field transformations to be expressed in a coordinate-aware and mathematically rigorous way.


Vectors, Tensors, and Beyond
----------------------------

In curvilinear geometry, every object of physical interest—scalars, vectors, forms, and higher-order tensors—can be described
as a field defined over the coordinate domain. That is, a **field** assigns a mathematical object to every point in space,
with the object transforming in a specific way under changes of coordinates.

At the core of this structure are **vectors** and **covectors**.

Covectors: Linear Maps on Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **covector** (also called a **dual vector** or **1-form**) is a linear map from the tangent space to the real numbers:

.. math::

   \omega : T_p \mathbb{R}^N \to \mathbb{R}

That is, given a tangent vector :math:`\mathbf{v} \in T_p \mathbb{R}^N`, the covector returns a scalar :math:`\omega(\mathbf{v})`.

In a coordinate basis :math:`\{ \mathbf{e}_\mu \}`, there exists a corresponding dual basis :math:`\{ \mathbf{e}^\mu \}` such that:

.. math::

   \mathbf{e}^\mu(\mathbf{e}_\nu) = \delta^\mu_\nu

Any covector :math:`\omega` can thus be written in terms of its components:

.. math::

   \omega = \omega_\mu \, \mathbf{e}^\mu

The index on :math:`\omega_\mu` is **lowered**, reflecting its membership in the cotangent space.

Tensors as Multilinear Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **tensor** is a multilinear map that accepts vectors and covectors as arguments and returns a real number:

.. math::

   T : \underbrace{T_p^\star \mathbb{R}^N \times \cdots \times T_p^\star \mathbb{R}^N}_{k \text{ times}} \times
       \underbrace{T_p \mathbb{R}^N \times \cdots \times T_p \mathbb{R}^N}_{\ell \text{ times}} \to \mathbb{R}

We say such a tensor is of **type (k, ℓ)**, with:

- :math:`k` **contravariant** (vector) slots
- :math:`\ell` **covariant** (covector) slots

In a coordinate basis, this tensor can be expressed via its components as:

.. math::

   T = T^{\mu_1 \dots \mu_k}_{\nu_1 \dots \nu_\ell} \; \mathbf{e}_{\mu_1} \otimes \cdots \otimes \mathbf{e}_{\mu_k}
                                     \otimes \mathbf{e}^{\nu_1} \otimes \cdots \otimes \mathbf{e}^{\nu_\ell}

Here, the **tensor product** :math:`\otimes` constructs a new basis for the space of multilinear maps. The indices on the
components encode their variance: **upper indices** for contravariant directions (vectors), and **lower indices** for
covariant directions (covectors).

Tensor Fields in PyMetric
~~~~~~~~~~~~~~~~~~~~~~~~~

In PyMetric, most objects are internally represented as a **tensor field**. This means that at each point in space, the
object carries both:

- A **tensor type** (its signature), defined by its number of covariant and contravariant indices
- A **buffer of values** that vary over the coordinate domain

For example:

- A scalar field has type (0, 0) and stores a single value at each point.
- A vector field has type (1, 0) and stores components along basis vectors.
- A (0, 2) field is a covariant rank-2 tensor—e.g., the metric tensor :math:`g_{\mu\nu}`.

The full power of PyMetric lies in its ability to operate on these tensor fields **with awareness of their structure**,
ensuring that mathematical operations like differentiation, contraction, or index permutation obey the correct transformation
rules.

.. note::

   The type signature of a tensor field in PyMetric determines how it interacts with coordinate changes, differential
   operators, and other tensors. Internally, this signature is used to validate operations and enforce geometric consistency.

From Geometry to Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we transition to the next section, it's important to recognize that **the behavior of operations like gradient,
divergence, or Laplacian depends intimately on the type and structure of the tensor field** being operated on.

The distinction between scalar fields, vector fields, and general tensors is not just semantic—it determines how the field
transforms, how derivatives are taken, and how integrals are computed in curved spaces.

In the next section, we'll explore how calculus adapts to curvilinear coordinates, and how PyMetric ensures correctness
when working with complex geometries and tensorial data.



Calculations in Curvilinear Coordinates
---------------------------------------

Differential calculus in curvilinear coordinates differs fundamentally from the Cartesian case because the **basis vectors
vary with position**. As a result, derivatives must account not only for the variation of the components of a field,
but also for the variation of the basis vectors themselves.

In Cartesian coordinates, the gradient of a scalar is simply the vector of partial derivatives, and divergence is the
sum of partial derivatives. But in curvilinear coordinates, these operations must be **modified by the local geometry**,
captured by the **metric tensor** and its derivatives.

.. note::

   All differential operations in PyMetric—gradient, divergence, Laplacian, and curl—are computed with explicit awareness of the metric
   and coordinate basis. They are implemented using standardized methods in the :mod:`~pisces_geometry.differential_geometry.dense_ops`
   module and dispatched through `grid` and `field` methods.

Coordinate-Invariant Foundations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Despite coordinate dependence in expressions, **physical quantities remain coordinate invariant**. Key operations that retain this
invariance include:

- **Inner products**, lengths, and angles via the metric tensor.
- **Covariant derivatives** that account for basis vector variation.
- **Geometric integrals** (flux, volume, work) with proper metric volume elements.

These operations are built into the PyMetric tensor infrastructure and obey transformation laws derived from the field
type (e.g., `(1,0)` vector, `(0,2)` metric).

Displacements, Areas, and Volumes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The building blocks of calculus in curved coordinates are the infinitesimal geometric quantities:

**Line Element**:
The squared differential displacement is given by the metric:

.. math::

   ds^2 = g_{\mu\nu} \, dq^\mu dq^\nu


**Area Element**:
For a surface spanned by two coordinate directions :math:`q^\mu` and :math:`q^\nu`, the differential area is:

.. math::

   dA^{\mu\nu} = \left| \mathbf{e}_\mu \times \mathbf{e}_\nu \right| \, dq^\mu dq^\nu


**Volume Element**:
In :math:`n` dimensions, the infinitesimal volume element is:

.. math::

   dV = \sqrt{\det g} \; dq^1 dq^2 \dots dq^n


This ensures that integrals over scalar or tensor fields account for geometric distortion.

Basic Operations
~~~~~~~~~~~~~~~~

Gradient
^^^^^^^^

The gradient of a scalar field :math:`\phi` is the covariant vector:

.. math::

   \nabla \phi = \frac{\partial \phi}{\partial q^\mu} \, \mathbf{e}^\mu

In PyMetric:

.. code-block:: python

   phi = DenseTensorField(...)
   grad_phi = phi.gradient()

This automatically dispatches to the correct method based on the grid's metric structure.
For orthogonal coordinates, it uses precomputed scale factors; otherwise, it uses metric inverse tensors.

Divergence
^^^^^^^^^^

The divergence of a vector field :math:`V^\mu` in a curvilinear coordinate system is defined as:

.. math::

   \nabla \cdot \mathbf{V} = \frac{1}{\sqrt{g}} \frac{\partial}{\partial q^\mu} \left( \sqrt{g} V^\mu \right)

This form ensures that conservation laws (e.g., for flux or charge) are preserved under general coordinate transformations.

In the Pisces Geometry implementation, this is expressed through a **product rule expansion**:

.. math::

   \nabla \cdot \mathbf{V} = \left( D^\mu + \partial_\mu \right) V^\mu

Here:

- :math:`\partial_\mu` denotes the partial derivative operator.
- :math:`D^\mu = \frac{1}{\sqrt{g}} \frac{\partial \sqrt{g}}{\partial q^\mu}` encodes the **volume distortion** induced by the metric determinant.

This expansion allows the divergence to be rewritten as a **first-order operator** acting on the vector components, where :math:`D^\mu` is precomputed from the grid and stored as a buffer field.

In PyMetric, divergence is implemented as:

.. code-block:: python

   V = DenseTensorField(...)
   div_V = V.divergence()

This method:

- Automatically detects the field's variance signature (e.g., contravariant or covariant).
- Applies finite differences to compute :math:`\partial_\mu V^\mu`.
- Adds the precomputed :math:`D^\mu V^\mu` product via broadcasting.

This allows divergence to be computed generically across arbitrary curvilinear systems, including spherical, cylindrical, and user-defined coordinates.


Laplacian
^^^^^^^^^

The Laplacian of a scalar field :math:`\phi` is given by:

.. math::

   \nabla^2 \phi = \frac{1}{\sqrt{g}} \frac{\partial}{\partial q^\mu} \left( \sqrt{g} g^{\mu\nu} \frac{\partial \phi}{\partial q^\nu} \right)

To compute this, PyMetric expands the expression using a product rule in two stages. The expansion introduces two geometric terms:

.. math::

   \nabla^2 \phi = \left( L^{\mu\nu} + D^\mu g^{\mu\nu} + g^{\mu\nu} \partial_\mu \right) \partial_\nu \phi

Where:

- :math:`g^{\mu\nu}` is the **inverse metric tensor**
- :math:`D^\mu = \frac{1}{\sqrt{g}} \frac{\partial \sqrt{g}}{\partial q^\mu}` as above
- :math:`L^{\mu\nu} = \frac{\partial g^{\mu\nu}}{\partial q^\mu}` is the **metric variation tensor**, capturing how the inverse metric changes over space

These geometric derivatives (:math:`D^\mu`, :math:`L^{\mu\nu}`) are automatically computed and cached by the grid object.

In PyMetric, this is evaluated via:

.. code-block:: python

   phi = DenseTensorField(...)
   lap_phi = phi.laplacian

Internally:

- First derivatives :math:`\partial_\nu \phi` are computed using central finite differences.
- Then, the above expression is evaluated using tensor contractions and broadcasting, with all geometric coefficients precomputed from the grid.
