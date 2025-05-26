.. _theory:
===============================
Curvilinear Coordinate Systems
===============================

In the `Pisces Project <https://www.github.com/Pisces-Project/Pisces>`_, every physical model you can generate is backed
up by a specific coordinate system defined here in Pisces-Geometry. These coordinate systems play a critical role in determining
the behavior of various operations and are a necessary step towards doing physics in these exotic coordinate systems. In this
guide, we'll introduce the theory of coordinate systems in a manner akin to that seen in the study of `differential geometry <https://en.wikipedia.org/wiki/Differential_geometry>`_.

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
The coordinate curves are traced by holding all but one :math:` q^i ` constant and letting :math:`q^i` vary.

The **tangent vectors** to these curves form the **coordinate basis**:

.. math::

   \mathbf{e}_i = \frac{\partial \mathbf{x}}{\partial q^i}

These basis vectors vary across space and are generally **not unit vectors** and **not orthogonal**. Their inner products define the components of the **metric tensor**:

.. math::

   g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j

This tensor captures how distances, angles, and volumes behave locally in the curvilinear space.


Defining a Coordinate System
----------------------------

A coordinate system in Pisces-Geometry is defined by a smooth, invertible mapping from a set of curvilinear coordinates to Cartesian space:

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
    rigorous in the context of differentiable manifolds (see `Tangent Spaces <https://en.wikipedia.org/wiki/Tangent_space>`_) and leads
    to the notion of the `Tangent Bundle <https://en.wikipedia.org/wiki/Tangent_bundle>`_.

As is the case for **all vector spaces**, the space of all **linear maps** :math:`f: T_p \mathbb{R}^N \to \mathbb{R}` also forms
a vector space called the `dual space <https://en.wikipedia.org/wiki/Dual_space>`_ denoted :math:`T^\star_p \mathbb{R}^N`. It is a
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

   In Pisces-Geometry, the metric is represented as a tensor field defined by the coordinate system. This enables differential
   operators and field transformations to be expressed in a coordinate-aware and mathematically rigorous way.


Vectors, Tensors, and Beyond
----------------------------

- What is a field on the coordinate system? What is a tensor
  as a map from the dual and tangent spaces, etc.


Calculations in Curvilinear Coordinates
---------------------------------------

- Why does calculus differ between coordinate systems?
- What coordinate agnostic operations matter in physics?

Displacements, Areas, and Volumes
''''''''''''''''''''''''''''''''''

- Define the volume, area, and line infinitesmals.

Basic Operations
''''''''''''''''

- Gradient
- Divergence
- Curl
- Laplacian
