---
title: 'PyMetric: A Library for Geometric Computation'
tags:
  - Python
  - differential geometry
  - modeling
authors:
  - name: Eliza C. Diggins
    orcid: 0009-0005-9389-9098
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Daniel R. Wik
    orcid: 0000-0001-9110-2245
    affiliation: 1
affiliations:
 - name: University of Utah, USA
   index: 1
date: 11 June 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
---

# Summary

PyMetric is a lightweight Python library designed to streamline differential geometry
and vector calculus operations in user-defined coordinate systems, with a focus on
applications in astrophysics and computational physics. In many physical modeling
tasks, especially in astrophysics, it is both natural and advantageous to work in
non-Cartesian coordinate systems that reflect the underlying symmetries of a system.
These coordinate systems may be highly nontrivial, such as ellipsoidal (homoeoidal) or spheroidal
coordinates, where the ability to express and manipulate tensorial quantities is essential
for deriving accurate and efficient models.

The core design of PyMetric emphasizes a seamless blend of symbolic and numerical computation.
Symbolic expressions are used to derive key geometric quantities such as metric tensors,
Christoffel symbols, and Jacobians, which are then converted into efficient numerical
functions suitable for use in array-based workflows. This enables users to write models
that respect the underlying geometry while retaining the performance and flexibility of
NumPy-style programming.

PyMetric further supports structured grid abstractions and offers extensibility
for multiple coordinate systems, buffer types (including HDF5 storage),
and geometric operations. It automates tasks such as computing gradients, divergences, and Laplacians—critical
for solving PDEs or modeling physical systems—while maintaining awareness of the coordinate system
and metric context. This makes it particularly useful for scientific domains that demand geometric
precision, such as general relativity, magnetohydrodynamics, and planetary dynamics.

By bringing geometric context to array mathematics, PyMetric provides a flexible
and modern foundation for coordinate-aware scientific computing.

# Statement of need

Astrophysical modeling frequently requires the use of complex, non-Cartesian coordinate
systems such as spherical, spheroidal, or homoeoidal coordinates to exploit the underlying symmetries
of physical systems. These symmetries are crucial for accurately and efficiently modeling galaxy
shapes, gravitational potentials, and relativistic spacetimes. For example, in galactic dynamics,
ellipsoidal models often benefit from the use of homoeoidal coordinates, which align with the geometry
of the system and simplify the underlying equations. Similarly, in general relativity, expressing
metrics and computing curvature quantities in adapted coordinates is a standard practice.

The Pisces project is an emerging computational framework designed to support high-level modeling
and simulation in relativistic astrophysics, galactic dynamics, and related domains. A core requirement
of Pisces is the ability to perform differential geometry and vector calculus operations generically
across coordinate systems and grid structures, with support for both symbolic derivation and numerical
evaluation. Prior to PyMetric, no lightweight and extensible Python package provided a unified backend
to handle these computations robustly.

PyMetric was developed to address this gap. It provides a foundation for the Pisces ecosystem by automating
the construction of metric tensors, differential operators (e.g., gradients, divergences, Laplacians),
and coordinate-aware transformations in arbitrary coordinate systems. Its ability to interoperate with
NumPy arrays and disk-backed storage formats (e.g., HDF5) ensures that it can be deployed
in both interactive research and large-scale simulation contexts.

By embedding geometric context directly into array-based workflows, PyMetric makes it easier to construct
models that respect the intrinsic symmetries of complex astrophysical systems. This reduces the need
for hard-coded coordinate-specific logic and enables more maintainable, reusable, and physically faithful
software in both research and educational settings.

# Methodology

The core methodology behind PyMetric centers on a mathematically rigorous yet computationally practical
framework for performing differential geometry operations in arbitrary coordinate systems. The library
is designed to support seamless transitions between symbolic derivation and numerical evaluation, allowing
for precise, efficient, and geometry-aware modeling.

A coordinate system in PyMetric is defined minimally by:

- A set of axes labels $(x^1, x^2, \ldots)$,
- Forward and inverse transformations between these coordinates and Cartesian Space $T(x,y,z)$ and
  $T^{-1}(x^1,x^2,x^3)$.
- A symbolically defined metric tensor $g_{\mu\nu}$.

From this core specification, PyMetric constructs key geometric quantities, such as the inverse metric
$g^{\mu\nu}$, the metric density $\sqrt{g}$, and terms appearing in differential operations, such as
$L^\nu = g^{-1/2} \partial_\mu (g^{1/2} g^{\mu\nu})$, which appears in the scalar Laplacian
$\nabla^2 \phi = L^\nu \partial_\nu \phi + g^{\mu\nu} \partial^2_{\mu\nu} \phi$. These are represented both as symbolic
expressions using **Sympy** and as numerical equivalents which are converted from Sympy into
native NumPy functions. hese quantities are computed lazily: they are only derived when required for a specific operation,
avoiding unnecessary overhead.

Coordinate systems are categorized into types (e.g., orthogonal or curvilinear) that determine
how symbolic properties are derived and which simplifications may apply. This abstraction enables users
to model highly symmetric systems (e.g., spherical or ellipsoidal coordinates) as easily
as more general curvilinear systems.

## Field and Grid Operations

Fields in PyMetric are array-backed data structures (typically NumPy or HDF5 buffers) that are explicitly
associated with a coordinate system and grid. While fields behave like standard NumPy arrays,
they also carry metadata about their geometric context, including coordinate labels, spacing,
and metric-aware tensor properties.

Operations on fields—such as computing covariant derivatives, applying Laplacians,
or transforming between bases—are automatically dispatched to appropriate symbolic
expressions and numerical kernels based on the field’s variance and the geometry of the underlying
coordinate system.

This design allows users to write high-level, reusable code that is agnostic to the specific geometry,
while still benefiting from the mathematical correctness and efficiency of coordinate-aware computation.


## Future Development

The development roadmap for PyMetric is focused on deepening its mathematical capabilities and expanding
its utility in advanced physical modeling contexts, particularly those involving curved and relativistic
spacetimes. While the current implementation supports a robust suite of differential operators in orthogonal
and curvilinear coordinate systems, several avenues for future growth are planned:

1. Expanded Differential Operator Support:

   PyMetric will be extended to support a broader range of tensor calculus operations, including:

   - Covariant derivatives of higher-rank tensors, enabling modeling of tensor transport and geodesic deviation.
   - Higher-order differential operators, such as biharmonic or fourth-order Laplacians,
     which are essential in elasticity theory, advanced fluid dynamics, and quantum field theory.
   - Tensor contractions and curvature operations, including the Riemann, Ricci, and Einstein tensors,
     to support simulations in general relativity and cosmology.

These features will allow PyMetric to serve as a general-purpose differential geometry engine suitable
for high-fidelity modeling in physics, engineering, and applied mathematics.

2. Relativistic and Non-Flat Coordinate Systems

  - A key area of expansion is support for relativistic geometries, where the metric tensor is no longer positive-definite and may depend dynamically on spacetime coordinates. Planned features include:
  - General Lorentzian manifolds, including Schwarzschild, Kerr, and FLRW spacetimes, enabling direct modeling of astrophysical systems governed by Einstein’s field equations.

PyMetric is explicitly intended as a modeling and analysis tool, not a time-domain simulation engine.
It provides geometric infrastructure for constructing and analyzing equations defined on curved spacetimes,
but does not aim to solve dynamical systems or perform numerical integration of time-evolving fields.

# References
