---
title: 'PyMetric: A Geometry Informed Array Mathematics Package'
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

---

# Summary

PyMetric is a lightweight Python library designed to streamline differential geometry
and vector calculus operations in user-defined coordinate systems, with a focus on applications
in astrophysics and computational physics. In many physical modeling tasks, it is both natural
and advantageous to work in non-Cartesian coordinate systems that align with the inherent
symmetries of the system. These systems can be highly nontrivial—such as ellipsoidal
(homoeoidal) or spheroidal coordinates—where explicitly handling coordinate-specific
expressions becomes tedious and error-prone. PyMetric provides a unified abstraction
that decouples the underlying coordinate representation from the operations themselves,
allowing users to accurately compute gradients, divergences, Laplacians, and related geometric
quantities through a consistent interface. This makes it easier to prototype and scale
models in complex geometries without having to rewrite operations for each coordinate system.

The core design of PyMetric relies on a hybrid symbolic-numeric model that balances efficiency,
flexibility, and accuracy. Symbolic computation is used to derive key geometric quantities, such
as metric tensors, Christoffel symbols, and Jacobians, directly from the structure of the coordinate
system. These symbolic expressions preserve the full geometric context and can be reused across
multiple evaluations. Once derived, they are compiled into optimized numerical functions that
can be efficiently applied to array data on structured grids. This approach allows PyMetric to
support coordinate-aware computation with minimal overhead, avoiding the need for repeated symbolic
manipulation during runtime, while maintaining high accuracy through analytically correct geometric
expressions. The result is a powerful and extensible framework that enables NumPy-style workflows
in complex coordinate systems without sacrificing physical fidelity.

In addition to its symbolic-numeric foundation, PyMetric provides structured abstractions for grids
and field data, supporting a range of coordinate systems and buffer backends—including in-memory arrays and
HDF5[@hdf5] storage for scalable computation. Users can define fields over geometric grids and apply
differential operators without manually managing coordinate-dependent logic.

PyMetric automates core operations such as gradients, divergences, and Laplacians in a
geometry-aware fashion, enabling accurate and efficient modeling of physical systems across
disciplines like general relativity, magnetohydrodynamics, and planetary dynamics. By embedding
geometric structure directly into array-based workflows, PyMetric offers a modern, extensible
foundation for scientific computing in complex coordinate geometries.

# Statement of need

Modern astrophysical modeling requires a high degree of flexibility—both in physical assumptions
and in computational infrastructure. The Pisces Project (of which ``PyMetric`` is a part)
is a general-purpose model-building framework for astrophysics that aims to unify and extend
existing tools for generating models and initial conditions (e.g., DICE[@perret2016dice], GALIC[@yurrin2014galic])
under a common, modular API.
Its goal is to make it easier to construct complex, physically motivated models
of systems such as galaxies, black holes, or relativistic fluids by exposing a simple and extensible
interface for defining models, fields, and dynamics.

A persistent challenge in building such extensible modeling tools is the limited and inconsistent
support for coordinate systems found in most existing software. Codes like EinsteinPy[@einsteinpy],
DICE[@perret2016dice], and yt[@turk2010yt] often
hard-code assumptions about coordinate geometry, making them difficult to generalize to new physical
contexts or non-Cartesian coordinate systems. This lack of abstraction limits reusability and complicates
the construction of unified modeling workflows across domains such as general relativity,
galactic dynamics, and fluid mechanics.

To address this limitation, PyMetric was developed to be a lightweight library that standardizes
coordinate-aware geometric computation. The library is designed to serve as the geometric backend
for Pisces and similar modeling systems. It provides a consistent abstraction layer for defining
coordinate systems, computing differential geometric quantities, and evaluating operators like gradients,
divergences, and Laplacians—all without requiring the user to manage low-level
details of tensor algebra or coordinate transformations.

PyMetric emphasizes extensibility and modularity through four core interfaces:

- **Coordinate System API** – Enables the definition and use of arbitrary coordinate systems with minimal required knowledge, while supporting symbolic derivation of metric-dependent quantities.
- **Buffer API** – Provides a backend-agnostic interface for array storage, allowing seamless integration with systems like HDF5, XArray, Dask, and unit-aware arrays.
- **Differential Geometry API** – Implements low-level, coordinate-independent formulations of core operations such as gradients, divergences, Laplacians, and volume elements.
- **Grid and Field API** – Supports flexible discretization strategies and a variety of field types, including sparse and dense scalar, vector, and tensor fields.

Together, these abstractions form a unified symbolic-numeric pipeline that allows high-level modeling code to operate naturally across diverse geometries and data representations. By standardizing geometric computation and decoupling it from specific coordinate assumptions or backend implementations, PyMetric addresses a longstanding gap in scientific Python infrastructure. This foundation enables Pisces to offer a powerful, composable, and user-friendly environment for building physically accurate models in astrophysics and beyond.

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
expressions-using SymPy[@sympy]-and as NumPy-backed callables. These quantities are computed lazily: they are only derived when required for a specific operation,
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
