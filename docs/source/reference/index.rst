.. _reference:
Pisces-Geometry User Guide
===========================

The Pisces-Geometry package is a sub-component of the larger Pisces ecosystem for astrophysical modeling. It started
out as the geometric backend for Pisces but has since grown to warrant a self-contained code base with its own documentation
and installation. In this guide, we'll introduce the basis of Pisces-Geometry and describe how it can be used for various
purposes.

Overview
--------

Pisces-Geometry has the following core goals in its development:

1. To support **differential** and **algebraic** operations in a wide variety of coordinate systems.
2. To optimize operations in complex coordinate systems to take advantage of **natural symmetries**.
3. To provide data structures for **self-consistently** storing / manipulating data in general coordinate systems.

Theory
------

The theory of Pisces-Geometry rests firmly in (somewhat elementary) differential geometry, which is a deep and complex topic
all its own. Below, we have composed a few documents giving a basic summary of the core results that are relevant to Pisces
and its implementation.

.. toctree::
    :titlesonly:
    :glob:

    ./theory/coordinate_theory

Using Pisces-Geometry
---------------------

Once you're comfortable with some of the underlying theory, these documents will provide a guide to using and building the
various components of Pisces-Geometry.

Coordinate Systems
++++++++++++++++++

Coordinate systems define the geometry of space in Pisces-Geometry. These documents walk through how coordinate systems
are structured, how they support differential operations, and how users can implement custom coordinate systems tailored
to their own physical domains. Whether you're using built-in systems or designing your own, this is the place to start.

.. toctree::
    :titlesonly:
    :glob:

    ./coordinates/user
    ./coordinates/dev

Grids
++++++++++++++++++

Grids are discrete representations of space built on top of coordinate systems. This section explains the different types
of grids supported by Pisces-Geometry, how they are constructed, and how they interact with coordinate systems. You’ll also
find guidance for building custom grids for advanced use cases.

.. toctree::
    :titlesonly:
    :glob:

    ./grids/grids_overview
    ./grids/grids_developer

Fields
++++++++++++++++++

Fields are the primary data structures for storing values over grids—scalars, vectors, tensors, and beyond.
These guides cover how to create, manipulate, and operate on fields in a coordinate-aware way. You'll learn how fields
support broadcasting, NumPy compatibility, and differential operators like gradients and divergences.

.. toctree::
    :titlesonly:
    :glob:

    ./fields/fields_overview
    ./fields/fields_developer
