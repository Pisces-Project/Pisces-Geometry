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

Coordinate systems form the backbone of Pisces-Geometry and are the most useful class to interact with. These guides will
familiarize you with how to interact with them and how to write custom ones.

.. toctree::
    :titlesonly:
    :glob:

    ./coordinates/user
    ./coordinates/dev

Grids
++++++++++++++++++

Coordinate systems form the backbone of Pisces-Geometry and are the most useful class to interact with. These guides will
familiarize you with how to interact with them and how to write custom ones.