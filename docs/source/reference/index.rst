.. _reference:

.. image:: ../images/PyMetric.png
   :width: 300px
   :align: center

PyMetric User Guide
=======================

The PyMetric package is a sub-component of the larger Pisces ecosystem for astrophysical modeling. It started
out as the geometric backend for Pisces but has since grown to warrant a self-contained code base with its own documentation
and installation. In this guide, we'll introduce the basis of PyMetric and describe how it can be used for various
purposes.

.. contents::
   :local:
   :depth: 2

Overview
--------

PyMetric has the following core goals in its development:

1. To support **differential** and **algebraic** operations in a wide variety of coordinate systems.
2. To optimize operations in complex coordinate systems to take advantage of **natural symmetries**.
3. To provide data structures for **self-consistently** storing / manipulating data in general coordinate systems.

Background
----------

As a mathematics package, PyMetric relies on some advanced concepts and methods from differential geometry, tensor
analysis, and related fields. Many functionalities of the library are easy enough to use without deep knowledge of
these areas; however, some functionalities do require a degree of background in the relevant fields. Here we
provide a few documents and external references for interested readers to brush up on the necessary mathematics.

.. hint::

    The new user might find it worthwhile to dig into the code documentation until (if) they find
    theoretical stumbling blocks, at which point returning here will likely answer any questions that
    have arisen.

.. toctree::
    :titlesonly:
    :glob:

    ./theory/coordinate_theory

PyMetric Core Modules
---------------------

There are 3 core modules in PyMetric: :mod:`coordinates`, :mod:`grids`, and :mod:`fields`. In this section, we'll walk
through these modules in some detail to provide all of the necessary information to make effective use of the library.

Coordinate Systems
++++++++++++++++++

Coordinate systems define the geometry of space in PyMetric and are housed in the :mod:`coordinates` module.
These documents walk through how coordinate systems are structured, how they support differential operations,
and how users can implement custom coordinate systems tailored to their own physical domains.
Whether you're using built-in systems or designing your own, this is the place to start.

.. toctree::
    :titlesonly:
    :glob:

    ./coordinates/overview
    ./coordinates/building_coordinate_systems

Grids
++++++++++++++++++

Grids are discrete representations of space built on top of coordinate systems. They are effectively the second
layer of abstraction in PyMetric. The :mod:`grids` module holds all of the relevant methods and classes. This section
explains the different types of grids supported by PyMetric, how they are constructed, and how they interact
with coordinate systems. You’ll also find guidance for building custom grids for advanced use cases.

.. toctree::
    :titlesonly:
    :glob:

    ./grids/overview
    ./grids/build_custom_grids

Fields
++++++++++++++++++

Fields are the primary data structures for storing values over grids—scalars, vectors, tensors, and beyond.
These guides cover how to create, manipulate, and operate on fields in a coordinate-aware way. You'll learn how fields
support broadcasting, NumPy compatibility, and differential operators like gradients and divergences.

.. toctree::
    :titlesonly:
    :glob:

    ./fields/overview
    ./fields/numpy
    ./fields/buffers
    ./fields/components

PyMetric Auxiliary Modules
---------------------------

In support of the 3 core modules of the library, there are a number of auxiliary modules which are
described briefly in the following documents.

Low-Level Differential Geometry
++++++++++++++++++++++++++++++++

.. toctree::
    :titlesonly:
    :glob:

    ./differential_geometry/overview
    ./differential_geometry/dependence

PyMetric Developer Documents
----------------------------

PyMetric is an open source project and encourages contributions from the community. In this section,
we provide a number of documents about how to develop the software and also provide more detailed guides
about how each of the relevant modules actually works.

.. note::

    Documentation in Progress.

Core Module Dev Documentation
++++++++++++++++++++++++++++++

.. toctree::
    :titlesonly:
    :glob:

    ./coordinates/dev
    ./grids/dev
    ./fields/dev
