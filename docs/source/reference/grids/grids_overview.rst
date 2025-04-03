.. _grids:

====================
Geometric Grids
====================

Geometric grids in Pisces-Geometry form the spatial backbone for all data fields and computations. A grid is a combination
of a **generic grid structure** and a **coordinate system**. This design allows users to precisely map data to physical
space, manage boundary conditions with ghost zones, and leverage efficient chunked operations on large domains.

Basic Usage
-----------

In this section you will learn the fundamentals of setting up a grid, including:

- Defining a grid from a coordinate system, a boundary, and a few other core parameters. You'll also be introduced
  to a number of important auxiliary arguments that can significantly improve the quality of your code.
- Grids in Pisces-Geometry are designed to optimize memory usage and computational performance. You’ll see how chunking
  (dividing the grid into smaller subdomains) can be enabled to efficiently process large-scale data and how grids store
  minimal required data to avoid flooding the memory space.
- Understand how the grid stores essential metadata such as the domain dimensions, bounding boxes, and axis names,
  which are critical for coordinate-aware operations.
- Understand how to perform differential operations on arrays over the specified coordinate grid.

Choosing A Grid Type
''''''''''''''''''''

The first step in using Pisces-Geometry grids is to select what kind of grid suits your purpose best.
Pisces-Geometry offers several grid classes to suit different application needs:

- **Generic Grids** (:py:class:`~grids.base.GenericGrid`) are grids which are defined by providing the
  grid points along each axis. This is the most generic grid available in Pisces-Geometry and allows for users to define
  grids with selective resolution and other properties.
- **Scaled Grids** (:py:class:`~grids.base.ScaledGrid`) are grids which permit the user to define a
  "scaling" under which the grid becomes uniform. The most common form of this is so-called log-scaled grids where one
  or many of the axes are uniform in log-space. Some processes are simplified in these grids and they take up less memory
  space but are less flexible.
- **Uniform Grids** (:py:class:`~grids.base.UniformGrid`) are grids with uniform spacing between elements.
  This is a very simple grid structure and very efficient, but without much flexibility.

.. note::

    In future releases of the code, we intent to also allow for meshgrid structures which are really composed of many
    uniform grids with different locations covering a single larger area. This will then be the most flexible (but most complex)
    of the available grids.

Initializing a Grid
''''''''''''''''''''

- Setting the coordinate system, the ghost zones, the chunking, etc.

Extracting Coordinates from a Grid
''''''''''''''''''''''''''''''''''

- Extract the coordinate arrays.
- Extract the coordinate grid.
- Extract in chunks.
- Extract for specific axes.
