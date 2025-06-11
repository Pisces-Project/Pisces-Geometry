:orphan:

Core Modules: Fields
====================

This gallery showcases examples based on the :mod:`fields` module, PyMetric's core abstraction
for physical quantities defined over space—scalars, vectors, tensors, and more.

The :mod:`fields` module provides a unified interface for working with field components,
buffers, and coordinate-aware differential operations. These examples walk through how to:

- Construct scalar, vector, and tensor fields over structured grids,
- Manage physical units and buffer types (e.g., NumPy, HDF5),
- Broadcast, expand, reduce, and reshape field components,
- Apply differential geometry operations like gradients, divergence, and Laplacians,
- Combine symbolic metadata with numerical buffers for hybrid workflows.

From initializing a field from an analytic expression to computing its divergence in curved space,
these examples demonstrate the expressive and extensible field API at the heart of PyMetric.

.. hint::

    For complete documentation, see :ref:`fields`.
