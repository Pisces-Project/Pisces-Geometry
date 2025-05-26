.. _api:

API
===

This page contains the complete API documentation for all public components of the Pisces-Geometry library. The
various modules have been broken down by the category into which they best fit.


Core Modules
-------------------
These modules provide the primary interface for interacting with coordinate systems, structured grids, and data fields.
Most users will work directly with these components.


.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    coordinates
    grids
    fields

Mathematical Modules
--------------------
These modules implement core mathematical operations—such as differential geometry—that extend Pisces-Geometry to support physical modeling and symbolic computation.


.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    differential_geometry


Developer Utilities
-------------------
These are internal utility modules used throughout the codebase. While not typically needed for everyday use, they are helpful for contributors and advanced users.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    utilities
