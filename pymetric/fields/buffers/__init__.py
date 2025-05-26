"""
Unified buffer system for backend-agnostic field storage in PyMetric.

This module defines the abstract infrastructure for managing numerical data buffers
in PyMetric fields. The buffer system enables :mod:`fields` and its components
to store and manipulate data independently of the underlying storage backend.

Each buffer backend implements a common interface via :py:class:`~fields.buffers.base.BufferBase`, ensuring consistent behavior
for array operations, unit handling, and I/O across field components.

This infrastructure allows developers to build fields and perform computations without
committing to a particular memory or storage strategy, facilitating scalability, portability,
and unit-safe numerical workflows.

For more details on buffers, see :ref:`buffers`.
"""
__all__ = [
    "ArrayBuffer",
    "UnytArrayBuffer",
    "HDF5Buffer",
    "BufferRegistry",
    "__DEFAULT_BUFFER_REGISTRY__",
    "resolve_buffer_class",
    "buffer_zeros",
    "buffer_ones",
    "buffer_full",
    "buffer_empty",
    "buffer_zeros_like",
    "buffer_ones_like",
    "buffer_full_like",
    "buffer_empty_like",
    "buffer",
    "buffer_from_array",
]
from .base import buffer_from_array
from .core import ArrayBuffer, HDF5Buffer, UnytArrayBuffer
from .registry import __DEFAULT_BUFFER_REGISTRY__, BufferRegistry, resolve_buffer_class
from .utilities import (
    buffer,
    buffer_empty,
    buffer_empty_like,
    buffer_full,
    buffer_full_like,
    buffer_ones,
    buffer_ones_like,
    buffer_zeros,
    buffer_zeros_like,
)
