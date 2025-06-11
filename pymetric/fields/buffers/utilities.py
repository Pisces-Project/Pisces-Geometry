"""
Utility functions for working with PyMetric buffer backends.

This module provides high-level entry points to the buffer system, including
the :py:func:`~fields.buffers.base.buffer_from_array` function which resolves the appropriate buffer class
for a given array-like input.

These utilities simplify buffer construction from generic data formats and
are recommended for use in user-facing APIs or internal preprocessing steps
that must remain backend-agnostic.
"""
from typing import TYPE_CHECKING, Any, Optional, Sequence, Type

from numpy.typing import ArrayLike

from pymetric.fields.buffers.registry import (
    __DEFAULT_BUFFER_REGISTRY__,
    resolve_buffer_class,
)

if TYPE_CHECKING:
    from .base import BufferBase
    from .core import ArrayBuffer
    from .registry import BufferRegistry


# ================================= #
# Buffer Creation Logic             #
# ================================= #
# These functions all handle creating new
# buffers.
def buffer(
    array: ArrayLike,
    *args,
    buffer_class: Optional[Type["BufferBase"]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    **kwargs,
) -> "BufferBase":
    """
    Wrap an array-like object in a Pisces buffer instance.

    This is the high-level constructor for buffer creation, analogous to ``np.array(...)``.
    It resolves the appropriate buffer backend (e.g., NumPy, :mod:`unyt`, HDF5) and returns
    a fully initialized buffer instance from the input array.

    This is the recommended entry point for turning arbitrary user data into a
    structured buffer within the Pisces field system.

    Parameters
    ----------
    array : array-like
        Any object that can be converted into a backend-compatible array, such as a
        :class:`list`, :class:`tuple`, :class:`numpy.ndarray`, :class:`~unyt.array.unyt_array`, or other supported type.
    *args :
        Additional positional arguments forwarded to the resolved buffer class's :meth:`~fields.buffers.base.BufferBase.from_array` method.
    buffer_class : :class:`~fields.buffers.base.BufferBase`, optional
        Explicit buffer class to use for wrapping. If provided, :meth:`~fields.buffers.base.BufferBase.from_array` is called on this class.
    buffer_registry : :class:`~fields.buffers.registry.BufferRegistry`, optional
        Registry to use when resolving `array` into a buffer class. Defaults to the global registry.
    **kwargs :
        Additional keyword arguments forwarded to the buffer class's :meth:`~fields.buffers.base.BufferBase.from_array` method. This includes
        things like `dtype`, `copy`, `order`, or HDF5-specific parameters.

    Returns
    -------
    :class:`~fields.buffers.base.BufferBase`
        A fully constructed buffer instance containing the wrapped array.

    Raises
    ------
    TypeError
        If the input cannot be resolved into a supported buffer class.
    ValueError
        If resolution fails due to misconfiguration of buffer class or registry.

    See Also
    --------
    buffer_zeros
    buffer_ones
    buffer_full
    buffer_empty

    Examples
    --------
    Convert a list to an :class:`~fields.buffers.core.ArrayBuffer`:

    >>> from pymetric.fields.buffers.utilities import buffer
    >>> import numpy as np
    >>>
    >>> b = buffer([1, 2, 3])
    >>> type(b).__name__
    'ArrayBuffer'
    >>> b.as_array()
    array([1, 2, 3])

    Creating an HDF5 buffer from a list:

    >>> b = buffer([1, 2, 3],file='test.hdf5',path='test',overwrite=True,buffer_class='HDF5Buffer')
    >>> type(b).__name__
    'HDF5Buffer'
    >>> b.as_array()
    array([1, 2, 3])
    >>> np.add(b,b,out=b)
    >>> b.as_array()

    """
    # Determine what class we are creating.
    if buffer_registry is None:
        buffer_registry = __DEFAULT_BUFFER_REGISTRY__

    if buffer_class is None:
        return buffer_registry.resolve(array, *args, **kwargs)
    else:
        buffer_class = resolve_buffer_class(
            buffer_class=buffer_class, buffer_registry=buffer_registry, default=None
        )
        return buffer_class.from_array(array, *args, **kwargs)


def buffer_zeros(
    shape: Sequence[int],
    *args,
    buffer_class: Optional[Type["BufferBase"]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    **kwargs,
) -> "BufferBase":
    """
    Create a new buffer filled with zeros.

    Parameters
    ----------
    shape : list of int
        The desired shape of the buffer.
    *args :
        Positional arguments passed through to the buffer constructor.
    buffer_class : :class:`~fields.buffers.base.BufferBase`, optional
        Specific buffer class to use. If None, uses the default (ArrayBuffer).
    buffer_registry : :class:`~fields.buffers.registry.BufferRegistry`, optional
        Registry to resolve buffer class by type or name.
    **kwargs :
        Additional keyword arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A zero-initialized buffer instance.
    """
    buffer_class = resolve_buffer_class(
        buffer_class=buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
    )
    return buffer_class.zeros(shape, *args, **kwargs)


def buffer_empty(
    shape: Sequence[int],
    *args,
    buffer_class: Optional[Type["BufferBase"]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    **kwargs,
) -> "BufferBase":
    """
    Create a new buffer with uninitialized values.

    Parameters
    ----------
    shape : list of int
        The desired shape of the buffer.
    *args :
        Positional arguments passed through to the buffer constructor.
    buffer_class : :class:`~fields.buffers.base.BufferBase`, optional
        Specific buffer class to use. If None, uses the default (ArrayBuffer).
    buffer_registry : :class:`~fields.buffers.registry.BufferRegistry`, optional
        Registry to resolve buffer class by type or name.
    **kwargs :
        Additional keyword arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A buffer with uninitialized contents.
    """
    buffer_class = resolve_buffer_class(
        buffer_class=buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
    )
    return buffer_class.empty(shape, *args, **kwargs)


def buffer_ones(
    shape: Sequence[int],
    *args,
    buffer_class: Optional[Type["BufferBase"]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    **kwargs,
) -> "BufferBase":
    """
    Create a new buffer filled with ones.

    Parameters
    ----------
    shape : list of int
        The desired shape of the buffer.
    *args :
        Positional arguments passed through to the buffer constructor.
    buffer_class : :class:`~fields.buffers.base.BufferBase`, optional
        Specific buffer class to use. If None, uses the default (ArrayBuffer).
    buffer_registry : :class:`~fields.buffers.registry.BufferRegistry`, optional
        Registry to resolve buffer class by type or name.
    **kwargs :
        Additional keyword arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A one-initialized buffer instance.
    """
    buffer_class = resolve_buffer_class(
        buffer_class=buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
    )
    return buffer_class.ones(shape, *args, **kwargs)


def buffer_full(
    shape: Sequence[int],
    *args,
    fill_value: Any = 0.0,
    buffer_class: Optional[Type["BufferBase"]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    **kwargs,
) -> "BufferBase":
    """
    Create a new buffer filled with a constant value.

    Parameters
    ----------
    shape : list of int
        The desired shape of the buffer.
    fill_value : Any, default 0.0
        The value to fill the buffer with.
    *args :
        Positional arguments passed through to the buffer constructor.
    buffer_class : :class:`~fields.buffers.base.BufferBase`, optional
        Specific buffer class to use. If None, uses the default (ArrayBuffer).
    buffer_registry : :class:`~fields.buffers.registry.BufferRegistry`, optional
        Registry to resolve buffer class by type or name.
    **kwargs :
        Additional keyword arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A constant-filled buffer instance.
    """
    buffer_class = resolve_buffer_class(
        buffer_class=buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
    )
    return buffer_class.full(shape, *args, fill_value=fill_value, **kwargs)


def buffer_zeros_like(other: "BufferBase", *args, **kwargs) -> "BufferBase":
    """
    Create a buffer filled with zeros, matching the shape of another buffer.

    Parameters
    ----------
    other : ~fields.buffers.base.BufferBase
        The buffer whose shape is used for the new one.
    *args, **kwargs :
        Additional arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A zero-initialized buffer with the same shape as `other`.
    """
    return buffer_zeros(other.shape, *args, **kwargs)


def buffer_ones_like(other: "BufferBase", *args, **kwargs) -> "BufferBase":
    """
    Create a buffer filled with ones, matching the shape of another buffer.

    Parameters
    ----------
    other : ~fields.buffers.base.BufferBase
        The buffer whose shape is used for the new one.
    *args, **kwargs :
        Additional arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A one-initialized buffer with the same shape as `other`.
    """
    return buffer_ones(other.shape, *args, **kwargs)


def buffer_full_like(
    other: "BufferBase", fill_value: Any = 0.0, *args, **kwargs
) -> "BufferBase":
    """
    Create a buffer filled with a constant value, matching the shape of another buffer.

    Parameters
    ----------
    other : ~fields.buffers.base.BufferBase
        The buffer whose shape is used for the new one.
    fill_value : Any, default 0.0
        The value to fill the buffer with.
    *args, **kwargs :
        Additional arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A buffer filled with `fill_value` and the same shape as `other`.
    """
    return buffer_full(other.shape, *args, fill_value=fill_value, **kwargs)


def buffer_empty_like(other: "BufferBase", *args, **kwargs) -> "BufferBase":
    """
    Create a buffer with uninitialized values, matching the shape of another buffer.

    Parameters
    ----------
    other : ~fields.buffers.base.BufferBase
        The buffer whose shape is used for the new one.
    *args, **kwargs :
        Additional arguments forwarded to the buffer constructor.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        An uninitialized buffer with the same shape as `other`.
    """
    return buffer_empty(other.shape, *args, **kwargs)
