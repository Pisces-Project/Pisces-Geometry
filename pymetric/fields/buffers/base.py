"""
Buffer base classes and buffer resolution support.

This module defines the core :py:class:`BufferBase` class, which all buffer types must subclass,
and the metaclass :class:`_BufferMeta`, which manages registration into the
default buffer registry and enforces interface correctness.

The buffer system abstracts different data storage backends (NumPy, unyt, HDF5, etc.)
behind a common interface so that field operations can delegate storage concerns. Novel buffer
classes can be implemented with relative ease vis-a-vis subclasses of :py:class:`BufferBase`.
"""
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import unyt
from numpy.typing import ArrayLike

from pymetric.fields.mixins._generic import NumpyArithmeticMixin

from .registry import __DEFAULT_BUFFER_REGISTRY__, BufferRegistry


# ========================================= #
# Buffer Meta Class                         #
# ========================================= #
class _BufferMeta(ABCMeta):
    """
    Metaclass for all Pisces buffer classes.

    This metaclass automatically registers concrete buffer classes with the
    global `__DEFAULT_BUFFER_REGISTRY__` if they are not abstract and define
    the `__can_resolve__` attribute.

    Expected class attributes:
    ---------------------------
    - __is_abc__ : bool
        Whether the class is abstract. If True, registration is skipped.

    - __can_resolve__ : List[Type] or None
        A list of array-like types (e.g. `np.ndarray`, `unyt_array`) that this buffer can wrap.
        Must be defined on all concrete buffer classes.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        # Create the generic class object with a call to super().
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Extract the class flags and use them to determine the triaging
        # behavior.
        is_abstract = getattr(cls, "__is_abc__", False)

        if is_abstract:
            return cls

        # Validate the __can_resolve__ attribute. This requires managing
        # the various possibly typing conventions.
        can_resolve = getattr(cls, "__can_resolve__", None)
        if can_resolve is None:
            raise TypeError(
                f"Concrete buffer subclass '{name}' must define "
                f"`__can_resolve__` (a type or iterable of types)."
            )

        # Accept single type or iterable of types
        if isinstance(can_resolve, type):
            can_resolve = (can_resolve,)

        if not isinstance(can_resolve, Iterable):
            raise TypeError(
                f"'{name}.__can_resolve__' must be a type or an iterable of types, "
                f"got object of type {type(can_resolve).__name__}."
            )

        # Ensure every entry is itself a 'type'
        bad_entries = [t for t in can_resolve if not isinstance(t, type)]
        if bad_entries:
            bad_str = ", ".join(repr(t) for t in bad_entries)
            raise TypeError(
                f"All entries in '{name}.__can_resolve__' must be types; "
                f"found invalid entries: {bad_str}"
            )

        # Store back the normalised tuple so the rest of the code can rely on it
        cls.__can_resolve__ = tuple(can_resolve)

        __DEFAULT_BUFFER_REGISTRY__.register(cls)
        return cls


# ========================================= #
# Abstract Base Class (BufferBase)          #
# ========================================= #
class BufferBase(NumpyArithmeticMixin, ABC, metaclass=_BufferMeta):
    """
    Abstract base class for Pisces Geometry-compatible field buffers.

    This interface abstracts data storage details so that field operations can be
    performed uniformly regardless of whether the underlying data is NumPy, unyt, HDF5, etc.
    """

    # === Class Attributes === #
    # These attributes configure behavior and registration rules for buffer subclasses.
    # All **concrete** buffer classes MUST define these attributes explicitly.
    #
    # Abstract classes may omit them by setting __is_abc__ = True.
    __is_abc__: bool = True
    """
    Marks the class as abstract (not to be registered).
    Set to `False` on all concrete subclasses.
    """
    __can_resolve__: List[Type] = NotImplemented
    """
    A list of data types (e.g., [np.ndarray, unyt_array]) that this buffer
    can wrap via the `coerce()` method.

    This is required for automatic buffer resolution.
    Must be defined on concrete subclasses.
    """
    __core_array_types__: Optional[Tuple[Type, ...]] = None
    """
    Type(s) that the buffer expects to wrap in its constructor (`__init__`).

    Used by `__validate_array_object__()` to ensure the input array object is valid.
    Typically set to `np.ndarray`, `unyt_array`, or `h5py.Dataset`.
    If `None`, no validation is performed.
    """
    __representation_types__: Optional[Tuple[Type, ...]] = None
    """
    The type(s) that can be represented with this buffer class.

    Currently only used when performing tests. Should still be implemented
    in all subclasses.
    """
    __resolution_priority__: int = 0
    """
    Optional integer priority used during buffer resolution.

    Lower numbers are prioritized first when resolving unknown array-like inputs.
    Used by buffer registries that support resolution ordering.
    """
    __array_priority__ = 2.0
    """
    The priority of the buffer class in numpy operations.
    """
    __array_function_dispatch__: Optional[Dict[Callable, Callable]] = {
        # Internally defined methods.
        np.copy: lambda self, *args, **kwargs: self.copy(*args, **kwargs),
        np.transpose: lambda self, *args, **kwargs: self.transpose(*args, **kwargs),
        np.reshape: lambda self, *args, **kwargs: self.reshape(*args, **kwargs),
        np.ravel: lambda self, *args, **kwargs: self.flatten(*args, **kwargs),
        np.squeeze: lambda self, *args, **kwargs: self.squeeze(*args, **kwargs),
        np.expand_dims: lambda self, *args, **kwargs: self.expand_dim(*args, **kwargs),
        np.broadcast_to: lambda self, *args, **kwargs: self.broadcast_to(
            *args, **kwargs
        ),
        # Internally defined properties.
        np.shape: lambda self, *_, **__: self.shape,
        np.ndim: lambda self, *_, **__: self.ndim,
        np.size: lambda self, *_, **__: self.size,
        # Simple redirect transformations.
        np.moveaxis: lambda self, s, d, *args, **kwargs: self._apply_numpy_transform_on_repr(
            np.moveaxis, [s, d], {}, *args, **kwargs
        ),
        np.swapaxes: lambda self, ax1, ax2, *args, **kwargs: self._apply_numpy_transform_on_repr(
            np.swapaxes, [ax1, ax2], {}, *args, **kwargs
        ),
        np.tile: lambda self, reps, *args, **kwargs: self._apply_numpy_transform_on_repr(
            np.tile, [reps], {}, *args, **kwargs
        ),
    }
    """
    `__array_function_dispatch__` is a dictionary which can optionally map
    NumPy callables to internal implementations to allow overriding of default behavior.

    By default, when a NumPy function (non ufunc) is called on a Buffer, the buffer
    is stripped and the operation occurs on the underlying representation. If a callable
    is specified here, then `__array_function__()` will catch the redirect and
    triage accordingly.
    """

    # === Initialization === #
    # The initialization procedure should be meta stable
    # in the sense that it always behaves the same way: __init__
    # requires a pre-coerced type and simply checks for type compliance.
    # Other methods can be used for more adaptive behavior.
    def __init__(self, array_object: ArrayLike):
        """
        Initialize a buffer from a validated array-like object.

        This constructor assumes that the input is already fully compatible with the
        expected core array type for this buffer class (e.g., :py:class:`numpy.ndarray`, :py:class:`unyt.unyt_array`, etc.).
        No validation or coercion of the data is performed beyond checking its type.

        .. warning::

            This method does **not** attempt to coerce or sanitize the input array.
            If you pass an incompatible or incorrect array-like object, a ``TypeError``
            will be raised. For flexible or user-facing buffer construction, use
            :meth:`from_array` or :meth:`coerce` instead.

        Parameters
        ----------
        array_object : ArrayLike
            A pre-validated, backend-specific array object that will be wrapped
            by this buffer. Must be an instance of the class’s ``__core_array_types__``,
            if that attribute is defined.

        Raises
        ------
        TypeError
            If the array does not match the expected core type(s).

        See Also
        --------
        BufferBase.from_array : Preferred interface for safe buffer construction.
        BufferBase.coerce : Coerces arbitrary array-like objects into valid buffers.
        """
        self.__array_object__: ArrayLike = array_object
        self.__validate_array_object__()

    def __validate_array_object__(self):
        """
        Validate that the wrapped array object is of the expected core type.

        This method checks that the buffer's internal array (``__array_object__``)
        matches the type or types declared in ``__core_array_types__``. If this condition
        fails, a `TypeError` is raised.

        This method may be extended in subclasses to include stricter or domain-specific
        validation logic.

        Raises
        ------
        TypeError
            If the internal array object does not match any type in ``__core_array_types__``.
        """
        core_types = self.__class__.__core_array_types__

        # No validation requested → simply return.
        if core_types is None:
            return

        # Accept a single type or an iterable of types.
        if not isinstance(core_types, tuple):
            core_types = (core_types,)

        if not isinstance(self.__array_object__, core_types):
            expected_names = ", ".join(t.__name__ for t in core_types)
            raise TypeError(
                f"{self.__class__.__name__} expects array of type {expected_names}, "
                f"but got {type(self.__array_object__).__name__}. "
                "Use '.from_array()' or '.coerce()' if conversion is possible."
            )

    @classmethod
    @abstractmethod
    def from_array(
        cls, obj: Any, *args, dtype: Optional[Any] = None, **kwargs
    ) -> "BufferBase":
        """
        Attempt to construct a new buffer instance from an array-like object.

        This method is the canonical entry point for converting arbitrary array-like
        inputs into a buffer of this type. It behaves similarly to a cast operation,
        and will coerce the input as needed to match the expected backend format
        (e.g., :class:`~numpy.ndarray`, class:`~unyt.unyt_array`, etc.).

        The method should be overridden in subclasses to handle type conversion,
        unit attachment, memory layout, or any other backend-specific behavior.

        Parameters
        ----------
        obj : array-like
            Input data to be wrapped. This can be any object that is compatible with
            the backend's array casting rules—such as lists, tuples,
            NumPy arrays, unyt arrays, or backend-native types (e.g., HDF5 datasets).
            The input will be coerced into a backend-compatible array before being
            wrapped in a buffer instance. If coercion fails, a `TypeError` will be raised.
        dtype : data-type, optional
            Desired data type of the resulting array. If not specified, the type is
            inferred from `obj`.
        *args, **kwargs :
            Additional arguments to customize the construction. These may include:

            - `units` for unit-aware buffers
            - `order`, `copy`, or `device` for backend-specific configuration
            - Any arguments accepted by the backend constructor

        Returns
        -------
        BufferBase
            A new buffer instance wrapping the coerced array.

        Raises
        ------
        TypeError
            If the input cannot be coerced into a valid array for this backend.
        """
        pass

    # === Resolution Logic === #
    @classmethod
    def can_handle(cls, obj: Any) -> bool:
        """
        Return ``True`` if *obj* can be wrapped by this buffer class.

        The test is simply ``isinstance(obj, t)`` for at least one *t* in
        ``__can_resolve__``.  Subclasses should set ``__can_resolve__`` to either:

        - a single type  (e.g. ``np.ndarray``), or
        - an iterable/tuple of types (e.g. ``(np.ndarray, np.ma.MaskedArray)``).

        If a subclass leaves ``__can_resolve__`` as ``NotImplemented`` the
        method always returns ``False`` so the registry can skip it.
        """
        can = cls.__can_resolve__

        if can is NotImplemented:
            return False

        # Accept both a single type and an iterable of types.
        if not isinstance(can, (tuple, list)):
            can = (can,)

        return isinstance(obj, tuple(can))

    @classmethod
    def can_handle_list(cls) -> List[str]:
        """
        Return a list of type names that this buffer can wrap.

        This method provides a human-readable list of supported types defined in
        ``__can_resolve__``. It is typically used for debugging, diagnostics,
        or generating documentation for supported backends.

        Returns
        -------
        list of str
            The names of the supported types (e.g., ``['ndarray', 'unyt_array']``).

        Raises
        ------
        TypeError
            If ``__can_resolve__`` is not defined or not iterable.
        """
        if cls.__can_resolve__ is NotImplemented:
            return []
        return [t.__name__ for t in cls.__can_resolve__]

    # === Required Constructors === #
    @classmethod
    @abstractmethod
    def zeros(cls, shape, *args, **kwargs) -> "BufferBase":
        """
        Create a new buffer filled with zeros.

        This method constructs a new backend-specific array of the given shape,
        filled with zeros, and wraps it in a buffer instance.

        Parameters
        ----------
        shape : tuple of int
            The desired shape of the buffer, including both grid and element dimensions.

        *args :
            Positional arguments passed through to the array constructor (backend-specific).

        **kwargs :
            Additional keyword arguments passed to the array constructor. May include:

            - ``dtype``: Data type of the array (e.g., ``float32``, ``int64``)
            - ``units``: Units of the array (if supported)

        Returns
        -------
        BufferBase
            A buffer instance wrapping a zero-initialized array.
        """
        pass

    @classmethod
    @abstractmethod
    def empty(cls, shape, *args, **kwargs) -> "BufferBase":
        """
        Create a new buffer with a window into unaltered memory.

        This method constructs a new backend-specific array of the given shape, and wraps it in a buffer instance.

        Parameters
        ----------
        shape : tuple of int
            The desired shape of the buffer, including both grid and element dimensions.

        *args :
            Positional arguments passed through to the array constructor (backend-specific).

        **kwargs :
            Additional keyword arguments passed to the array constructor. May include:

            - ``dtype``: Data type of the array (e.g., ``float32``, ``int64``)
            - ``units``: Units of the array (if supported)

        Returns
        -------
        BufferBase
            A buffer instance wrapping an uninitialized array.
        """
        pass

    @classmethod
    @abstractmethod
    def ones(cls, shape, *args, **kwargs) -> "BufferBase":
        """
        Create a new buffer filled with ones.

        Constructs a backend-compatible array filled with ones and wraps it
        in a buffer instance.

        Parameters
        ----------
        shape : tuple of int
            The desired shape of the buffer, including both grid and element dimensions.

        *args :
            Positional arguments forwarded to the array constructor.

        **kwargs :
            Additional keyword arguments passed to the array constructor. May include:

            - ``dtype``: Data type of the array (e.g., ``float32``, ``int64``)
            - ``units``: Units of the array (if supported)

        Returns
        -------
        BufferBase
            A buffer instance wrapping a one-filled array.
        """
        pass

    @classmethod
    @abstractmethod
    def full(cls, shape, *args, fill_value=0.0, **kwargs) -> "BufferBase":
        """
        Create a new buffer filled with a constant value.

        This method builds a backend-specific array of the given shape and fills it
        with the provided `fill_value`. The resulting array is wrapped and returned
        as a buffer instance.

        Parameters
        ----------
        shape : tuple of int
            The desired shape of the buffer (grid + element dimensions).

        *args :
            Additional positional arguments passed to the backend constructor.

        fill_value : float, default 0.0
            The constant value to use for every element in the array.

        **kwargs :
            Additional keyword arguments passed to the array constructor. May include:

            - ``dtype``: Data type of the array (e.g., ``float32``, ``int64``)
            - ``units``: Units of the array (if supported)

        Returns
        -------
        BufferBase
            A buffer instance wrapping a constant-filled array.
        """
        pass

    @classmethod
    def zeros_like(cls, other: "BufferBase", *args, **kwargs) -> "BufferBase":
        """
        Create a new buffer filled with zeros and matching the shape of another buffer.

        This method delegates to the class's `zeros` constructor, using the shape of
        the provided buffer instance.

        Parameters
        ----------
        other : BufferBase
            The buffer whose shape will be used.
        *args :
            Additional positional arguments forwarded to `zeros`.
        **kwargs :
            Additional keyword arguments forwarded to `zeros`. Common options include:
            - `dtype` : data type of the buffer
            - `units` : physical units (for unit-aware buffers)

        Returns
        -------
        BufferBase
            A buffer filled with zeros and the same shape as `other`.
        """
        return cls.zeros(other.shape, *args, **kwargs)

    @classmethod
    def ones_like(cls, other: "BufferBase", *args, **kwargs) -> "BufferBase":
        """
        Create a new buffer filled with ones and matching the shape of another buffer.

        This method delegates to the class's `ones` constructor, using the shape of
        the provided buffer instance.

        Parameters
        ----------
        other : BufferBase
            The buffer whose shape will be used.
        *args :
            Additional positional arguments forwarded to `ones`.
        **kwargs :
            Additional keyword arguments forwarded to `ones`. Common options include:
            - `dtype` : data type of the buffer
            - `units` : physical units (for unit-aware buffers)

        Returns
        -------
        BufferBase
            A buffer filled with ones and the same shape as `other`.
        """
        return cls.ones(other.shape, *args, **kwargs)

    @classmethod
    def full_like(
        cls, other: "BufferBase", fill_value: Any = 0.0, *args, **kwargs
    ) -> "BufferBase":
        """
        Create a new buffer filled with a constant value and matching the shape of another buffer.

        This method delegates to the class's `full` constructor, using the shape of
        the provided buffer instance.

        Parameters
        ----------
        other : BufferBase
            The buffer whose shape will be used.
        fill_value : scalar or quantity, default 0.0
            The constant value to fill the buffer with.
        *args :
            Additional positional arguments forwarded to `full`.
        **kwargs :
            Additional keyword arguments forwarded to `full`. Common options include:
            - `dtype` : data type of the buffer
            - `units` : physical units (for unit-aware buffers)

        Returns
        -------
        BufferBase
            A buffer filled with the specified value and the same shape as `other`.
        """
        return cls.full(other.shape, *args, fill_value=fill_value, **kwargs)

    @classmethod
    def empty_like(cls, other: "BufferBase", *args, **kwargs) -> "BufferBase":
        """
        Create a new buffer allocation matching the shape of another buffer.

        This method delegates to the class's `empty` constructor, using the shape of
        the provided buffer instance.

        Parameters
        ----------
        other : BufferBase
            The buffer whose shape will be used.
        *args :
            Additional positional arguments forwarded to `empty`.
        **kwargs :
            Additional keyword arguments forwarded to `empty`. Common options include:
            - `dtype` : data type of the buffer
            - `units` : physical units (for unit-aware buffers)

        Returns
        -------
        BufferBase
            An unallocated buffer like `other`.
        """
        return cls.empty(other.shape, *args, **kwargs)

    # === NumPy-Like Interface === #
    def __getitem__(self, idx):
        return self.__array_object__[idx]

    def __setitem__(self, idx, value):
        self.__array_object__[idx] = value

    def __array__(self, dtype=None):
        return np.asarray(self.__array_object__, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Forward semantics for numpy operations on arrays.

        The heuristic of buffer numpy interaction is that we perform the operations
        between the `RepresentationType`s of each of the input buffers. Our returned value
        is determined by the `RepresentationType`s of each of the input buffers.

        If `out` is specified, then an attempt is made to place the result into the relevant
        buffer.
        """
        # Convert all of the inputs into their corresponding representation type. This
        # will break any lazy-loading behavior in the inputs and convert everything to
        # numpy compatible types.
        core_inputs = [
            x.as_repr() if isinstance(x, self.__class__) else x for x in inputs
        ]

        # Handle `out`: We fetch the out kwarg, check if it is a buffer type, and then
        # attempt to place the result into the buffer by specifying out=self.__array_object__.
        out = kwargs.get("out", None)
        if out is not None:
            # Normalize to a tuple for uniform processing
            is_tuple = isinstance(out, tuple)
            out_tuple = out if is_tuple else (out,)

            # Unwrap buffers
            unwrapped_out = tuple(
                o.as_core() if isinstance(o, self.__class__) else o for o in out_tuple
            )
            kwargs["out"] = unwrapped_out if is_tuple else unwrapped_out[0]

            # Apply the ufunc
            result = getattr(ufunc, method)(*core_inputs, **kwargs)

            # Pass result through based on the typing.
            if isinstance(result, tuple):
                return out_tuple
            elif result is not None:
                return out_tuple[0]
            else:
                return None

        else:
            # out was not specified, we simply return the unwrapped behavior.
            return getattr(ufunc, method)(*core_inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """
        Override NumPy high-level functions for BufferBase.

        The heuristic for this behavior is to simply delegate operations to
        the buffer representation unless there is a specific override in place.
        """
        # Check for custom forwarding implementations via
        # the __array_functions_dispatch__.
        if all(issubclass(t, self.__class__) for t in types):
            # Fetch the dispatch and check for the override of
            # this function.
            redirect_func = getattr(self, "__array_function_dispatch__", {}).get(
                func, None
            )
            if redirect_func is not None:
                # We have a redirection, we now delegate to that.
                return redirect_func(*args, **kwargs)

        # No valid dispatch found. We now strip the args down and
        # pass through without and further alterations.
        unwrapped_args = tuple(
            a.as_repr() if isinstance(a, self.__class__) else a for a in args
        )
        unwrapped_kwargs = {
            _k: _v.as_core() if isinstance(_v, self.__class__) else _v
            for _k, _v in kwargs.items()
        }
        return func(*unwrapped_args, **unwrapped_kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"

    def __str__(self):
        return self.__array_object__.__str__()

    def __len__(self) -> int:
        """
        Return the length of the buffer along its first axis.

        This is equivalent to `len(buffer.as_core())`, and will raise an error
        if the buffer has zero dimensions.

        Returns
        -------
        int
            The size of the first dimension.

        Raises
        ------
        TypeError
            If the buffer is scalar (zero-dimensional).
        """
        return len(self.__array_object__)

    def __iter__(self):
        """
        Return an iterator over the outermost dimension of the buffer.

        This allows iteration like `for row in buffer`, where each row is returned
        as a slice of the buffer. Slices are returned as NumPy arrays or `unyt_array`,
        depending on the underlying backend.

        Returns
        -------
        Iterator[Any]
            An iterator over the first dimension of the wrapped array.
        """
        return iter(self.__array_object__)

    def __eq__(self, other: Any) -> bool:
        """
        Check for equality with another buffer or array-like object.

        This uses NumPy-style broadcasting and comparison. If `other` is not a buffer,
        it will be coerced to an array for comparison. This performs an *element-wise*
        comparison and returns a boolean scalar only if the entire contents are equal.

        Parameters
        ----------
        other : Any
            Another buffer or array-like object.

        Returns
        -------
        bool
            True if the contents are equal (element-wise). False otherwise.
        """
        return self.as_core() == other

    # === Public Properties === #
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying array."""
        return self.__array_object__.shape

    @property
    def units(self):
        """
        Physical units attached to the buffer data.

        Returns
        -------
        unyt.unit_registry.Unit
            The physical units associated with this buffer’s array values.
        """
        return None

    @units.setter
    def units(self, units):
        raise ValueError(f"Class {self.__class__.__name__} does not support units.")

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.__array_object__.size

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.__array_object__.ndim

    @property
    def has_units(self) -> bool:
        """
        Whether the buffer carries unit metadata.

        This returns `True` if the buffer has an attached physical unit
        (i.e., `self.units` is not `None`), and `False` otherwise.

        Returns
        -------
        bool
            `True` if the buffer has units, `False` if it is unitless.
        """
        return self.units is not None

    @property
    def dtype(self) -> Any:
        """Data type of the array."""
        return self.__array_object__.dtype

    @property
    def c(self):
        """
        Shorthand for `as_core()`.

        This returns the raw backend-specific array (e.g., `np.ndarray`, `unyt_array`, or HDF5 dataset),
        without applying any conversions or wrapping. Useful for advanced users who want direct access.

        Equivalent to: `self.as_core()`

        Returns
        -------
        ArrayLike
            The backend-native data structure stored in this buffer.
        """
        return self.__array_object__

    @property
    def d(self):
        """
        Shorthand for `as_array()`.

        This returns the buffer data as a plain `numpy.ndarray`, stripping any units or backend context.

        Equivalent to: `self.as_array()`

        Returns
        -------
        numpy.ndarray
            The numerical contents of the buffer as a standard array.
        """
        return self.as_array()

    @property
    def v(self):
        """
        Shorthand for `as_unyt_array()`.

        This returns the buffer data as a `unyt_array`, preserving any attached physical units.

        Equivalent to: `self.as_unyt_array()`

        Returns
        -------
        unyt.unyt_array
            Unit-tagged array of the buffer's data.
        """
        return self.as_unyt_array()

    def as_array(self) -> np.ndarray:
        """
        Return the buffer as a NumPy array.

        Returns
        -------
        numpy.ndarray
        """
        return self.__array__(dtype=self.dtype)

    def as_unyt_array(self) -> unyt.unyt_array:
        """
        Convert the buffer contents into a `unyt_array` with attached units.

        This method returns the contents of the buffer as a `unyt.unyt_array`,
        using the physical units defined by the buffer (via the `units` property).
        This is particularly useful when working with buffers that store physical
        quantities and need to interoperate with unit-aware calculations.

        If the buffer has no defined units (`self.units` is `None`), the array
        is returned as a dimensionless `unyt_array`.

        Returns
        -------
        unyt.unyt_array
            A unit-aware array representing the contents of this buffer.

        See Also
        --------
        BufferBase.units : The unit system attached to the buffer.
        BufferBase.as_array : Returns the underlying array as a NumPy array.
        """
        if self.units is None:
            units = ""
        else:
            units = self.units
        return unyt.unyt_array(self.as_array(), units)

    def as_core(self) -> ArrayLike:
        """
        Return the raw backend array object stored in this buffer.

        This method provides direct access to the internal array-like object
        (e.g., :py:class:`numpy.ndarray`, :py:class:`unyt.unyt_array`, or :py:class:`h5py.Dataset`) without any conversion
        or wrapping. It is useful for advanced users who need to access backend-specific
        methods or metadata not exposed through the generic buffer interface.

        Unlike :meth:`as_array`, this method returns the native format of the
        underlying backend, preserving units or lazy behavior if applicable.

        Returns
        -------
        ArrayLike
            The unmodified internal array object stored in the buffer.
        """
        return self.__array_object__

    def as_repr(self) -> ArrayLike:
        """
        Return a NumPy-compatible array for use in NumPy operations.

        This method is used internally to provide a consistent interface for applying
        NumPy ufuncs and broadcasting logic across different buffer backends. It returns
        an array-like object that is suitable for NumPy operations such as `np.sin()`,
        `np.add()`, or reductions like `np.sum()`.

        By default, this returns the result of :meth:`as_array`, which typically coerces
        the internal buffer into a standard `numpy.ndarray`. Subclasses may override this
        method to return more specialized representations (e.g., a `unyt_array` that
        preserves units, or a lazily sliced `h5py.Dataset`).

        Returns
        -------
        ArrayLike
            A NumPy-operable array object, such as `numpy.ndarray`, `unyt_array`, or a
            backend-compatible equivalent.

        See Also
        --------
        BufferBase.__array_ufunc__ : How NumPy dispatches operations on buffers.
        BufferBase.as_array : Returns the NumPy array representation used here by default.
        """
        return self.as_array()

    # ------------------------------ #
    # Standard Numpy Transformations #
    # ------------------------------ #
    def _apply_numpy_transform_on_repr(
        self, func: Callable, fargs, fkwargs, *args, **kwargs
    ) -> "BufferBase":
        """
        Apply a NumPy-compatible transformation to the buffer's representation.

        This method is used to wrap high-level NumPy functions or methods (e.g., `np.reshape`,
        `np.copy`, `np.squeeze`) that operate on the buffer's `as_repr()` array, which may be
        a NumPy array or a unit-tagged `unyt_array`.

        Parameters
        ----------
        func : Callable
            The NumPy-compatible function or method to apply.
        fargs : tuple
            Positional arguments for the function (not for `from_array`).
        fkwargs : dict
            Keyword arguments for the function (not for `from_array`).
        *args :
            Positional arguments passed to `.from_array()` after transformation.
        **kwargs :
            Keyword arguments passed to `.from_array()` after transformation.

        Returns
        -------
        BufferBase
            A new buffer wrapping the transformed representation.
        """
        repr_view = self.as_repr()
        return self.__class__.from_array(
            func(repr_view, *fargs, **fkwargs), *args, **kwargs
        )

    def _apply_numpy_transform_on_core(
        self, func: Callable, fargs, fkwargs, *args, **kwargs
    ) -> "BufferBase":
        """
        Apply a transformation to the raw backend array (`as_core()`).

        This method applies the given NumPy-compatible function directly to the
        buffer’s raw storage (e.g., `np.ndarray`, `unyt_array`, `h5py.Dataset`),
        and wraps the result in a new buffer instance.

        This is useful when you want to preserve the underlying structure and avoid
        unnecessary coercion to a representation type (e.g., for memory efficiency or
        backend-specific manipulations).

        Parameters
        ----------
        func : Callable
            The function or method to apply to the core array.
        fargs : tuple
            Positional arguments for the function (not for `from_array`).
        fkwargs : dict
            Keyword arguments for the function (not for `from_array`).
        *args :
            Positional arguments passed to `.from_array()` after transformation.
        **kwargs :
            Keyword arguments passed to `.from_array()` after transformation.

        Returns
        -------
        BufferBase
            A new buffer wrapping the transformed core array.
        """
        core = self.as_core()
        return self.__class__.from_array(func(core, *fargs, **fkwargs), *args, **kwargs)

    def copy(self, *args, **kwargs) -> "BufferBase":
        """
        Return a deep copy of this buffer.

        This creates a new buffer instance containing a copy of the underlying array data.
        Any units or backend metadata are preserved, and the copy is fully detached from the original.

        Parameters
        ----------
        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        **kwargs :
            Additional keyword arguments forwarded to :meth:`from_array`.

        Returns
        -------
        BufferBase
            A deep copy of the current buffer.
        """
        return self._apply_numpy_transform_on_repr(np.copy, [], {}, *args, **kwargs)

    def astype(self, dtype: Any, *args, **kwargs) -> "BufferBase":
        """
        Return a copy of this buffer with a different data type.

        This performs a type conversion using the underlying array and returns
        a new buffer of the same class with the updated `dtype`.

        Parameters
        ----------
        dtype : data-type
            The target data type for the returned array.
        *args :
            Additional positional arguments forwarded to `from_array`.
        **kwargs :
            Additional keyword arguments forwarded to `astype()` and `from_array`.

        Returns
        -------
        BufferBase
            A new buffer instance with the specified data type.
        """
        return self.__class__.from_array(self.as_repr().astype(dtype), *args, **kwargs)

    def reshape(self, shape, *args, **kwargs) -> "BufferBase":
        """
        Return a reshaped copy of this buffer.

        This reshapes the buffer into a new shape and returns a new buffer instance.
        The reshaping is done using the NumPy-compatible view returned by `as_repr()`.

        Parameters
        ----------
        shape : tuple of int
            Target shape for the new buffer.
        *args :
            Additional positional arguments forwarded to `from_array`.
        **kwargs :
            Additional keyword arguments forwarded to `from_array`.

        Returns
        -------
        BufferBase
            A new buffer with reshaped data.
        """
        return self._apply_numpy_transform_on_repr(
            np.reshape, [shape], {}, *args, **kwargs
        )

    def transpose(self, axes=None, *args, **kwargs) -> "BufferBase":
        """
        Return this buffer with axes transposed. See :func:`numpy.transpose`.

        Parameters
        ----------
        axes : tuple or list of ints, optional
            If specified, it must be a tuple or list which contains a permutation
            of [0, 1, ..., N-1] where N is the number of axes of `self`. Negative
            indices can also be used to specify axes. The i-th axis of the returned
            array will correspond to the axis numbered ``axes[i]`` of the input.
            If not specified, defaults to ``range(self.ndim)[::-1]``, which reverses
            the order of the axes.
        *args :
            Additional positional arguments forwarded to `from_array`.
        **kwargs :
            Additional keyword arguments forwarded to `from_array`.

        Returns
        -------
        BufferBase
            A new transposed buffer.
        """
        return self._apply_numpy_transform_on_repr(
            np.transpose, [axes], {}, *args, **kwargs
        )

    def flatten(self, *args, order="C", **kwargs) -> "BufferBase":
        """
        Return a flattened 1D view of this buffer.

        This flattens the buffer using the specified memory layout and returns a new buffer instance.

        Parameters
        ----------
        order : {'C','F', 'A', 'K'}, optional

            The elements of `a` are read using this index order. 'C' means
            to index the elements in row-major, C-style order,
            with the last axis index changing fastest, back to the first
            axis index changing slowest.  'F' means to index the elements
            in column-major, Fortran-style order, with the
            first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of
            the memory layout of the underlying array, and only refer to
            the order of axis indexing.  'A' means to read the elements in
            Fortran-like index order if `a` is Fortran *contiguous* in
            memory, C-like order otherwise.  'K' means to read the
            elements in the order they occur in memory, except for
            reversing the data when strides are negative.  By default, 'C'
            index order is used.

        *args :
            Additional positional arguments forwarded to `from_array`.
        **kwargs :
            Additional keyword arguments forwarded to `from_array`.

        Returns
        -------
        BufferBase
            A contiguous 1-D array of the same subtype as `self`,
            with shape ``(self.size,)``.
            Note that matrices are special cased for backward compatibility,
            if `self` is a matrix, then y is a 1-D ndarray.
        """
        return self.__class__.from_array(
            self.as_repr().flatten(order=order), *args, **kwargs
        )

    def squeeze(self, *args, axis=None, **kwargs) -> "BufferBase":
        """
        Remove axes of length one from `self`.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Selects a subset of the entries of length one in the
            shape. If an axis is selected with shape entry greater than
            one, an error is raised.
        *args :
            Additional positional arguments forwarded to `from_array`.
        **kwargs :
            Additional keyword arguments forwarded to `from_array`.

        Returns
        -------
        BufferBase
            The input buffer, but with all or a subset of the
            dimensions of length 1 removed. This is always `self` itself
            or a view into `self`. Note that if all axes are squeezed,
            the result is a 0d array and not a scalar.
        """
        return self._apply_numpy_transform_on_repr(
            np.squeeze, [axis], {}, *args, **kwargs
        )

    def expand_dims(self, axis: int, *args, **kwargs) -> "BufferBase":
        """
        Expand the shape of an array.

        Insert a new axis that will appear at the `axis` position in the expanded
        array shape.

        Parameters
        ----------
        axis : int or tuple of ints
            Position in the expanded axes where the new axis (or axes) is placed.
        *args :
            Additional positional arguments forwarded to `from_array`.
        **kwargs :
            Additional keyword arguments forwarded to `from_array`.

        Returns
        -------
        BufferBase
            A buffer with the expanded shape.
        """
        return self._apply_numpy_transform_on_repr(
            np.expand_dims, [axis], {}, *args, **kwargs
        )

    def broadcast_to(self, shape: Any, *args, **kwargs) -> "BufferBase":
        """
        Broadcast an array to a new shape.

        Parameters
        ----------
        shape : tuple or int
            The shape of the desired output array. A single integer ``i`` is interpreted
            as ``(i,)``.

        Returns
        -------
        broadcast : BufferBase
            A readonly view on the original array with the given shape. It is
            typically not contiguous. Furthermore, more than one element of a
            broadcasted array may refer to a single memory location.

        Raises
        ------
        ValueError
            If the array is not compatible with the new shape according to NumPy's
            broadcasting rules.
        """
        return self._apply_numpy_transform_on_repr(
            np.broadcast_to, [shape], {}, *args, **kwargs
        )

    # ------------------------------ #
    # Unit Handling                  #
    # ------------------------------ #
    # These method supplement those above to help with
    # unit handling.

    # === Inplace unit manipulation === #
    @abstractmethod
    def convert_to_units(
        self, units: Union[str, unyt.Unit], equivalence=None, **kwargs
    ):
        """
        Convert this buffer's data to the specified physical units (in-place).

        This operation replaces the buffer's internal data with a unit-converted
        equivalent. It modifies the object directly.

        Not all buffer classes support in-place unit assignment. Subclasses that
        do not should override this method to raise an appropriate error.

        Parameters
        ----------
        units : str or unyt.Unit
            Target units to convert the data to.
        equivalence : str, optional
            Unit equivalence to apply during conversion (e.g., "mass_energy").
        **kwargs :
            Additional keyword arguments forwarded to the equivalence logic.

        Raises
        ------
        UnitConversionError
            If the conversion is not dimensionally consistent.
        NotImplementedError
            If the subclass does not support in-place unit modification.

        Notes
        -----
        If the buffer's units are `None`, this method assigns the new units directly
        without modifying data. Otherwise, it performs a physical conversion.
        """
        raise ValueError(
            f"Cannot set units for buffer of class {self.__class__.__name__}."
        )

    def convert_to_base(self, unit_system=None, equivalence=None, **kwargs):
        """
        Convert this buffer in-place to base units for the given unit system.

        The base units are those defined by `unyt` for the specified unit system.
        This is equivalent to calling `convert_to_units` with `.get_base_equivalent()`.

        Parameters
        ----------
        unit_system : str, optional
            Unit system to use for base units (e.g., "mks", "cgs"). If not provided,
            defaults to MKS.
        equivalence : str, optional
            Equivalence scheme to use during the conversion (if applicable).
        **kwargs :
            Additional keyword arguments forwarded to the equivalence.

        Raises
        ------
        UnitConversionError
            If the conversion is not dimensionally valid.
        NotImplementedError
            If the buffer does not support unit conversion.
        """
        self.convert_to_units(
            self.units.get_base_equivalent(unit_system),
            equivalence=equivalence,
            **kwargs,
        )

    # === Casting Unit Manipulation === #
    def in_units(
        self,
        units,
        *args,
        equivalence=None,
        buffer_class=None,
        buffer_registry=None,
        as_array: bool = False,
        equiv_kw: Optional[dict] = None,
        **kwargs,
    ):
        """
        Return a new copy of this buffer cast to the specified physical units.

        This method is non-destructive and returns either a new buffer or a
        raw unit-tagged `unyt_array`. It is the preferred way to convert units
        for downstream usage or manipulation.

        Parameters
        ----------
        units : str or unyt.Unit
            Target physical units to cast to.
        equivalence : str, optional
            Name of a supported `unyt` equivalence scheme (e.g., "mass_energy").
        buffer_class : type, optional
            If provided, explicitly wrap result in this buffer class.
        buffer_registry : BufferRegistry, optional
            If resolving from array, use this registry.
        as_array : bool, default False
            If True, return a `unyt_array` instead of re-wrapping as a buffer.
        equiv_kw : dict, optional
            Keyword arguments for the equivalence function.
        *args, **kwargs :
            Passed to the buffer constructor if wrapping is performed.

        Returns
        -------
        unyt_array or BufferBase
            Either a raw unit-tagged array or a new buffer with the requested units.

        Raises
        ------
        UnitConversionError
            If the units are incompatible.
        """
        # Cast to unyt array.
        uarr = self.as_unyt_array()

        # Cast to the correct units.
        equiv_kw = equiv_kw or {}
        converted = uarr.to(units, equivalence=equivalence, **equiv_kw)

        # Figure out the returning system.
        if as_array:
            return converted
        else:
            return buffer_from_array(
                converted,
                *args,
                buffer_class=buffer_class,
                buffer_registry=buffer_registry,
                **kwargs,
            )

    def to(
        self,
        units,
        *args,
        equivalence=None,
        buffer_class=None,
        buffer_registry=None,
        as_array: bool = False,
        **kwargs,
    ):
        """
        Return a new buffer (or array) with values cast to the specified units.

        This is a shorthand for `.in_units(...)`, and fully equivalent in functionality.

        Parameters
        ----------
        units : str or unyt.Unit
            Desired output units.
        equivalence : str, optional
            Optional equivalence name for converting between dimensionally different types.
        buffer_class : type, optional
            Explicit buffer type for re-wrapping.
        buffer_registry : BufferRegistry, optional
            Optional registry to use for resolution.
        as_array : bool, default False
            If True, return raw `unyt_array` instead of a buffer.
        *args, **kwargs :
            Forwarded to `.in_units`.

        Returns
        -------
        BufferBase or unyt_array
            Buffer (or array) in the new units.

        See Also
        --------
        in_units : Underlying method.
        """
        return self.in_units(
            units,
            *args,
            equivalence=equivalence,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            as_array=as_array,
            **kwargs,
        )

    def to_value(
        self,
        units,
        equivalence=None,
        **kwargs,
    ):
        """
        Return a NumPy array of values converted to the specified physical units.

        This is equivalent to calling `.in_units(..., as_array=True).value`. It strips
        unit information and returns a plain NumPy array for interoperability.

        Parameters
        ----------
        units : str or unyt.Unit
            Target units for conversion.
        equivalence : str, optional
            Equivalence name (e.g., "mass_energy").
        **kwargs :
            Additional arguments passed to `unyt.to`.

        Returns
        -------
        numpy.ndarray
            Data in the specified units, stripped of unit tags.
        """
        return self.as_unyt_array().to_value(units, equivalence=equivalence, **kwargs)

    # === Registry Integration === #
    @classmethod
    def resolve(
        cls,
        array_like: Any,
        *args,
        buffer_registry: Optional["BufferRegistry"] = None,
        **kwargs,
    ) -> "BufferBase":
        """
        Resolve and instantiate a buffer subclass for an arbitrary array-like input.

        This method delegates to :func:`buffer_from_array`, which attempts to find
        a compatible buffer backend and coerce the input into it. The registry
        dispatch system is used unless explicitly overridden.

        Parameters
        ----------
        array_like : Any
            An array-like object to be wrapped as a buffer. Supported types vary depending
            on the registered buffer classes (e.g., `np.ndarray`, `unyt_array`, `h5py.Dataset`).

        buffer_registry : _BufferRegistry, optional
            A custom buffer registry to use for dispatch. If None (default), the global
            `__DEFAULT_BUFFER_REGISTRY__` will be used.

        *args, **kwargs : dict
            Additional arguments passed to the :func:`buffer_from_array` method of the resolved buffer class.
            These may include unit annotations, dtype specifications, HDF5 parameters, etc.

        Returns
        -------
        BufferBase
            An instance of the appropriate buffer subclass, wrapping the adapted array data.

        Raises
        ------
        TypeError
            If no compatible buffer class is found in the registry for the given object type.

        See Also
        --------
        buffer_from_array : General-purpose resolution utility.
        BufferBase.from_array : Class-based coercion.
        BufferBase.coerce : Direct array conversion method.
        """
        return buffer_from_array(
            array_like, *args, buffer_registry=buffer_registry, **kwargs
        )


def buffer_from_array(
    obj: Any,
    *args,
    buffer_class: Optional[Type["BufferBase"]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    **kwargs,
) -> "BufferBase":
    """
    Construct a buffer from a raw array-like object.

    This function performs the **buffer resolution** process (see :ref:`buffers`)
    to determine a suitable buffer to wrap the provided object.

    It is the recommended high-level interface for constructing buffers when
    the underlying storage format is not known in advance (e.g., NumPy, unyt, HDF5).

    Parameters
    ----------
    obj : array-like
        Input data to wrap (e.g., :py:class:`list`, :py:class:`~numpy.ndarray`,
        :py:class:`~unyt.array.unyt_array`, :py:class:`~h5py.Dataset`, etc.).

        By default, the type of `obj` will be used in conjunction
        with `registry` to determine which buffer class is
        used to wrap the array. If `buffer_class` is explicitly provided,
        then an attempt will be made to wrap the array
        with that class instead (regardless of the registry).
    buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
        A custom buffer registry to use for automatic resolution.
        If None (default), uses the global ``__DEFAULT_BUFFER_REGISTRY__``.
    buffer_class : :py:class:`~fields.buffers.base.BufferBase`, optional
        An explicit buffer class to use instead of registry resolution.
        If specified, the function bypasses registry lookup and directly
        calls :meth:`~BufferBase.from_array`.
    *args, **kwargs :
        Additional arguments forwarded to the :meth:`~BufferBase.from_array` method.

    Returns
    -------
    ~fields.buffers.base.BufferBase
        A fully constructed buffer instance wrapping the input data.

    Raises
    ------
    TypeError
        If no compatible buffer type is found in the registry (when ``buffer_class`` is not specified),
        or if the input is not valid for the explicitly provided ``buffer_class``.

    Examples
    --------
    By default, the correct buffer class is resolved vis-a-vis the registry. As such,
    if you simply support an array-like input, a valid buffer will be constructed:

    - A :py:class:`list`, :py:class:`tuple`, etc. will be interpreted as an array:

      >>> from pymetric.fields.buffers.core import ArrayBuffer, UnytArrayBuffer
      >>> buffer_from_array([1, 2, 3])
      ArrayBuffer(shape=(3,), dtype=int64)

    - A :py:class:`~unyt.array.unyt_array` will be interpreted as an :py:class:`~fields.buffers.core.UnytArrayBuffer`:

      >>> from unyt import unyt_array
      >>> from pymetric.fields.buffers.core import ArrayBuffer, UnytArrayBuffer
      >>> buffer_from_array(unyt_array([1, 2, 3],units='keV'))
      UnytArrayBuffer(shape=(3,), dtype=int64)

    You can also **enforce** a particular buffer class by specifying the ``buffer_class``:

    >>> from pymetric.fields.buffers.core import ArrayBuffer, UnytArrayBuffer
    >>> u = buffer_from_array([1, 2, 3],buffer_class=UnytArrayBuffer)
    >>>
    >>> # Let's look at the type and the units
    >>> print(type(u), u.units)
    <class 'pymetric.fields.buffers.core.UnytArrayBuffer'> dimensionless
    >>>
    >>> # The units can be specified in kwargs:
    >>> u = buffer_from_array([1, 2, 3],buffer_class=UnytArrayBuffer, units='keV')
    >>> print(type(u), u.units)
    <class 'pymetric.fields.buffers.core.UnytArrayBuffer'> keV

    Notes
    -----
    - If `buffer_class` is provided, the registry is ignored.
    - If `buffer_class` is not provided, resolution proceeds via the registry,
      honoring the ``__resolution_priority__`` values of registered buffer classes.
    - This method is especially useful in backend-agnostic workflows, field
      initialization logic, or serialization pipelines.

    See Also
    --------
    :py:meth:`~fields.buffers.base.BufferBase.from_array`: Class-specific buffer creation method.
    ~fields.buffers.registry.BufferRegistry.resolve : Resolve from a specific buffer registry.
    """
    if buffer_class is not None:
        return buffer_class.from_array(obj, *args, **kwargs)
    if buffer_registry is None:
        from pymetric.fields.buffers.registry import __DEFAULT_BUFFER_REGISTRY__

        buffer_registry = __DEFAULT_BUFFER_REGISTRY__
    return buffer_registry.resolve(obj, **kwargs)
