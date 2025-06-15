"""
Buffer base classes and buffer resolution support.

This module defines the core :py:class:`BufferBase` class, which all buffer types must subclass,
and the metaclass :class:`_BufferMeta`, which manages registration into the
default buffer registry and enforces interface correctness.

The buffer system abstracts different data storage backends (NumPy, HDF5, etc.)
behind a common interface so that field operations can delegate storage concerns. Novel buffer
classes can be implemented with relative ease vis-a-vis subclasses of :py:class:`BufferBase`.

For a detailed overview of the buffer system, see :ref:`buffers`.
"""
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from numpy.typing import ArrayLike

from pymetric.fields.mixins._generic import NumpyArithmeticMixin

from .registry import __DEFAULT_BUFFER_REGISTRY__, BufferRegistry


# ========================================= #
# Buffer Meta Class                         #
# ========================================= #
class _BufferMeta(ABCMeta):
    """
    Metaclass to automatically register newly created buffer classes in
    the default registry so that they are available for buffer resolution.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        # Create the generic superclass as normal. We will
        # make alterations to the class object after it is created.
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Extract the class flags and use them to determine the triaging
        # behavior.
        is_abstract = getattr(cls, "__is_abc__", False)
        if is_abstract:
            return cls

        # Determine the compatible array types so that they can
        # be registered and used for resolution.
        compatible_classes = getattr(
            cls, "__buffer_resolution_compatible_classes__", None
        )
        if compatible_classes is None:
            raise TypeError(
                f"Concrete buffer subclass '{name}' must define "
                f"`__buffer_resolution_compatible_classes__` (a type or iterable of types)."
            )

        cls.__buffer_resolution_compatible_classes__ = list(compatible_classes)

        # Now register the class in the default buffer.
        __DEFAULT_BUFFER_REGISTRY__.register(cls)
        return cls


# ========================================= #
# Abstract Base Class (BufferBase)          #
# ========================================= #
class BufferBase(NumpyArithmeticMixin, ABC, metaclass=_BufferMeta):
    """
    Abstract base class for PyMetric-compatible field buffers.

    All field data in PyMetric is stored using concrete subclasses of `BufferBase`, which provide
    a uniform, backend-agnostic interface for array access, transformation, and participation in
    numerical operations. This abstraction allows PyMetric to operate seamlessly over a variety of
    storage backends, including:

    - In-memory arrays via :class:`~fields.buffers.core.ArrayBuffer`
    - On-disk storage via :class:`~fields.buffers.core.HDF5Buffer` or similar implementations

    Subclasses of :class:`BufferBase` must implement the full buffer API protocol as documented in
    the reference section: :ref:`buffers`. This includes:

    - NumPy-style indexing via `__getitem__` and `__setitem__`
    - Array attributes such as `.shape`, `.dtype`, `.ndim`, and `.size`
    - NumPy ufunc participation via `__array_ufunc__`
    - Fallbacks for `__array_function__`, which must materialize a NumPy array

    All field constructors and numerical operators in PyMetric internally operate on :class:`BufferBase` instances.
    User-provided arrays (such as raw `np.ndarray`) will be automatically wrapped into
    the appropriate buffer subclass at runtime using the buffer resolution mechanism.

    Subclasses are encouraged to document any backend-specific limitations (e.g., restricted indexing behavior
    in HDF5 datasets), and must raise appropriate exceptions (e.g., `NotImplementedError`) when features
    are unavailable.

    See Also
    --------
    :ref:`buffers` The PyMetric buffer specification.

    """

    # =================================== #
    # Class Flags                         #
    # =================================== #
    # These flags are class level markers indicating the behavior
    # of the class. They should be configured in subclasses.
    __is_abc__: bool = True
    """
    Flag indicating whether this class is abstract and should be excluded from buffer registry resolution.

    Set this to ``True`` in base classes or abstract buffer prototypes that define interface logic
    but are not meant to be directly instantiated or selected during automatic resolution.

    Concrete buffer implementations **must** override this with ``False`` to be discoverable by
    :class:`~fields.buffers.registry.BufferRegistry` and participate in buffer dispatch.
    """

    __buffer_resolution_priority__: int = 0
    """
    Integer value used to rank buffer classes during automatic backend resolution.

    When multiple buffer types report compatibility with the same object,
    the resolution system selects the one with the **highest priority value**.

    This enables preference for more specialized buffers (e.g., disk-backed)
    over fallback types like raw NumPy arrays.

    Defaults to 0; subclasses should assign higher values based on specificity or performance.

    See Also
    --------
    :class:`~fields.buffers.registry.BufferRegistry`
    """

    __buffer_resolution_compatible_classes__: Optional[List[Type]] = None
    """
    List of Python types that this buffer class can directly recognize and wrap.

    Each entry in this list should be a type (e.g., ``numpy.ndarray``, ``h5py.Dataset``).
    These types will be checked against inputs during automatic resolution by the
    :class:`~fields.buffers.registry.BufferRegistry`.

    This attribute **must be defined** for a buffer to be registered and resolved automatically.
    """

    __array_function_dispatch__: Dict[Callable, Callable] = {}
    """
    Mapping of high-level NumPy functions to custom handler implementations for this buffer type.

    This dictionary allows buffers to override behavior for specific NumPy functions
    (e.g., ``np.sum``, ``np.moveaxis``) when using the ``__array_function__`` protocol.

    Each key should be a NumPy function, and each value should be a callable that implements
    custom logic. These handlers receive unwrapped core arrays (via ``as_core()``) and are
    responsible for returning the appropriate result (buffer or array).

    If a NumPy function is not listed here, it will fall back to default delegation logic.

    See Also
    --------
    :meth:`~fields.buffers.base.BufferBase.__array_function__`
    """
    __array_priority__ = 1.0
    """
    The priority of the component class in numpy operations.
    """

    # =================================== #
    # Initialization Methods              #
    # =================================== #
    def __init__(self, array_object: ArrayLike, *args, **kwargs):
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
            by this buffer.
        *args, **kwargs:
            Additional positional and keyword arguments. These are not used in the base
            class but may be relevant for subclasses that require additional initialization.

        Raises
        ------
        TypeError
            If the array does not match the expected core type(s).

        See Also
        --------
        BufferBase.from_array : Preferred interface for safe buffer construction.
        """
        self.__array_object__: ArrayLike = array_object
        self.__validate_array_object__()

    # noinspection PyMethodMayBeStatic
    def __validate_array_object__(self):
        """
        Check that the provided underlying array object
        is a valid choice. By default, this check is automatically passed; however,
        it may be specified in subclasses to provide custom checks.
        """
        return None

    # =================================== #
    # Dunder Methods and Numpy Support    #
    # =================================== #
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
        between numpy representations of the arrays. Our returned value
        is determined by the `RepresentationType`s of each of the input buffers.

        If `out` is specified, then an attempt is made to place the result into the relevant
        buffer.
        """
        # Convert all of the inputs into their corresponding representation type. This
        # will break any lazy-loading behavior in the inputs and convert everything to
        # numpy compatible types.
        core_inputs = [x[...] if isinstance(x, self.__class__) else x for x in inputs]

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
            a[...] if isinstance(a, self.__class__) else a for a in args
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

    # =================================== #
    # Generator Methods                   #
    # =================================== #
    # These methods control the generation of buffer subclasses.
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

        The method should be overridden in subclasses to handle type conversion, memory layout,
        or any other backend-specific behavior.

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

        Returns
        -------
        BufferBase
            An unallocated buffer like `other`.
        """
        return cls.empty(other.shape, *args, **kwargs)

    def as_array(self) -> np.ndarray:
        """
        Return the buffer as a NumPy array.

        Returns
        -------
        numpy.ndarray
        """
        return self.__array__(dtype=self.dtype)

    def as_core(self) -> ArrayLike:
        """
        Return the raw backend array object stored in this buffer.

        This method provides direct access to the internal array-like object
        (e.g., :py:class:`numpy.ndarray`, :py:class:`unyt.unyt_array`, or :py:class:`h5py.Dataset`) without any conversion
        or wrapping. It is useful for advanced users who need to access backend-specific
        methods or metadata not exposed through the generic buffer interface.

        Unlike :meth:`as_array`, this method returns the native format of the
        underlying backend, preserving special structures or lazy behavior if applicable.

        Returns
        -------
        ArrayLike
            The unmodified internal array object stored in the buffer.
        """
        return self.__array_object__

    # =================================== #
    # Properties                          #
    # =================================== #
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.__array_object__.size

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.__array_object__.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying array."""
        return self.__array_object__.shape

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
    def T(self):
        """
        Transpose of the array (shortcut for `.transpose()`).

        See Also
        --------
        numpy.ndarray.T
        """
        return self.transpose()

    # =================================== #
    # Utility Methods                     #
    # =================================== #
    @classmethod
    def list_compatible_classes(cls):
        """
        Return the list of compatible types for this buffer class.

        These are the Python types (e.g., `numpy.ndarray`, `h5py.Dataset`) that this buffer
        can accept and wrap directly during resolution. The types are specified by the
        class-level attribute `__buffer_resolution_compatible_classes__`.

        This method is primarily used by the buffer registry to determine whether a buffer
        class can handle a given input during automatic resolution.

        Returns
        -------
        list
            A shallow copy of the compatible Python types that this buffer can wrap.
        """
        return cls.__buffer_resolution_compatible_classes__[:]

    @classmethod
    def can_resolve(cls, obj):
        """
        Check whether this buffer class can resolve the given object.

        This method returns True if the type of `obj` matches any of the types listed
        in ``__buffer_resolution_compatible_classes__``. It is used internally by the
        buffer registry to dispatch input objects to the correct backend.

        Parameters
        ----------
        obj : Any
            The input object to test for compatibility.

        Returns
        -------
        bool
            True if `obj` is compatible with this buffer class, False otherwise.
        """
        return type(obj) in cls.list_compatible_classes()

    # =================================== #
    # Numpy-Like Methods                  #
    # =================================== #
    # These methods are ONLY necessary to support transformations of
    # a single buffer in ways which might need to return a buffer directly (i.e. reshaping).
    def _cast_numpy_op(
        self,
        func: Callable,
        *args,
        numpy: bool = False,
        bargs: Optional[Tuple] = (),
        bkwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Apply a NumPy transformation to this buffer’s array and optionally wrap the result.

        This method enables concise, consistent support for buffer-returning operations such as
        `.reshape()`, `.transpose()`, `.copy()`, etc. When `numpy=True`, the result is returned
        as a raw NumPy array. Otherwise, a new buffer of the same class is constructed from the result.

        Parameters
        ----------
        func : Callable
            A function that takes an array and returns a transformed array (e.g., `np.transpose`).
        *args :
            Positional arguments to pass to `func`.
        numpy : bool, optional
            If True, return the result as a raw array. If False (default), return as a buffer.
        bargs : tuple, optional
            Positional arguments to forward to the buffer constructor when `numpy=False`.
        bkwargs : dict, optional
            Keyword arguments to forward to the buffer constructor when `numpy=False`.
        **kwargs :
            Additional keyword arguments passed to `func`.

        Returns
        -------
        BufferBase or array-like
            Result of applying the transformation, either as a buffer or raw array.
        """
        bkwargs = bkwargs or {}
        result = func(self.as_array(), *args, **kwargs)
        return result if numpy else self.__class__.from_array(result, *bargs, **bkwargs)

    def astype(self, dtype, copy=True, *, numpy=False, bargs=(), bkwargs=None):
        """
        Cast to a specified dtype.

        See Also
        --------
        numpy.ndarray.astype
        """
        return self._cast_numpy_op(
            lambda arr: arr.astype(dtype, copy=copy),
            numpy=numpy,
            bargs=bargs,
            bkwargs=bkwargs,
        )

    def conj(self, *, numpy=False, bargs=(), bkwargs=None):
        """
        Return the complex conjugate.

        See Also
        --------
        numpy.ndarray.conj
        """
        return self._cast_numpy_op(np.conj, numpy=numpy, bargs=bargs, bkwargs=bkwargs)

    def conjugate(self, *, numpy=False, bargs=(), bkwargs=None):
        """
        Return the complex conjugate.

        See Also
        --------
        numpy.ndarray.conjugate
        """
        return self._cast_numpy_op(
            np.conjugate, numpy=numpy, bargs=bargs, bkwargs=bkwargs
        )

    def copy(self, *, numpy=False, bargs=(), bkwargs=None):
        """
        Return a copy of the array.

        See Also
        --------
        numpy.ndarray.copy
        """
        return self._cast_numpy_op(np.copy, numpy=numpy, bargs=bargs, bkwargs=bkwargs)

    def flatten(self, order="C", *, numpy=False, bargs=(), bkwargs=None):
        """
        Return a flattened copy of the array.

        See Also
        --------
        numpy.ndarray.flatten
        """
        return self._cast_numpy_op(
            lambda arr: arr.flatten(order=order),
            numpy=numpy,
            bargs=bargs,
            bkwargs=bkwargs,
        )

    def ravel(self, order="C", *, numpy=False, bargs=(), bkwargs=None):
        """
        Return a flattened view of the array.

        See Also
        --------
        numpy.ndarray.ravel
        """
        return self._cast_numpy_op(
            lambda arr: np.ravel(arr, order=order),
            numpy=numpy,
            bargs=bargs,
            bkwargs=bkwargs,
        )

    def reshape(self, *shape, numpy=False, bargs=(), bkwargs=None):
        """
        Return a reshaped view or copy of the array.

        See Also
        --------
        numpy.ndarray.reshape
        """
        return self._cast_numpy_op(
            lambda arr: np.reshape(arr, *shape),
            numpy=numpy,
            bargs=bargs,
            bkwargs=bkwargs,
        )

    def resize(self, *shape, numpy=False, bargs=(), bkwargs=None):
        """
        Resize the array to a new shape (copy-based emulation of NumPy's in-place behavior).

        See Also
        --------
        numpy.ndarray.resize
        """
        return self._cast_numpy_op(
            lambda arr: np.resize(arr.copy(), *shape),
            numpy=numpy,
            bargs=bargs,
            bkwargs=bkwargs,
        )

    def swapaxes(self, axis1, axis2, *, numpy=False, bargs=(), bkwargs=None):
        """
        Interchange two axes of the array.

        See Also
        --------
        numpy.ndarray.swapaxes
        """
        return self._cast_numpy_op(
            lambda arr: np.swapaxes(arr, axis1, axis2),
            numpy=numpy,
            bargs=bargs,
            bkwargs=bkwargs,
        )

    def transpose(self, *axes, numpy=False, bargs=(), bkwargs=None):
        """
        Permute the dimensions of the array.

        See Also
        --------
        numpy.ndarray.transpose
        """
        return self._cast_numpy_op(
            lambda arr: np.transpose(arr, axes if axes else None),
            numpy=numpy,
            bargs=bargs,
            bkwargs=bkwargs,
        )


# ========================================= #
# Buffer Resolution Methods                 #
# ========================================= #
# This is the main entry point for buffer resolution.
def buffer_from_array(
    obj: Any,
    *args,
    buffer_class: Optional[Type["BufferBase"]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    **kwargs,
) -> "BufferBase":
    """
    Construct a PyMetric-compatible buffer from a raw array-like object.

    This function serves as the high-level entry point for buffer resolution in PyMetric.
    It accepts any array-like input (e.g., NumPy arrays, lists, unyt arrays, or HDF5 datasets)
    and returns a fully constructed :class:`~fields.buffers.base.BufferBase` subclass instance
    that wraps the data.

    Resolution behavior is determined as follows:

    1. If ``buffer_class`` is specified, it bypasses registry logic and directly calls
       :meth:`BufferBase.from_array` on that class.
    2. If ``buffer_class`` is not specified, the function dispatches to the buffer registry,
       which selects a compatible backend based on type compatibility and resolution priority.

    Parameters
    ----------
    obj : Any
        The input array-like object to be wrapped. Can be any supported format, including:
        - `list`, `tuple`, `numpy.ndarray`
        - `unyt.unyt_array`
        - `h5py.Dataset`
    buffer_class : Type[BufferBase], optional
        An explicit buffer class to use. If provided, registry resolution is skipped.
    buffer_registry : BufferRegistry, optional
        A custom registry to use for resolution. Defaults to the global
        :data:`__DEFAULT_BUFFER_REGISTRY__`.
    *args :
        Positional arguments forwarded to the resolved buffer's `from_array()` method.
    **kwargs :
        Keyword arguments forwarded to `from_array()` (e.g., `dtype`, etc.).

    Returns
    -------
    BufferBase
        An instance of a registered buffer type wrapping the input.

    Raises
    ------
    TypeError
        If no compatible buffer type can be resolved, or if `buffer_class` is invalid.

    Examples
    --------
    >>> buffer_from_array([1, 2, 3])
    ArrayBuffer(shape=(3,), dtype=int64)

    Notes
    -----
    - This function is backend-agnostic and safe to use in any context where the
      precise buffer type is not known in advance.
    - If `buffer_class` is provided, registry logic is skipped entirely.
    - If `buffer_registry` is not provided, the global default registry is used.

    See Also
    --------
    BufferBase.from_array : Class-specific buffer constructor.
    BufferRegistry.resolve : Registry-based resolution logic.
    resolve_buffer_class : Utility to normalize buffer class identifiers.
    """
    if buffer_class is not None:
        return buffer_class.from_array(obj, *args, **kwargs)

    if buffer_registry is None:
        from pymetric.fields.buffers.registry import __DEFAULT_BUFFER_REGISTRY__

        buffer_registry = __DEFAULT_BUFFER_REGISTRY__

    return buffer_registry.resolve(obj, *args, **kwargs)
