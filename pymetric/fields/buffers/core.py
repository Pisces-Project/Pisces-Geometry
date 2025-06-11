"""
Core buffer types for PyMetric data management.

This module defines the core buffer backends used by the PyMetric framework.
Each buffer subclass implements the shared interface defined in :class:`~fields.buffers.base.BufferBase`,
but supports a distinct storage strategy.

Usage Notes
-----------

These buffers serve as drop-in storage layers for data fields, components, and other high-level PyMetric components.
They ensure consistent semantics for field creation, manipulation, and resolution, regardless of backend format.

All buffer classes:

See Also
--------
:class:`~fields.buffers.base.BufferBase`
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import h5py
import numpy as np
from numpy.typing import ArrayLike

from pymetric.fields.buffers.base import BufferBase


class ArrayBuffer(BufferBase):
    """
    A lightweight buffer wrapper around a plain `NumPy <https://numpy.org/doc/stable/index.html>`__ array.

    This class provides a minimal backend for storing field data using
    standard :class:`numpy.ndarray` objects. It is designed for general-purpose use cases
    where advanced I/O (e.g., HDF5) is not required.


    Examples
    --------
    Create a buffer from a 2D list:

    >>> buf = ArrayBuffer.from_array([[1, 2], [3, 4]])
    >>> buf
    ArrayBuffer(shape=(2, 2), dtype=int64)

    Create a zero-initialized buffer with shape (3, 3):

    >>> ArrayBuffer.zeros((3, 3)).as_array()
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])


    See Also
    --------
    HDF5Buffer: HDF5 backed buffer.
    ~fields.buffers.base.BufferBase : Abstract interface for all buffer backends.
    """

    # =================================== #
    # Class Flags                         #
    # =================================== #
    # These flags are class level markers indicating the behavior
    # of the class. They should be configured in subclasses.
    __is_abc__ = False
    __buffer_resolution_compatible_classes__ = [np.ndarray, list, tuple]
    __buffer_resolution_priority__ = 0
    __array_function_dispatch__: Dict[Callable, Callable] = {}

    # =================================== #
    # Initialization Methods              #
    # =================================== #
    def __init__(self, array: np.ndarray):
        """
        Initialize an ArrayBuffer from a NumPy array.

        This constructor wraps a standard in-memory :class:`numpy.ndarray` and exposes
        it via the PyMetric buffer interface. It does not copy the input unless required
        by downstream operations; the buffer holds a direct reference to the array.

        Parameters
        ----------
        array : numpy.ndarray
            The array to wrap. Must be a valid NumPy array of any shape or dtype.

        Raises
        ------
        TypeError
            If the input is not a :class:`numpy.ndarray`.

        Notes
        -----
        - This constructor is typically called internally by :meth:`from_array`.
        - Use :meth:`ArrayBuffer.from_array` for a safe, flexible entry point that
          can accept non-NumPy types (e.g., lists, scalars).
        - This class provides lightweight, zero-copy wrapping for in-memory data.

        Examples
        --------
        >>> import numpy as np
        >>> from pymetric.fields.buffers import ArrayBuffer
        >>> arr = np.arange(6).reshape(2, 3)
        >>> buf = ArrayBuffer(arr)
        >>> print(buf.shape)
        (2, 3)
        """
        super().__init__(array)

    def __validate_array_object__(self):
        if not isinstance(self.__array_object__, np.ndarray):
            raise TypeError(
                f"Expected a NumPy array, got {type(self.__array_object__)}."
            )

    # =================================== #
    # Generator Methods                   #
    # =================================== #
    @classmethod
    def from_array(cls, obj: Any, *args, **kwargs) -> "ArrayBuffer":
        """
        Construct a new :class:`ArrayBuffer` from an arbitrary array-like input.

        This method attempts to coerce the input into a valid :class:`numpy.ndarray` and wrap
        it in an :class:`ArrayBuffer` instance. It serves as the standard mechanism
        for converting raw or structured numerical data (e.g., lists, tuples, or other
        array-compatible types) into a compatible buffer for numerical operations.

        This constructor mimics the behavior of :func:`numpy.array` and accepts
        keyword arguments like `dtype`, `order`, and `copy` to control how the data
        is materialized. Subclasses may override this method to enforce stricter
        requirements or attach additional metadata.

        Parameters
        ----------
        obj : array-like
            Input data to be wrapped. This can include Python lists, tuples,
            NumPy arrays, or other objects that support array conversion via
            :func:`numpy.array()`.
        *args, **kwargs :
            Additional keyword arguments passed to :func:`numpy.array` to control
            coercion behavior (e.g., `copy`, `order`, etc.).

        Returns
        -------
        ArrayBuffer
            A buffer instance wrapping the resulting :class:`numpy.ndarray`.

        Raises
        ------
        TypeError
            If the input cannot be coerced into a NumPy array.
        """
        return cls(np.array(obj, *args, **kwargs))

    @classmethod
    def zeros(cls, shape, *args, **kwargs) -> "ArrayBuffer":
        """
        Create a buffer initialized with zeros.

        Parameters
        ----------
        shape : tuple of int
            Shape of the buffer (grid + element dimensions).
        *args, **kwargs :
            arguments forwarded to :func:`numpy.zeros` (e.g., dtype).

        Returns
        -------
        ArrayBuffer
            A zero-filled buffer.
        """
        return cls(np.zeros(shape, *args, **kwargs))

    @classmethod
    def ones(cls, shape, *args, **kwargs) -> "ArrayBuffer":
        """
        Create a buffer initialized with ones.

        Parameters
        ----------
        shape : tuple of int
            Shape of the buffer (grid + element dimensions).
        *args, **kwargs :
            arguments forwarded to :func:`numpy.ones` (e.g., dtype).

        Returns
        -------
        ArrayBuffer
            A one-filled buffer.
        """
        return cls(np.ones(shape, *args, **kwargs))

    @classmethod
    def full(cls, shape, *args, fill_value=0.0, **kwargs) -> "ArrayBuffer":
        """
        Create a buffer filled with a constant value.

        Parameters
        ----------
        shape : tuple of int
            Shape of the buffer (grid + element dimensions).
        fill_value : scalar, optional
            The constant value to use for every element. By default, this is ``0.0``.
        *args, **kwargs :
            arguments forwarded to :func:`numpy.full` (e.g., dtype).

        Returns
        -------
        ArrayBuffer
            A buffer filled with the given value.
        """
        return cls(np.full(shape, fill_value, *args, **kwargs))

    @classmethod
    def empty(cls, shape, *args, **kwargs) -> "ArrayBuffer":
        """
        Return a new buffer of given shape and type, without initializing entries.

        Parameters
        ----------
        shape : tuple of int
            Shape of the buffer (grid + element dimensions).
        *args, **kwargs :
            arguments forwarded to :func:`numpy.empty` (e.g., dtype).

        Returns
        -------
        ArrayBuffer
            An uninitialized buffer (contents may be arbitrary).
        """
        return cls(np.empty(shape, *args, **kwargs))


class HDF5Buffer(BufferBase):
    """
    A disk-backed buffer that wraps an `h5py.Dataset` for persistent field storage.

    This class provides a PyMetric-compatible buffer interface for data stored in
    HDF5 files. It is particularly suited for large-scale or long-lived datasets
    that exceed in-memory limits or require persistent storage across sessions.

    Unlike in-memory buffers (e.g., :class:`ArrayBuffer`), `HDF5Buffer` instances
    operate on datasets saved to disk via `h5py`. They support standard buffer semantics
    including indexing, broadcasting, NumPy interoperability, and context management,
    while handling backend-specific logic such as file ownership and I/O flushing.

    Examples
    --------
    Create a new HDF5 buffer from an in-memory array:

    >>> from pymetric.fields.buffers import HDF5Buffer
    >>> buf = HDF5Buffer.from_array([[1, 2], [3, 4]], file="data.h5", path="/example")

    Open an existing dataset:

    >>> with HDF5Buffer.open("data.h5", "/example", close_on_exit=True) as buf:
    ...     print(buf.shape)

    Update a buffer in place using a NumPy ufunc:

    >>> import numpy as np
    >>> np.add(buf, 1.0, out=buf)

    Notes
    -----
    - When `make_owner=True`, the buffer is responsible for closing the file.
    - You can manually call :meth:`flush()` to write buffered changes to disk.
    - Use context management (``with``) to ensure safe file handling.

    See Also
    --------
    ArrayBuffer : Lightweight in-memory buffer using :class:`numpy.ndarray`.
    ~fields.buffers.base.BufferBase : Abstract base class for all buffers.
    h5py.Dataset : Underlying dataset format used in this buffer.
    """

    # =================================== #
    # Class Flags                         #
    # =================================== #
    # These flags are class level markers indicating the behavior
    # of the class. They should be configured in subclasses.
    __is_abc__: bool = False
    __buffer_resolution_priority__: int = 30
    __buffer_resolution_compatible_classes__: Optional[List[Type]] = [h5py.Dataset]
    __array_function_dispatch__: Dict[Callable, Callable] = {}

    # =================================== #
    # Initialization Methods              #
    # =================================== #
    def __init__(self, array: h5py.Dataset, owns_file: bool = False):
        """
        Create an :class:`HDF5Buffer` object.

        Parameters
        ----------
        array : h5py.Dataset
            Open dataset to wrap. *Must* remain valid for the buffer's lifetime.
        owns_file : bool, optional
            If True, the buffer will close the file when exiting its context.
        """
        super().__init__(array)
        self.__is_owner__: bool = owns_file

    def __validate_array_object__(self):
        if not isinstance(self.__array_object__, h5py.Dataset):
            raise TypeError(
                f"Expected an HDF5 dataset, got {type(self.__array_object__)}."
            )

        # Check that the h5py object is successfully opened
        # and has a valid status.
        if not self.__array_object__.id.valid:
            raise ValueError("HDF5 dataset is not valid or has been closed.")

    def __enter__(self) -> "HDF5Buffer":
        """Enter the buffer context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context: flush and optionally close file if owned."""
        self.__array_object__.flush()
        if self.__is_owner__:
            self.__array_object__.file.close()

    # =================================== #
    # Generator Methods                   #
    # =================================== #
    @classmethod
    def create_dataset(
        cls,
        file: Union[str, Path, h5py.File],
        name: str,
        shape: Optional[tuple] = None,
        dtype: Optional[Any] = None,
        *,
        data: Optional[ArrayLike] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> h5py.Dataset:
        """
        Create an HDF5 dataset in a file and return the resulting `h5py.Dataset`.

        This utility is used internally by `HDF5Buffer` to create datasets from raw data.
        If a dataset with the given name already exists, it can be optionally overwritten.

        Parameters
        ----------
        file : str, Path, or h5py.File
            Path to the HDF5 file or an open `h5py.File` object. This is the parent object
            into which the dataset will be created.
        name : str
            Name of dataset to create. May be an absolute or relative path.
            Provide ``None`` to create an anonymous dataset, to be linked into the file later.
        shape : tuple of int, optional
            Desired shape of the dataset. Required if `data` is not provided.
        dtype : data-type, optional
            Data type of the dataset. Inferred from `data` if not provided.
        data : array-like, optional
            Input data to populate the dataset. If provided, `shape` and `dtype` are optional.
        overwrite : bool, default False
            Whether to delete and replace an existing dataset with the same name.
        **kwargs :
            Additional keyword arguments passed to `h5py.File.create_dataset`.

        Returns
        -------
        h5py.Dataset
            A created dataset ready to be wrapped in a buffer.

        Raises
        ------
        FileNotFoundError
            If the HDF5 file path does not exist and file creation is not permitted.
        ValueError
            If the dataset already exists and `overwrite` is not set.
        TypeError
            If the `file` parameter is neither a path nor an `h5py.File`.
        """
        # Open and validate the file. Ensure it exists / create it
        # and then do the necessary logic progression to ensure consistency.
        if isinstance(file, (str, Path)):
            file = Path(file)

            if not file.exists():
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()

            file = h5py.File(file, mode="r+")

        else:
            raise TypeError(f"`file` must be a path or h5py.File, got {type(file)}.")

        # Check for pre-existing dataset
        if name in file:
            if overwrite:
                del file[name]
            else:
                raise ValueError(
                    f"Dataset '{name}' already exists. Use `overwrite=True` to replace it."
                )

        # Create the dataset
        dset = file.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)
        return dset

    # noinspection PyIncorrectDocstring
    @classmethod
    def from_array(
        cls,
        obj: Any,
        file: Optional[Union[str, Path, h5py.File]] = None,
        path: Optional[str] = None,
        *args,
        dtype: Optional[Any] = None,
        **kwargs,
    ) -> "HDF5Buffer":
        """
        Construct an HDF5-backed buffer from an array-like object or existing HDF5 dataset.

        This method wraps either:

        - an existing `h5py.Dataset`, or
        - arbitrary in-memory data by creating a new dataset in the specified HDF5 file.

        Parameters
        ----------
        obj : array-like or :class:`~h5py.Dataset`
            The data to store. If already a :class:`h5py.Dataset`, it is wrapped directly.
            Otherwise, it is coerced to a `unyt_array` and stored in a new dataset.
        file : str, Path, or :class:`~h5py.File`, optional
            Path to the HDF5 file or an open file object. Required if `obj` is not already a dataset.
        path : str, optional
            Internal path for the new dataset in the HDF5 file. Required if `obj` is not already a dataset.
        dtype : data-type, optional
            Desired data type of the dataset. Defaults to the dtype inferred from `obj`.
        overwrite : bool, default False
            If True, any existing dataset at the given path will be deleted and replaced.
        **kwargs :
            Additional keyword arguments forwarded to :meth:`create_dataset`.

        Returns
        -------
        HDF5Buffer
            A lazily-loaded buffer backed by an HDF5 dataset.

        Raises
        ------
        ValueError
            If `file` or `path` is not specified when required.
        TypeError
            If input types are not supported or invalid.

        See Also
        --------
        HDF5Buffer.create_dataset : Core method used for new dataset creation.
        buffer_from_array : Registry-based entry point to buffer resolution.
        """
        # Case 1: Wrap an existing HDF5 dataset directly
        if isinstance(obj, h5py.Dataset):
            is_owner = kwargs.pop("make_owner", False)
            return cls(obj, owns_file=is_owner)

        # Case 2: Create a new dataset from in-memory data
        if file is None or path is None:
            raise ValueError(
                "Both `file` and `path` must be specified to create a new HDF5 dataset."
            )

        is_owner = kwargs.pop("make_owner", True)
        array = np.asarray(obj, dtype=dtype)
        dataset = cls.create_dataset(
            file, path, data=array, dtype=array.dtype, **kwargs
        )
        return cls(dataset, owns_file=is_owner)

    @classmethod
    def open(
        cls,
        file: Union[str, Path, h5py.File],
        path: str,
        *,
        mode: str = "r+",
        close_on_exit: bool = False,
    ) -> "HDF5Buffer":
        """
        Open an existing HDF5 dataset and wrap it as a buffer.

        This method provides controlled access to an HDF5 dataset stored within
        a file, supporting both file paths and already opened `h5py.File` objects.

        If `close_on_exit=True`, the returned buffer will automatically close the
        underlying file when used in a `with` block or when `close()` is called.

        Parameters
        ----------
        file : str, Path, or h5py.File
            The HDF5 file containing the dataset. May be a file path or an open handle.
        path : str
            The internal path to the dataset within the file (e.g., "/my/data").
        mode : str, default "r+"
            Mode to use when opening the file if a path is provided. Ignored if `file` is already open.
        close_on_exit : bool, default False
            Whether the buffer should take ownership of the file and close it on exit.

        Returns
        -------
        HDF5Buffer
            A buffer wrapping the specified HDF5 dataset.

        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.
        KeyError
            If the specified dataset does not exist in the file.
        TypeError
            If `file` is not a string, Path, or h5py.File.

        Examples
        --------
        >>> with HDF5Buffer.open("data.h5", "temperature", close_on_exit=True) as buf:
        ...     print(buf.shape)
        """
        # Open the file if it's a path
        if isinstance(file, (str, Path)):
            file = Path(file)
            if not file.exists():
                raise FileNotFoundError(f"HDF5 file '{file}' does not exist.")
            file_obj = h5py.File(file, mode=mode)
        elif isinstance(file, h5py.File):
            file_obj = file
        else:
            raise TypeError(f"`file` must be a path or h5py.File, got {type(file)}")

        # Look up the dataset inside the file
        if path not in file_obj:
            raise KeyError(f"Dataset '{path}' not found in file.")

        dataset = file_obj[path]
        return cls(dataset, owns_file=close_on_exit)

    # ------------------------------ #
    # Properties                     #
    # ------------------------------ #
    @property
    def is_open(self) -> bool:
        """True if the underlying dataset is still attached to an open file."""
        return self.__array_object__ is not None and bool(
            self.__array_object__.id.valid
        )

    @property
    def filename(self) -> Optional[str]:
        """
        Absolute path to the file backing this buffer.

        Returns
        -------
        str or None
            Full path of the file, or None if unavailable.
        """
        return getattr(self.__array_object__.file, "filename", None)

    @property
    def file(self) -> h5py.File:
        """
        The open `h5py.File` object backing this buffer.

        This provides direct access to the HDF5 file handle used internally.
        It is useful for advanced workflows that require access to groups,
        attributes, or metadata outside the dataset.

        Returns
        -------
        h5py.File
            The HDF5 file containing the dataset.
        """
        return self.__array_object__.file

    @property
    def name(self) -> str:
        """
        Internal HDF5 path to the dataset.

        Returns
        -------
        str
            The full dataset path within the HDF5 file (e.g., "/data/temperature").
        """
        return self.__array_object__.name

    @property
    def is_owner(self) -> bool:
        """
        Whether this buffer owns the file (i.e., responsible for closing it).

        Returns
        -------
        bool
            True if this buffer should close the file on `__exit__` or `close()`.
        """
        return self.__is_owner__

    # ------------------------------ #
    # Numpy Semantics                #
    # ------------------------------ #
    def __repr__(self):
        return f"HDF5Buffer(shape={self.shape}, dtype={self.dtype}, path='{self.__array_object__.name}')"

    def __str__(self):
        return self.__array_object__.__str__()

    def __getitem__(self, idx):
        ret = self.__array_object__.__getitem__(idx)
        return ret

    def __setitem__(self, item, value):
        _raw_value = np.asarray(value, dtype=self.dtype)

        # Avoid broadcasting if value is scalar#
        if _raw_value.ndim == 0:
            _raw_value = _raw_value.item()

        self.__array_object__[item] = _raw_value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override BufferBase.__array_ufunc__ to catch instances where HDF5Buffer
        is a specified output of the operation. Because HDF5 Dataset types cannot
        handle in place operations, we use a sleight of hand to perform the comp.
        in memory and then assign under the hood.
        """
        # Extract the `out` kwarg if it is present.
        out = kwargs.pop("out", None)
        if out is not None:
            # Normalize to tuple for uniform handling
            is_tuple = isinstance(out, tuple)
            out_tuple = out if is_tuple else (out,)

            # Run the operation and capture result(s).
            # This doesn't have `out` in it because of .pop() so
            # we're just forwarding.
            result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

            # Coerce the typing and check that the lengths are
            # correctly matched.
            result_tuple = result if isinstance(result, tuple) else (result,)
            if len(result_tuple) != len(out_tuple):
                raise ValueError(
                    f"Expected {len(out_tuple)} outputs, got {len(result_tuple)}"
                )

            # Assign results into HDF5-backed targets
            for r, o in zip(result_tuple, out_tuple):
                if isinstance(o, self.__class__):
                    o.__array_object__[
                        ...
                    ] = r  # assign in-memory result into HDF5 buffer

                else:
                    raise TypeError("All `out=` targets must be HDF5Buffer instances")

            # Pass result through based on the typing.
            if isinstance(result, tuple):
                return out_tuple
            elif result is not None:
                return out_tuple[0]
            else:
                return None

        # No `out` to catch, simply push on to the super method.
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    # ------------------------------ #
    # Generator Methods              #
    # ------------------------------ #
    @classmethod
    def zeros(
        cls, shape: Tuple[int, ...], *args, dtype: Any = float, **kwargs
    ) -> "HDF5Buffer":
        """
        Create an HDF5-backed buffer filled with zeros.

        Parameters
        ----------
        shape : tuple of int
            Desired shape of the dataset.
        dtype : data-type, default float
            Element type of the dataset.
        *args :
            Additional positional arguments passed to `create_dataset`.
        **kwargs :
            Keyword arguments forwarded to `create`, such as:

            - parent : h5py.File or h5py.Group
            - dataset_name : str
            - overwrite : bool

        Returns
        -------
        HDF5Buffer
            A buffer wrapping a zero-filled HDF5 dataset.
        """
        return cls(
            cls.create_dataset(*args, data=np.zeros(shape, dtype=dtype), **kwargs)
        )

    @classmethod
    def ones(
        cls, shape: Tuple[int, ...], *args, dtype: Any = float, **kwargs
    ) -> "HDF5Buffer":
        """
        Create an HDF5-backed buffer filled with ones.

        Parameters
        ----------
        shape : tuple of int
            Desired shape of the dataset.
        dtype : data-type, default float
            Element type of the dataset.
        *args :
            Additional positional arguments passed to `create_dataset`.
        **kwargs :
            Keyword arguments forwarded to `create`, such as:

            - parent : h5py.File or h5py.Group
            - dataset_name : str
            - overwrite : bool

        Returns
        -------
        HDF5Buffer
            A buffer wrapping a one-filled HDF5 dataset.
        """
        return cls(
            cls.create_dataset(*args, data=np.ones(shape, dtype=dtype), **kwargs)
        )

    @classmethod
    def full(
        cls,
        shape: Tuple[int, ...],
        *args,
        fill_value: Any = 0.0,
        dtype: Any = float,
        **kwargs,
    ) -> "HDF5Buffer":
        """
        Create an HDF5-backed buffer filled with a constant value or quantity.

        Parameters
        ----------
        shape : tuple of int
            Desired shape of the dataset.
        fill_value : scalar, array-like, or unyt_quantity, default 0.0
            Value to initialize the dataset with. Can be:
            - A plain scalar (e.g., 3.14)
            - A NumPy array or similar array-like object
        dtype : data-type, default float
            Element type of the dataset.
        *args :
            Additional positional arguments passed to `create_dataset`.
        **kwargs :
            Keyword arguments forwarded to `create`, such as:

            - parent : h5py.File or h5py.Group
            - dataset_name : str
            - overwrite : bool

        Returns
        -------
        HDF5Buffer
            A buffer wrapping a constant-filled HDF5 dataset.
        """
        return cls(
            cls.create_dataset(
                *args, data=np.full(shape, fill_value=fill_value, dtype=dtype), **kwargs
            )
        )

    @classmethod
    def empty(
        cls, shape: Tuple[int, ...], *args, dtype: Any = float, **kwargs
    ) -> "HDF5Buffer":
        """
        Create an HDF5-backed buffer filled with zeros. Equivalent to :meth:`zeros`.
        """
        file, name, *args = args
        return cls(cls.create_dataset(file, name, shape, dtype, *args, **kwargs))

    def as_array(self) -> np.ndarray:
        """Return a NumPy array view of the full dataset."""
        return np.asarray(self[:])

    def close(self, force: bool = False):
        """
        Close the underlying HDF5 file, if owned by this buffer.

        This method closes the file used by this buffer if:
        - It was opened by the buffer (i.e., the buffer owns it), or
        - `force=True` is explicitly specified.

        After calling this method, the buffer becomes invalid for further
        access or modification. Attempting to access data will raise errors.

        Parameters
        ----------
        force : bool, default False
            If True, closes the file regardless of ownership.
            Use with care to avoid interfering with shared file handles.

        Raises
        ------
        AttributeError
            If the file handle is already invalid or closed.

        See Also
        --------
        flush : For non-destructive persistence.
        """
        if self.__is_owner__ or force:
            self.file.close()

    def flush(self):
        """
        Flush any buffered data to disk without closing the file.

        This method ensures that any pending write operations are committed
        to the backing HDF5 file. It is safe to call multiple times and does
        nothing if the file is read-only or already closed.

        This is useful in long-running operations where intermediate data
        should be saved but the file should remain open.

        See Also
        --------
        close : For final closure of the file.
        """
        self.file.flush()
