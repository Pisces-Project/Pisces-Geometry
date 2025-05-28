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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import h5py
import numpy as np
import unyt
from numpy.typing import ArrayLike
from unyt import Unit, unyt_array, unyt_quantity
from unyt.exceptions import UnitConversionError

from pymetric.fields.buffers.base import BufferBase
from pymetric.fields.buffers.utilities import _to_unyt_array

if TYPE_CHECKING:
    from unyt import UnitRegistry


class ArrayBuffer(BufferBase):
    """
    A lightweight buffer wrapper around a plain `NumPy <https://numpy.org/doc/stable/index.html>`__ array.

    This class provides a minimal, unitless backend for storing field data using
    standard :class:`numpy.ndarray` objects. It is designed for general-purpose use cases
    where unit handling or advanced I/O (e.g., HDF5) is not required.

    Because it does not attach physical units, this class is best suited for purely
    numerical workflows or as a baseline buffer in performance-sensitive tasks.

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
    UnytArrayBuffer : A unit-aware buffer backend.
    HDF5Buffer: HDF5 backed buffer.
    ~fields.buffers.base.BufferBase : Abstract interface for all buffer backends.
    """

    # === Class Attributes === #
    # These attributes configure behavior and registration rules for buffer subclasses.
    # All **concrete** buffer classes MUST define these attributes explicitly.
    #
    # Abstract classes may omit them by setting __is_abc__ = True.
    __is_abc__ = False
    __can_resolve__ = [np.ndarray, list, tuple]
    __core_array_types__ = (np.ndarray,)
    __representation_types__ = (np.ndarray,)
    __resolution_priority__ = 0

    def __init__(self, array: np.ndarray):
        """
        Initialize the buffer with a NumPy array.

        Parameters
        ----------
        array : numpy.ndarray
            A NumPy array to wrap. This should already be an instance
            of :py:class:`numpy.ndarray`. Use :py:meth:`from_array` or :py:meth:`coerce` for flexible inputs.
        """
        super().__init__(array)

    @classmethod
    def from_array(
        cls, obj: Any, *args, dtype: Optional[Any] = None, **kwargs
    ) -> "ArrayBuffer":
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
        dtype : data-type, optional
            Desired data type for the resulting array. If omitted, the data type
            is inferred from `obj`.
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
        return cls(np.array(obj, *args, dtype=dtype, **kwargs))

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
        super().convert_to_units(units, equivalence=equivalence, **kwargs)

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


class UnytArrayBuffer(BufferBase):
    """
    A buffer that wraps a :py:class:`~unyt.array.unyt_array`, providing unit-aware numerical storage.

    This buffer is designed for use cases that require physical units. Internally,
    it stores data as a :py:class:`~unyt.array.unyt_array` and ensures consistent behavior when interacting
    with values that carry or lack units.

    The class supports coercion from raw array-like types (NumPy arrays, lists, tuples)
    and automatically wraps them in a :py:class:`~unyt.array.unyt_array`.

    Examples
    --------
    Create an :py:class:`UnytArrayBuffer` from a list of values.

    >>> UnytArrayBuffer.from_array([1, 2, 3], units="m")
    UnytArrayBuffer(shape=(3,), dtype=int64)

    Generate a buffer using the :meth:`ones` method.

    >>> UnytArrayBuffer.ones((2, 2), units="kg").as_core()
    unyt_array([[1., 1.],
           [1., 1.]], 'kg')

    Wrap a generic :py:class:`~unyt.array.unyt_array`.

    >>> UnytArrayBuffer.from_array(unyt_array([1.,1.],units='keV'))
    UnytArrayBuffer(shape=(2,), dtype=float64)

    """

    # === Class Attributes === #
    # These attributes configure behavior and registration rules for buffer subclasses.
    # All **concrete** buffer classes MUST define these attributes explicitly.
    #
    # Abstract classes may omit them by setting __is_abc__ = True.
    __is_abc__ = False
    __can_resolve__ = [unyt_array]
    __core_array_types__ = (unyt_array,)
    __representation_types__ = (unyt_array,)
    __resolution_priority__ = 10

    def __init__(self, array: unyt_array):
        """
        Initialize the buffer with a NumPy array.

        Parameters
        ----------
        array : unyt.array.unyt_array
            An Unyt array to wrap. This should already be an instance
            of :py:class:`unyt.array.unyt_array`. Use :py:meth:`from_array` or :py:meth:`coerce` for flexible inputs.
        """
        super().__init__(array)

    @property
    def units(self):
        """
        Physical units attached to the buffer data.

        Returns
        -------
        unyt.unit_registry.Unit
            The physical units associated with this buffer’s array values.
        """
        return self.__array_object__.units

    @units.setter
    def units(self, value: Union[Unit, str]):
        """
        Set the physical units in the HDF5 dataset metadata.

        Parameters
        ----------
        value : str or unyt.Unit
            Unit string / instance to attach.
        """
        self.__array_object__.units = value

    @classmethod
    def from_array(
        cls,
        obj: Any,
        dtype: Optional[Any] = None,
        units: Optional[Union["Unit", str]] = None,
        *args,
        registry: Optional["UnitRegistry"] = None,
        bypass_validation: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> "UnytArrayBuffer":
        """
        Construct a new unit-aware buffer from any array-like input.

        This method wraps the input in an :class:`~unyt.array.unyt_array`, preserving or applying physical units
        as needed. It handles input that is already a :class:`~unyt.array.unyt_array` or :class:`~unyt.array.unyt_quantity`, a plain
        :class:`numpy.ndarray`, or any array-like object (e.g., list, tuple) that can be converted via
        :func:`numpy.array`.

        Parameters
        ----------
        obj : array-like
            The input data to interpret as a buffer. Regardless of the input type,
            the data will be coerced to a :class:`~unyt.array.unyt_array` and then wrapped
            in :class:`UnytArrayBuffer`.
        units : str or :class:`~unyt.unit_object.Unit`, optional
            Physical units to attach to the resulting array. If not provided and `obj` has
            attached units, they are preserved. Otherwise defaults to dimensionless.
        dtype : data-type, optional
            Desired data type of the resulting array. If not specified, inferred from input.
        registry: :class:`~unyt.unit_registry.UnitRegistry`, optional
            Registry to associate with the units, if applicable.
        bypass_validation : bool, default False
            If True, skip unit and value validation (faster but unsafe for malformed input).
        name : str, optional
            Optional name for the array (useful for annotation).
        **kwargs :
            Additional keyword arguments passed to :func:`numpy.array` if the input must be coerced.

        Returns
        -------
        UnytArrayBuffer
            A buffer instance wrapping the resulting :class:`~unyt.array.unyt_array`.

        Raises
        ------
        TypeError
            If the input cannot be converted into a :class:`~unyt.array.unyt_array`.

        Notes
        -----
        **Input Type Behavior:**

        - If `obj` is already a :class:`~unyt.array.unyt_array` or :class:`~unyt.array.unyt_quantity`, it is forwarded directly
          (converted if `units` is specified).
        - If `obj` is a :class:`numpy.ndarray`, it is passed to :class:`~unyt.array.unyt_array` with all metadata.
        - Otherwise, the input is cast via :func:`numpy.array` and then wrapped.
        """
        if isinstance(obj, unyt_array):
            # Convert if user has requested a new unit.
            if units is not None:
                return cls(obj.to(units))
            return cls(obj)

        if isinstance(obj, unyt_quantity):
            # Promote scalar to unyt_array; preserve or convert units.
            array = unyt_array(
                obj,
                units=units or obj.units,
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                name=name,
            )
            return cls(array)

        if isinstance(obj, np.ndarray):
            # Wrap numpy arrays directly in unyt_array
            return cls(
                unyt_array(
                    obj,
                    units=units or "",
                    registry=registry,
                    dtype=dtype,
                    bypass_validation=bypass_validation,
                    name=name,
                )
            )

        # Final fallback: try to coerce to np.array first, then wrap
        coerced = np.array(obj, dtype=dtype, **kwargs)
        return cls(
            unyt_array(
                coerced,
                units=units or "",
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                name=name,
            )
        )

    def convert_to_units(
        self, units: Union[str, unyt.Unit], equivalence=None, **kwargs
    ):
        """
        Convert the buffer data to the specified physical units in-place.

        This method performs an in-place conversion of the HDF5 dataset to the target
        physical units and updates the unit metadata stored in the HDF5 file. The
        conversion is only valid if the target units are dimensionally compatible with
        the current units.

        The method preserves the structure and layout of the underlying dataset while
        modifying its numerical values according to the specified units. This is
        useful when standardizing units across datasets or applying unit-sensitive
        transformations in physical simulations or analysis pipelines.

        Parameters
        ----------
        units : str or ~unyt.unit_object.Unit
            Target units to convert the buffer data to. Can be a unit string (e.g., "km")
            or a :class:`unyt.unit_object.Unit` instance.
        equivalence : str, optional
            Optional unit equivalence (e.g., "mass_energy") to use when converting
            between units that are not strictly dimensionally identical but are
            convertible under certain physical principles.
        **kwargs :
            Additional keyword arguments forwarded to `unyt`'s unit conversion routines.

        Raises
        ------
        UnitConversionError
            If the target units are not compatible with the buffer's existing units.
        """
        self.__array_object__.convert_to_units(units, equivalence=equivalence, **kwargs)

    @classmethod
    def zeros(cls, shape, *args, units: str = "", **kwargs) -> "UnytArrayBuffer":
        """
        Create a buffer filled with zeros and optional units.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array to create.
        units : str, optional
            Units to assign to the buffer.
        *args :
            Additional positional arguments forwarded to :func:`numpy.zeros`.
        **kwargs :
            Keyword arguments forwarded to :func:`numpy.zeros`.

        Returns
        -------
        UnytArrayBuffer
            A zero-filled, unit-tagged buffer.
        """
        return cls.from_array(np.zeros(shape, *args, **kwargs), units=units)

    @classmethod
    def ones(cls, shape, *args, units: str = "", **kwargs) -> "UnytArrayBuffer":
        """
        Create a buffer filled with ones and optional units.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array to create.
        units : str, optional
            Units to assign to the buffer.
        *args :
            Additional positional arguments forwarded to :func:`numpy.ones`.
        **kwargs :
            Keyword arguments forwarded to :func:`numpy.ones`.

        Returns
        -------
        UnytArrayBuffer
            A one-filled, unit-tagged buffer.
        """
        return cls.from_array(np.ones(shape, *args, **kwargs), units=units)

    @classmethod
    def full(
        cls, shape: Tuple[int, ...], *args, fill_value: Any = 0.0, **kwargs
    ) -> "UnytArrayBuffer":
        """
        Create a buffer filled with a constant value and attached units.

        This method accepts either scalar values or :class:`~unyt.array.unyt_quantity` inputs for `fill_value`.
        If a :class:`~unyt.array.unyt_quantity` is provided, its units are extracted automatically unless
        overridden by the `units` keyword.

        Parameters
        ----------
        shape : tuple of int
            Desired shape of the array.
        fill_value : scalar or unyt_quantity, default 0.0
            The value to fill the array with. Units are inferred if `fill_value` is a :class:`~unyt.array.unyt_quantity`.
        *args :
            Additional arguments passed to :func:`numpy.full`.
        **kwargs :
            Additional keyword arguments passed to :func:`numpy.full`. May include:
            - units : str — to override or explicitly declare output units.

        Returns
        -------
        UnytArrayBuffer
            A buffer filled with the specified value and tagged with physical units.

        Raises
        ------
        TypeError
            If the resulting array cannot be coerced into a `unyt_array`.
        """
        units = kwargs.pop("units", "")

        if isinstance(fill_value, unyt_quantity):
            if not units:
                units = str(fill_value.units)
            fill_scalar = fill_value.value
        else:
            fill_scalar = fill_value

        array = np.full(shape, fill_scalar, *args, **kwargs)
        return cls.from_array(array, units=units)

    @classmethod
    def empty(cls, shape, *args, units: str = "", **kwargs) -> "UnytArrayBuffer":
        """
        Return a new array of given shape and type, without initializing entries.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array to create.
        units : str, optional
            Units to assign to the buffer.
        *args :
            Additional positional arguments forwarded to :func:`numpy.empty`.
        **kwargs :
            Keyword arguments forwarded to :func:`numpy.empty`.

        Returns
        -------
        UnytArrayBuffer
            A one-filled, unit-tagged buffer.
        """
        return cls.from_array(np.empty(shape, *args, **kwargs), units=units)

    def as_repr(self) -> ArrayLike:
        """
        Return this buffer as an unyt array.
        """
        return self.as_core()


class HDF5Buffer(BufferBase):
    """
    A buffer that wraps a lazily-loaded HDF5 dataset using :class:`h5py.Dataset`.

    This buffer enables disk-backed storage of field data, making it ideal for large datasets
    that exceed memory constraints. Internally, it wraps a persistent HDF5 dataset and exposes
    a NumPy-compatible interface for slicing, indexing, and arithmetic operations.

    Unlike in-memory buffers like :class:`ArrayBuffer` or :class:`UnytArrayBuffer`, this buffer
    does not eagerly load its contents into memory. Instead, all reads and writes are performed
    through a memory-mapped interface, and computations are supported through seamless integration
    with NumPy ufuncs.

    If units are provided (either at creation or via a unit-aware input), they are stored as
    a string attribute on the dataset and automatically reapplied when data is accessed.

    Examples
    --------
    Create a disk-backed buffer with units:

    >>> buff = HDF5Buffer.from_array([1, 2, 3],
    ...                       "mydata.h5",
    ...                       "dataset",
    ...                       units=None,
    ...                       overwrite=True,dtype='f8',
    ...                       create_file=True)
    >>> buff
    HDF5Buffer(shape=(3,), dtype=float64, path='/dataset')

    See Also
    --------
    ArrayBuffer : In-memory NumPy array backend.
    UnytArrayBuffer : In-memory unit-aware buffer.
    ~fields.buffers.base.BufferBase : Abstract buffer interface.
    """

    # ------------------------------ #
    # Class Flags                    #
    # ------------------------------ #
    # These attributes configure behavior and
    # registration rules for buffer subclasses.
    # All **concrete** buffer classes MUST define
    # these attributes explicitly.
    #
    # Abstract classes may omit them by setting __is_abc__ = True.
    __is_abc__ = False
    __can_resolve__ = [h5py.Dataset]
    __core_array_types__ = (h5py.Dataset,)
    __representation_types__ = (unyt_array, np.ndarray)
    __resolution_priority__ = 30

    # Reduce the scope of directed array dispatching to
    # provide a more intuitive behavioral context.
    __array_function_dispatch__: Optional[Dict[Callable, Callable]] = {
        # Internally defined properties.
        np.shape: lambda self, *_, **__: self.shape,
        np.ndim: lambda self, *_, **__: self.ndim,
        np.size: lambda self, *_, **__: self.size,
    }

    # ------------------------------ #
    # Initialization                 #
    # ------------------------------ #
    # The initialization procedure should be meta stable
    # in the sense that it always behaves the same way: __init__
    # requires a pre-coerced type and simply checks for type compliance.
    # Other methods can be used for more adaptive behavior.
    def __init__(self, array: h5py.Dataset, __is_file_owner__: bool = False):
        """
        Create an :class:`HDF5Buffer` object.

        Parameters
        ----------
        array : h5py.Dataset
            Open dataset to wrap. *Must* remain valid for the buffer's lifetime.
        """
        super().__init__(array)
        self.__is_owner__: bool = __is_file_owner__

    def __enter__(self) -> "HDF5Buffer":
        """Enter the buffer context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context: flush and optionally close file if owned."""
        self.__array_object__.flush()
        if self.__is_owner__:
            self.__array_object__.file.close()

    # noinspection PyIncorrectDocstring
    @classmethod
    def from_array(
        cls,
        obj: Any,
        file: Optional[Union[str, Path, h5py.File]] = None,
        path: Optional[str] = None,
        *args,
        dtype: Optional[Any] = None,
        units: Optional[Union["Unit", str]] = None,
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
        units : str or :class:`~unyt.unit_object.Unit`, optional
            Units to attach to the dataset. If `obj` already has units, this overrides them.
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
        if isinstance(obj, h5py.Dataset):
            __is_file_owner__ = kwargs.pop("make_owner", False)
            return cls(obj, __is_file_owner__=__is_file_owner__)

        if file is None or path is None:
            raise ValueError(
                "When creating a new dataset, both `file` and `path` must be provided."
            )

        __is_file_owner__ = kwargs.pop("make_owner", True)
        dataset = cls.create_dataset(
            file, path, dtype=dtype, data=obj, units=units, **kwargs
        )
        return cls(dataset, __is_file_owner__=__is_file_owner__)

    def convert_to_units(
        self, units: Union[str, unyt.Unit], equivalence=None, **kwargs
    ):
        """
        Convert the buffer data to the specified physical units in-place.

        This method performs an in-place conversion of the HDF5 dataset to the target
        physical units and updates the unit metadata stored in the HDF5 file. The
        conversion is only valid if the target units are dimensionally compatible with
        the current units.

        The method preserves the structure and layout of the underlying dataset while
        modifying its numerical values according to the specified units. This is
        useful when standardizing units across datasets or applying unit-sensitive
        transformations in physical simulations or analysis pipelines.

        Parameters
        ----------
        units : str or ~unyt.unit_object.Unit
            Target units to convert the buffer data to. Can be a unit string (e.g., "km")
            or a :class:`unyt.unit_object.Unit` instance.
        equivalence : str, optional
            Optional unit equivalence (e.g., "mass_energy") to use when converting
            between units that are not strictly dimensionally identical but are
            convertible under certain physical principles.
        **kwargs :
            Additional keyword arguments forwarded to `unyt`'s unit conversion routines.

        Raises
        ------
        UnitConversionError
            If the target units are not compatible with the buffer's existing units.
        """
        self.__array_object__[...] = self[...].to_value(
            units, equivalence=equivalence, **kwargs
        )
        self.units = units

    # === Internal Utilities === #
    @classmethod
    def create_dataset(
        cls,
        file: Union[str, Path, h5py.File],
        name: str,
        shape: Optional[tuple] = None,
        dtype: Optional[Any] = None,
        *,
        data: Optional[Union[np.ndarray, list, "unyt_array"]] = None,
        overwrite: bool = False,
        units: Optional[Union[str, Unit]] = None,
        **kwargs,
    ) -> h5py.Dataset:
        """
        Create a new HDF5 dataset with optional unit metadata.

        Parameters
        ----------
        file : str, Path, or :class:`h5py.File`
            Path to HDF5 file or open file handle.
        name : str
            Name of the dataset inside the HDF5 file.
        shape : tuple of int, optional
            Shape of the dataset, if not using `data`.
        dtype : data-type, optional
            Desired data type. If not given, inferred from `data`.
        data : array-like, optional
            Initial values. Will be coerced to a `unyt_array`.
        overwrite : bool, default False
            Whether to overwrite an existing dataset.
        units : str or :class:`~unyt.unit_object.Unit`, optional
            Units to attach to the dataset. If not provided and `data` has units, they are preserved.
        **kwargs :
            Additional arguments to :meth:`~h5py.File.create_dataset`.

        Returns
        -------
        ~h5py.Dataset
            The created dataset.
        """
        # Open the file if it is provided as a path
        # and not as an HDF5 file. Ensure it exists before
        # proceeding. By default, we will allow the
        # creation of the file.
        if isinstance(file, (str, Path)):
            file = Path(file)

            # Handle the non-existence options. By default,
            # we raise an error unless create_file = True
            __create_file_flag__ = kwargs.pop("create_file", False)
            if not file.exists():
                if not __create_file_flag__:
                    raise FileNotFoundError(
                        f"File '{file}' does not exist. To create a new file, ensure create_file=True."
                    )
                else:
                    file.touch()

            # Replace the file with an HDF5 reference.
            file = h5py.File(file, mode="r+")
        elif isinstance(file, h5py.File):
            _ = kwargs.pop("create_file", False)
        else:
            raise TypeError(
                f"`file` must be a file path or an h5py.File, got {type(file)}"
            )

        # With the file open, we need to now resolve the
        # path, ensure overwrite conventions are respected
        # and prepare to generate the dataset.
        if name in file:
            if overwrite:
                del file[name]
            else:
                raise ValueError(
                    f"Dataset '{name}' already exists. Use `overwrite=True` to replace it."
                )

        # Handle data provided. If we received data, we force it into
        # the our conventional unyt type and then grab out the
        # true data and the units. Otherwise, we just skip through.
        # Normalize units that we got from kwargs.
        # Normalize units provided by user
        raw_data = None  # Default to None unless data is provided
        if data is not None:
            # We got handed data. We need to standardize units and then
            # build the unyt array.
            arg_units = unyt.Unit(units) if units is not None else None
            arr_units = getattr(data, "units", None)

            if (arg_units is not None) and (arr_units is not None):
                data_array = _to_unyt_array(data, dtype=dtype).to(arg_units).d
                units = arg_units
            elif (arg_units is not None) and (arr_units is None):
                data_array = _to_unyt_array(data, dtype=dtype, units=arg_units).d
                units = arg_units
            elif (arr_units is not None) and (arg_units is None):
                data_array = _to_unyt_array(data, dtype=dtype).d
                units = arr_units
            else:
                data_array = _to_unyt_array(data, dtype=dtype).d
                units = None

            raw_data = data_array

        # Create the dataset
        dset = file.create_dataset(
            name, shape=shape, dtype=dtype, data=raw_data, **kwargs
        )

        # Set units metadata if available
        if units is not None:
            dset.attrs["units"] = str(units)

        return dset

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
        Open an existing HDF5 dataset and return it as a buffer.

        This method provides safe access to datasets within an HDF5 file,
        whether the file is passed as a string path or an already opened
        :class:`h5py.File` object.

        When used in a `with` statement (context manager), the file will be
        automatically flushed and closed at exit if ``close_on_exit=True``.

        Parameters
        ----------
        file : str, Path, or h5py.File
            File path or an open HDF5 file containing the dataset.
        path : str
            Path to the dataset inside the HDF5 file (e.g., ``"/my/data"``).
        mode : str, default "r+"
            File mode to use when `file` is a path. Ignored if `file` is a :class:`h5py.File`.
            Common values:

                - ``"r"``: read-only
                - ``"r+"``: read/write
                - ``"a"``: read/write or create

        close_on_exit : bool, default False
            If True, the file will be closed when the buffer is used in a context manager.

        Returns
        -------
        HDF5Buffer
            A buffer object wrapping the specified dataset.

        Raises
        ------
        FileNotFoundError
            If the file or dataset path does not exist.
        TypeError
            If `file` is neither a path nor an `h5py.File`.

        Examples
        --------
        >>> with HDF5Buffer.open("data.h5", "temperature", close_on_exit=True) as buf:
        ...     print(buf.shape)
        """
        if isinstance(file, (str, Path)):
            file = Path(file)
            if not file.exists():
                raise FileNotFoundError(f"File '{file}' does not exist.")
            file_obj = h5py.File(file, mode=mode)
        elif isinstance(file, h5py.File):
            file_obj = file
        else:
            raise TypeError(f"`file` must be a path or h5py.File, got {type(file)}")

        if path not in file_obj:
            raise KeyError(f"Dataset '{path}' not found in file.")

        dataset = file_obj[path]
        buffer = cls(dataset, __is_file_owner__=close_on_exit)

        return buffer

    # ------------------------------ #
    # Properties                     #
    # ------------------------------ #
    @property
    def units(self) -> Optional[Unit]:
        """
        Return the physical units stored as an HDF5 attribute.

        Returns
        -------
        str or None
            The unit string stored in the dataset, or None if not present.
        """
        _units = self.__array_object__.attrs.get("units", None)
        if _units is None:
            return _units
        return Unit(_units)

    @units.setter
    def units(self, value: Union[Unit, str]):
        """
        Set the physical units in the HDF5 dataset metadata.

        Parameters
        ----------
        value : str or unyt.Unit
            Unit string / instance to attach.
        """
        self.__array_object__.attrs["units"] = str(value)

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
        return self.as_unyt_array().__str__()

    def __getitem__(self, idx):
        """
        Retrieve the values of the buffer for a specified index.

        If units are defined, return a :class:`~unyt.array.unyt_array` that preserves the correct metadata.
        This guarantees safe behavior even for scalar slices or dtype edge cases.

        Parameters
        ----------
        idx : int, slice, or tuple
            Index expression.

        Returns
        -------
        numpy.ndarray or unyt_array
            Data slice, optionally wrapped with units.
        """
        ret = self.__array_object__.__getitem__(idx)

        if self.units is not None:
            if getattr(ret, "shape", None) == ():
                ret = unyt_quantity(ret, bypass_validation=True)
            else:
                ret = unyt_array(ret, bypass_validation=True)
            ret.units = self.units
        return ret

    def __setitem__(self, item, value):
        # Handle the units first to ensure that we are not changing
        # units.
        if hasattr(value, "units"):
            try:
                _raw_value = np.asarray(value.to_value(self.units), dtype=self.dtype)
            except UnitConversionError:
                raise ValueError(
                    "Cannot use __setitem__ with inconsistent units. To replace the entire array with "
                    "one of new units, use `.replace`."
                )
        else:
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

                    unit = getattr(o, "units", None)
                    if unit is not None:
                        o.__array_object__.attrs["units"] = str(unit)

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
            - A :class:`~unyt.array.unyt_quantity` with units
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
        """Return a NumPy array view of the full dataset (units stripped)."""
        return np.asarray(self[:])

    def as_unyt_array(self) -> unyt_array:
        """Return a unit-aware unyt_array view of the full dataset."""
        return self[:]  # __getitem__ already wraps in unyt_array if needed

    def as_repr(self):
        """
        Return this buffer as an :class:`~unyt.array.unyt_array` if
        the buffer has units; otherwise as a :class:`~numpy.ndarray`.
        """
        # Use a memory-mapped view, but preserve units if defined
        out = self.__array_object__[:]
        return unyt_array(out, self.units) if self.units else out

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

    # ---------------------------------- #
    # Reconstructed array manip. methods #
    # ---------------------------------- #
    def replace_dataset(
        self,
        shape: Optional[tuple] = None,
        dtype: Optional[Any] = None,
        data: Optional[Union[np.ndarray, list, "unyt_array"]] = None,
        *,
        units: Optional[Union[str, Unit]] = None,
        **kwargs,
    ) -> "HDF5Buffer":
        """
        Replace the current HDF5 dataset with a newly created one.

        This method deletes and recreates the dataset at the same internal HDF5 path (`self.name`)
        with optionally modified shape, dtype, data, and units. It is useful for performing
        operations like `reshape`, `astype`, or structural updates in-place, without requiring
        the caller to manage HDF5 file I/O manually.

        Parameters
        ----------
        shape : tuple of int, optional
            New shape for the dataset. If not provided, uses the current shape.
        dtype : data-type, optional
            Desired data type for the new dataset. Defaults to the current dtype.
        data : array-like, optional
            Data to populate the new dataset. If not provided, the current buffer data is reused.
        units : str or unyt.Unit, optional
            Units to assign to the new dataset. If not provided, preserves the current units.
        **kwargs :
            Additional keyword arguments passed to :meth:`create_dataset`, such as compression options.

        Returns
        -------
        HDF5Buffer
            The same buffer instance, now referencing the newly created dataset.

        Notes
        -----
        - The underlying HDF5 file must be writable.
        - This will *delete* the existing dataset at `self.name`.
        - Units are preserved by default unless explicitly overridden.
        """
        new_dataset = self.__class__.create_dataset(
            file=self.file,
            name=self.name,
            shape=shape,
            dtype=dtype or self.dtype,
            data=data if data is not None else self.as_repr(),
            overwrite=True,
            units=units or self.units,
            **kwargs,
        )
        self.__array_object__ = new_dataset
        return self

    # ------------------------------ #
    # Standard Numpy Transformations #
    # ------------------------------ #
    def _apply_numpy_transform_on_repr(
        self, func: Callable, fargs, fkwargs, *args, inplace: bool = True, **kwargs
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

        if inplace:
            self.replace_dataset(data=func(repr_view, *fargs, **fkwargs), **kwargs)
            return self

        return self.__class__.from_array(
            func(repr_view, *fargs, **fkwargs), *args, **kwargs
        )

    def _apply_numpy_transform_on_core(
        self, func: Callable, fargs, fkwargs, *args, inplace: bool = True, **kwargs
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
        repr_view = self.as_core()

        if inplace:
            self.replace_dataset(data=func(repr_view, *fargs, **fkwargs), **kwargs)
            return self

        return self.__class__.from_array(
            func(repr_view, *fargs, **fkwargs), *args, **kwargs
        )

    def astype(
        self, dtype: Any, *args, inplace: bool = False, **kwargs
    ) -> "BufferBase":
        """
        Return a copy of this buffer with a different data type.

        This performs a type conversion using the underlying array and returns
        a new buffer of the same class with the updated `dtype`.

        Parameters
        ----------
        dtype : data-type
            The target data type for the returned array.
        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        inplace : bool, optional
            If `True`, values for `file`, `name`, and `overwrite` are ignored; the
            dataset underlying the calling buffer is directly replaced and the object is
            updated and returned in-place.
        **kwargs :
            Additional keyword arguments forwarded to and :meth:`from_array`. If `inplace` is
            ``True``, then these are forwarded to :meth:`replace_dataset`.

        Returns
        -------
        BufferBase
            A new buffer instance with the specified data type.
        """
        if inplace:
            self.replace_dataset(dtype=dtype, **kwargs)
            return self
        else:
            return self.__class__.from_array(
                self.as_repr().astype(dtype), *args, **kwargs
            )

    def reshape(self, shape, *args, inplace: bool = False, **kwargs) -> "BufferBase":
        """
        Return a reshaped copy of this buffer.

        This reshapes the buffer into a new shape and returns a new buffer instance.
        The reshaping is done using the NumPy-compatible view returned by `as_repr()`.

        Parameters
        ----------
        shape : tuple of int
            Target shape for the new buffer.
        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        inplace : bool, optional
            If `True`, values for `file`, `name`, and `overwrite` are ignored; the
            dataset underlying the calling buffer is directly replaced and the object is
            updated and returned in-place.
        **kwargs :
            Additional keyword arguments forwarded to and :meth:`from_array`. If `inplace` is
            ``True``, then these are forwarded to :meth:`replace_dataset`.

        Returns
        -------
        BufferBase
            A new buffer with reshaped data.
        """
        if inplace:
            self.replace_dataset(
                shape=shape, data=self.as_repr().reshape(shape), **kwargs
            )
            return self
        return self.__class__.from_array(self.as_repr().reshape(shape), *args, **kwargs)

    def transpose(
        self, axes=None, *args, inplace: bool = False, **kwargs
    ) -> "BufferBase":
        """
        Return a transposed copy of this buffer.

        The transposition is performed using the NumPy-compatible array returned by `as_repr()`.

        Parameters
        ----------
        axes : tuple of int, optional
            By default, reverses the dimensions. If specified, reorders the axes accordingly.
        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        inplace : bool, optional
            If `True`, values for `file`, `name`, and `overwrite` are ignored; the
            dataset underlying the calling buffer is directly replaced and the object is
            updated and returned in-place.
        **kwargs :
            Additional keyword arguments forwarded to and :meth:`from_array`. If `inplace` is
            ``True``, then these are forwarded to :meth:`replace_dataset`.

        Returns
        -------
        BufferBase
            A new transposed buffer.
        """
        if inplace:
            self.replace_dataset(data=self.as_repr().transpose(axes), **kwargs)
            return self
        return self.__class__.from_array(
            self.as_repr().transpose(axes), *args, **kwargs
        )

    def flatten(
        self, *args, order="C", inplace: bool = False, **kwargs
    ) -> "BufferBase":
        """
        Return a flattened 1D view of this buffer.

        This flattens the buffer using the specified memory layout and returns a new buffer instance.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, default 'C'
            Memory layout to use when flattening:
            - 'C' → row-major (C-style),
            - 'F' → column-major (Fortran-style),
            - 'A' → 'F' if input is Fortran contiguous, else 'C',
            - 'K' → as close to input layout as possible.

        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        inplace : bool, optional
            If `True`, values for `file`, `name`, and `overwrite` are ignored; the
            dataset underlying the calling buffer is directly replaced and the object is
            updated and returned in-place.
        **kwargs :
            Additional keyword arguments forwarded to and :meth:`from_array`. If `inplace` is
            ``True``, then these are forwarded to :meth:`replace_dataset`.

        Returns
        -------
        BufferBase
            A new 1D buffer.
        """
        if inplace:
            self.replace_dataset(data=self.as_repr().flatten(order=order), **kwargs)
            return self
        return self.__class__.from_array(
            self.as_repr().flatten(order=order), *args, **kwargs
        )

    def squeeze(
        self, *args, axis=None, inplace: bool = False, **kwargs
    ) -> "BufferBase":
        """
        Remove singleton dimensions from this buffer.

        Parameters
        ----------
        axis : int or tuple of int, optional
            If specified, only squeezes the given axes. Otherwise, all single-dim axes are removed.
        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        inplace : bool, optional
            If `True`, values for `file`, `name`, and `overwrite` are ignored; the
            dataset underlying the calling buffer is directly replaced and the object is
            updated and returned in-place.
        **kwargs :
            Additional keyword arguments forwarded to and :meth:`from_array`. If `inplace` is
            ``True``, then these are forwarded to :meth:`replace_dataset`.

        Returns
        -------
        BufferBase
            A squeezed buffer.
        """
        if inplace:
            self.replace_dataset(data=self.as_repr().squeeze(axis=axis), **kwargs)
            return self
        return self.__class__.from_array(
            self.as_repr().squeeze(axis=axis), *args, **kwargs
        )

    def expand_dims(
        self, axis: int, *args, inplace: bool = False, **kwargs
    ) -> "BufferBase":
        """
        Expand the shape of this buffer by inserting a new axis.

        Parameters
        ----------
        axis : int
            Position at which to insert the new axis.
        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        inplace : bool, optional
            If `True`, values for `file`, `name`, and `overwrite` are ignored; the
            dataset underlying the calling buffer is directly replaced and the object is
            updated and returned in-place.
        **kwargs :
            Additional keyword arguments forwarded to and :meth:`from_array`. If `inplace` is
            ``True``, then these are forwarded to :meth:`replace_dataset`.

        Returns
        -------
        BufferBase
            A buffer with the expanded shape.
        """
        if inplace:
            self.replace_dataset(
                data=np.expand_dims(self.as_repr(), axis=axis), **kwargs
            )
            return self
        return self.__class__.from_array(
            np.expand_dims(self.as_repr(), axis=axis), *args, **kwargs
        )

    def broadcast_to(
        self, shape: Any, *args, inplace: bool = False, **kwargs
    ) -> "BufferBase":
        """
        Broadcast an array to a new shape.

        Parameters
        ----------
        shape : tuple or int
            The shape of the desired output array. A single integer ``i`` is interpreted
            as ``(i,)``.
        *args :
            Additional positional arguments forwarded to :meth:`from_array`.
        inplace : bool, optional
            If `True`, values for `file`, `name`, and `overwrite` are ignored; the
            dataset underlying the calling buffer is directly replaced and the object is
            updated and returned in-place.
        **kwargs :
            Additional keyword arguments forwarded to and :meth:`from_array`. If `inplace` is
            ``True``, then these are forwarded to :meth:`replace_dataset`.

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
