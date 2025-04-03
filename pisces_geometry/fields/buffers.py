"""
Buffer classes for uniform data access in fields.

The general intention of the :py:mod:`fields.buffers` module is to ensure that :py:class:`fields.base.GenericField` classes only
need to manage their unique behavior as fields and not the behavior of their backend buffers. As such, the ``_Buffer`` class
provides a uniform layer of abstraction so that :py:class:`fields.base.GenericField` objects can effectively
delegate all of the details of actual data storage and interaction.
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple

import h5py
import numpy as np
from numpy.typing import ArrayLike
from unyt import unyt_array


class _Buffer(ABC):
    """
    Abstract base class for all Pisces Geometry-compliant field buffers.

    Buffers abstract the storage backend for a field (NumPy array, unyt array,
    or HDF5 dataset), providing a uniform interface regardless of implementation.
    """

    def __init__(self):
        # This is the core internal storage attribute for the buffer.
        # Subclasses must assign a valid array-like object (e.g., np.ndarray, unyt_array, h5py.Dataset)
        # to `self.__array_object__` during their own initialization logic.
        self.__array_object__: ArrayLike = None

    # --- Core Protocol --- #
    def __getitem__(self, idx):
        return self.__array_object__[idx]

    def __setitem__(self, idx, value):
        self.__array_object__[idx] = value

    def __array__(self, dtype=None):
        return np.asarray(self.__array_object__, dtype=dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"

    # --- Properties --- #
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the buffer.

        This includes both grid and element dimensions, and corresponds to
        the full shape of the underlying array-like object.

        Returns
        -------
        tuple of int
            The shape of the buffer.
        """
        return self.__array_object__.shape

    @property
    def size(self) -> int:
        """
        The total number of elements in the buffer.

        This is equivalent to the product of all dimensions in `shape`.

        Returns
        -------
        int
            The total number of elements.
        """
        return self.__array_object__.size

    @property
    def ndim(self) -> int:
        """
        The number of dimensions in the buffer.

        This represents the number of axes in the array-like object,
        including both grid and element dimensions.

        Returns
        -------
        int
            Number of dimensions.
        """
        return self.__array_object__.ndim

    @property
    def dtype(self) -> Any:
        """
        The data type of the underlying buffer.

        This reflects the type of each element stored in the buffer (e.g., float64, int32).

        Returns
        -------
        dtype
            The data type of the buffer.
        """
        return self.__array_object__.dtype

    @property
    def units(self) -> Any:
        """
        Return units if the buffer supports them.
        Subclasses should override.
        """
        return NotImplemented

    def as_array(self) -> np.ndarray:
        """
        Return the buffer as a NumPy array (eager).
        """
        return np.asarray(self.__array_object__)

    # --- Abstract Constructors --- #

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> "_Buffer":
        """
        Create a new buffer instance from scratch.

        This method is responsible for constructing a fresh array-like object
        (e.g., NumPy array, unyt array, or HDF5 dataset) based on the provided
        shape and metadata, and returning it wrapped in the appropriate buffer class.

        Subclasses are expected to handle any arguments required to construct their
        underlying array types. These often include:

        - `shape` : tuple of int — required shape of the buffer
        - `dtype` : data type of the array (e.g., float, int)
        - `fill_value` : default fill value
        - `units` : optional physical units (for unit-aware buffers)
        - `file`, `dataset_name`, `overwrite`, etc. (for HDF5 buffers)

        Returns
        -------
        _Buffer
            A new instance of the buffer subclass containing the initialized storage.
        """
        pass

    @classmethod
    @abstractmethod
    def coerce(cls, array_like: Any, **kwargs) -> "_Buffer":
        """
        Wrap or convert an existing array-like object into a buffer instance.

        This method is used when a preexisting object (e.g., a NumPy array,
        a unyt array, or an h5py.Dataset) is passed into a field and needs to
        be adapted to conform to the buffer interface.

        Subclasses should verify that the object is compatible with their backend
        and convert or wrap it as needed. This may also include:

        - Validating or applying units
        - Copying or referencing memory (depending on behavior)
        - Raising if the object is not compatible

        Parameters
        ----------
        array_like : Any
            An existing array-like object to adapt into this buffer class.

        Returns
        -------
        _Buffer
            A buffer instance wrapping or referencing the input array-like object.
        """
        pass


class ArrayBuffer(_Buffer):
    """
    A buffer wrapper around a plain NumPy array.

    This class stores and exposes a standard :py:class:`numpy.ndarray`, with no unit-awareness
    or external dependencies. Suitable for most general-purpose, unitless fields.
    """

    def __init__(self, array: ArrayLike):
        super().__init__()
        self.__array_object__ = np.asarray(array)

    @classmethod
    def coerce(cls, array_like: Any, **kwargs) -> "_Buffer":
        """
        Convert or wrap an existing array-like object into a NumPy-backed buffer.

        Parameters
        ----------
        array_like : Any
            An array-like object to wrap (e.g., list, np.ndarray, etc.).

        Returns
        -------
        ArrayBuffer
            A buffer wrapping the input array.
        """
        return cls(np.asarray(array_like))

    @classmethod
    def create(
        cls, shape: Tuple[int, ...], dtype: Any = float, fill_value: Any = 0, **kwargs
    ) -> "_Buffer":
        """
        Create a new NumPy buffer of the given shape and dtype.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array to create.
        dtype : data-type, default float
            Element type.
        fill_value : scalar, default 0
            Initial value to fill.

        Returns
        -------
        ArrayBuffer
            A buffer containing a new NumPy array.
        """
        return cls(np.full(shape, fill_value, dtype=dtype))


class UnytArrayBuffer(_Buffer):
    """
    A buffer wrapper around a :py:class:`unyt.unyt_array`, supporting physical units.

    This buffer uses :py:class:`unyt.unyt_array` to store and manipulate data with attached units.
    """

    def __init__(self, array: ArrayLike, units: str = ""):
        super().__init__()

        if isinstance(array, unyt_array):
            self.__array_object__ = array.to(units) if units else array
        else:
            self.__array_object__ = unyt_array(array, units=units)

    @property
    def units(self):
        """Return the physical units attached to this buffer."""
        return self.__array_object__.units

    @classmethod
    def coerce(cls, array_like: Any, units: str = "", **kwargs) -> "UnytArrayBuffer":
        """
        Convert or wrap an existing object into a unit-aware buffer.

        Parameters
        ----------
        array_like : Any
            An array-like object (NumPy array, list, or `unyt_array`).
        units : str, optional
            Desired units. If the input is a `unyt_array`, the units will be
            preserved unless explicitly overridden.

        Returns
        -------
        UnytArrayBuffer
            A buffer wrapping the input array with units.
        """
        return cls(array_like, units=units)

    @classmethod
    def create(
        cls,
        shape: Tuple[int, ...],
        dtype: Any = float,
        fill_value: Any = 0,
        units: str = "",
        **kwargs,
    ) -> "UnytArrayBuffer":
        """
        Create a new unit-aware buffer.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array to create.
        dtype : data-type, default float
            Element type.
        fill_value : scalar, default 0
            Value to fill.
        units : str, optional
            Physical units to attach.

        Returns
        -------
        UnytArrayBuffer
            A buffer containing a new `unyt_array`.
        """
        arr = np.full(shape, fill_value, dtype=dtype)
        return cls(unyt_array(arr, units=units))


class HDF5Buffer(_Buffer):
    """
    A buffer wrapper for a lazy-loaded HDF5 dataset.

    This buffer accesses an `h5py.Dataset` directly and only loads data
    when it is indexed, making it efficient for large datasets.

    Units (if any) are stored as a string attribute `attrs['units']`.
    """

    def __init__(self, dataset: h5py.Dataset):
        super().__init__()

        # Load the dataset.
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError("HDF5Buffer expects an h5py.Dataset instance.")
        self.__array_object__ = dataset
        self._units = dataset.attrs.get("units", None)

    def __getitem__(self, idx):
        """
        Fetch data from the underlying HDF5 dataset.

        If the buffer has units, returns a `unyt_array` with the same units.
        Otherwise, returns a NumPy array view of the data.

        Parameters
        ----------
        idx : int, slice, or tuple
            Indexing expression.

        Returns
        -------
        np.ndarray or unyt_array
            The sliced data, wrapped in units if applicable.
        """
        data = self.__array_object__[idx]
        if self._units is not None:
            return unyt_array(data, self._units)
        return data

    def __setitem__(self, idx, value):
        """
        Assign values to the underlying HDF5 dataset.

        If the incoming value is a `unyt_array` or has units,
        it's converted to raw NumPy values with units stripped.

        Parameters
        ----------
        idx : int, slice, or tuple
            Indexing expression.
        value : array-like
            The values to write.
        """
        if hasattr(value, "to") and self._units:
            value = value.to(self._units).value  # Convert and strip units
        self.__array_object__[idx] = value

    @property
    def units(self):
        """Return the units stored as metadata in the HDF5 dataset, if present."""
        return self._units

    @units.setter
    def units(self, value):
        self.__array_object__.attrs["units"] = value
        self._units = value

    @classmethod
    def coerce(cls, dataset: h5py.Dataset, **kwargs) -> "HDF5Buffer":
        """
        Wrap an existing HDF5 dataset in an HDF5Buffer.

        Parameters
        ----------
        dataset : h5py.Dataset
            The dataset to wrap.

        Returns
        -------
        HDF5Buffer
        """
        return cls(dataset)

    @classmethod
    def create(
        cls,
        shape: Tuple[int, ...],
        file: h5py.File,
        dataset_name: str,
        dtype: Any = float,
        fill_value: Any = 0,
        units: str = None,
        overwrite: bool = False,
        **kwargs,
    ) -> "HDF5Buffer":
        """
        Create (and optionally overwrite) a dataset in an HDF5 file.

        Parameters
        ----------
        shape : tuple of int
            Shape of the dataset.
        file : h5py.File
            Open HDF5 file or path to file.
        dataset_name : str
            Name of the dataset to create.
        dtype : type, default float
            Data type of the dataset.
        fill_value : scalar, default 0
            Initial fill value.
        units : str, optional
            Units to store in metadata.
        overwrite : bool, default False
            If True, deletes and recreates the dataset if it already exists.

        Returns
        -------
        HDF5Buffer
            A buffer wrapping the new dataset.
        """
        # Now look for / overwrite the dataset if it already
        # exists and we have permission to do so.
        if dataset_name in file:
            if overwrite:
                del file[dataset_name]
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' already exists. Use overwrite=True to replace it."
                )

        # Create the dataset and fill the dataset with the fill value.
        dset = file.create_dataset(dataset_name, shape=shape, dtype=dtype)
        if fill_value != 0:
            dset[...] = fill_value
        if units:
            dset.attrs["units"] = units

        return cls(dset)


__buffer_registry__ = {
    "array": ArrayBuffer,
    "unyt": UnytArrayBuffer,
    "hdf5": HDF5Buffer,
}
