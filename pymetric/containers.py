"""
Container environments for geometric fields.

This module defines the abstract base class :class:`FieldContainer` for managing
collections of geometric fields that share a common structured grid. It is designed
to organize and operate on scalar, vector, or tensor-valued fields (e.g., density,
velocity, metric tensor) in a coordinated spatial domain.

The container enforces consistency of the grid across all fields and provides
dictionary-like access to stored fields, as well as generator utilities for
constructing new fields from common initialization routines.

See Also
--------
:class:`~fields.base.DenseField`
:class:`~fields.base.SparseField`
:class:`~fields.tensors.DenseTensorField`
:mod:`~grids.base`
"""
from typing import Dict, Iterator, Optional, Tuple, Union

from pymetric.fields.base import DenseField, SparseField
from pymetric.fields.tensors import DenseTensorField
from pymetric.grids.base import GridBase

_FieldType = Union[DenseField, SparseField, DenseTensorField]
_DenseField = Union[DenseField, DenseTensorField]

# --- State Parameters --- #
__field_aliases__ = dict(s=SparseField, d=DenseField, dt=DenseTensorField)


class FieldContainer:
    """
    Base container for managing multiple geometric fields on a common structured grid.

    This class ensures all contained fields share a consistent spatial structure, and provides
    dictionary-like access and management utilities.

    Notes
    -----
    - All fields must be defined on the same `GridBase` instance.
    - This base class is designed to be subclassed for domain-specific applications.

    """

    def __init__(
        self,
        grid: "GridBase",
        *args,
        fields: Optional[Dict[str, _FieldType]] = None,
        **kwargs,
    ):
        """
        Initialize a :class:`FieldContainer` instance.

        Parameters
        ----------
        grid : GridBase
            The structured grid shared by all fields.
        fields : dict of str -> FieldType, optional
            Initial mapping of field names to field objects.
        *args, **kwargs
            Arbitrary additional arguments, for use in subclasses.
        """
        if not isinstance(grid, GridBase):
            raise ValueError("`grid` must be an instance of `GridBase`.")
        self.__grid__: GridBase = grid
        self.__fields__: Dict[str, _FieldType] = {}

        if fields:
            for name, field in fields.items():
                if field.grid is not self.__grid__:
                    raise ValueError(
                        f"Field '{name}' is defined on a different grid than the container."
                    )
                self.add_field(name, field)

    # ----------------------- #
    # Container Management    #
    # ----------------------- #
    def add_field(self, name: str, field: _FieldType) -> None:
        """
        Add a field to the container.

        Parameters
        ----------
        name : str
            Name to register the field under.
        field : DenseField or SparseField
            Field object to register.

        Raises
        ------
        ValueError
            If the field's grid does not match the container's grid.
        """
        if field.grid is not self.__grid__:
            raise ValueError(f"Field '{name}' must use the same grid as the container.")

        if name in self.__fields__:
            raise ValueError(
                "A field already exists with this name. Please remove it before adding a new one."
            )

        self.__fields__[name] = field

    def get_field(self, name: str) -> _FieldType:
        """
        Retrieve a field by name.

        Parameters
        ----------
        name : str
            Name of the field to retrieve.

        Returns
        -------
        DenseField or SparseField
            Field object associated with the name.

        Raises
        ------
        KeyError
            If the name does not correspond to a field.
        """
        return self.__fields__[name]

    def has_field(self, name: str) -> bool:
        """
        Check if a field with the given name is registered.

        Parameters
        ----------
        name : str
            Field name to check.

        Returns
        -------
        bool
            True if the field exists, False otherwise.
        """
        return name in self.__fields__

    def remove_field(self, name: str, missing_ok: bool = False) -> None:
        """
        Remove a field by name.

        Parameters
        ----------
        name : str
            Name of the field to remove.
        missing_ok : bool, optional
            If True, ignore if field doesn't exist. If False (default), raise KeyError.
        """
        if name in self.__fields__:
            del self.__fields__[name]
        elif not missing_ok:
            raise KeyError(f"Field '{name}' not found in container.")

    def filter_by_type(self, field_type: type) -> Dict[str, _FieldType]:
        """
        Return a dictionary of fields matching a specific type.

        Parameters
        ----------
        field_type : type
            The field class to match (e.g., DenseField).

        Returns
        -------
        dict
            Dictionary of (name, field) pairs matching the type.
        """
        return {k: v for k, v in self.__fields__.items() if isinstance(v, field_type)}

    def __getitem__(self, name: str) -> _FieldType:
        return self.get_field(name)

    def __setitem__(self, name: str, field: _FieldType) -> None:
        self.add_field(name, field)

    def __delitem__(self, name: str) -> None:
        del self.__fields__[name]

    def __contains__(self, name: str) -> bool:
        return name in self.__fields__

    def __len__(self) -> int:
        return len(self.__fields__)

    def __iter__(self):
        return iter(self.__fields__)

    # ----------------------- #
    # Properties              #
    # ----------------------- #
    @property
    def grid(self) -> "GridBase":
        """
        The shared grid of all fields in the container.

        Returns
        -------
        GridBase
        """
        return self.__grid__

    @property
    def coordinate_system(self):
        """
        Coordinate system associated with the grid.

        Returns
        -------
        _CoordinateSystemBase
        """
        return self.__grid__.coordinate_system

    @property
    def field_names(self) -> list:
        """
        List of all registered field names.

        Returns
        -------
        list of str
        """
        return list(self.__fields__.keys())

    # ----------------------- #
    # Dunder Methods          #
    # ----------------------- #
    def __repr__(self) -> str:
        return f"<FieldContainer(grid={self.__grid__}, fields={list(self.__fields__.keys())})>"

    # ------------------------ #
    # Generic Methods          #
    # ------------------------ #
    def items(self) -> Iterator[Tuple[str, _FieldType]]:
        """
        Return an iterator over (name, field) pairs.

        Returns
        -------
        iterator of (str, FieldType)
        """
        return self.__fields__.items()

    def keys(self) -> Iterator[str]:
        """
        Return an iterator over field names.

        Returns
        -------
        iterator of str
        """
        return self.__fields__.keys()

    def values(self) -> Iterator[_FieldType]:
        """
        Return an iterator over field values.

        Returns
        -------
        iterator of FieldType
        """
        return self.__fields__.values()

    def summary(self) -> str:
        """
        Return a formatted summary of all registered fields and their basic properties.

        Returns
        -------
        str
            Tabular-style summary string for inspection.
        """
        if not self.__fields__:
            return "FieldContainer is empty."

        lines = ["{:<15} {:<20} {:<20}".format("Name", "Type", "Shape")]
        lines.append("-" * 60)
        for name, field in self.__fields__.items():
            lines.append(
                "{:<15} {:<20} {:<20}".format(
                    name, type(field).__name__, str(getattr(field, "shape", "—"))
                )
            )
        return "\n".join(lines)

    # -------------------------- #
    # Generator Methods          #
    # -------------------------- #
    # These are methods that allow the user to add fields
    # via constructor to the field collection.
    @staticmethod
    def _forward_op_to_field(op: str, field_class: type, *args, **kwargs):
        """
        Forward a named class method (e.g., 'zeros', 'from_array') to a specified field class
        and return the resulting field instance.

        This helper supports generator-style utilities for field creation inside the container,
        ensuring the method exists and is callable on the provided field class.

        Parameters
        ----------
        op : str
            Name of the class method to forward (e.g., "zeros", "from_function").
        field_class : type
            The field class to which the method call should be forwarded (e.g., `DenseField`).
        *args
            Positional arguments to pass to the class method.
        **kwargs
            Keyword arguments to pass to the class method.

        Returns
        -------
        FieldType
            A newly created field instance.

        Raises
        ------
        TypeError
            If `field_class` does not define a callable method named `op`.
        """
        method = getattr(field_class, op, None)
        if not callable(method):
            raise TypeError(
                f"Field class `{field_class.__name__}` does not define a callable method `{op}`."
            )
        return method(*args, **kwargs)

    def zeros(self, name: str, *args, field_type: str = "d", **kwargs) -> _FieldType:
        """
        Create a zero-initialized field and register it in the container.

        This method delegates to the appropriate field class's `zeros` constructor,
        passing in the grid and any user-defined arguments.

        .. hint::

            For details on the available args and kwargs, see the following
            depending on `field_type`:

            - ``'d'`` for :class:`~fields.base.DenseField`
            - ``'s'`` for :class:`~fields.base.SparseField`.
            - ``'dt'`` for :class:`~fields.tensors.DenseTensorField`.

        Parameters
        ----------
        name : str
            The name under which to store the generated field.
        *args
            Positional arguments passed to the field class's `zeros()` constructor.
        field_type : str, optional
            Field type alias:

            - ``'d'`` for :class:`~fields.base.DenseField`
            - ``'s'`` for :class:`~fields.base.SparseField`.
            - ``'dt'`` for :class:`~fields.tensors.DenseTensorField`.

            Defaults to ``'d'``.
        **kwargs
            Additional keyword arguments passed to the field class's `zeros()` constructor.

        Raises
        ------
        ValueError
            If `type` is not a recognized alias.
        TypeError
            If the corresponding field class does not implement a `zeros` method.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]

        field = self._forward_op_to_field(
            "zeros", field_cls, self.grid, *args, **kwargs
        )
        self.add_field(name, field)

        return field

    def ones(self, name: str, *args, field_type: str = "d", **kwargs) -> _FieldType:
        """
        Create a one-initialized field and register it in the container.

        This method delegates to the appropriate field class's `ones()` constructor.

        .. hint::

            For details on the available args and kwargs, see the following
            depending on `field_type`:

            - ``'d'`` for :class:`~fields.base.DenseField`
            - ``'s'`` for :class:`~fields.base.SparseField`.
            - ``'dt'`` for :class:`~fields.tensors.DenseTensorField`.

        Parameters
        ----------
        name : str
            The name under which to store the generated field.
        *args
            Positional arguments passed to the field class's `ones()` constructor.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        FieldType
            The generated field.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field("ones", field_cls, self.grid, *args, **kwargs)
        self.add_field(name, field)
        return field

    def empty(self, name: str, *args, field_type: str = "d", **kwargs) -> _FieldType:
        """
        Create an uninitialized field and register it in the container.

        This method delegates to the appropriate field class's `empty()` constructor.

        .. hint::

            For details on the available args and kwargs, see the following
            depending on `field_type`:

            - ``'d'`` for :class:`~fields.base.DenseField`
            - ``'s'`` for :class:`~fields.base.SparseField`.
            - ``'dt'`` for :class:`~fields.tensors.DenseTensorField`.

        Parameters
        ----------
        name : str
            The name under which to store the generated field.
        *args
            Positional arguments passed to the field class's `empty()` constructor.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        FieldType
            The generated field.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field(
            "empty", field_cls, self.grid, *args, **kwargs
        )
        self.add_field(name, field)
        return field

    def full(
        self, name: str, fill_value, *args, field_type: str = "d", **kwargs
    ) -> _FieldType:
        """
        Create a field filled with a specified value and register it in the container.

        This method delegates to the appropriate field class's `full()` constructor.

        .. hint::

            For details on the available args and kwargs, see the following
            depending on `field_type`:

            - ``'d'`` for :class:`~fields.base.DenseField`
            - ``'s'`` for :class:`~fields.base.SparseField`.
            - ``'dt'`` for :class:`~fields.tensors.DenseTensorField`.

        Parameters
        ----------
        name : str
            The name under which to store the generated field.
        fill_value : scalar
            The value to fill the field with.
        *args
            Additional positional arguments passed to the `full()` constructor.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        FieldType
            The generated field.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field(
            "full", field_cls, self.grid, fill_value, *args, **kwargs
        )
        self.add_field(name, field)
        return field

    def ones_like(
        self, name: str, reference: _FieldType, field_type: str = "d", **kwargs
    ) -> _FieldType:
        """
        Create a one-initialized field with the same shape and metadata as `reference`.

        Parameters
        ----------
        name : str
            Name to assign to the generated field.
        reference : FieldType
            The field to match shape and structure with.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments passed to the `ones_like()` constructor.

        Returns
        -------
        FieldType
            The generated field.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field("ones_like", field_cls, reference, **kwargs)
        self.add_field(name, field)
        return field

    def zeros_like(
        self, name: str, reference: _FieldType, field_type: str = "d", **kwargs
    ) -> _FieldType:
        """
        Create a zero-initialized field with the same shape and metadata as `reference`.

        Parameters
        ----------
        name : str
            Name to assign to the generated field.
        reference : FieldType
            The field to match shape and structure with.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments passed to the `zeros_like()` constructor.

        Returns
        -------
        FieldType
            The generated field.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field("zeros_like", field_cls, reference, **kwargs)
        self.add_field(name, field)
        return field

    def empty_like(
        self, name: str, reference: _FieldType, field_type: str = "d", **kwargs
    ) -> _FieldType:
        """
        Create an uninitialized field with the same shape and metadata as `reference`.

        Parameters
        ----------
        name : str
            Name to assign to the generated field.
        reference : FieldType
            The field to match shape and structure with.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments passed to the `empty_like()` constructor.

        Returns
        -------
        FieldType
            The generated field.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field("empty_like", field_cls, reference, **kwargs)
        self.add_field(name, field)
        return field

    def full_like(
        self,
        name: str,
        reference: _FieldType,
        fill_value,
        field_type: str = "d",
        **kwargs,
    ) -> _FieldType:
        """
        Create a field filled with a specified value and structured like `reference`.

        Parameters
        ----------
        name : str
            Name to assign to the generated field.
        reference : FieldType
            The field to match shape and structure with.
        fill_value : scalar
            The value to fill the field with.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments passed to the `full_like()` constructor.

        Returns
        -------
        FieldType
            The generated field.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field(
            "full_like", field_cls, reference, fill_value, **kwargs
        )
        self.add_field(name, field)
        return field

    def from_array(
        self, name: str, array, field_type: str = "d", **kwargs
    ) -> _FieldType:
        """
        Create a field from an existing array and register it in the container.

        This method delegates to the appropriate field class's `from_array()` constructor,
        passing the container's grid and the user-supplied array.

        .. hint::

            The shape and layout of `array` must match the expected format for the field type.

        Parameters
        ----------
        name : str
            The name under which to store the generated field.
        array : array-like
            Input array to wrap as a field.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments passed to the field class's `from_array()` method.

        Returns
        -------
        FieldType
            The generated field.

        Raises
        ------
        ValueError
            If the provided `field_type` is not recognized.
        TypeError
            If the field class does not support `from_array`.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field(
            "from_array", field_cls, self.grid, array, **kwargs
        )
        self.add_field(name, field)
        return field

    def from_function(
        self, name: str, func, field_type: str = "d", **kwargs
    ) -> _FieldType:
        """
        Create a field by evaluating a function on the container's grid and register it.

        This method delegates to the appropriate field class's `from_function()` constructor,
        passing in the container's grid and a user-defined function.

        .. hint::

            The function `func` should be callable and accept grid coordinates or meshgrid input.

        Parameters
        ----------
        name : str
            The name under which to store the generated field.
        func : callable
            Function to evaluate across the grid. Must be compatible with the expected
            signature for the field type.
        field_type : str, optional
            Field type alias (default is ``'d'``).
        **kwargs
            Additional keyword arguments passed to the field class's `from_function()` method.

        Returns
        -------
        FieldType
            The generated field.

        Raises
        ------
        ValueError
            If the provided `field_type` is not recognized.
        TypeError
            If the field class does not support `from_function`.
        """
        if field_type not in __field_aliases__:
            raise ValueError(
                f"Invalid field type alias '{field_type}'. Expected one of {list(__field_aliases__.keys())}."
            )

        field_cls = __field_aliases__[field_type]
        field = self._forward_op_to_field(
            "from_function", field_cls, self.grid, func, **kwargs
        )
        self.add_field(name, field)
        return field
