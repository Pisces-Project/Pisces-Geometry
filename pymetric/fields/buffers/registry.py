"""
Buffer resolution and registration logic for Pisces Geometry.

This module defines the infrastructure for registering and resolving buffer
types used in the Pisces field system. Buffers are the backend data containers
for field values, and may be implemented using NumPy, `unyt`, HDF5 datasets,
or any other array-like structure that adheres to the `BufferBase` interface.

The registry system ensures that field constructors can automatically determine
how to wrap arbitrary user-provided input (such as `np.ndarray`, `list`, `unyt_array`)
into the correct buffer backend without requiring explicit user configuration.
"""
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

if TYPE_CHECKING:
    from pymetric.fields.buffers.base import BufferBase


# ========================================== #
# Define the buffer registry class           #
# ========================================== #
# In principle, users / developers could define multiples of
# these if (for some reason), they wanted to preempt the standard
# registration ordering etc.
class BufferRegistry:
    """
    Registry for mapping array-like objects to Pisces buffer backends.

    This class maintains an ordered list of buffer types and handles dispatching
    to the appropriate one when resolving unknown array-like input.

    Registries can be instantiated per subsystem, or a default global instance can be used.
    """

    def __init__(self):
        self._registry: List[Type["BufferBase"]] = []

    def __getitem__(self, key: Union[str, Type["BufferBase"]]) -> Type["BufferBase"]:
        """
        Allow dictionary-style access to buffer classes by name or type.

        Parameters
        ----------
        key : str or type
            Name of the buffer class, or the type itself.

        Returns
        -------
        BufferBase subclass
            The matched buffer class.

        Raises
        ------
        KeyError
            If the buffer is not registered.
        """
        if isinstance(key, str):
            for cls in self._registry:
                if cls.__name__ == key:
                    return cls
            raise KeyError(f"Buffer class '{key}' not found in registry.")
        elif isinstance(key, type) and issubclass(key, BufferBase):
            if key in self._registry:
                return key
            raise KeyError(f"Buffer class '{key.__name__}' is not registered.")
        else:
            raise TypeError(f"Invalid key type {type(key)} for buffer registry lookup.")

    def __contains__(self, key: Union[str, Type["BufferBase"]]) -> bool:
        """
        Support `in` checks on registered buffer classes.

        Parameters
        ----------
        key : str or type
            Class name or BufferBase subclass.

        Returns
        -------
        bool
            True if key is registered, False otherwise.
        """
        try:
            _ = self[key]
            return True
        except (KeyError, TypeError):
            return False

    def register(self, buffer_cls: Type["BufferBase"], prepend: bool = False) -> None:
        """
        Register a new buffer type.

        Parameters
        ----------
        buffer_cls : Type[BufferBase]
            The buffer class to register.
        prepend : bool, default False
            If True, the class is inserted at the front of the list (overrides priority).
            Otherwise, it is inserted and re-sorted by resolution priority.

        Notes
        -----
        Buffer classes should define an integer `__resolution_priority__` attribute.
        Higher values mean higher precedence. Classes with equal priority retain
        insertion order unless `prepend=True` is used.
        """
        if prepend:
            self._registry.insert(0, buffer_cls)
        else:
            self._registry.append(buffer_cls)
            self._registry.sort(
                key=lambda cls: getattr(cls, "__resolution_priority__", 0),
                reverse=True,  # Higher priority first
            )

    def get_buffer_class(self, array_like: Any) -> Type["BufferBase"]:
        """
        Identify the appropriate buffer class for a given array-like object.

        This method searches through the registry to find a buffer class that can
        handle the input object. It does not instantiate the buffer or coerce the
        input—only returns the class that would be used to handle it.

        This is useful for introspection, dispatch logic, or diagnostics when you
        need to know which buffer backend will be selected without actually creating
        the buffer.

        Parameters
        ----------
        array_like : Any
            An object to test against registered buffer classes.

        Returns
        -------
        Type[BufferBase]
            The buffer class that can handle the input.

        Raises
        ------
        TypeError
            If no registered buffer class can handle the input.
        """
        for buffer_cls in self._registry:
            if buffer_cls.can_handle(array_like):
                return buffer_cls
        raise TypeError(
            f"No registered buffer class can handle input of type {type(array_like)}."
        )

    def resolve(self, array_like: Any, *args, **kwargs) -> "BufferBase":
        """
        Resolve and wrap an array-like object using an appropriate buffer class.

        This method attempts to convert an input array-like object (such as a NumPy
        array, list, tuple, or unyt array) into a concrete instance of a buffer class
        registered in this registry.

        If `array_like` is already an instance of a registered buffer class, it is
        returned unchanged.

        If it is not a buffer, the method attempts to find a registered class whose
        `can_handle()` method returns True for the input, and calls that class's
        `from_array()` constructor to wrap it.

        Parameters
        ----------
        array_like : Any
            The object to resolve into a buffer. May be a raw array-like object or
            an existing buffer instance.
        *args :
            Positional arguments forwarded to the resolved class's `from_array()` method.
        **kwargs :
            Keyword arguments forwarded to the resolved class's `from_array()` method.

        Returns
        -------
        BufferBase
            An instance of a registered buffer class wrapping the input.

        Raises
        ------
        TypeError
            If no registered buffer class can handle the input type.
        """
        # Check if the input array_like is already a buffer
        # in the registry. If not, we strip it.
        if type(array_like) in self._registry:
            return array_like

        for buffer_cls in self._registry:
            if hasattr(buffer_cls, "can_handle") and buffer_cls.can_handle(array_like):
                return buffer_cls.from_array(array_like, *args, **kwargs)

        raise TypeError(
            f"No compatible buffer type found for object: {type(array_like)}"
        )

    def clear(self):
        """Clear all registered buffer types."""
        self._registry.clear()

    def list_registered_types(self) -> List[str]:
        """Return a list of the names of registered buffer classes."""
        return [cls.__name__ for cls in self._registry]


# ========================================== #
# Create the default buffer registry         #
# ========================================== #
# This is the buffer that is automatically registered by
# the meta class in base.
__DEFAULT_BUFFER_REGISTRY__ = BufferRegistry()
"""
Default global buffer registry used by Pisces fields.

This registry is populated automatically by the `_BufferMeta` metaclass
during buffer class definition. It serves as the canonical dispatch table
for resolving raw array-like objects into wrapped `BufferBase` subclasses.

Unless explicitly overridden, all `GenericField` and `TensorField` objects
use this registry to determine how to handle their input buffers.

See Also
--------
_BufferRegistry
_BufferMeta
BufferBase
"""


def resolve_buffer_class(
    buffer_class: Optional[Union[str, Type["BufferBase"]]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    default: Optional[Type["BufferBase"]] = None,
) -> Type["BufferBase"]:
    """
    Resolve a buffer class from a string, type, or fallback.

    Parameters
    ----------
    buffer_class : str or BufferBase subclass, optional
        The buffer class to resolve.
    buffer_registry : BufferRegistry, optional
        The registry to use for resolving string names. Defaults to the global registry.
    default : BufferBase subclass, optional
        Fallback to use if `buffer_class` is None.

    Returns
    -------
    :py:class:`~pymetric.fields.buffer.base.BufferBase`
        The resolved buffer class.

    Raises
    ------
    ValueError
        If the string name is not found in the registry, or the type is invalid.
    """
    from pymetric.fields.buffers.base import BufferBase

    if buffer_registry is None:
        buffer_registry = __DEFAULT_BUFFER_REGISTRY__

    if buffer_class is None:
        if default is not None:
            return default
        raise ValueError(
            "No buffer_class provided and no default fallback was specified."
        )

    if isinstance(buffer_class, str):
        try:
            return buffer_registry[buffer_class]
        except KeyError:
            raise ValueError(f"Unknown buffer class name '{buffer_class}' in registry.")

    if isinstance(buffer_class, type) and issubclass(buffer_class, BufferBase):
        return buffer_class

    raise ValueError(
        f"Invalid buffer_class argument: must be string, BufferBase subclass, or None. Got {type(buffer_class)}."
    )
