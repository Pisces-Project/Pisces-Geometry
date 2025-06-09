"""
Buffer resolution and registration system for PyMetric field buffers.

This module defines the infrastructure for managing and dispatching buffer types used in
the PyMetric field system. Buffers are backend-specific containers that store field data,
and may wrap a variety of storage formats including:

- In-memory arrays (e.g., :class:`numpy.ndarray`)
- Disk-backed arrays (e.g., :class:`h5py.Dataset`)

To support generic and extensible workflows, PyMetric uses a registry-based system to resolve
raw array-like inputs into appropriate subclasses of :class:`~pymetric.fields.buffers.base.BufferBase`.
This allows field constructors and APIs to accept flexible input types without requiring the user
to explicitly select or instantiate a specific buffer backend.

Key components of this module include:

- :class:`BufferRegistry`:
    Maintains an ordered list of buffer classes and provides resolution,
    introspection, registration, and deregistration APIs.
- :attr:`__DEFAULT_BUFFER_REGISTRY__`:
    The canonical global registry used by field constructors and the `_BufferMeta` metaclass.
- :func:`resolve_buffer_class`:
    Utility for resolving a buffer class from a name, type, or fallback—used internally
    by configuration systems and field factories.

Buffer classes must declare:

- ``__buffer_resolution_compatible_classes__``: A tuple of types they can wrap.
- ``__buffer_resolution_priority__``: An integer priority used to determine resolution order.

Resolution works by inspecting registered classes in descending priority order and selecting the
first that supports the input type. Once matched, the buffer's `from_array()` method is used
to wrap the input.
"""
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

if TYPE_CHECKING:
    from pymetric.fields.buffers.base import BufferBase


class BufferRegistry:
    """
    Registry for resolving array-like objects into PyMetric-compatible buffer backends.

    The `BufferRegistry` class maintains an ordered list of registered buffer types,
    each of which is a subclass of :class:`~fields.buffers.base.BufferBase`.
    This registry enables PyMetric to convert raw array-like inputs—such as `numpy.ndarray`,
    `unyt_array`, lists, or HDF5 datasets—into the appropriate wrapped buffer implementation.

    Registries can be initialized with a list of buffer classes or populated incrementally
    via the :meth:`register` or :meth:`__iadd__` methods. Resolution is driven by
    class-level metadata:

    - ``__buffer_resolution_compatible_classes__`` defines the supported types a buffer can wrap.
    - ``__buffer_resolution_priority__`` defines sorting precedence. Higher values are matched first.

    Core features of the registry include:

    - Dynamic resolution of unknown objects using :meth:`resolve`
    - Type introspection using :meth:`get_buffer_class`, :meth:`list_registered_types`, and :meth:`as_dict`
    - Mutation control using :meth:`register`, :meth:`unregister`, and :meth:`clear`

    The default global registry (:data:`__DEFAULT_BUFFER_REGISTRY__`) is populated automatically
    using the buffer metaclass :class:`~fields.buffers.base._BufferMeta`.

    Examples
    --------
    >>> registry = BufferRegistry(initial=[ArrayBuffer, HDF5Buffer])
    >>> buffer = registry.resolve(np.zeros((64, 64)))
    """

    def __init__(self, initial: Optional[List[Type["BufferBase"]]] = None):
        """
        Initialize the buffer registry.

        Parameters
        ----------
        initial : list of BufferBase subclasses, optional
            A list of buffer classes to register at creation. These will be inserted
            in the order given and then sorted by resolution priority.
        """
        self._registry: List[Type["BufferBase"]] = []

        if initial is not None:
            for cls in initial:
                self.register(cls)

    # ===================================== #
    # Dunder Methods                        #
    # ===================================== #
    # These methods implement the getting and setting behavior
    # for the buffer registry.
    def __getitem__(self, key: Union[str, Type["BufferBase"]]) -> Type["BufferBase"]:
        # Handle the case where we have a string key provided.
        # This will look for a match to the class name.
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
        try:
            _ = self[key]
            return True
        except (KeyError, TypeError):
            return False

    def __bool__(self) -> bool:
        """Return True if the registry contains any entries."""
        return bool(self._registry)

    def __iter__(self):
        """Iterate over registered buffer classes."""
        return iter(self._registry)

    def __len__(self) -> int:
        """Return the number of registered buffer types."""
        return len(self._registry)

    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the registry.

        Returns
        -------
        str
            A full list of registered buffer types with priorities.
        """
        lines = ["<BufferRegistry:"]
        if not self._registry:
            lines.append("  (empty)")
        else:
            for cls in self._registry:
                prio = getattr(cls, "__buffer_resolution_priority__", 0)
                lines.append(f"  - {cls.__name__} (priority={prio})")
        lines.append(">")
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Return a human-readable string summarizing the registry contents.

        Returns
        -------
        str
            A description of the registered buffer types and their resolution priorities.
        """
        if not self._registry:
            return "<BufferRegistry (empty)>"
        return (
            "<BufferRegistry: "
            + ", ".join(
                f"{cls.__name__}[p={getattr(cls, '__buffer_resolution_priority__', 0)}]"
                for cls in self._registry
            )
            + ">"
        )

    def __iadd__(self, buffer_cls: Type["BufferBase"]) -> "BufferRegistry":
        """
        Support in-place addition of buffer classes via `+=`.

        Parameters
        ----------
        buffer_cls : Type[BufferBase]
            A buffer class to register.

        Returns
        -------
        BufferRegistry
            The updated registry instance.
        """
        self.register(buffer_cls)
        return self

    # ===================================== #
    # Core Methods                          #
    # ===================================== #
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
                key=lambda cls: getattr(cls, "__buffer_resolution_priority__", 0),
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
        Type
            The buffer class that can handle the input.

        Raises
        ------
        TypeError
            If no registered buffer class can handle the input.
        """
        for buffer_cls in self._registry:
            if type(array_like) in buffer_cls.list_compatible_classes():
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

        Parameters
        ----------
        array_like : Any
            The object to resolve into a buffer.
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
        from pymetric.fields.buffers.base import BufferBase

        # Short-circuit if already a valid BufferBase instance
        if isinstance(array_like, BufferBase):
            return array_like

        # Use compatible type declarations for static dispatch
        for buffer_cls in self._registry:
            compatible = getattr(
                buffer_cls, "__buffer_resolution_compatible_classes__", None
            )
            if compatible is None:
                continue
            if isinstance(array_like, tuple(compatible)):
                return buffer_cls.from_array(array_like, *args, **kwargs)

        raise TypeError(
            f"No compatible buffer type found for object of type {type(array_like).__name__}."
        )

    def clear(self):
        """Clear all registered buffer types."""
        self._registry.clear()

    def list_registered_types(self) -> List[str]:
        """Return a list of the names of registered buffer classes."""
        return [cls.__name__ for cls in self._registry]

    def unregister(self, buffer_cls: Union[str, Type["BufferBase"]]) -> None:
        """
        Unregister a buffer class from the registry.

        Parameters
        ----------
        buffer_cls : str or Type[BufferBase]
            The buffer class to remove, specified by class or class name.

        Raises
        ------
        KeyError
            If the class is not registered.
        """
        cls_to_remove = self[buffer_cls]  # uses __getitem__ for resolution
        self._registry.remove(cls_to_remove)

    def is_registered(self, buffer_cls: Union[str, Type["BufferBase"]]) -> bool:
        """
        Check whether a buffer class is currently registered.

        Parameters
        ----------
        buffer_cls : str or Type[BufferBase]

        Returns
        -------
        bool
            True if registered, False otherwise.
        """
        return buffer_cls in self

    def sort(self) -> None:
        """
        Re-sort the registry in-place based on resolution priority.

        Highest-priority buffer classes will be listed first.
        """
        self._registry.sort(
            key=lambda cls: getattr(cls, "__buffer_resolution_priority__", 0),
            reverse=True,
        )

    def as_dict(self) -> dict:
        """
        Return a dictionary of registered buffer classes keyed by class name.

        Returns
        -------
        dict
            A mapping from class name to class object.
        """
        return {cls.__name__: cls for cls in self._registry}

    def get_registered_classes(self) -> List[Type["BufferBase"]]:
        """
        Return a list of all registered buffer class objects.

        Returns
        -------
        list of Type[BufferBase]
        """
        return list(self._registry)


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

"""


def resolve_buffer_class(
    buffer_class: Optional[Union[str, Type["BufferBase"]]] = None,
    buffer_registry: Optional["BufferRegistry"] = None,
    default: Optional[Type["BufferBase"]] = None,
) -> Type["BufferBase"]:
    """
    Resolve a buffer class from a string, type, or fallback.

    This utility function is used to normalize user-provided buffer class
    arguments into a concrete subclass of :class:`~pymetric.fields.buffers.base.BufferBase`.
    It supports:

    - Resolving from a string name using a provided or default registry
    - Accepting an already valid buffer class
    - Falling back to a default if `buffer_class` is not specified

    Parameters
    ----------
    buffer_class : str or type, optional
        The buffer class to resolve. May be a string (name of a registered class)
        or an actual subclass of :class:`BufferBase`. If None, the `default`
        is used.
    buffer_registry : BufferRegistry, optional
        The registry to use for string-based lookup. If not specified, the
        global :data:`__DEFAULT_BUFFER_REGISTRY__` is used.
    default : type, optional
        Fallback buffer class to use if `buffer_class` is None.

    Returns
    -------
    Type
        The resolved buffer class.

    Raises
    ------
    ValueError
        If `buffer_class` is a string not found in the registry,
        or an invalid type is provided.

    Examples
    --------
    >>> resolve_buffer_class("ArrayBuffer")
    <class 'ArrayBuffer'>

    >>> resolve_buffer_class(ArrayBuffer)
    <class 'ArrayBuffer'>

    >>> resolve_buffer_class(None, default=ArrayBuffer)
    <class 'ArrayBuffer'>
    """
    from pymetric.fields.buffers.base import BufferBase

    # Use the global registry unless overridden
    if buffer_registry is None:
        buffer_registry = __DEFAULT_BUFFER_REGISTRY__

    # Use the default fallback if no class is specified
    if buffer_class is None:
        if default is not None:
            return default
        raise ValueError(
            "No buffer_class provided and no default fallback was specified."
        )

    # Handle string name: look it up in the registry
    if isinstance(buffer_class, str):
        try:
            return buffer_registry[buffer_class]
        except KeyError:
            raise ValueError(f"Unknown buffer class name '{buffer_class}' in registry.")

    # Handle direct class input: validate
    if isinstance(buffer_class, type) and issubclass(buffer_class, BufferBase):
        return buffer_class

    # Invalid input
    raise ValueError(
        f"Invalid buffer_class argument: expected a string name, a BufferBase subclass, or None. "
        f"Got object of type {type(buffer_class).__name__}."
    )
