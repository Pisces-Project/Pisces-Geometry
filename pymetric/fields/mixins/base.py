"""
Core mixin classes for the central
field classes.
"""
from typing import TYPE_CHECKING, Any, Generic, Iterable, Optional, Type, TypeVar

from pymetric.fields.components import FieldComponent
from pymetric.fields.utils.utilities import validate_rank_signature

# =============================== #
# Configure Typing                #
# =============================== #
# This section of the module supports typing
# hints and protocols for static type checking.
if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pymetric.fields.utils._typing import (
        ComponentDictionary,
        ComponentIndex,
        SignatureInput,
    )

    from ._typing import (
        _SupportsDFieldCore,
        _SupportsDTFieldCore,
        _SupportsFieldCore,
        _SupportsSFieldCore,
    )

_SupFieldCore = TypeVar("_SupFieldCore", bound="_SupportsFieldCore")
_SupSFieldCore = TypeVar("_SupSFieldCore", bound="_SupportsSFieldCore")
_SupDFieldCore = TypeVar("_SupDFieldCore", bound="_SupportsDFieldCore")
_SupDTFieldCore = TypeVar("_SupDTFieldCore", bound="_SupportsDTFieldCore")


# =============================== #
# Mixin Classes                   #
# =============================== #
class FieldCoreMixin(Generic[_SupFieldCore]):
    """
    Core mixin methods for the :class:`~fields.base.Field` class.
    """

    # --- Utility Methods --- #
    # These are simple utility methods for use
    # when performing various checks.
    def validate_components(self, components: "ComponentDictionary") -> None:
        """
        Validate all components in a dictionary for consistency with the field.

        Parameters
        ----------
        components : dict
            A dictionary mapping component indices to FieldComponent instances.

        Raises
        ------
        TypeError, ValueError
            If any component is invalid or inconsistent with the field’s grid.
        """
        for index, component in components.items():
            self.validate_component(index, component)

    def validate_component(
        self: _SupFieldCore, index: "ComponentIndex", component: Any
    ) -> None:
        """
        Validate a single field component and its index.

        Checks that:

        - the component is a FieldComponent,
        - the component’s grid matches the field’s grid,
        - the index is of the correct type (int or tuple of ints).

        Parameters
        ----------
        index : int or tuple of int
            The index key for the component in the field.
        component : Any
            The component object to validate.

        Raises
        ------
        TypeError
            If the component is not a FieldComponent or index is not valid.
        ValueError
            If the component uses a different grid or index format is incorrect.
        """
        if not isinstance(component, FieldComponent):
            raise TypeError(
                f"Component {index!r} is not a FieldComponent (got {type(component).__name__})."
            )

        if component.grid is not self.__grid__:
            raise ValueError(
                f"Component {index!r} uses a different grid (got {component.grid!r}, "
                f"expected {self.__grid__!r})."
            )

        # Ensure the index is valid
        if isinstance(index, tuple):
            if not all(isinstance(i, int) for i in index):
                raise TypeError(
                    f"Component index {index!r} must be a tuple of integers, "
                    f"but got invalid entries: {index}."
                )
        elif not isinstance(index, int):
            raise TypeError(
                f"Component index {index!r} must be an int or tuple of ints."
            )


class SFieldCoreMixin(Generic[_SupSFieldCore]):
    """
    Core mixin methods for the :class:`~fields.base.SparseField` class.
    """

    # ========================== #
    # Utility Methods: Access    #
    # ========================== #
    # Simple methods to allow various access patterns
    # for elements in the SparseFields.
    def items(self: _SupSFieldCore) -> Iterable:
        """
        Return a view of (index, component) pairs in the field.

        This is equivalent to calling `.items()` on the underlying
        component dictionary.

        Returns
        -------
        ItemsView
            A view over the field’s component index-value pairs.
        """
        return self.__components__.items()

    def keys(self: _SupSFieldCore) -> Iterable:
        """
        Return a view of the component indices in the field.

        This is equivalent to calling `.keys()` on the underlying
        component dictionary.

        Returns
        -------
        KeysView
            A view over the field’s component indices.
        """
        return self.__components__.keys()

    def values(self: _SupSFieldCore) -> Iterable:
        """
        Return a view of the field components.

        This is equivalent to calling `.values()` on the underlying
        component dictionary.

        Returns
        -------
        ValuesView
            A view over the field’s components.
        """
        return self.__components__.values()


class DFieldCoreMixin(Generic[_SupDFieldCore]):
    """
    Core mixin methods for the :class:`~fields.base.DenseField` class.
    """

    # ============================= #
    # Data Access Methods           #
    # ============================= #
    # These methods provide entry points for getting data from
    # FieldComponents.
    def as_array(self: _SupDFieldCore):
        """
        Return the underlying buffer as a NumPy array.

        This method extracts the field's raw data buffer and returns it
        as a standard NumPy array. This strips any associated units or
        metadata.

        Returns
        -------
        np.ndarray
            The raw array representing the field's data.
        """
        return self.__component__.buffer.as_array()

    # ============================= #
    # Generator Methods             #
    # ============================= #
    # These methods provide entry points for creating
    # FieldComponents.
    @classmethod
    def _wrap_comp_from_op(
        cls: Type[_SupDFieldCore], operation: str, *args, **kwargs
    ) -> "_SupDFieldCore":
        """
        Apply an operation from `FieldComponent` and wrap the result in the current class.

        This method takes the name of an operation (as a string) or a direct reference to
        a method defined in the `FieldComponent` class, applies it to the provided arguments,
        and wraps the resulting `FieldComponent` instance in a new instance of `cls`.

        Parameters
        ----------
        operation : str or callable
            The name of a method defined in `FieldComponent` or a callable method itself.
        *args : tuple
            Positional arguments to pass to the operation.
        **kwargs : dict
            Keyword arguments to pass to the operation.

        Returns
        -------
        cls
            A new instance of the calling class (`cls`), wrapping the result of the operation.

        Raises
        ------
        ValueError
            If the operation is not a valid attribute of `FieldComponent`.

        Examples
        --------
        >>> MyFieldClass._wrap_comp_from_op("some_operation", arg1, arg2, kwarg1=value)
        """
        try:
            operation_fn = getattr(FieldComponent, operation)
        except AttributeError:
            raise ValueError(
                f"Operation {operation!r} is not a valid operation of FieldComponent"
            )

        c = operation_fn(*args, **kwargs)
        return cls(c.grid, c)

    # noinspection PyIncorrectDocstring
    @classmethod
    def from_function(cls: Type[_SupDFieldCore], *args, **kwargs) -> _SupDFieldCore:
        """
        Construct a :class:`~fields.components.FieldComponent` by evaluating a function on the grid.

        This method evaluates a coordinate-dependent function `func` over a physical coordinate mesh
        generated from `grid` along the specified `axes`. It is useful for initializing field data from
        analytic expressions.

        Parameters
        ----------
        func : callable
            A function that takes coordinate arrays (one per axis) and returns an array of values with
            shape `(*grid_shape, *result_shape)`, matching the evaluated field.
        grid : ~grids.base.GridBase
            The grid over which to evaluate the function.
        axes : list of str
            Coordinate axes along which the field is defined.
        *args :
            Additional arguments forwarded to the buffer constructor (e.g., dtype).
        result_shape : tuple of int, optional
            Shape of trailing element-wise structure (e.g., for vectors/tensors). Defaults to scalar `()`.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer backend used to hold data.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry used to resolve buffer class strings.
        buffer_args:
            Additional positional arguments forwarded to the buffer constructor (e.g., `dtype`).
        buffer_kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `units` if supported).
        *args, **kwargs :
            Additional arguments forwarded to the function `func`.

        Returns
        -------
        FieldComponent
            A new field component with data populated from `func`.

        Raises
        ------
        ValueError
            If the output shape of `func` does not match the expected field shape.
        """
        return cls._wrap_comp_from_op("from_function", *args, **kwargs)

    # noinspection PyIncorrectDocstring
    @classmethod
    def from_array(cls: Type[_SupDFieldCore], *args, **kwargs) -> _SupDFieldCore:
        """
        Construct a :class:`~fields.components.FieldComponent` from an existing array-like object.

        This method creates a :class:`~fields.components.FieldComponent` by wrapping a NumPy array, unyt array,
        or similar backend-supported array in a compatible buffer. The shape of the input
        array must match the combined spatial and element-wise shape implied by `grid` and `axes`.

        Parameters
        ----------
        array_like : array-like
            The array to wrap. This can be a :class:`numpy.ndarray`, :class:`unyt.unyt_array`, or other compatible type.
        grid : ~grids.base.GridBase
            The grid over which the field is defined.
        axes : list of str
            The spatial axes of the coordinate system over which the array is defined.
            The shape of the array must begin with the grid shape corresponding to these axes.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for wrapping the data. This can be a class or string identifier.
            If not provided, defaults to `ArrayBuffer`.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Custom registry to use for resolving string buffer types.
        buffer_args:
            Additional positional arguments forwarded to the buffer constructor (e.g., `dtype`).
        buffer_kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `units` if supported).

        Returns
        -------
        FieldComponent
            A new component wrapping the given array.

        Raises
        ------
        ValueError
            If the input array shape is incompatible with the grid and axes.
        """
        return cls._wrap_comp_from_op("from_array", *args, **kwargs)

    # noinspection PyIncorrectDocstring
    @classmethod
    def zeros(cls: Type[_SupDFieldCore], *args, **kwargs) -> _SupDFieldCore:
        """
        Create a dense field filled with zeros.

        This is a convenience constructor that builds a zero-initialized :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        element_shape : tuple of int, optional
            Shape of the element-wise data structure (e.g., vector or tensor dimensions). This
            does **not** include the spatial shape, which is fixed by the grid. The resulting
            buffer will have an overall shape of ``(*spatial_shape, *element_shape)``.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.zeros()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        buffer_args:
            Additional positional arguments forwarded to the buffer constructor (e.g., `dtype`).
        buffer_kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `units` if supported).

        Returns
        -------
        ~fields.base.DenseField
            A new field object filled with zeros and defined over the specified grid and axes.
        """
        # Start with generation of the component so that
        # it can be wrapped.
        return cls._wrap_comp_from_op("zeros", *args, **kwargs)

    # noinspection PyIncorrectDocstring
    @classmethod
    def ones(cls: Type[_SupDFieldCore], *args, **kwargs) -> _SupDFieldCore:
        """
        Create a dense field filled with ones.

        This is a convenience constructor that builds a ones-initialized :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        element_shape : tuple of int, optional
            Shape of the element-wise data structure (e.g., vector or tensor dimensions). This
            does **not** include the spatial shape, which is fixed by the grid. The resulting
            buffer will have an overall shape of ``(*spatial_shape, *element_shape)``.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.ones()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        buffer_args:
            Additional positional arguments forwarded to the buffer constructor (e.g., `dtype`).
        buffer_kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `units` if supported).

        Returns
        -------
        ~fields.base.DenseField
            A new field object filled with ones and defined over the specified grid and axes.
        """
        # Start with generation of the component so that
        # it can be wrapped.
        return cls._wrap_comp_from_op("ones", *args, **kwargs)

    # noinspection PyIncorrectDocstring
    @classmethod
    def empty(cls: Type[_SupDFieldCore], *args, **kwargs) -> _SupDFieldCore:
        """
        Create a dense field without initializing any data.

        This is a convenience constructor that builds an empty :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        element_shape : tuple of int, optional
            Shape of the element-wise data structure (e.g., vector or tensor dimensions). This
            does **not** include the spatial shape, which is fixed by the grid. The resulting
            buffer will have an overall shape of ``(*spatial_shape, *element_shape)``.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.ones()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        buffer_args:
            Additional positional arguments forwarded to the buffer constructor (e.g., `dtype`).
        buffer_kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `units` if supported).

        Returns
        -------
        ~fields.base.DenseField
            A new field object filled with ones and defined over the specified grid and axes.
        """
        # Start with generation of the component so that
        # it can be wrapped.
        return cls._wrap_comp_from_op("empty", *args, **kwargs)

    # noinspection PyIncorrectDocstring
    @classmethod
    def full(cls: Type[_SupDFieldCore], *args, **kwargs) -> _SupDFieldCore:
        """
        Create a dense field filled with a fill value.

        This is a convenience constructor that builds a :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        element_shape : tuple of int, optional
            Shape of the element-wise data structure (e.g., vector or tensor dimensions). This
            does **not** include the spatial shape, which is fixed by the grid. The resulting
            buffer will have an overall shape of ``(*spatial_shape, *element_shape)``.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.full()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        buffer_args:
            Additional positional arguments forwarded to the buffer constructor (e.g., `dtype`).
        buffer_kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `units` if supported).

        Returns
        -------
        ~fields.base.DenseField
            A new field object filled with `fill_value` and defined over the specified grid and axes.
        """
        # Start with generation of the component so that
        # it can be wrapped.
        return cls._wrap_comp_from_op("full", *args, **kwargs)

    @classmethod
    def empty_like(
        cls: Type[_SupDFieldCore], other: Type[_SupDFieldCore], *args, **kwargs
    ) -> _SupDFieldCore:
        """
        Create an empty field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`zeros` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and zero-initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.empty(grid, axes, *args, element_shape=shape, **kwargs)

    @classmethod
    def zeros_like(
        cls: Type[_SupDFieldCore], other: Type[_SupDFieldCore], *args, **kwargs
    ) -> _SupDFieldCore:
        """
        Create a zero-filled field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`zeros` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and zero-initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.zeros(grid, axes, *args, element_shape=shape, **kwargs)

    @classmethod
    def ones_like(
        cls: Type[_SupDFieldCore], other: Type[_SupDFieldCore], *args, **kwargs
    ) -> _SupDFieldCore:
        """
        Create a one-filled field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`ones` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and one-initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.ones(grid, axes, *args, element_shape=shape, **kwargs)

    @classmethod
    def full_like(
        cls: Type[_SupDFieldCore], other: Type[_SupDFieldCore], *args, **kwargs
    ) -> _SupDFieldCore:
        """
        Create a full-valued field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`full` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and filled with a constant value.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.full(grid, axes, *args, element_shape=shape, **kwargs)


class DTensorFieldCoreMixin(DFieldCoreMixin, Generic[_SupDTFieldCore]):
    """
    Core mixin methods for the :class:`~fields.tensors.DenseTensorField` class.
    """

    # ============================= #
    # Generator Methods             #
    # ============================= #
    # These methods provide entry points for creating
    # DenseTensorFields.
    # noinspection PyIncorrectDocstring
    @classmethod
    def from_function(
        cls: Type[_SupDTFieldCore],
        *args,
        signature: Optional["SignatureInput"] = None,
        **kwargs,
    ) -> _SupDTFieldCore:
        """
        Construct a :class:`~fields.components.FieldComponent` by evaluating a function on the grid.

        This method evaluates a coordinate-dependent function `func` over a physical coordinate mesh
        generated from `grid` along the specified `axes`. It is useful for initializing field data from
        analytic expressions.

        Parameters
        ----------
        func : callable
            A function that takes coordinate arrays (one per axis) and returns an array of values with
            shape `(*grid_shape, *result_shape)`, matching the evaluated field.
        grid : ~grids.base.GridBase
            The grid over which to evaluate the function.
        axes : list of str
            Coordinate axes along which the field is defined.
        *args :
            Additional arguments forwarded to the buffer constructor (e.g., dtype).
        signature : int or list of int, optional
            The signature of the tensor field being generated. This should be a sequence of
            `1` and `-1` with `1` marking a contravariant index and `-1` a covariant index. The
            length must match that of `rank`. If not specified, `signature` defaults to a fully
            contravariant form.
        result_shape : tuple of int, optional
            Shape of trailing element-wise structure (e.g., for vectors/tensors). Defaults to scalar `()`.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer backend used to hold data.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry used to resolve buffer class strings.
        buffer_kwargs : dict, optional
            Extra keyword arguments passed to the buffer constructor (e.g., units).
        **kwargs :
            Additional keyword arguments forwarded to the function `func`.

        Returns
        -------
        FieldComponent
            A new field component with data populated from `func`.

        Raises
        ------
        ValueError
            If the output shape of `func` does not match the expected field shape.
        """
        # create the component vis-a-vis the lower level
        c = FieldComponent.from_function(*args, **kwargs)
        return cls(c.grid, c, signature=signature)

    # noinspection PyIncorrectDocstring
    @classmethod
    def from_array(
        cls: Type[_SupDTFieldCore],
        *args,
        signature: Optional["SignatureInput"] = None,
        **kwargs,
    ) -> _SupDTFieldCore:
        """
        Construct a :class:`~fields.components.FieldComponent` from an existing array-like object.

        This method creates a :class:`~fields.components.FieldComponent` by wrapping a NumPy array, unyt array,
        or similar backend-supported array in a compatible buffer. The shape of the input
        array must match the combined spatial and element-wise shape implied by `grid` and `axes`.

        Parameters
        ----------
        array_like : array-like
            The array to wrap. This can be a :class:`numpy.ndarray`, :class:`unyt.unyt_array`, or other compatible type.
        grid : ~grids.base.GridBase
            The grid over which the field is defined.
        axes : list of str
            The spatial axes of the coordinate system over which the array is defined.
            The shape of the array must begin with the grid shape corresponding to these axes.
        signature : int or list of int, optional
            The signature of the tensor field being generated. This should be a sequence of
            `1` and `-1` with `1` marking a contravariant index and `-1` a covariant index. The
            length must match that of `rank`. If not specified, `signature` defaults to a fully
            contravariant form.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for wrapping the data. This can be a class or string identifier.
            If not provided, defaults to `ArrayBuffer`.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Custom registry to use for resolving string buffer types.
        **kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `units` if supported).

        Returns
        -------
        FieldComponent
            A new component wrapping the given array.

        Raises
        ------
        ValueError
            If the input array shape is incompatible with the grid and axes.
        """
        # create the component vis-a-vis the lower level
        c = FieldComponent.from_array(*args, **kwargs)
        return cls(c.grid, c, signature=signature)

    # noinspection PyIncorrectDocstring
    @classmethod
    def zeros(
        cls: Type[_SupDTFieldCore],
        *args,
        signature: Optional["SignatureInput"] = None,
        **kwargs,
    ) -> _SupDTFieldCore:
        """
        Create a dense field filled with zeros.

        This is a convenience constructor that builds a zero-initialized :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        rank: int, optional
            The rank of the tensor field being generated. This can be supplemented by
            specifying the `signature` below to determine the variance of each rank index.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        signature : int or list of int, optional
            The signature of the tensor field being generated. This should be a sequence of
            `1` and `-1` with `1` marking a contravariant index and `-1` a covariant index. The
            length must match that of `rank`. If not specified, `signature` defaults to a fully
            contravariant form.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.zeros()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        **kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `dtype`, `units`).

        Returns
        -------
        ~fields.base.DenseField
            A new field object filled with zeros and defined over the specified grid and axes.
        """
        # Validate the rank and the signature.
        grid, axes, rank, *xargs = args
        args = (grid, axes, *xargs)

        signature = validate_rank_signature(rank, signature=signature)

        # Fix the element shape
        element_shape = (grid.ndim,) * rank
        kwargs["element_shape"] = element_shape

        # create the component vis-a-vis the lower level
        c = FieldComponent.zeros(*args, **kwargs)

        return cls(grid, c, signature=signature)

    # noinspection PyIncorrectDocstring
    @classmethod
    def empty(
        cls: Type[_SupDTFieldCore],
        *args,
        signature: Optional["SignatureInput"] = None,
        **kwargs,
    ) -> _SupDTFieldCore:
        """
        Create a dense field which is not initialized.

        This is a convenience constructor that builds an uninitialized :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        rank: int, optional
            The rank of the tensor field being generated. This can be supplemented by
            specifying the `signature` below to determine the variance of each rank index.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        signature : int or list of int, optional
            The signature of the tensor field being generated. This should be a sequence of
            `1` and `-1` with `1` marking a contravariant index and `-1` a covariant index. The
            length must match that of `rank`. If not specified, `signature` defaults to a fully
            contravariant form.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.zeros()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        **kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `dtype`, `units`).

        Returns
        -------
        ~fields.base.DenseField
            A new field (unfilled) object and defined over the specified grid and axes.
        """
        # Validate the rank and the signature.
        grid, axes, rank, *xargs = args
        args = (grid, axes, *xargs)

        signature = validate_rank_signature(rank, signature=signature)

        # Fix the element shape
        element_shape = (grid.ndim,) * rank
        kwargs["element_shape"] = element_shape

        # create the component vis-a-vis the lower level
        c = FieldComponent.empty(*args, **kwargs)

        return cls(grid, c, signature=signature)

    # noinspection PyIncorrectDocstring
    @classmethod
    def ones(
        cls: Type[_SupDTFieldCore],
        *args,
        signature: Optional["SignatureInput"] = None,
        **kwargs,
    ) -> _SupDTFieldCore:
        """
        Create a dense field filled with ones.

        This is a convenience constructor that builds a ones-initialized :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        rank: int, optional
            The rank of the tensor field being generated. This can be supplemented by
            specifying the `signature` below to determine the variance of each rank index.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        signature : int or list of int, optional
            The signature of the tensor field being generated. This should be a sequence of
            `1` and `-1` with `1` marking a contravariant index and `-1` a covariant index. The
            length must match that of `rank`. If not specified, `signature` defaults to a fully
            contravariant form.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.ones()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        **kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `dtype`, `units`).

        Returns
        -------
        ~fields.base.DenseField
            A new field object filled with ones and defined over the specified grid and axes.
        """
        # Validate the rank and the signature.
        grid, axes, rank, *xargs = args
        args = (grid, axes, *xargs)

        signature = validate_rank_signature(rank, signature=signature)

        # Fix the element shape
        element_shape = (grid.ndim,) * rank
        kwargs["element_shape"] = element_shape

        # create the component vis-a-vis the lower level
        c = FieldComponent.ones(*args, **kwargs)

        return cls(grid, c, signature=signature)

    # noinspection PyIncorrectDocstring
    @classmethod
    def full(
        cls: Type[_SupDTFieldCore],
        *args,
        signature: Optional["SignatureInput"] = None,
        **kwargs,
    ) -> _SupDTFieldCore:
        """
        Create a dense field filled with a fill value.

        This is a convenience constructor that builds a :class:`~fields.components.FieldComponent`
        using the provided grid and axes, then wraps it in a  :class:`~fields.base.DenseField`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        rank: int, optional
            The rank of the tensor field being generated. This can be supplemented by
            specifying the `signature` below to determine the variance of each rank index.
        *args :
            Additional positional arguments forwarded to the buffer constructor. The specific
            available args will depend on ``buffer_class`` and ``buffer_registry``.
        signature : int or list of int, optional
            The signature of the tensor field being generated. This should be a sequence of
            `1` and `-1` with `1` marking a contravariant index and `-1` a covariant index. The
            length must match that of `rank`. If not specified, `signature` defaults to a fully
            contravariant form.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer class to use for holding the data. This may be specified as a string, in which
            case the ``buffer_registry`` is queried for a matching class or it may be a specific buffer class.

            The relevant ``*args`` and ``**kwargs`` arguments will be passed underlying
            buffer class's ``.full()`` method.
        buffer_registry : ~fields.buffer.registry.BufferRegistry, optional
            Custom registry to use for resolving buffer class strings. By default, the standard
            registry is used.
        **kwargs :
            Additional keyword arguments forwarded to the buffer constructor (e.g., `dtype`, `units`).

        Returns
        -------
        ~fields.base.DenseField
            A new field object filled with `fill_value` and defined over the specified grid and axes.
        """
        # Validate the rank and the signature.
        grid, axes, rank, *xargs = args
        args = (grid, axes, *xargs)

        signature = validate_rank_signature(rank, signature=signature)

        # Fix the element shape
        element_shape = (grid.ndim,) * rank
        kwargs["element_shape"] = element_shape

        # create the component vis-a-vis the lower level
        c = FieldComponent.full(*args, **kwargs)

        return cls(grid, c, signature=signature)

    @classmethod
    def zeros_like(
        cls: Type[_SupDTFieldCore], other: Type[_SupDTFieldCore], *args, **kwargs
    ) -> _SupDTFieldCore:
        """
        Create a zero-filled field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`zeros` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and zero-initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, rank, signature = (
            other.grid,
            other.axes,
            other.rank,
            other.signature,
        )
        return cls.zeros(grid, axes, rank, *args, signature=signature, **kwargs)

    @classmethod
    def empty_like(
        cls: Type[_SupDTFieldCore], other: Type[_SupDTFieldCore], *args, **kwargs
    ) -> _SupDTFieldCore:
        """
        Create an unfilled field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`zeros` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and no initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, rank, signature = (
            other.grid,
            other.axes,
            other.rank,
            other.signature,
        )
        return cls.empty(grid, axes, rank, *args, signature=signature, **kwargs)

    @classmethod
    def ones_like(
        cls: Type[_SupDTFieldCore], other: Type[_SupDTFieldCore], *args, **kwargs
    ) -> _SupDTFieldCore:
        """
        Create a one-filled field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`ones` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and one-initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, rank, signature = (
            other.grid,
            other.axes,
            other.rank,
            other.signature,
        )
        return cls.ones(grid, axes, rank, *args, signature=signature, **kwargs)

    @classmethod
    def full_like(
        cls: Type[_SupDTFieldCore], other: Type[_SupDTFieldCore], *args, **kwargs
    ) -> _SupDTFieldCore:
        """
        Create a full-valued field with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : ~fields.base.DenseField
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`full` constructor.

        Returns
        -------
        ~fields.base.DenseField
            A new field with the same layout as `other` and filled with a constant value.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, rank, signature = (
            other.grid,
            other.axes,
            other.rank,
            other.signature,
        )
        return cls.full(grid, axes, rank, *args, signature=signature, **kwargs)
