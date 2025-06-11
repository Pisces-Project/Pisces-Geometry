"""
Core mixins for :class:`~fields.components.FieldComponent`.
"""
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from pymetric.fields.buffers import ArrayBuffer, resolve_buffer_class

if TYPE_CHECKING:
    from pymetric.fields.buffers.base import BufferBase
    from pymetric.fields.buffers.registry import BufferRegistry

    # noinspection PyUnresolvedReferences
    from pymetric.fields.mixins._typing import (
        _SupportsFieldComponentCore,
        _SupportsFieldCore,
    )
    from pymetric.grids.base import GridBase
    from pymetric.grids.utils._typing import AxesInput

_SupFCCore = TypeVar("_SupFCCore", bound="_SupportsFieldComponentCore")
_SupFieldCore = TypeVar("_SupFieldCore", bound="_SupportsFieldCore")


class FieldComponentCoreMixin(Generic[_SupFCCore]):
    """
    Core mixin methods for the :py:class:`~fields.components.FieldComponent` class.

    This class handles various entry point methods, some utility methods, etc.
    """

    # ================================= #
    # Broadcasting Support              #
    # ================================= #
    # Broadcasting support allows for users to cast buffers
    # to be compatible with new shapes and new axes during
    # computations.
    #
    # Because a broadcast typically doesn't imply
    # continued compatibility, these generally return buffers.
    def broadcast_to_array_in_axes(
        self: _SupFCCore, axes: "AxesInput", **kwargs
    ) -> np.ndarray:
        """
        Return the field data as a NumPy array broadcasted to a specified set of axes.

        This reshapes the array so that its leading dimensions align with `axes`,
        inserting singleton dimensions as necessary to ensure compatibility.

        Parameters
        ----------
        axes : list of str
            Target axes to broadcast the array over.
        **kwargs :
            Additional keyword arguments passed to `np.broadcast_to`.

        Returns
        -------
        np.ndarray
            The buffer contents as a NumPy array aligned with the given axes.
        """
        axes = self.__grid__.standardize_axes(axes)
        return self.__grid__.broadcast_array_to_axes(
            self.as_array(), self.__axes__, axes, **kwargs
        )

    def broadcast_buffer_to_axes(
        self: _SupFCCore,
        axes: "AxesInput",
        *args,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        **kwargs,
    ) -> "BufferBase":
        """
        Return a new buffer instance with data broadcasted to a specified set of axes.

        This returns a wrapped buffer aligned with the new axes, suitable for downstream use
        in field construction or data manipulation.

        Parameters
        ----------
        axes : list of str
            Target axes to broadcast over.
        buffer_class : BufferBase, optional
            Optional override of the buffer class to use.
        buffer_registry : BufferRegistry, optional
            If using a string identifier for buffer_class, this registry is used to resolve it.
        *args, **kwargs :
            Additional arguments forwarded to the buffer constructor.

        Returns
        -------
        BufferBase
            A new buffer object whose shape and layout is compatible with `axes`.
        """
        axes = self.__grid__.standardize_axes(axes)

        # Broadcast underlying data to new axes
        data = self.__grid__.broadcast_array_to_axes(
            self.as_array(), self.__axes__, axes
        )

        # Use default or override buffer type
        buffer_class: "BufferBase" = resolve_buffer_class(
            buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
        )

        return buffer_class.from_array(data, *args, **kwargs)

    def expand_axes(
        self: _SupFCCore,
        axes: "AxesInput",
        *args,
        out: Optional[Union[_SupFCCore, "BufferBase"]] = None,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        **kwargs,
    ) -> _SupFCCore:
        """
        Broadcast and tile an existing :class:`FieldComponent` to an expanded set of axes.

        Parameters
        ----------
        axes : list of str
            The full set of axes to expand to. Must include all existing axes. All axes must
            be coordinate axes of the underlying coordinate system.
        out : FieldComponent or BufferBase, optional
            Optional target to write the expanded data into. If a :class:`FieldComponent`,
            its shape is checked. If a :class:`~fields.buffer.base.BufferBase`, only the buffer
            is reused.
        buffer_class : type, optional
            If no `out` is given, this buffer class is used to construct a new one.
        buffer_registry : BufferRegistry, optional
            Registry used to resolve string buffer types.
        *args, **kwargs :
            Forwarded to buffer constructor if a new one is created.

        Returns
        -------
        FieldComponent
            A new field component with fully realized axes.

        Raises
        ------
        ValueError
            If axes are not compatible or `out` is incompatible.
        """
        # Standardize the axes and broadcast the shape so
        # that we can use it when checking things.
        axes = self.grid.standardize_axes(axes)
        if not self.grid.__cs__.is_axes_subset(self.axes, axes):
            raise ValueError("Cannot expand axes to a smaller set of axes.")

        # Determine how to perform the expansion in the
        # context of the necessary output.
        expanded_data = self.grid.tile_array_to_axes(
            self.as_array(), self.__axes__, axes, include_ghosts=True
        )

        # Reconstruct the necessary downstream wrappers
        # around the produced data array.
        if out is None:
            # We will be creating a new field component from
            # this. We'll case the buffer_repr and then re-wrap.
            buffer_class: "BufferBase" = resolve_buffer_class(
                buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
            )
            buff = buffer_class.from_array(expanded_data, *args, **kwargs)
            return self.__class__(self.grid, buff, axes)
        else:
            # We have out= specified and would like to not
            # cast to a new buffer.
            if tuple(out.shape) != expanded_data.shape:
                raise ValueError(
                    f"Output buffer shape mismatch: expected {expanded_data.shape}, got {out.shape}."
                )

            out[...] = expanded_data
            return out

    def reduce_axes(
        self: _SupFCCore,
        axes: "AxesInput",
        indices: Sequence[int],
        *args,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        out: Optional[Union[_SupFCCore, "BufferBase"]] = None,
        **kwargs,
    ) -> _SupFCCore:
        """
        Reduce a component to a smaller set of axes by slicing.

        Parameters
        ----------
        axes : list of str
            Axes to index remove from this component. These must be active axes
            in the existing component.
        indices : list of int
            Indices to use for each axis to remove. This should be the same length
            as `axes`.
        buffer_class : type, optional
            Optional override for the new buffer class.
        buffer_registry : BufferRegistry, optional
            Registry to resolve string-based buffer classes.
        out : FieldComponent or BufferBase, optional
            Optional destination for the result. Can be a FieldComponent or buffer.
        *args, **kwargs :
            Forwarded to buffer constructor if a new buffer is created.

        Returns
        -------
        FieldComponent
            A new component with reduced dimensionality.
        """
        # Standardize axes and ensure they are part of self.axes. Then
        # validate the provided indices so that we know there are enough of
        # those.
        axes = self.grid.standardize_axes(axes)
        if not self.grid.__cs__.is_axes_subset(axes, self.axes):
            raise ValueError(
                f"Cannot reduce axes {axes} not present in component axes {self.axes}."
            )

        if len(axes) != len(indices):
            raise ValueError(f"Expected {len(axes)} indices, got {len(indices)}.")

        # Determine which axes will remain (in canonical order) and
        # which will leave. We always have fields in canonical order.
        kept_axes, _ = [ax for ax in self.axes if ax not in axes], axes
        if len(kept_axes) == 0:
            raise ValueError("Cannot reduce field to an empty set of axes.")

        # Slice down the array with the provided indices. Because we are
        # ensured canonical order, we simply index the axes.
        slc_map = {ax: idx for ax, idx in zip(axes, indices)}
        reduced_array = self.as_array()[
            tuple(slice(None) if ax not in slc_map else slc_map[ax] for ax in self.axes)
        ]

        # Now dump the reduced array to a valid output.
        if out is None:
            # Create new buffer and wrap it
            buffer_class: "BufferBase" = resolve_buffer_class(
                buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
            )
            buff = buffer_class.from_array(reduced_array, *args, **kwargs)
            return self.__class__(self.grid, buff, kept_axes)
        else:
            if tuple(out.shape) != reduced_array.shape:
                raise ValueError(
                    f"Output shape mismatch: expected {reduced_array.shape}, got {out.shape}."
                )

            out[...] = reduced_array
            return out

    def reshape_element(
        self: _SupFCCore,
        new_element_shape: Sequence[int],
        *args,
        **kwargs,
    ) -> _SupFCCore:
        """
        Reshape the element-wise portion of the buffer (i.e., trailing dimensions) to a new shape.

        This allows for reshaping vector/tensor field components, while keeping the spatial shape and
        axes unchanged.

        Parameters
        ----------
        new_element_shape : sequence of int
            The new shape of the element-wise (non-spatial) portion of the buffer.
        *args, **kwargs :
            Forwarded to the buffer's `reshape` method.

        Returns
        -------
        FieldComponent
            A new field component with reshaped element dimensions.

        Raises
        ------
        ValueError
            If the total number of elements is incompatible with the original buffer shape.
        """
        # Determine current spatial + element-wise shape
        spatial_shape = self.shape[: self.spatial_ndim]
        old_element_shape = self.shape[self.spatial_ndim :]

        old_numel = np.prod(old_element_shape, dtype=int)
        new_numel = np.prod(new_element_shape, dtype=int)

        if old_numel != new_numel:
            raise ValueError(
                f"Cannot reshape element dimensions from {old_element_shape} to {new_element_shape} "
                f"(incompatible sizes: {old_numel} vs {new_numel})."
            )

        reshaped = self.__buffer__.reshape(
            spatial_shape + tuple(new_element_shape), *args, **kwargs
        )

        # Rewrap in the current context
        return self.__class__(self.grid, reshaped, self.axes)

    # ============================= #
    # Generator Methods             #
    # ============================= #
    # These methods provide entry points for creating
    # FieldComponents.
    @classmethod
    def from_array(
        cls: Type[_SupFCCore],
        array_like: Any,
        grid: "GridBase",
        axes: Sequence[str],
        buffer_args: Sequence[Any] = (),
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> _SupFCCore:
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
            Additional keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        FieldComponent
            A new component wrapping the given array.

        Raises
        ------
        ValueError
            If the input array shape is incompatible with the grid and axes.
        """
        # Resolve the buffer class.
        buffer_kwargs = buffer_kwargs or {}
        buffer_class: Type["BufferBase"] = resolve_buffer_class(
            buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
        )

        # Standardize axes and compute expected spatial shape.
        axes = grid.standardize_axes(axes)
        axes_indices = grid.__cs__.convert_axes_to_indices(axes)
        spatial_shape = tuple(grid.gdd[axes_indices])

        # Extract array shape and check spatial compatibility.
        array_shape = np.shape(array_like)
        if array_shape[: len(spatial_shape)] != spatial_shape:
            raise ValueError(
                f"Array shape {array_shape} does not match expected spatial shape {spatial_shape} "
                f"for axes {axes}."
            )

        # Compute element shape and wrap in buffer.
        buff = buffer_class.from_array(array_like, *buffer_args, **buffer_kwargs)

        return cls(grid, buff, axes)

    @classmethod
    def from_function(
        cls: Type[_SupFCCore],
        func: Callable[..., Any],
        grid: "GridBase",
        axes: Sequence[str],
        result_shape: Optional[Sequence[int]] = None,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Sequence[Any] = (),
        buffer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> _SupFCCore:
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
        result_shape : tuple of int, optional
            Shape of trailing element-wise structure (e.g., for vectors/tensors). Defaults to scalar `()`.
        buffer_class : str or ~fields.buffers.base.BufferBase, optional
            The buffer backend used to hold data.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry used to resolve buffer class strings.
        buffer_args:
            Additional positional arguments forwarded to the buffer constructor (e.g., `dtype`).
        buffer_kwargs :
            Additional keyword arguments forwarded to the buffer constructor.
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
        axes = grid.standardize_axes(axes)
        buffer_kwargs = buffer_kwargs or {}

        # Allocate the buffer using the zeros constructor
        buff = cls.zeros(
            grid,
            axes,
            buffer_args=buffer_args,
            element_shape=result_shape,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_kwargs=buffer_kwargs,
        )

        # Evaluate the function and store the result in the buffer
        grid.compute_function_on_grid(
            func, result_shape=result_shape, out=buff, output_axes=axes, **kwargs
        )

        # Return the wrapped FieldComponent
        return buff

    @classmethod
    def zeros(
        cls: Type[_SupFCCore],
        grid: "GridBase",
        axes: Sequence[str],
        element_shape: Optional[Sequence[int]] = None,
        buffer_class: Optional[Type[_SupFCCore]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = (),
        buffer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> _SupFCCore:
        """
        Create a component filled with zeros.

        This is a convenience constructor that builds a zero-initialized :class:`~fields.components.FieldComponent`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
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
            Additional keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        ~fields.components.FieldComponent
            A new field component filled with zeros and defined over the specified grid and axes.
        """
        # Identify a buffer class.
        buffer_kwargs = buffer_kwargs or {}
        buffer_class: Type["BufferBase"] = resolve_buffer_class(
            buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
        )

        # Now determine the shape given the axes.
        axes = grid.standardize_axes(axes)
        axes_indices = grid.__cs__.convert_axes_to_indices(axes)
        spatial_shape = tuple(grid.gdd[axes_indices])
        element_shape = tuple() if element_shape is None else tuple(element_shape)
        shape = spatial_shape + element_shape

        # Now construct the buffer with the relevant args and kwargs.
        buff = buffer_class.zeros(shape, *buffer_args, **buffer_kwargs)

        # return the resulting object.
        return cls(grid, buff, axes)

    @classmethod
    def ones(
        cls: Type[_SupFCCore],
        grid: "GridBase",
        axes: Sequence[str],
        element_shape: Optional[Sequence[int]] = None,
        buffer_class: Optional[Type[_SupFCCore]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = (),
        buffer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> _SupFCCore:
        """
        Create a component filled with ones.

        This is a convenience constructor that builds a ones-initialized :class:`~fields.components.FieldComponent`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
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
            Additional keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        ~fields.components.FieldComponent
            A new field component filled with ones and defined over the specified grid and axes.
        """
        # Identify a buffer class.
        # noinspection DuplicatedCode
        buffer_kwargs = buffer_kwargs or {}
        buffer_class: Type["BufferBase"] = resolve_buffer_class(
            buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
        )

        # Now determine the shape given the axes.
        axes = grid.standardize_axes(axes)
        axes_indices = grid.__cs__.convert_axes_to_indices(axes)
        spatial_shape = tuple(grid.gdd[axes_indices])
        element_shape = tuple() if element_shape is None else tuple(element_shape)
        shape = spatial_shape + element_shape

        # Now construct the buffer with the relevant args and kwargs.
        buff = buffer_class.ones(shape, *buffer_args, **buffer_kwargs)

        # return the resulting object.
        return cls(grid, buff, axes)

    @classmethod
    def empty(
        cls: Type[_SupFCCore],
        grid: "GridBase",
        axes: Sequence[str],
        element_shape: Optional[Sequence[int]] = None,
        buffer_class: Optional[Type[_SupFCCore]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = (),
        buffer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> _SupFCCore:
        """
        Create a component with an empty buffer.

        This is a convenience constructor that builds a ones-initialized :class:`~fields.components.FieldComponent`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
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
            Additional keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        ~fields.components.FieldComponent
            A new field component with an empty buffer and defined over the specified grid and axes.
        """
        # Identify a buffer class.
        # noinspection DuplicatedCode
        buffer_kwargs = buffer_kwargs or {}
        buffer_class: Type["BufferBase"] = resolve_buffer_class(
            buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
        )

        # Now determine the shape given the axes.
        axes = grid.standardize_axes(axes)
        axes_indices = grid.__cs__.convert_axes_to_indices(axes)
        spatial_shape = tuple(grid.gdd[axes_indices])
        element_shape = tuple() if element_shape is None else tuple(element_shape)
        shape = spatial_shape + element_shape

        # Now construct the buffer with the relevant args and kwargs.
        buff = buffer_class.empty(shape, *buffer_args, **buffer_kwargs)

        # return the resulting object.
        return cls(grid, buff, axes)

    @classmethod
    def full(
        cls: Type[_SupFCCore],
        grid: "GridBase",
        axes: Sequence[str],
        fill_value: float = 0.0,
        element_shape: Optional[Sequence[int]] = None,
        buffer_class: Optional[Type[_SupFCCore]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = (),
        buffer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> _SupFCCore:
        """
        Create a component filled with a particular fill value.

        This is a convenience constructor that builds a :class:`~fields.components.FieldComponent`.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
        axes : list of str
            The spatial axes of the underlying coordinate system over which the field is
            defined. This must be some subset of the axes available in the coordinate system
            of `grid`.
        fill_value : float, optional
            The fill value to fill the newly initialized component with. This should be a floating point
            value. By default, the fill value is 0.0.
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
            Additional keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        ~fields.components.FieldComponent
            A new field component filled with `fill_value` and defined over the specified grid and axes.
        """
        # Identify a buffer class.
        # noinspection DuplicatedCode
        buffer_kwargs = buffer_kwargs or {}
        buffer_class: Type["BufferBase"] = resolve_buffer_class(
            buffer_class, buffer_registry=buffer_registry, default=ArrayBuffer
        )

        # Now determine the shape given the axes.
        axes = grid.standardize_axes(axes)
        axes_indices = grid.__cs__.convert_axes_to_indices(axes)
        spatial_shape = tuple(grid.gdd[axes_indices])
        element_shape = tuple() if element_shape is None else tuple(element_shape)
        shape = spatial_shape + element_shape

        # Now construct the buffer with the relevant args and kwargs.
        buff = buffer_class.full(
            shape, *buffer_args, fill_value=fill_value, **buffer_kwargs
        )

        # return the resulting object.
        return cls(grid, buff, axes)

    @classmethod
    def zeros_like(
        cls: Type[_SupFCCore], other: Type[_SupFCCore], *args, **kwargs
    ) -> _SupFCCore:
        """
        Create a zero-filled component with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : FieldComponent
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`zeros` constructor.

        Returns
        -------
        FieldComponent
            A new component with the same layout as `other` and zero-initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.zeros(grid, axes, *args, element_shape=shape, **kwargs)

    @classmethod
    def ones_like(
        cls: Type[_SupFCCore], other: Type[_SupFCCore], *args, **kwargs
    ) -> _SupFCCore:
        """
        Create a one-filled component with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : FieldComponent
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`ones` constructor.

        Returns
        -------
        FieldComponent
            A new component with the same layout as `other` and one-initialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.ones(grid, axes, *args, element_shape=shape, **kwargs)

    @classmethod
    def empty_like(
        cls: Type[_SupFCCore], other: Type[_SupFCCore], *args, **kwargs
    ) -> _SupFCCore:
        """
        Create an empty component with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : FieldComponent
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`ones` constructor.

        Returns
        -------
        FieldComponent
            A new component with the same layout as `other` and uninitialized data.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.empty(grid, axes, *args, element_shape=shape, **kwargs)

    @classmethod
    def full_like(
        cls: Type[_SupFCCore], other: Type[_SupFCCore], *args, **kwargs
    ) -> _SupFCCore:
        """
        Create a full-valued component with the same grid, axes, and element shape as another.

        Parameters
        ----------
        other : FieldComponent
            The reference component whose layout is used.
        *args, **kwargs:
            Forwarded to the :meth:`full` constructor.

        Returns
        -------
        FieldComponent
            A new component with the same layout as `other` and filled with a constant value.
        """
        # Extract the other's grid, axes, and shape.
        grid, axes, shape = other.grid, other.axes, other.element_shape
        return cls.full(grid, axes, *args, element_shape=shape, **kwargs)
