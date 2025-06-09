"""
Field component classes for structured geometric data.

This module defines the :class:`FieldComponent` class, the core data container for single-buffer
components of structured geometric fields in Pisces Geometry. A :class:`FieldComponent` binds together a
numerical buffer (such as a NumPy array, `unyt_array`, or HDF5-backed dataset) with metadata that
describes how it aligns with a structured grid and how it should behave under NumPy-style operations.

Field components are the atomic building blocks of all field objects in the Pisces framework.
They support spatial axis labeling, broadcasting, ufunc overrides, and type coercion,
and are designed to integrate cleanly with NumPy and scientific Python workflows.

See Also
--------
:class:`~fields.base.DenseField`
:class:`~fields.tensors.DenseTensorField`
:class:`~fields.buffers.base.BufferBase`
:func:`~fields.buffers.base.buffer_from_array`
:class:`~grids.base.GridBase`
"""
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import ArrayLike

from pymetric.fields.buffers import buffer_from_array
from pymetric.fields.mixins._generic import NumpyArithmeticMixin
from pymetric.fields.mixins.components import FieldComponentCoreMixin
from pymetric.utilities.arrays import apply_ufunc_to_labels

if TYPE_CHECKING:
    from pymetric.fields.buffers.base import BufferBase
    from pymetric.grids.base import GridBase


class FieldComponent(
    NumpyArithmeticMixin,
    FieldComponentCoreMixin,
):
    r"""
    A single buffer-aligned component of a geometric field.

    :class:`FieldComponent` represents a dense, spatially resolved data block
    associated with a single component of a geometric field. It couples a numerical buffer
    (e.g., NumPy array, `unyt`, or HDF5 dataset) with metadata about its spatial axes and
    the structured grid on which it resides.

    This class provides the foundational storage and computation interface for fields such as
    :class:`~fields.base.DenseField` and :class:`~fields.tensors.DenseTensorField`. Each
    `FieldComponent` knows which axes it spans, how it aligns with a coordinate grid, and
    can interact seamlessly with NumPy operations through ufunc overrides and broadcasting.

    Key Responsibilities
    --------------------
    - Wrap a structured array-like buffer with grid-aware metadata.
    - Track which grid axes the data aligns with and support broadcasting across compatible fields.
    - Provide NumPy ufunc and array function compatibility.
    - Serve as a backend-neutral data container for differential geometry operations.
    - Support views into the underlying buffer in various formats (e.g., array, `unyt`, raw core).

    Notes
    -----
    Each field in the library is composed of one or more :class:`FieldComponent` instances.
    Most dense fields only contain one, but multi-component or symbolic fields may
    combine many for blockwise decomposition or coordinate basis expansion.

    Unlike higher-level field objects, :class:`FieldComponent` does not track symbolic dependence,
    tensor variance, or units at the field level. It focuses purely on the aligned numerical
    representation of a single data block.

    See Also
    --------
    ~fields.base.DenseField :
        Field class that wraps a single `FieldComponent`.
    ~fields.tensors.DenseTensorField :
        Tensor-valued field with variance and signature tracking.
    ~fields.buffers.base.BufferBase :
        Underlying abstract buffer class used to store the field data.
    ~grids.base.GridBase :
        Structured grid object that defines spatial coordinates and chunking.
    """

    # --- Class Level Flags --- #
    # These are logical flags for determining
    # finer details of class behavior and triage.
    __array_priority__ = 3.0
    """
    The priority of the component class in numpy operations.
    """
    __array_function_dispatch__: Optional[Dict[Callable, Callable]] = None
    """
    `__array_function_dispatch__` is a dictionary which can optionally map
    NumPy callables to internal implementations to allow overriding of default behavior.

    By default, when a NumPy function (non ufunc) is called on a Component, the buffer
    is stripped and the operation occurs on the underlying representation. If a callable
    is specified here, then `__array_function__()` will catch the redirect and
    triage accordingly.
    """

    # --- Initialization Logic --- #
    # This section of the class manages the initialization
    # logic. It can be altered in subclasses as needed.
    def __init__(
        self, grid: "GridBase", buffer: ArrayLike, axes: Sequence[str], /, **kwargs
    ):
        """
        Initialize a FieldComponent instance.

        Parameters
        ----------
        grid : GridBase
            The structured grid over which the field is defined.

        buffer : ArrayLike
            The data buffer representing field values. This can be a NumPy array,
            unyt array, h5py dataset, or other supported backend. It will be
            wrapped automatically using `buffer_from_array`.

        axes : Sequence[str]
            The sequence of axis labels corresponding to spatial dimensions
            of the buffer. These should match dimensions in the grid.

        **kwargs :
            Additional keyword arguments passed to `buffer_from_array`.

        Raises
        ------
        ValueError
            If the shape of the buffer is inconsistent with the grid and axes.
        """
        # Assign the grid and the buffer. Attempt to coerce to
        # a buffer successfully.
        self.__grid__: "GridBase" = grid
        self.__buffer__: "BufferBase" = buffer_from_array(buffer, **kwargs)

        # Now utilize the axes to standardize and then check that the field
        # has the correct shape.
        self.__axes__: List[str] = self.__grid__.standardize_axes(axes)
        self.__grid__.check_field_shape(buffer.shape, axes=self.__axes__)

    # --- Dunder Methods --- #
    # Dunder methods should not be altered to any significant
    # degree to ensure that field components have relatively standard
    # behaviors.
    def __getitem__(self, idx: Any) -> Any:
        return self.__buffer__.__getitem__(idx)

    def __setitem__(self, idx: Any, value: Any) -> Any:
        return self.__buffer__.__setitem__(idx, value)

    def __repr__(self) -> str:
        return f"<FieldComponent grid={self.__grid__} buffer={self.__buffer__.__repr__()} axes={self.__axes__}>"

    def __str__(self) -> str:
        return self.__buffer__.__str__()

    def __len__(self) -> int:
        return len(self.__buffer__)

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__buffer__)

    # --- Numpy Semantics --- #
    # Methods determining the semantics of numpy operations.
    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return self.__buffer__.__array__(dtype=dtype)

    def __standardize_ufunc_inputs__(
        self, *inputs
    ) -> Tuple[Sequence[Any], Sequence[Any], Sequence[str]]:
        """
        Normalize and prepare inputs for NumPy ufunc operations on field components.

        This method ensures that all `FieldComponent` inputs are coerced to use a common
        set of axes, extracted as the union of all axes present in the input components.
        Each component is converted to its underlying array buffer and broadcasted to
        match the full `broadcast_axes` layout.

        Non-`FieldComponent` inputs are returned unchanged. Their axis labels are marked as `None`.

        Parameters
        ----------
        *inputs : Any
            The positional inputs passed to a NumPy ufunc. These may include a mix of
            `FieldComponent` instances and raw arrays, scalars, etc.

        Returns
        -------
        unwrapped_inputs : list
            The actual buffer-backed data used in computation. For `FieldComponent`s,
            this is the `.as_buffer_repr_in_axes(...)` representation.
        input_axes : list of list[str] or None
            The axes each input is aligned to. Non-`FieldComponent` inputs will have `None`.
        broadcast_axes : list of str
            The unified set of axes shared by all participating `FieldComponent`s.

        Notes
        -----
        This is intended for internal use inside `__array_ufunc__` to ensure that field buffers
        are broadcast-compatible and semantically consistent during elementwise operations.
        """
        # Identify all of the elements of the inputs
        # which are self.__class__ typed so that we can
        # catch and coerce them.
        input_axes: List = [None] * len(inputs)
        component_inputs = [inp for inp in inputs if isinstance(inp, self.__class__)]

        # Check that we have components that need management. If not,
        # we can return immediately without any further computation.
        if len(component_inputs) == 0:
            return inputs, input_axes, []

        # Determine the set of shared axes from the provided
        # components. These will be the axes that get broadcast to.
        broadcast_axes = [ci.__axes__ for ci in component_inputs]
        broadcast_axes = self.__grid__.standardize_axes(
            list(set().union(*broadcast_axes)),
        )

        # Start managing the inputs. We enforce that everything ends up
        # in the Buffer Representation class and is coerced to the shared axes.
        unwrapped_inputs = []
        for _i, inp in enumerate(inputs):
            if isinstance(inp, FieldComponent):
                unwrapped_inputs.append(inp.broadcast_buffer_to_axes(broadcast_axes))
                input_axes[_i] = broadcast_axes
            else:
                unwrapped_inputs.append(inp)

        # Return the unwrapped inputs, the input axes, and the broadcast
        # axes.
        return unwrapped_inputs, input_axes, broadcast_axes

    # noinspection PyMethodMayBeStatic
    def __apply_ufunc_to_axes__(
        self, ufunc, method, *inputs, **kwargs
    ) -> Sequence[str]:
        """
        Apply a NumPy universal function to a set of input shapes and axes to
        determine the anticipated output shape and axes.

        Inputs should be a list of tuples containing (shape,axes).
        """
        # Standardize the inputs by extending them to have
        # a length matching the number of dimensions for each
        # element.
        inputs = [
            (s, tuple(a) + (len(s) - len(a)) * (None,)) if a is not None else (s, None)
            for s, a in inputs
        ]

        # Run the label logic to compute output axes
        _, output_labels = apply_ufunc_to_labels(ufunc, method, *inputs, **kwargs)

        # Extract grid axes: we expect (..., None, None, ... )
        # so we extract up to the first instance of None and then
        # return.
        # TODO: This is a little inelegant. PRIORITY: low
        grid_axes = []
        for ax in output_labels:
            if ax is None:
                break
            grid_axes.append(ax)

        return grid_axes

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Standardize the inputs. This requires that we go through the
        # inputs, identify the relevant axes for each input, and broadcast
        # components to a shared set of axes.
        unwrapped_inputs, input_axes, output_axes = self.__standardize_ufunc_inputs__(
            *inputs
        )

        # Propagate the shape and axes through the ufunc to determine the
        # anticipated size and axes of the resulting object.
        input_shapes_and_axes = [
            (np.shape(_input), _input_axes)
            for _input, _input_axes in zip(unwrapped_inputs, input_axes)
        ]
        output_axes: Sequence[str] = self.__apply_ufunc_to_axes__(
            ufunc, method, *input_shapes_and_axes, **kwargs
        )

        # Handle any instances where `out` is specified.
        out = kwargs.get("out", None)
        if out is not None:
            # Normalize to a tuple for uniform processing
            is_tuple = isinstance(out, tuple)
            out_tuple = out if is_tuple else (out,)

            # Unwrap buffers
            unwrapped_out = tuple(
                o.__buffer__ if isinstance(o, self.__class__) else o for o in out_tuple
            )
            kwargs["out"] = unwrapped_out if is_tuple else unwrapped_out[0]

            # Apply the ufunc
            result = getattr(ufunc, method)(*unwrapped_inputs, **kwargs)

            # check if result is None or if we got something back.
            # If something came back we actually want to pass out.
            if result is not None:
                return out
            else:
                return result
        else:
            # out was not specified, we need to compute the result and
            # determine if we treat it as a component or not.
            result = getattr(ufunc, method)(*unwrapped_inputs, **kwargs)

            # Determine what typing the output should get.
            if len(output_axes) > 0:
                return self.__class__(
                    self.__grid__,
                    result,
                    output_axes,
                )
            else:
                return result

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
            a.__buffer__ if isinstance(a, self.__class__) else a for a in args
        )
        unwrapped_kwargs = {
            _k: _v.__buffer__ if isinstance(_v, self.__class__) else _v
            for _k, _v in kwargs.items()
        }
        return func(*unwrapped_args, **unwrapped_kwargs)

    def as_array(self) -> np.ndarray:
        """
        Return the buffer as a standard NumPy array.

        This strips away units and any backend-specific metadata. It is useful for
        performing standard NumPy operations or exporting data in a backend-agnostic format.

        Returns
        -------
        np.ndarray
            A NumPy array copy of the field's data buffer.
        """
        return self.__buffer__.as_array()

    # --- Attributes --- #
    # Standardized attributes for component classes.
    @property
    def grid(self) -> "GridBase":
        """
        The structured grid over which this field component is defined.

        Returns
        -------
        ~grids.base.GridBase
            The grid object associated with this field.
        """
        return self.__grid__

    @property
    def axes(self) -> List[str]:
        """
        The spatial axes along which this field component is defined.

        These axes correspond to the leading dimensions of the field buffer.

        Returns
        -------
        list of str
            Canonical list of spatial axis labels.
        """
        return self.__axes__[:]

    @property
    def naxes(self) -> int:
        """
        Number of spatial axes the field is defined over.

        Returns
        -------
        int
            The number of named spatial axes.
        """
        return len(self.__axes__)

    @property
    def buffer(self) -> "BufferBase":
        """
        The internal buffer storing this field’s data.

        This buffer provides backend-specific logic (NumPy, unyt, HDF5, etc.)
        for data access, arithmetic, and I/O.

        Returns
        -------
        ~fields.buffers.base.BufferBase
            The underlying storage buffer.
        """
        return self.__buffer__

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the full data array, including spatial and element dimensions.

        Returns
        -------
        tuple of int
            Full shape of the field buffer.
        """
        return self.__buffer__.shape

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """
        The shape of the field over the spatial axes (grid-aligned dimensions).

        Returns
        -------
        tuple of int
            Shape of the field over the named spatial axes only.
        """
        return self.shape[: self.spatial_ndim]

    @property
    def element_shape(self) -> Tuple[int, ...]:
        """
        The shape of the element-wise structure (e.g., vector or tensor components).

        Returns
        -------
        tuple of int
            Shape of the trailing element-wise dimensions.
        """
        return self.shape[self.spatial_ndim :]

    @property
    def is_scalar(self) -> bool:
        """Return True if the field has no element-wise structure."""
        return self.element_ndim == ()

    @property
    def size(self) -> int:
        """
        Total number of elements in the buffer.

        Returns
        -------
        int
            Product of all dimensions in the field shape.
        """
        return self.__buffer__.size

    @property
    def element_size(self) -> int:
        """Total number of element-wise components."""
        return int(np.prod(self.element_shape)) if self.element_ndim > 0 else 1

    @property
    def spatial_size(self) -> int:
        """Total number of spatial elements (grid cells)."""
        return int(np.prod(self.spatial_shape))

    @property
    def ndim(self) -> int:
        """
        Total number of dimensions in the field buffer.

        This includes both spatial (grid-aligned) and trailing element dimensions.

        Returns
        -------
        int
            Total rank of the field array.
        """
        return self.__buffer__.ndim

    @property
    def spatial_ndim(self) -> int:
        """
        Number of spatial dimensions (i.e., number of named axes).

        Returns
        -------
        int
            Number of dimensions aligned with the grid.
        """
        return len(self.__axes__)

    @property
    def element_ndim(self) -> int:
        """
        Number of trailing element-wise dimensions (e.g., vector or tensor structure).

        Returns
        -------
        int
            Number of dimensions not aligned with spatial grid axes.
        """
        return self.ndim - self.spatial_ndim

    @property
    def dtype(self) -> Any:
        """
        The data type of the elements stored in the buffer.

        Returns
        -------
        dtype
            The NumPy dtype or equivalent backend type.
        """
        return self.__buffer__.dtype

    @property
    def c(self):
        """
        Shorthand to obtain the underlying buffer's core
        object.

        Returns
        -------
        ArrayLike
            The backend-native data structure stored in this buffer.
        """
        return self.__buffer__.__array_object__
