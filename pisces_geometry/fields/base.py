"""
Base classes for Pisces-Geometry fields.

This module contains a single class (:py:class:`GenericField`), which serves as the base class on which
all other forms of fields are defined.
"""
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from pisces_geometry.fields.boundary_conditions import (
    __bc_registry__,
    _GenericBoundaryCondition,
)
from pisces_geometry.fields.buffers import ArrayBuffer, _Buffer
from pisces_geometry.fields.utilities import identify_grid_axes_from_array

if TYPE_CHECKING:
    from pisces_geometry.grids._typing import BoundingBox
    from pisces_geometry.grids.base import _BaseGrid
    from pisces_geometry.utilities._typing import IndexType


class _BaseField(ABC):
    # ================================= #
    # Class Variables (core settings)   #
    # ================================= #
    # These are core variables that determine some of the underlying
    # behavior of field classes. They can be modified or added to in subclasses.
    __default_buffer_type__: Type[_Buffer] = ArrayBuffer

    # ================================= #
    # Numpy Interactivity               #
    # ================================= #
    # Field classes should behave (basically) like numpy arrays with some
    # special features. To that end, we implement a number of numpy specific
    # internal methods to ensure that numpy integration is fully operable.
    def __array__(self, **kwargs) -> np.ndarray:
        return np.asarray(self.__buffer__.__array_object__, **kwargs)

    @staticmethod
    def __axes_ufunc__(_, method, *inputs, **kwargs):
        """
        Determines how axes are propagated when performing ufuncs on
        or between fields.

        Logic:
        - For standard elementwise calls (method="__call__"):
            → Output axes = union of all grid axes from input fields.
            → No axes are removed.

        - For reduction-like methods ("reduce", "reduceat", etc.):
            → Grid axes specified in 'axis' may be removed.
            → If any grid axis is reduced, it's removed from the output axes.
            → If axis=None (reduce all), then all grid axes are removed.

        Parameters
        ----------
        ufunc : np.ufunc
            The NumPy ufunc being applied.
        method : str
            The ufunc method: '__call__', 'reduce', 'accumulate', 'reduceat', etc.
        *inputs : Any
            Positional arguments passed to the ufunc.
        **kwargs : Any
            Keyword arguments passed to the ufunc (particularly 'axis').

        Returns
        -------
        output_axes : set[str]
            Set of grid axes that should remain in the output field.
        removed_axes : set[str]
            Axes that were removed during the operation (e.g., by reduction).
        modifies_grid : bool
            Whether any grid axis was removed (used for validation).
        """
        # Parse the inputs and identify the fields that are present in
        # the inputs. We then want to also fetch all of the axes from that
        # list of inputs.
        input_fields = [x for x in inputs if isinstance(x, _BaseField)]
        all_axes = set().union(*(f.axes for f in input_fields))
        if not len(all_axes):
            raise ValueError(
                "__axes_ufunc__ called in scenario when no fields were present in *inputs. This would"
                " be a very weird thing to have happen."
            )

        # Construct the set of removed axes and then start working through the
        # reduction logic. To start, if method == `__call__`, then we have a really
        # simple case because things are always performed element wise and we simply lose
        # no axes.
        removed_axes = set()
        if method in ["__call__", "outer", "reduceat", "at", "accumulate"]:
            # Elementwise — preserve all grid axes. Outer won't reduce
            # away any axes, only add them. By our convention, those should
            # all be non-grid axes so the axes are the same.
            return all_axes, removed_axes, False

        # In all other cases, the `method` might cause things to change regarding
        # which axes are present in the output. To contend with that, we need to
        # determine the shapes / dimensions of all the inputs and then explicitly
        # determine what happens to the axes in each case.
        field_ndim = input_fields[0].full_ndim
        field_axes = input_fields[0].axes

        if method == "reduce":
            # The reduction will act on one field and is performed on one / many axis(es).
            # if that axis is a grid axis, then we're going to lose it.
            axis = kwargs.get("axis", None)
            if axis is None:
                # We reduce over ALL axes, all of the input axes (including the grid)
                # will be removed.
                removed_axes = set(field_axes)
            else:
                # We will lose some (but not necessarily all) axes. We normalize
                # and then remove them sequentially.
                _normed_idx = np.core.numeric.normalize_axis_tuple(axis, field_ndim)
                removed_axes.update(
                    *[
                        field_axes[_nix]
                        for _nix in _normed_idx
                        if _nix < len(field_axes)
                    ]
                )

        return all_axes - removed_axes, removed_axes, bool(removed_axes)

    @staticmethod
    def __broadcast_ufunc__(*inputs):
        """
        Manage broadcasting rules for universal functions. The idea here is that
        when we have 2 fields there is a way to expand on the typical broadcasting
        rules by casting them to a buffer on a shared grid.

        Parameters
        ----------
        *inputs :
            The inputs passed to the ufunc.

        Returns
        -------
        tuple
            A tuple of broadcasted buffers ready to pass to the ufunc.

        Notes
        -----
        The broadcasting rules are as follows:
            - If there is a SINGLE field in the inputs, then we simply grab the
              buffer and return. There is no manipulation necessary.
            - If there is a SINGLE field and ANY OTHER INPUT, then we again simply
              grab and go. If there is a broadcasting error, it's the user's problem.
            - If there are TWO OR MORE FIELDS, we will broadcast the buffers to
              a standardized buffer grid.
        """
        # Identify the input fields. These should be all of the field-class
        # inputs that are delivered. There should be at least 1 of them.
        input_fields = [x for x in inputs if isinstance(x, _BaseField)]
        if not input_fields:
            raise ValueError("No GenericField inputs found in `__broadcast_ufunc__`.")

        # Check if we need to actually do anything here. Basically, unless
        # we have multiple fields, we don't care to do anything.
        if len(input_fields) < 2:
            return tuple(
                [
                    x.__buffer__.__array_object__ if isinstance(x, _BaseField) else x
                    for x in inputs
                ]
            )

        # Because we have more than 1 field, we will instead construct a
        # shared set of axes and broadcast the array objects to that set of
        # axes.
        # Start by checking that all the fields are on the same grid.
        _basegrid_ = input_fields[0].__grid__
        if any(_field.__grid__ != _basegrid_ for _field in input_fields):
            raise ValueError(
                "Cannot perform ufunc operations between fields on different grids."
            )

        # Identify the target axes.
        target_axes = set().union(*[_field.axes for _field in input_fields])
        target_axes = [ax for ax in _basegrid_.axes if ax in target_axes]

        # Now we start to perform the field by field broadcasting procedure
        # to standardize the field inputs.
        broadcasted = []
        for x in inputs:
            if isinstance(x, _BaseField):
                broadcasted.append(x.broadcast_buffer_to_new_axes(target_axes))
            else:
                broadcasted.append(x)  # literals, scalars, or arrays

        return tuple(broadcasted)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Overrides the behavior of numpy universal functions (ufuncs) when
        performing operations between Pisces-Geometry fields.

        This modification is written with the intention of defining
        standardized semantics for numpy operations on fields including
        managing advanced broadcasting behavior and grid standardization.

        The method is composed of a few submethods which can be altered
        or extended as needed in subclasses.

        Rules:
        - Elementwise ufuncs work on aligned buffers with matching axes.
        - Reductions on grid axes are disallowed and raise an error.
        - Resulting field has output_axes as defined by __axes_ufunc__.
        - Scalar outputs (from full reductions) return raw arrays.
        """
        # To begin, we determine what the expected output axes are going to be
        # given the ufunc that is being performed and the method being called.
        # This layer of logic allows us to comprehensively account for
        # changes in the field's grid connection when performing ufuncs.
        output_axes, removed_axes, modifies_grid = self.__axes_ufunc__(
            ufunc, method, *inputs, **kwargs
        )

        # Standardize the buffers by broadcasting. This ONLY AFFECTS
        # FIELDS. Fields can be made consistent by broadcasting against
        # grids; however, if any other broadcasting issues are not
        # solved by numpy, we simply raise an error.
        buffers = self.__broadcast_ufunc__(*inputs)

        # Perform the universal function on the buffers.
        result = getattr(ufunc, method)(*buffers, **kwargs)

        # Determine the behavior of returned values. If the
        # output buffer can be turned onto a grid (with the output axes),
        # we will try to do so; however, if not, an error will arise.
        if not len(output_axes):
            # There are no output axes left, so this is not
            # a grid-bound field anymore. Return the buffer.
            return result
        else:
            try:
                axes = [ax for ax in self.grid if ax in output_axes]
                return self.__class__(self.grid, axes=axes, buffer=result)
            except Exception as e:
                raise ValueError(
                    f"Failed to interpret result of ufunc {ufunc.__name__} as a field: {e}"
                )

    # ================================= #
    # Dunder Methods                    #
    # ================================= #
    def __str__(self):
        return f"<{self.__class__.__name__} | (grid={self.__grid__.shape})>"

    def __repr__(self):
        return f"{self.__class__.__name__}(grid={self.grid})"

    def __getitem__(self, index: "IndexType") -> Any:
        return self.__buffer__[index]

    def __setitem__(self, index: "IndexType", value: Any) -> None:
        self.__buffer__[index] = value

    def __iter__(self):
        """
        Iterate over each grid point, yielding the field value at that point.
        This assumes the field is defined over a regular grid and the buffer is
        indexed with grid-shaped leading dimensions.

        Yields
        ------
        Any
            Field value at each grid point.
        """
        # Iterate over the grid index space
        for index in np.ndindex(*self.grid.shape):
            yield self.__buffer__[index, ...]

    def __len__(self) -> int:
        return self.__buffer__.size

    def __contains__(self, item) -> bool:
        return item in self.__buffer__

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _BaseField):
            return False
        return np.array_equal(self.__buffer__, other.buffer) and self.grid == other.grid

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.divide(self, other)

    def __rtruediv__(self, other):
        return np.divide(other, self)

    # ================================= #
    # Properties                        #
    # ================================= #
    @property
    def axes(self) -> List[str]:
        """
        The grid axes that this field is defined over.

        This is a subset of the coordinate axes from the grid's coordinate system.
        It defines the spatial subdomain over which the field is sampled.

        For example, if the full grid has axes ``('x', 'y', 'z')``, and a field
        is defined over ``('x', 'z')``, then the field is treated as a 2D slice of
        the full 3D domain.

        Returns
        -------
        List[str]
            The names of the grid axes this field spans.
        """
        return self.__axes__

    @property
    def axes_ndim(self) -> int:
        """
        The number of grid axes this field spans.

        This corresponds to the number of leading dimensions in the buffer
        that represent the spatial grid portion of the field.

        Returns
        -------
        int
            Dimensionality of the field's grid slice.
        """
        return len(self.axes)

    @property
    def buffer(self) -> _Buffer:
        """
        The underlying buffer object that stores the field values.

        This buffer may wrap a NumPy array, HDF5-backed array, or other
        compatible memory model. The buffer has shape:

            ``(grid_shape for axes) + (element_shape)``

        Returns
        -------
        _Buffer
            The data buffer object holding field values.
        """
        return self.__buffer__

    @property
    def buffer_shape(self) -> Tuple[int, ...]:
        """
        Full shape of the field buffer, including both grid and element dimensions.

        This includes the leading spatial (grid) axes followed by trailing element-wise
        dimensions.

        Returns
        -------
        Tuple[int, ...]
            The shape of the underlying buffer.
        """
        return self.__buffer__.shape

    @property
    def buffer_ndim(self) -> int:
        """
        Total number of dimensions in the buffer.

        This includes both the grid and element-wise dimensions.

        Returns
        -------
        int
            The number of dimensions in the full field buffer.
        """
        return self.__buffer__.ndim

    @property
    def element_shape(self) -> Tuple[int, ...]:
        """
        The shape of the field values at each grid point.

        This excludes the grid dimensions and reflects the tensorial nature of the field.
        For example:
            - Scalars: ()
            - Vectors: (3,)
            - Tensors: (3, 3)

        Returns
        -------
        Tuple[int, ...]
            The trailing shape of the buffer after the grid axes.
        """
        return self.buffer.shape[self.axes_ndim :]

    @property
    def element_ndim(self) -> int:
        """
        Number of element-wise (non-grid) dimensions.

        This corresponds to the rank of the field's value at each grid point.

        Returns
        -------
        int
            Dimensionality of the field values (scalar = 0, vector = 1, etc.)
        """
        return self.buffer.ndim - self.axes_ndim

    @property
    def rank(self) -> int:
        """
        The tensor rank of the field.

        This is an alias for :py:attr:`element_ndim`, indicating the dimensionality
        of the field value at each point (e.g., scalar=0, vector=1, matrix=2).

        Returns
        -------
        int
            Tensor rank of the field values.
        """
        return self.element_ndim

    @property
    def grid_axes_mask(self) -> np.ndarray:
        """
        Boolean mask over the grid coordinate system indicating active axes.

        This array has one entry per coordinate axis in the full grid. It is `True`
        at positions where the field is defined, and `False` otherwise.

        Example
        -------
        If the grid has axes ``['x', 'y', 'z']`` and the field spans ``['x', 'z']``,
        the mask will be:

            ``array([True, False, True])``

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(grid.ndim,)`` marking used axes.
        """
        return self.grid.coordinate_system.build_axes_mask(self.axes)

    @property
    def grid_bbox(self) -> "BoundingBox":
        """
        The physical bounding box of the grid in coordinate space.

        This describes the extent of the grid in each spatial dimension.
        Returned as a shape ``(2, ndim)`` array of lower and upper bounds.

        Returns
        -------
        BoundingBox
            Physical extent of the coordinate grid (excluding ghost zones).
        """
        return self.grid.bbox

    @property
    def grid(self) -> "_BaseGrid":
        """
        Returns the grid object associated with this field.

        This grid defines the coordinate system, domain shape, and metadata
        that underpins the field’s spatial representation.

        Returns
        -------
        _BaseGrid
            The structured grid instance tied to this field.
        """
        return self.__grid__

    @property
    def grid_shape(self) -> Sequence[int]:
        """
        The full shape of the underlying grid (all spatial axes).

        This is distinct from the grid portion of the field's buffer,
        which may span only a subset of these axes.

        Returns
        -------
        Sequence[int]
            Shape of the full coordinate grid.
        """
        return self.grid.shape

    @property
    def grid_ndim(self) -> int:
        """
        Total number of dimensions in the coordinate grid.

        This includes all axes supported by the grid’s coordinate system,
        regardless of whether this field spans them.

        Returns
        -------
        int
            Dimensionality of the full coordinate grid.
        """
        return self.grid.ndim

    @property
    def T(self) -> "_BaseField":
        """
        Transpose the element-wise dimensions of the field at each grid point.

        This operation does not affect the spatial (grid) axes of the field. It only
        permutes the trailing dimensions of the buffer, which represent the tensorial
        structure of the field at each grid point. For example:

            - A scalar field (shape: (Nx, Ny)) is unaffected.
            - A vector field (shape: (Nx, Ny, 3)) becomes (Nx, Ny, 3) — no change.
            - A matrix field (shape: (Nx, Ny, 3, 3)) becomes (Nx, Ny, 3, 3). The last two dims are transposed.

        Returns
        -------
        _BaseField
            A new field instance with transposed element-wise dimensions.

        Raises
        ------
        NotImplementedError
            If the field is rank < 2 and transpose has no meaningful effect.
        """
        if self.rank < 2:
            return self  # No change for scalar or vector

        # Transpose the element-wise dimensions (only trailing dims)
        grid_ndim = self.axes_ndim
        element_shape = self.buffer_shape[grid_ndim:]

        # If the element is not 2D or higher, no-op
        if len(element_shape) < 2:
            return self

        # Compute full transpose: grid axes untouched, element dims reversed
        perm = list(range(grid_ndim)) + list(
            range(grid_ndim + self.rank - 1, grid_ndim - 1, -1)
        )
        transposed = self.buffer.__array_object__.transpose(perm)

        return self.__class__(
            buffer=transposed,
            grid=self.grid,
            axes=self.axes,
        )

    # ================================= #
    # Initialization                    #
    # ================================= #
    # These methods are all delegated to during __init__ and can
    # be modified as needed when subclassing. Read the docstrings
    # for each in order to get a clear explanation of required tasks
    # for each method.
    def __coerce_buffer__(self, *args, **kwargs) -> _Buffer:
        """
        This method takes (at the minimum) the following arguments:

        - args[0]: buffer: the buffer that got passed in to __init__ that now
          needs to be coerced.
        - __buffer_type__: The type of buffer to try to obtain. This supplies
          the coerce method.
        - __buffer_kwargs__: The kwargs that get passed to the coercion method.

        In return, the method must supply a VALID _Buffer instance.
        """
        # By default, we simply obtain the buffer and attempt to return
        # the coerced result.
        buffer = args[0]
        __buffer_type__ = kwargs.get(
            "__buffer_type__", self.__class__.__default_buffer_type__
        )
        __buffer_kwargs__ = kwargs.get("__buffer_kwargs__", {})
        __buffer_type__ = (
            __buffer_type__
            if __buffer_type__ is not None
            else self.__class__.__default_buffer_type__
        )
        __buffer_kwargs__ = __buffer_kwargs__ if __buffer_kwargs__ is not None else {}

        try:
            return __buffer_type__.coerce(buffer, **__buffer_kwargs__)
        except Exception as e:
            raise ValueError(
                f"Failed to coerce input buffer ({buffer}) to type {__buffer_type__.__name__}: {e}"
            )

    def __configure_and_validate_grid__(self, *args, **kwargs):
        """
        Configure the grid and validate the buffer's alignment with the grid axes.

        This method sets the grid and determines which subset of the grid's axes the field spans,
        either from user input or by inferring from the buffer shape.

        Parameters
        ----------
        args : tuple
            Expected to contain at least the grid object as the first element.
        kwargs :
            May include:
                - 'axes': optional list of axis names to explicitly set.

        Raises
        ------
        ValueError
            If no axes can be inferred or if the buffer is inconsistent with the grid.
        """
        # Extract the arguments from the function and then validate that
        # we have everything we need to have.
        if not args:
            raise ValueError(
                "Grid must be provided as the first positional argument to __configure_and_validate_grid__."
            )

        grid = args[0]
        provided_axes = kwargs.get("axes", None)

        # If no axes are given, attempt to infer them from the buffer shape
        if provided_axes is None:
            inferred_axes = identify_grid_axes_from_array(
                self.__buffer__.__array_object__, grid
            )
            if not inferred_axes:
                raise ValueError(
                    f"Could not infer grid axes from buffer shape {self.__buffer__.shape} "
                    f"for grid shape {grid.shape}."
                )
            axes = inferred_axes
        else:
            axes = list(provided_axes)

        # Validate that all specified axes exist on the grid
        invalid_axes = [ax for ax in axes if ax not in grid.axes]
        if invalid_axes:
            raise ValueError(
                f"The following axes are not valid for the grid: {invalid_axes}"
            )

        # Finalize configuration
        self.__grid__ = grid
        self.__axes__ = axes

    def __init__(
        self,
        buffer: Any,
        grid: Any,
        *args,
        axes: Sequence[str] = None,
        __buffer_type__: Type[_Buffer] = None,
        __buffer_kwargs__: Dict[str, Any] = None,
        **kwargs,
    ):
        # Begin by standardizing the buffer. Fields require a
        # _Buffer class (or subclass) buffer but might get anything passed
        # in `buffer`. Our first step is to therefore coerce the buffer to
        # meet our standards. The signature of __coerce_buffer__ might change
        # in subclasses and should be updated here to reflect that change.
        self.__buffer__ = self.__coerce_buffer__(
            buffer, __buffer_type__=__buffer_type__, __buffer_kwargs__=__buffer_kwargs__
        )

        # With the __buffer__ set, we proceed to configure the grid and
        # check the validity of the axes and the buffer's shape. This is
        # likely to change somewhat in subclasses.
        self.__configure_and_validate_grid__(grid, axes=axes)

        # Setup the boundary conditions. To avoid clutter in the __init__, we
        # don't actually allow these to be set during init but instead modified
        # as the user needs.
        self.__boundary_conditions__: np.ndarray[Callable] = np.full(
            (len(self.axes), 2), fill_value=None, dtype=object
        )

    # =================================== #
    # General Utility Functions           #
    # =================================== #
    def broadcast_buffer_to_new_axes(self, target_axes: list[str]) -> Any:
        """
        Broadcasts the grid portion of this field's buffer to match a target set of axes.

        This is useful for aligning fields with different axis subsets so they can be
        combined in ufuncs. The resulting buffer will have shape compatible with ``target_axes``,
        preserving the element-wise shape.

        Parameters
        ----------
        target_axes : list of str
            Target grid axes (must be a superset of this field's axes). The order must match
            the coordinate system's axis order.

        Returns
        -------
        Any
            A buffer reshaped for broadcasting, with shape:
                ``(grid_shape aligned to target_axes) + element_shape``

        Raises
        ------
        ValueError
            If ``target_axes`` does not include all of this field's axes.
        """
        # Validate that this field's axes are a subset of the target
        if any(ax not in target_axes for ax in self.__axes__):
            raise ValueError(
                f"Cannot broadcast {self} to {target_axes}: missing required axes {self.__axes__}."
            )

        cs_axes = self.grid.coordinate_system.axes
        grid_shape_by_axis = dict(zip(cs_axes, self.grid.dd))

        # Construct broadcast grid shape in coordinate system order
        broadcast_grid_shape = []
        for ax in cs_axes:
            if ax in target_axes:
                if ax in self.__axes__:
                    broadcast_grid_shape.append(grid_shape_by_axis[ax])
                else:
                    broadcast_grid_shape.append(1)

        final_shape = tuple(broadcast_grid_shape) + tuple(self.shape)
        reshaped_buffer = self.__buffer__.__array_object__.reshape(final_shape)

        return reshaped_buffer

    def set_boundary_condition(
        self,
        grid_axis: str,
        boundary_condition: Union[int, Type[_GenericBoundaryCondition], None],
        side: Union[int, Literal["left", "right", "both"]] = "both",
        **kwargs,
    ):
        """
        Register a boundary condition on this field for a given axis and side(s).

        Parameters
        ----------
        grid_axis : str
            The name of the grid axis (e.g., 'r', 'theta') to which the boundary applies.
        boundary_condition : int, class, or None
            The boundary condition specification:

            - If `int`, it is resolved via the global `__bc_registry__` dictionary.
            - If a subclass of `_GenericBoundaryCondition`, it is instantiated with this field.
            - If `None`, the boundary condition is cleared on the specified side(s).

        side : {'left', 'right', 'both'} or int
            Which side of the axis to apply the condition. Accepts string labels or integers:
            - 'left' or 0
            - 'right' or 1
            - 'both' applies the condition to both sides.

        **kwargs
            Additional keyword arguments passed to the boundary condition constructor
            (e.g., `value`, `gradient`, etc.).

        Raises
        ------
        ValueError
            If the axis is invalid or the boundary type is not recognized.
        """
        if grid_axis not in self.axes:
            raise ValueError(f"Axis '{grid_axis}' not found in field axes {self.axes}.")

        # Normalize side argument
        if isinstance(side, int):
            if side not in (0, 1):
                raise ValueError("Integer 'side' must be 0 (left) or 1 (right).")
            sides = [side]
        elif side == "both":
            sides = [0, 1]
        elif side == "left":
            sides = [0]
        elif side == "right":
            sides = [1]
        else:
            raise ValueError(f"Invalid side specifier: {side}")

        axis_index = self.axes.index(grid_axis)

        for s in sides:
            key = (axis_index, s)

            if boundary_condition is None:
                self.__boundary_conditions__[key] = None
                continue

            # Resolve the boundary condition class
            if isinstance(boundary_condition, int):
                bc_cls = __bc_registry__.get(boundary_condition)
                if bc_cls is None:
                    raise ValueError(
                        f"No boundary condition registered for ID {boundary_condition}"
                    )
            elif isinstance(boundary_condition, type) and issubclass(
                boundary_condition, _GenericBoundaryCondition
            ):
                bc_cls = boundary_condition
            else:
                raise TypeError(
                    "boundary_condition must be an int (registry key), a _GenericBoundaryCondition subclass, or None."
                )

            # Instantiate and store the boundary condition
            self.__boundary_conditions__[key] = bc_cls(
                field=self, axis=grid_axis, side=s, **kwargs
            )

    def get_boundary_condition(
        self, axis: str, side: Literal["left", "right"]
    ) -> Optional[_GenericBoundaryCondition]:
        """
        Retrieve a registered boundary condition for a specific axis and side.

        Parameters
        ----------
        axis : str
            Name of the axis (must be in self.axes).
        side : {'left', 'right'}
            Side of the axis.

        Returns
        -------
        _GenericBoundaryCondition or None
            The boundary condition instance, or None if not registered.

        Raises
        ------
        ValueError
            If the axis or side is invalid.
        """
        if axis not in self.axes:
            raise ValueError(f"Axis '{axis}' not found in field axes {self.axes}.")

        side_index = 0 if side == "left" else 1 if side == "right" else None
        if side_index is None:
            raise ValueError("Side must be 'left' or 'right'.")

        key = (self.axes.index(axis), side_index)
        return self.__boundary_conditions__.get(key, None)

    def list_boundary_conditions(self):
        """
        Print a summary of all boundary conditions registered to this field.
        """
        print(f"Registered boundary conditions for {self}:")
        for axis_index, axis in enumerate(self.axes):
            for side_index, side_label in enumerate(["left", "right"]):
                bc = self.__boundary_conditions__.get((axis_index, side_index))
                name = bc.__class__.__name__ if bc else "None"
                print(f"  Axis: {axis:<8} Side: {side_label:<5} → {name}")

    def assert_boundary_condition(
        self, order: Optional[list[tuple[str, Union[int, str]]]] = None
    ):
        """
        Apply all boundary conditions currently registered on this field.

        Parameters
        ----------
        order : list of tuple (axis, side), optional
            A list of (axis, side) pairs to specify the order of application.
            If None, all registered boundary conditions are applied in default order.
        """
        if order is None:
            order = [(axis, side) for axis in self.axes for side in [0, 1]]

        for axis, side in order:
            axis_index = self.axes.index(axis) if isinstance(axis, str) else axis
            side_index = 0 if side == "left" else 1 if side == "right" else side

            key = (axis_index, side_index)
            bc = self.__boundary_conditions__.get(key)

            if bc is None:
                continue  # Skip if no BC registered

            try:
                bc()
            except Exception as e:
                raise RuntimeError(
                    f"Error applying boundary condition on axis '{self.axes[axis_index]}' side '{['left', 'right'][side_index]}'."
                ) from e

    # =================================== #
    # I/O Operations                      #
    # =================================== #
    # These IO methods are used to save / load the field from
    # disk.
    def from_hdf5(self):
        pass

    def to_hdf5(self):
        pass


class GenericField(_BaseField):
    """
    A generic field with no additional tensor structure.
    """
