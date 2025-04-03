"""
Boundary conditions for geometric fields in Pisces-Geometry.

This module defines a flexible and extensible system for assigning and applying
boundary conditions on numerical fields defined over structured coordinate grids.

Boundary conditions are essential for simulating physical problems on finite domains.
They dictate how field values behave at the edges of the computational domain and
ensure mathematical consistency for differential equations and geometry-driven evolution.

Overview
--------
At the core of this module is the abstract base class ``_GenericBoundaryCondition``,
which defines the interface shared by all boundary condition types. Subclasses must
implement the ``__call__()`` method, which modifies the field buffer in-place to satisfy
the constraint along a specified axis and side.

All boundary conditions are tied to:
    - A field (typically a subclass of ``_BaseField``)
    - A specific axis (e.g., ``'r'``,``'theta'``)
    - A side of that axis (``'left'`` or ``'right'``)

Once constructed, boundary conditions must be applied manually or registered with the
field and enforced automatically during simulation steps.

Available Conditions
--------------------

- :py:class:`DirichletBC` : Imposes a fixed value at the boundary. This can be a scalar or an array
  matching the appropriate broadcast shape. Ghost cells are filled directly.

- :py:class:`NeumannBC` : Enforces a fixed gradient (first derivative) normal to the boundary.
  Values in ghost cells are extrapolated using a second-order centered stencil
  based on the gradient and interior field values.

- :py:class:`PeriodicBC` : Wraps values from the opposite end of the domain into the ghost region,
  creating a periodic (topologically closed) domain in that direction.

Interface
---------
All boundary conditions accept the following initialization arguments:

- ``field`` : the field object to which the condition applies
- ``axis`` : the axis (as a string) along which the condition is applied
- ``side`` : either "left" or "right", or 0/1
- additional keyword arguments for customization


Design Notes
------------

- The system uses precomputed slicing and buffer reshaping logic to efficiently
  apply constraints at runtime without allocating intermediate buffers.
- All conditions assume the presence of ghost zones, and their sizes are determined
  from the field's grid (``grid.__ghost_zones__``).
- Future boundary types (e.g., absorbing, reflective, higher-order extrapolation)
  can be implemented by subclassing ``_GenericBoundaryCondition``.

See Also
--------
fields.base._BaseField
grids.base._BaseGrid

"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

if TYPE_CHECKING:
    from pisces_geometry.fields.base import _BaseField
    from pisces_geometry.grids.base import _BaseGrid


class _GenericBoundaryCondition(ABC):
    """
    Abstract base class for geometric boundary conditions. All boundary conditions should
    be descended from this base class.

    The basic principle of boundary conditions is that each implements the ``__call__`` method
    which (using the bound parameters of the instance) operates on the boundary of a ``field`` to
    assert the boundary condition. To locate the array buffer to coerce and the position of the boundary,
    all BoundaryCondition subclasses take (at least) the following arguments at __init__:

    1. ``field``: The field on which this boundary condition is operating.
    2. ``axis``: The grid axis on which this boundary condition is operating.
    3. ``side``: The side of the boundary this boundary condition is operating on.

    Any additional configuration is passed as ``**kwargs``.
    """

    def __init__(
        self,
        field: "_BaseField",
        axis: str,
        side: Union[int, Literal["left", "right"]],
        **kwargs,
    ):
        """
        Initialize the generic boundary condition. This method should be extended
        or overwritten in subclasses to ensure custom behavior.

        Parameters
        ----------
        field: :py:class:`fields.base.GenericField`
            The field on which this boundary condition will apply.
        axis: str
            The grid axis of the field on which this boundary condition will be applied.
        side: "left" or "right"
            The side of the domain on which to apply this boundary condition
        kwargs:
            Additional keyword arguments passed to ``__init__``.
        """
        # Start by setting the field. This should generically be quite
        # easy. NOTE: we do NOT register the boundary condition when initializing
        # here to keep logic simple. Even once initialized, the boundary condition
        # needs to be handed to the field separately.
        self.__field__ = field

        # Set the axis on which the boundary condition applies. We require
        # that this axis is a legitimate grid axis of the field. From it, we can
        # also set the self.__axis_index__ attribute.
        if axis not in self.__field__.axes:
            raise ValueError(
                f"Field {self.__field__} does not span axis {axis}. Cannot set boundary condition."
            )

        # Set the parameters.
        self.__axis__ = axis
        self.__axis_index__ = self.__field__.axes.index(axis)

        # Handle the side parameter. This determines which side of the domain (left, lower) or
        # (right, upper) the boundary condition applies to.
        if isinstance(side, int):
            if 0 <= side < 2:
                self.__side__ = side
            else:
                raise ValueError("Side must be either 'left' or 'right' or 0/1.")
        elif side in ["left", "right"]:
            self.__side__ = 0 if side == "left" else 1
        else:
            raise ValueError("Boundary condition requires 'left' or 'right' side.")

        # Now pass off to the rest of the initialization process.
        self.__post_init__(**kwargs)

    def __post_init__(self, **kwargs):
        """
        Optional method for subclasses to perform setup or extract metadata
        (e.g., number of ghost cells, slicing indices, or precomputed stencils).
        """
        pass

    @abstractmethod
    def __call__(self):
        """
        Enforce the boundary condition on the attached field. This method should
        act directly on the underlying buffer of the field.
        """
        pass


class DirichletBC(_GenericBoundaryCondition):
    r"""
    `Dirichlet <https://en.wikipedia.org/wiki/Dirichlet_boundary_condition>`_ boundary conditions.

    For a given ``side`` and ``axis``, the dirichlet boundary condition is that
    the field :math:`\phi` satisfies the condition on the boundary :math:`\partial \Sigma` that

    .. math::

        \phi({\bf x}) = f({\bf x}),\; \forall {\bf x} \in \partial \Sigma.

    In practice, the boundary function (:math:`f({\bf x})`) is provided as either a :py:class:`numpy.ndarray` like object
    or as a scalar value.
    """

    def __init__(
        self,
        field: "_BaseField",
        axis: str,
        side: Union[int, Literal["left", "right"]],
        value: Union[float, np.ndarray],
        **kwargs,
    ):
        """
        Initialize a Dirichlet boundary condition on a field given a specified axis and
        side.

        Parameters
        ----------
        field: :py:class:`fields.base.GenericField`
            The field on which this boundary condition will apply. In order for the boundary condition
            to have any relevance, the field's grid must have at least 1 ghost zone on the boundary.
        axis: str
            The grid axis of the field on which this boundary condition will be applied.
        side: "left" or "right"
            The side of the domain on which to apply this boundary condition
        value: float or numpy.ndarray
            The value to use for this boundary condition. For a field with shape
            ``(N_1,...,N_k)``, if the dirichlet boundary is applied to axis ``i``, then
            the ``value`` should  be ``(N_1,...N_i-1,N_i+1,...,N_k)`` in shape in order
            to provide a valid boundary. If there are more than 1 layers of ghost zones available, this
            will be tiled to ensure that correct shape.
        kwargs:
            Additional keyword arguments passed to ``__init__``.
        """
        super().__init__(field, axis, side, value=value, **kwargs)

    def __post_init__(self, value=0.0):
        """
        Set up slicing and metadata for the boundary condition.

        Parameters
        ----------
        value : float or array-like
            The constant value to set in the ghost zone.
        """
        # Set and validate the value input to ensure that it is
        # reasonable / acceptable.
        _desired_shape_ = (
            self.__field__.buffer_shape[: self.__axis_index__]
            + (1,)
            + self.__field__.buffer_shape[self.__axis_index__ + 1 :]
        )
        _expected_shape_ = (
            self.__field__.buffer_shape[: self.__axis_index__]
            + self.__field__.buffer_shape[self.__axis_index__ + 1 :]
        )
        if isinstance(value, np.ndarray):
            if value.shape != _expected_shape_:
                raise ValueError(
                    f"Dirichlet boundary value had shape {value.shape}, but the {self.__axis__} boundary has shape"
                    f" {_expected_shape_}."
                )
        else:
            value *= np.ones(_desired_shape_)

        self.__value__ = value.reshape(_desired_shape_)

        # Determine the number of ghost cells from the grid's ghost zone table.
        grid = self.__field__.__grid__
        self.nghost = grid.__ghost_zones__[self.__axis_index__, self.__side__]

        # Precompute the slice for this boundary region
        slicer = [slice(None)] * self.__field__.buffer_ndim

        if self.__side__ == 0:  # left
            slicer[self.__axis_index__] = slice(0, self.nghost)
        else:  # right
            slicer[self.__axis_index__] = slice(-self.nghost, None)

        self.__slicer__ = tuple(slicer)

    def __call__(self):
        """
        Apply the Dirichlet boundary condition by setting ghost zones to the fixed value.
        The value is broadcast to match the shape of the ghost region.
        """
        buffer = self.__field__.buffer
        ghost_shape = buffer[self.__slicer__].shape

        try:
            # Attempt to broadcast and assign
            buffer[self.__slicer__] = np.broadcast_to(self.__value__, ghost_shape)
        except ValueError as e:
            raise ValueError(
                f"Could not broadcast value of shape {np.shape(self.__value__)} to ghost region "
                f"shape {ghost_shape}."
            ) from e


class NeumannBC(_GenericBoundaryCondition):
    """
    `Neumann <https://en.wikipedia.org/wiki/Neumann_boundary_condition>`_ (constant gradient) boundary condition.

    Enforces a fixed normal derivative at the domain boundary using a symmetric
    two-point stencil to compute ghost zone values based on interior values.
    """

    def __init__(
        self,
        field: "_BaseField",
        axis: str,
        side: Union[int, Literal["left", "right"]],
        value: Union[float, np.ndarray],
        **kwargs,
    ):
        """
        Initialize a Neumann boundary condition on a field along a given axis and side.

        Parameters
        ----------
        field : _BaseField
            The field to which this boundary condition is applied.
        axis : str
            The axis along which to apply the BC (e.g., 'r', 'theta').
        side : {'left', 'right'}
            Which boundary to apply it to.
        value : float or np.ndarray
            The desired gradient (∂u/∂n) to enforce at the boundary.
        kwargs : dict
            Additional options passed to base class.
        """
        super().__init__(field, axis, side, value=value, **kwargs)

    def __post_init__(self, value=0.0):
        """
        Precompute the slices and grid information needed to apply the BC.
        """
        # Compute expected shape for the incoming gradient array
        full_shape = self.__field__.buffer_shape
        axis = self.__axis_index__ = self.__field__.axes.index(self.__axis__)
        self.__axis__ = self.__field__.axes[axis]

        # Expected shape is the buffer shape with axis dim removed
        element_shape = full_shape[:axis] + full_shape[axis + 1 :]
        broadcast_shape = full_shape[:axis] + (1,) + full_shape[axis + 1 :]

        # Handle scalar or array-valued gradient
        if isinstance(value, np.ndarray):
            if value.shape != element_shape:
                raise ValueError(
                    f"Neumann gradient value shape {value.shape} does not match boundary shape {element_shape}."
                )
            self.__value__ = value.reshape(broadcast_shape)
        else:
            self.__value__ = np.full(broadcast_shape, value)

        # Get ghost zone width for this axis/side
        grid = self.__field__.__grid__
        self.nghost = grid.__ghost_zones__[axis, self.__side__]

        # Pull coordinate array including or excluding ghosts as needed
        coord_array = grid.get_coordinate_arrays(
            axes=[self.__axis__], include_ghosts=True
        )
        coord_vals = np.ndarray(coord_array)

        # Extract the coordinate values used for ghost/interior interpolation
        if self.__side__ == 0:
            # Left side: ghost spans [0 : nghost], use [0 : nghost+2]
            coords = coord_vals[: self.nghost + 2]
        else:
            # Right side: ghost spans [-nghost :], use [-nghost-2 :]
            coords = coord_vals[-(self.nghost + 2) :]

        # Compute effective 2*dx spacing for symmetric stencil
        self.ghost_loc_diff = coords[2:] - coords[:-2]  # length = nghost

        # Reshape to broadcast with gradient value
        shape = [1] * len(full_shape)
        shape[axis] = self.nghost
        self.ghost_loc_diff = self.ghost_loc_diff.reshape(shape)

        # Create slices for ghost region and its interior stencil
        ghost_slice = [slice(None)] * len(full_shape)
        interior_slice = [slice(None)] * len(full_shape)

        if self.__side__ == 0:
            ghost_slice[axis] = slice(0, self.nghost)
            interior_slice[axis] = slice(2, 2 + self.nghost)
            self.__sign__ = -1
        else:
            ghost_slice[axis] = slice(-self.nghost, None)
            interior_slice[axis] = slice(-(2 + self.nghost), -2)
            self.__sign__ = +1

        self.__slicer__ = tuple(ghost_slice)
        self.__interior_slicer__ = tuple(interior_slice)

    def __call__(self):
        """
        Apply the Neumann (constant gradient) condition by setting ghost zone values.

        The ghost zone value is computed using a second-order symmetric stencil:
            u_ghost = u_interior + sign * grad * dx
        """
        buffer = self.__field__.buffer

        # Interior values used to extrapolate
        interior_vals = buffer[self.__interior_slicer__]

        # Compute ghost zone values via Neumann condition
        ghost_vals = (
            interior_vals + self.__sign__ * self.__value__ * self.ghost_loc_diff
        )

        # Assign to buffer
        try:
            buffer[self.__slicer__] = ghost_vals
        except ValueError as e:
            raise ValueError(
                f"Failed to apply Neumann BC: could not broadcast shape {ghost_vals.shape} "
                f"to ghost region shape {buffer[self.__slicer__].shape}."
            ) from e


class PeriodicBC(_GenericBoundaryCondition):
    """
    Periodic boundary condition.

    This boundary condition wraps values from the opposite side of the domain
    into the ghost cells, effectively creating a topologically periodic domain
    along the specified axis.

    For a grid with ``nghost`` ghost zones:

    - Left ghost cells receive data from the last ``nghost`` interior cells on the right.
    - Right ghost cells receive data from the first ``nghost`` interior cells on the left.
    """

    def __post_init__(self):
        """
        Precompute source and destination slices to wrap periodic values.
        """
        grid: "_BaseGrid" = self.__field__.__grid__
        axis = self.__axis_index__
        ndim = self.__field__.buffer_ndim

        # Number of ghost cells on this side and the opposite
        self.nghost = grid.__ghost_zones__[axis, self.__side__]
        self.nghost_opposite = grid.__ghost_zones__[axis, 1 - self.__side__]

        src = [slice(None)] * ndim
        dst = [slice(None)] * ndim

        if self.__side__ == 0:
            # Left ghost cells ← wrap from far-right interior
            dst[axis] = slice(0, self.nghost)
            src_start = -(self.nghost_opposite + self.nghost)
            src_stop = -self.nghost_opposite
        else:
            # Right ghost cells ← wrap from far-left interior
            dst[axis] = slice(-self.nghost, None)
            src_start = self.nghost_opposite
            src_stop = self.nghost_opposite + self.nghost

        src[axis] = slice(src_start, src_stop)

        self.__slicer_dst__ = tuple(dst)
        self.__slicer_src__ = tuple(src)

    def __call__(self):
        """
        Apply the periodic boundary condition by copying values from the
        opposite boundary region into this side’s ghost zone.
        """
        buffer = self.__field__.buffer

        try:
            buffer[self.__slicer_dst__] = buffer[self.__slicer_src__]
        except ValueError as e:
            raise ValueError(
                f"PeriodicBC failed: cannot broadcast shape {buffer[self.__slicer_src__].shape} "
                f"to ghost region shape {buffer[self.__slicer_dst__].shape} "
                f"(axis='{self.__axis__}', side={self.__side__})."
            ) from e


__bc_registry__ = {
    0: None,
    1: DirichletBC,
    2: PeriodicBC,
    3: NeumannBC,
}
