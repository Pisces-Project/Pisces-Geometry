"""
Coordinate mixin classes inherited by all coordinate system subclasses.

This module provides the coordinate system support for coordinate operations like slicing
coordinate grids, constructing grids, etc.
"""
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np

# ================================== #
# TYPING SUPPORT                     #
# ================================== #
if TYPE_CHECKING:
    from pymetric.coordinates.mixins._typing import _SupportsCoordinateSystemAxes

_SupCSAxes = TypeVar("_SupCSAxes", bound="_SupportsCoordinateSystemAxes")


# ================================== #
# Mixin Classes                      #
# ================================== #
# These classes form the core mixins of the base coordinate system class.
class CoordinateOperationsMixin(Generic[_SupCSAxes]):
    """
    Provides coordinate manipulation operations to coordinate systems.

    This class provides methods for inserting missing coordinates into lists of
    coordinate representations to make operations at the coordinate system level
    as easy as possible.
    """

    def coordinate_meshgrid(
        self: _SupCSAxes,
        *coordinate_arrays,
        axes: Optional[Sequence[str]] = None,
        copy: bool = True,
        sparse: bool = False,
    ):
        r"""
        Construct a coordinate-aligned meshgrid from 1D coordinate arrays.

        This method returns a tuple of N-D coordinate arrays aligned with the coordinate system’s
        canonical axis order, given a subset of 1D coordinate arrays. It reorders the inputs to
        match the canonical axis order and wraps NumPy’s :func:`numpy.meshgrid`.

        Parameters
        ----------
        coordinate_arrays: array_like
            1-D arrays representing the coordinates of a grid.
        axes: list of str, optional
            The axes present in the provided ``coordinate_arrays``. If ``None`` (default),
            then the first ``len(coordinate_arrays)`` axes of the coordinate system will be used.
            Otherwise, the provided axes will be used to re-order the coordinates into canonical
            order before creating the mesh.
        sparse : bool, optional
            If True the shape of the returned coordinate array for dimension *i*
            is reduced from ``(N1, ..., Ni, ... Nn)`` to
            ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
            intended to be use with :ref:`basics.broadcasting`.  When all
            coordinates are used in an expression, broadcasting still leads to a
            fully-dimensional result array.

            Default is False.

        copy : bool, optional
            If False, a view into the original arrays are returned in order to
            conserve memory.  Default is True.  Please note that
            ``sparse=False, copy=False`` will likely return non-contiguous
            arrays.  Furthermore, more than one element of a broadcast array
            may refer to a single memory location.  If you need to write to the
            arrays, make copies first.

        Returns
        -------
        tuple of np.ndarray
            A tuple of N-D arrays suitable for evaluating vectorized fields over a structured grid.

        Raises
        ------
        ValueError
            If `axes` are invalid or their count does not match `coordinate_arrays`.

        Examples
        --------
        Create a grid in spherical coordinates composed of :math:`r` and :math:`\theta`
        coordinates.

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> from pymetric.utilities.logging import pg_log
        >>> pg_log.level = 50
        >>> cs = SphericalCoordinateSystem()
        >>> r = np.linspace(0, 1, 10)
        >>> theta = np.linspace(0, np.pi, 20)
        >>> rgrid, thetagrid = cs.coordinate_meshgrid(r, theta, axes=["r", "theta"])
        >>> rgrid.shape, thetagrid.shape
        ((10, 20), (10, 20))

        Axes may also be supplied out of order

        >>> cs = SphericalCoordinateSystem()
        >>> r = np.linspace(0, 1, 10)
        >>> theta = np.linspace(0, np.pi, 20)
        >>> rgrid, thetagrid = cs.coordinate_meshgrid(theta,r, axes=["theta","r"])
        >>> rgrid.shape, thetagrid.shape
        ((10, 20), (10, 20))

        If an invalid axis is specified, then an error occurs.

        >>> cs = SphericalCoordinateSystem()
        >>> r = np.linspace(0, 1, 10)
        >>> theta = np.linspace(0, np.pi, 20)
        >>> rgrid, thetagrid = cs.coordinate_meshgrid(theta,r, axes=["theta","psi"]) # doctest: +SKIP, +ELLIPSIS
        ValueError: Unknown axis/axes ['psi'] – valid axes are ['r', 'theta', 'phi']
        """
        # Validate and ensure that axes are legitimate.
        axes = self.resolve_axes(axes)

        # return the meshgrid with a pass through the
        # canonical ordering.
        return np.meshgrid(
            *self.in_canonical_order(coordinate_arrays, axes),
            copy=copy,
            sparse=sparse,
            indexing="ij",
        )

    def create_coordinate_list(
        self: _SupCSAxes,
        coordinate_arrays: Sequence[np.ndarray],
        /,
        axes: Optional[Sequence[str]] = None,
        *,
        fixed_axes: Optional[Dict[str, float]] = None,
    ) -> List[Union[np.ndarray, float]]:
        """
        Assemble a list of coordinate components (arrays or scalars) in canonical axis order.

        This method combines provided free axis arrays and fixed scalar values into a
        complete list of coordinates, ordered according to the coordinate system's axes.

        Parameters
        ----------
        coordinate_arrays : list of np.ndarray
            Array values corresponding to the free (swept) axes.
            Each array can be:

            - 1D (for simple swept coordinates, e.g., `[0, 1, 2, 3]`), or
            - ND (for pre-built grids created by :func:`numpy.meshgrid`).

        axes : list of str, optional
            The complete list of axes (both free and fixed) represented by the inputs.
            Defaults to the canonical axis order ``self.axes`` if omitted.
        fixed_axes : dict of {str: float}, optional
            A mapping of axis names to constant scalar values for axes held fixed.
            Scalars must be real numbers (floats or ints).

        Returns
        -------
        list of numpy.ndarray or float
            A list of coordinate components, one for each canonical axis in ``self.axes``.

            - Free axes are represented by the corresponding arrays (broadcast-compatible).
            - Fixed axes are represented by constant scalar values.

        Raises
        ------
        ValueError

            - If unknown axes are provided.
            - If the number of provided coordinate arrays does not match the number of free axes.
            - If a fixed axis is specified that is not present in the overall axes list.

        Examples
        --------
        Construct a list with a mix of free (swept) and fixed coordinates:

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> from pymetric.utilities.logging import pg_log
        >>> pg_log.level = 50
        >>> cs = SphericalCoordinateSystem()
        >>> r = np.linspace(0, 1, 100)
        >>> phi = np.linspace(0, 2*np.pi, 200)
        >>> coords = cs.create_coordinate_list([r, phi], fixed_axes={"theta": np.pi/2})
        >>> [c.shape if isinstance(c, np.ndarray) else c for c in coords]
        [(100,), 1.5707963267948966, (200,)]

        Providing axes explicitly and out of order:

        >>> coords = cs.create_coordinate_list([phi, r], axes=["phi", "theta", "r"], fixed_axes={"theta": np.pi/2})
        >>> [c.shape if isinstance(c, np.ndarray) else c for c in coords]
        [(100,), 1.5707963267948966, (200,)]
        """
        # Start by processing the input axes and ensuring that
        # they are all valid and listed out.
        axes = self.resolve_axes(axes, require_order=False)
        ordered_axes = self.order_axes_canonical(axes)
        free_axes, fixed_axes = self.get_free_fixed(axes=axes, fixed_axes=fixed_axes)

        # Build the full axis-to-value mapping
        axis_values = {ax: val for ax, val in zip(free_axes, coordinate_arrays)}
        axis_values.update(fixed_axes)

        # Assemble the output list in canonical self.axes order
        return [axis_values[ax] for ax in ordered_axes]
