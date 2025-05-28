"""
Mathematical mixins for coordinate systems in order to support basic mathematical
support.
"""
from typing import TYPE_CHECKING, Callable, Dict, Generic, Optional, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from pymetric.differential_geometry.dense_utils import (
    dense_adjust_tensor_signature,
    dense_lower_index,
    dense_raise_index,
)

# ================================== #
# TYPING SUPPORT                     #
# ================================== #
if TYPE_CHECKING:
    from pymetric.coordinates.mixins._typing import _SupportsCoordinateSystemCoordinates

_SupCSCoords = TypeVar("_SupCSCoords", bound="_SupportsCoordinateSystemCoordinates")


# ================================== #
# Mixin Classes                      #
# ================================== #
# These classes form the core mixins of the base coordinate system class.
class CoordinateSystemMathMixin(Generic[_SupCSCoords]):
    """
    Mixin class for coordinate systems which provides access to tensor methods.

    These methods do not include differential operations because the necessary grid logic is
    not present at this level of abstraction.
    """

    def compute_expression(
        self: _SupCSCoords,
        expression: str,
        coordinates: Sequence[ArrayLike],
        fixed_axes: Optional[Dict[str, float]] = None,
    ) -> ArrayLike:
        """
        Evaluate a named symbolic expression over mixed coordinate inputs.

        This method supports evaluating an expression by combining:
        - Positional `*coordinates` (1D arrays for free axes),
        - Keyword `fixed_axes` (scalars for fixed axes),
        into a full coordinate list in canonical axis order.

        Parameters
        ----------
        expression : str
            The name of the symbolic expression to evaluate.
        *coordinates : array-like
            Positional coordinate inputs for free axes. The number of entries must match
            the number of non-fixed axes in the system.
        fixed_axes : dict of {str: float}, optional
            A dictionary mapping axis names to fixed scalar values (e.g., `theta=np.pi/2`).
            These axes must be a subset of the coordinate system's axes and must not overlap with
            the axes supplied positionally.

        Returns
        -------
        array-like
            The result of evaluating the named expression at the provided coordinates.

        Raises
        ------
        KeyError
            If the expression is not found or has no numeric form.
        ValueError
            If coordinate axes or counts are inconsistent with the system definition.

        Examples
        --------
        >>> from pymetric.coordinates import CylindricalCoordinateSystem
        >>> from pymetric.utilities.logging import pg_log
        >>> pg_log.level = 50
        >>> cs = CylindricalCoordinateSystem()
        >>> r = np.linspace(0, 1, 100)
        >>> z = np.linspace(-1, 1, 100)
        >>> R,Z = cs.coordinate_meshgrid(r,z,axes=['rho','z'])
        >>> rho = cs.compute_expression_from_coordinates(
        ...     "metric_density",
        ...     [R,Z],
        ...     fixed_axes={"phi": np.pi/4}
        ... )
        >>> rho.shape
        (100, 100)
        """
        try:
            numeric_expr = self.get_numeric_expression(expression)
        except KeyError as e:
            raise KeyError(
                f"Expression '{expression}' is not available as a numeric callable for {self}."
            ) from e

        # Assemble full coordinate list in canonical axis order
        coord_list = self.create_coordinate_list(coordinates, fixed_axes=fixed_axes)

        # Evaluate and return result
        return numeric_expr(*coord_list)

    def compute_expression_from_coordinates(
        self: _SupCSCoords,
        expression: str,
        coordinates: Sequence[ArrayLike],
        fixed_axes: Optional[Dict[str, float]] = None,
        sparse: bool = False,
    ) -> ArrayLike:
        r"""
        Evaluate a named expression over 1D coordinate arrays with optional fixed axes.

        This method creates a broadcasted meshgrid internally from 1D input arrays
        and evaluates the expression over it.

        Parameters
        ----------
        expression : str
            The name of the symbolic expression to evaluate.
        coordinates : list of array-like
            1D arrays representing coordinate values for free axes.
        fixed_axes : dict of {str: float}, optional
            Dictionary of scalar axis values for fixed axes.
        sparse : bool, optional
            If True the shape of the returned coordinate array for dimension *i*
            is reduced from ``(N1, ..., Ni, ... Nn)`` to
            ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
            intended to be use with :ref:`basics.broadcasting`.  When all
            coordinates are used in an expression, broadcasting still leads to a
            fully-dimensional result array.

            Default is False.

        Returns
        -------
        array-like
            The evaluated result over the broadcasted grid.

        Raises
        ------
        KeyError
            If the expression is not available.
        ValueError
            If axis names are inconsistent.

        Examples
        --------
        In spherical coordinates, the metric density is :math:`r^2\sin \theta`. Thus, we can
        fairly easily create a plot of this!

        >>> # Imports
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> from scipy.interpolate import NearestNDInterpolator
        >>> import matplotlib.pyplot as plt
        >>> from pymetric.utilities.logging import pg_log
        >>> pg_log.level = 50
        >>> u = SphericalCoordinateSystem()
        >>>
        >>> # Create the r,theta coordinate grids.
        >>> r = np.linspace(0,1,100)
        >>> theta = np.linspace(0,np.pi,100)
        >>> R,THETA = np.meshgrid(r,theta, indexing='ij')
        >>>
        >>> # Compute the metric density on the grid.
        >>> md = u.compute_expression_from_coordinates("metric_density", [r, theta], fixed_axes={"phi": np.pi / 4})
        >>>
        >>> # Use SciPy to interpolate to a cartesian coordinate system.
        >>> x,z = np.linspace(-1/np.sqrt(2),1/np.sqrt(2),100),np.linspace(-1/np.sqrt(2),1/np.sqrt(2),100)
        >>> X,Z = np.meshgrid(x,z,indexing='ij')
        >>> cart_grid_r, cart_grid_theta, _ = u._convert_cartesian_to_native(X, 0, Z)
        >>> interp = NearestNDInterpolator(np.stack([R.ravel(), THETA.ravel()],axis=-1), md.ravel())
        >>> Z = interp(cart_grid_r, cart_grid_theta)
        >>>
        >>> # Create the plot.
        >>> ext = [-1/np.sqrt(2),1/np.sqrt(2),-1/np.sqrt(2),1/np.sqrt(2)]
        >>> plt.imshow(Z.T,extent=ext,origin='lower')
        >>> plt.show()
        """
        try:
            numeric_expr = self.get_numeric_expression(expression)
        except KeyError as e:
            raise KeyError(
                f"Expression '{expression}' is not available for {self}."
            ) from e

        # Get the free and fixed axes.
        free_ax, fixed_ax = self.get_free_fixed(fixed_axes=fixed_axes)
        grids = self.coordinate_meshgrid(
            *coordinates, axes=free_ax, sparse=sparse, copy=False
        )
        coordinates = self.insert_fixed_axes(grids, free_ax, fixed_axes=fixed_axes)
        return numeric_expr(*coordinates)

    def compute_function_from_coordinates(
        self: _SupCSCoords,
        func: Callable,
        coordinates: Sequence[ArrayLike],
        fixed_axes: Optional[Dict[str, float]] = None,
        sparse: bool = False,
    ) -> ArrayLike:
        r"""
        Evaluate a function over 1D coordinate arrays with optional fixed axes.

        This method creates a broadcasted meshgrid internally from 1D input arrays
        and evaluates the expression over it.

        Parameters
        ----------
        func : callable
            The function being evaluated.
        coordinates : list of array-like
            1D arrays representing coordinate values for free axes.
        fixed_axes : dict of {str: float}, optional
            Dictionary of scalar axis values for fixed axes.
        sparse : bool, optional
            If True the shape of the returned coordinate array for dimension *i*
            is reduced from ``(N1, ..., Ni, ... Nn)`` to
            ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
            intended to be use with :ref:`basics.broadcasting`.  When all
            coordinates are used in an expression, broadcasting still leads to a
            fully-dimensional result array.

            Default is False.

        Returns
        -------
        array-like
            The evaluated result over the broadcasted grid.

        Raises
        ------
        KeyError
            If the expression is not available.
        ValueError
            If axis names are inconsistent.
        """
        # Get the free and fixed axes.
        free_ax, fixed_ax = self.get_free_fixed(fixed_axes=fixed_axes)
        grids = self.coordinate_meshgrid(
            *coordinates, axes=free_ax, sparse=sparse, copy=False
        )
        coordinates = self.insert_fixed_axes(grids, free_ax, fixed_axes=fixed_axes)
        return func(*coordinates)

    def requires_expression(
        self: _SupCSCoords,
        value: Optional[ArrayLike],
        expression: str,
        coordinates: Sequence[ArrayLike],
        fixed_axes: Optional[Dict[str, float]] = None,
    ) -> ArrayLike:
        """
        Return `value` if not None; otherwise evaluate `compute_expression`.

        This is a convenience method for lazy evaluation over ND broadcasted grids.

        Parameters
        ----------
        value : array-like or None
            Precomputed result or None to trigger evaluation.
        expression : str
            The name of the expression to evaluate if `value` is None.
        coordinates : list of array-like
            ND broadcasted coordinate arrays.
        fixed_axes : dict of {str: float}, optional
            Scalar axes for fixed dimensions.

        Returns
        -------
        array-like
            Either the provided `value` or the computed result.
        """
        if value is not None:
            return value
        return self.compute_expression(expression, coordinates, fixed_axes=fixed_axes)

    def requires_expression_from_coordinates(
        self: _SupCSCoords,
        value: Optional[ArrayLike],
        expression: str,
        coordinates: Sequence[ArrayLike],
        fixed_axes: Optional[Dict[str, float]] = None,
        sparse: bool = False,
    ) -> ArrayLike:
        """
        Return `value` if not None; otherwise evaluate `compute_expression_from_coordinates`.

        This is a convenience wrapper for evaluating on 1D coordinate arrays with optional fixed axes.

        Parameters
        ----------
        value : array-like or None
            Precomputed result or None to trigger evaluation.
        expression : str
            The name of the expression to evaluate if `value` is None.
        coordinates : list of array-like
            1D arrays for each free coordinate axis.
        fixed_axes : dict of {str: float}, optional
            Fixed scalar coordinate values.
        sparse : bool, optional
            Whether to construct sparse coordinate arrays.

        Returns
        -------
        array-like
            Either the provided `value` or the computed result.
        """
        if value is not None:
            return value
        return self.compute_expression_from_coordinates(
            expression, coordinates, fixed_axes=fixed_axes, sparse=sparse
        )

    def raise_index_dense(
        self,
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        *coordinates: np.ndarray,
        inverse_metric_field: Optional[np.ndarray] = None,
        fixed_axes: Optional[Dict[str, float]] = None,
        out: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Raise a single tensor index using :math:`g^{ab}`.

        The routine contracts the supplied *inverse* metric with ``tensor_field``

        .. math::

            T^{\mu}{}_{\dots} = g^{\mu\nu} \, T_{\nu\dots}

        Parameters
        ----------
        tensor_field : numpy.ndarray
            Tensor field of shape ``(F₁, ..., F_m, N, ...)``, where the last `rank` axes
            are the tensor index dimensions.
        index
            Which tensor slot (``0, ..., rank-1``) to raise.
        rank : int
            Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
            the number of identified coordinate axes and therefore determines the shape of the returned array.
        *coordinates
            ND coordinate grids **in canonical axis order**.
            Only needed when ``inverse_metric_field`` is *None*.
        inverse_metric_field :  numpy.ndarray
            Inverse metric tensor with shape ``(F₁, ..., F_n, N, )`` or ``(F₁, ..., F_n, N, N)``,
            where ``N == n``. Must be broadcast-compatible with the field shape of `tensor_field`.
        fixed_axes
            Constant axis values to use when computing the metric.
        out
            Optional output buffer.
        **kwargs
            Forwarded verbatim to :func:`~differential_geometry.dense_utils.dense_raise_index`.

        Returns
        -------
        numpy.ndarray
            Tensor identical to ``tensor_field`` except the chosen
            slot is now contravariant.

        Examples
        --------
        In spherical coordinates, the vector :math:`V_\mu = (r^2,r^2,0)` becomes

        .. math::

            v^\mu = g^{\mu\mu} V_\mu = (r^2, 1, 0).

        To perform the operation computationally,

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> from pymetric.utilities.logging import pg_log
        >>> pg_log.level = 50
        >>>
        >>> # Build the coordinate system.
        >>> u = SphericalCoordinateSystem()
        >>>
        >>> # Create the R,THETA grid.
        >>> r = np.linspace(1e-3,1,10)
        >>> theta = np.linspace(1e-3,np.pi-1e-3,10)
        >>> R, THETA = np.meshgrid(r,theta)
        >>>
        >>> # Create the vector field.
        >>> V_co = np.stack([R**2,R**2,np.zeros_like(R)],axis=-1)
        >>>
        >>> # Raise the index.
        >>> V_contra = u.raise_index_dense(V_co,0,1,R, THETA, fixed_axes={'phi':0})
        >>> V_contra.shape
        (10, 10, 3)
        >>> np.allclose(V_contra[...,0],R**2)
        True
        >>> np.allclose(V_contra[...,1],np.ones_like(R))
        True

        It is also possible to provide the inverse metric immediately to
        avoid needing to compute it on the fly. This also allows us to
        get away without the coordinates.

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>>
        >>> # Build the coordinate system.
        >>> u = SphericalCoordinateSystem()
        >>>
        >>> # Create the R,THETA grid.
        >>> r = np.linspace(1e-3,1,10)
        >>> theta = np.linspace(1e-3,np.pi-1e-3,10)
        >>> R, THETA = np.meshgrid(r,theta)
        >>>
        >>> # Create the vector field.
        >>> V_co = np.stack([R**2,R**2,np.zeros_like(R)],axis=-1)
        >>>
        >>> # Create the inverse metric tensor
        >>> imt = np.stack([np.ones_like(R),R**-2,R**-2 * np.sin(THETA)**-1],axis=-1)
        >>>
        >>> # Raise the index.
        >>> V_contra = u.raise_index_dense(V_co,0,1, inverse_metric_field=imt, fixed_axes={'phi':0})
        >>> V_contra.shape
        (10, 10, 3)
        >>> np.allclose(V_contra[...,0],R**2)
        True
        >>> np.allclose(V_contra[...,1],np.ones_like(R))
        True
        """
        # lazily obtain the inverse metric if necessary
        inverse_metric_field = self.requires_expression(
            inverse_metric_field,
            "inverse_metric_tensor",
            coordinates,
            fixed_axes=fixed_axes,
        )

        return dense_raise_index(
            tensor_field,
            index,
            rank,
            inverse_metric_field=inverse_metric_field,
            out=out,
            **kwargs,
        )

    def lower_index_dense(
        self,
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        *coordinates: np.ndarray,
        metric_field: Optional[np.ndarray] = None,
        fixed_axes: Optional[Dict[str, float]] = None,
        out: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Lower a single tensor index using :math:`g_{ab}`.

        .. math::

            T_{\mu\dots} = g_{\mu\nu}\,T^{\nu}{}_{\dots}

        Parameters
        ----------
        tensor_field : numpy.ndarray
            Tensor field of shape ``(F₁, ..., F_m, N, ...)``, where the last `rank` axes
            are the tensor index dimensions.
        index
            Which tensor slot (``0, ..., rank-1``) to raise.
        rank : int
            Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
            the number of identified coordinate axes and therefore determines the shape of the returned array.
        *coordinates
            ND coordinate grids **in canonical axis order**.
            Only needed when ``inverse_metric_field`` is *None*.
        metric_field :  numpy.ndarray
            Metric tensor with shape ``(F₁, ..., F_n, N, )`` or ``(F₁, ..., F_n, N, N)``,
            where ``N == n``. Must be broadcast-compatible with the field shape of `tensor_field`.
        fixed_axes
            Constant axis values to use when computing the metric.
        out
            Optional output buffer.
        **kwargs
            Forwarded verbatim to :func:`~differential_geometry.dense_utils.dense_raise_index`.

        Returns
        -------
        numpy.ndarray
            Tensor identical to ``tensor_field`` except the chosen
            slot is now contravariant.

        Examples
        --------
        In spherical coordinates, the vector :math:`V^\mu = (r^2,r^{-2},0)` becomes

        .. math::

            v_\mu = g_{\mu\mu} V^\mu = (r^2, 1, 0).

        To perform the operation computationally,

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> from pymetric.utilities.logging import pg_log
        >>> pg_log.level = 50
        >>>
        >>> # Build the coordinate system.
        >>> u = SphericalCoordinateSystem()
        >>>
        >>> # Create the R,THETA grid.
        >>> r = np.linspace(1e-3,1,10)
        >>> theta = np.linspace(1e-3,np.pi-1e-3,10)
        >>> R, THETA = np.meshgrid(r,theta)
        >>>
        >>> # Create the vector field.
        >>> V_contra = np.stack([R**2,R**-2,np.zeros_like(R)],axis=-1)
        >>>
        >>> # Raise the index.
        >>> V_co = u.lower_index_dense(V_contra,0,1,R, THETA, fixed_axes={'phi':0})
        >>> V_co.shape
        (10, 10, 3)
        >>> np.allclose(V_co[...,0],R**2)
        True
        >>> np.allclose(V_co[...,1],np.ones_like(R))
        True

        It is also possible to provide the inverse metric immediately to
        avoid needing to compute it on the fly. This also allows us to
        get away without the coordinates.

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>>
        >>> # Build the coordinate system.
        >>> u = SphericalCoordinateSystem()
        >>>
        >>> # Create the R,THETA grid.
        >>> r = np.linspace(1e-3,1,10)
        >>> theta = np.linspace(1e-3,np.pi-1e-3,10)
        >>> R, THETA = np.meshgrid(r,theta)
        >>>
        >>> # Create the vector field.
        >>> V_contra = np.stack([R**2,R**-2,np.zeros_like(R)],axis=-1)
        >>>
        >>> # Create the inverse metric tensor
        >>> mt = np.stack([np.ones_like(R),R**2,R**2 * np.sin(THETA)],axis=-1)
        >>>
        >>> # Raise the index.
        >>> V_co = u.raise_index_dense(V_contra,0,1, inverse_metric_field=mt, fixed_axes={'phi':0})
        >>> V_co.shape
        (10, 10, 3)
        >>> np.allclose(V_co[...,0],R**2)
        True
        >>> np.allclose(V_co[...,1],np.ones_like(R))
        True
        """
        # lazily obtain the metric if necessary
        metric_field = self.requires_expression(
            metric_field,
            "metric_tensor",
            coordinates,
            fixed_axes=fixed_axes,
        )

        return dense_lower_index(
            tensor_field,
            index,
            rank,
            metric_field,
            out=out,
            **kwargs,
        )

    def adjust_dense_tensor_signature(
        self,
        tensor_field: np.ndarray,
        indices: Sequence[int],
        tensor_signature: ArrayLike,
        *coordinates: np.ndarray,
        metric_field: Optional[np.ndarray] = None,
        inverse_metric_field: Optional[np.ndarray] = None,
        fixed_axes: Optional[Dict[str, float]] = None,
        out: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Raise and/or lower multiple tensor slots in one call.

        Each entry of ``tensor_signature`` must be ``+1`` (contravariant) or
        ``-1`` (covariant).  Slots listed in *indices* will be flipped
        (``+1 ↔ -1``) using the metric

        .. math::
            g_{ab},\; g^{ab}

        evaluated on the supplied coordinate grid.

        Parameters
        ----------
        tensor_field
            Array with shape ``(F₁,…,F_m, I₁,…,I_rank)`` where the last *rank*
            axes hold the tensor indices.
        indices
            Positions (``0 ≤ i < rank``) of the slots to transform.
        tensor_signature
            Length-``rank`` vector with the current variance of every slot.
        rank
            Number of tensor indices (== ``len(tensor_signature)``).
        *coordinates
            ND broadcasted coordinate grids (canonical axis order).
            Only needed when a metric has to be computed.
        metric_field, inverse_metric_field
            Pre-computed ``g_{ab}`` / ``g^{ab}``.  If omitted they are
            evaluated from *coordinates*.
        fixed_axes
            Constant axis values inserted when computing a metric.
        out
            Optional output buffer.
        **kwargs
            Passed straight through to
            :func:`~pymetric.differential_geometry.dense_adjust_tensor_signature`.

        Returns
        -------
        numpy.ndarray
            ``tensor_field`` with the requested slots flipped.

        Examples
        --------
        Raise slot 0 and lower slot 1 of a rank-2 tensor in cylindrical
        coordinates:

        >>> import numpy as np
        >>> from pymetric.coordinates import CylindricalCoordinateSystem
        >>>
        >>> # Create the coordinate system.
        >>> cs = CylindricalCoordinateSystem()
        >>> r, z = np.linspace(1, 2, 4), np.linspace(-1, 1, 4)
        >>> R, Z = np.meshgrid(r, z, indexing='ij')
        >>>
        >>> # contravariant/covariant signature: (+1, -1)
        >>> T = np.zeros(R.shape + (3, 3))
        >>> T[..., 0, 1] = 1        # only T^{ρ}{}_{z} non-zero
        >>>
        >>> # Create the tensor signature.
        >>> sig = np.array([+1, -1])
        >>> T_new, sig_new = cs.adjust_dense_tensor_signature(
        ...     T, [0, 1],sig, R, Z,
        ...     fixed_axes={'phi': 0.0}
        ... )
        >>> T_new.shape   # unchanged grid-shape + (2,2)
        (4, 4, 3, 3)
        """
        # Only fetch what is really required
        if any(tensor_signature[i] == -1 for i in indices):
            metric_field = self.requires_expression(
                metric_field,
                "metric_tensor",
                coordinates,
                fixed_axes=fixed_axes,
            )

        if any(tensor_signature[i] == +1 for i in indices):
            inverse_metric_field = self.requires_expression(
                inverse_metric_field,
                "inverse_metric_tensor",
                coordinates,
                fixed_axes=fixed_axes,
            )

        return dense_adjust_tensor_signature(
            tensor_field,
            list(indices),
            tensor_signature,
            metric_field=metric_field,
            inverse_metric_field=inverse_metric_field,
            out=out,
            **kwargs,
        )
