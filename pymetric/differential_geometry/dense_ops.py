"""
Differential geometry operations on dense arrays.

These methods provide a differential geometry backend which is designed to interact well
with the "dense" representation of tensor fields in PyMetric; specifically, the use
of full arrays to represent tensor fields.
"""
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .dense_utils import (
    _dense_contract_index_with_diagonal_metric,
    _dense_contract_index_with_metric,
    infer_metric_type,
)


# =================================== #
# Gradient Methods                    #
# =================================== #
# These methods are used to compute the gradient of dense
# tensor fields.
def dense_gradient_covariant(
    tensor_field: np.ndarray,
    rank: int,
    ndim: int,
    *varargs,
    field_axes: Optional[Sequence[int]] = None,
    output_indices: Optional[Sequence[int]] = None,
    edge_order: Literal[1, 2] = 2,
    out: Optional[ArrayLike] = None,
    **_,
) -> np.ndarray:
    r"""
    Compute the element-wise covariant gradient of a tensor field.

    This function computes the raw partial derivatives :math:`\partial_\mu T` of a tensor (or array) field
    with respect to its grid coordinates, treating each tensor component independently.

    It returns the gradient along all spatial axes (or a subset, if specified), storing results
    in the final axis of the output. This is the covariant gradient in the sense of component-wise
    partial derivatives, without applying connection terms (i.e., no Christoffel symbols). It is **not**
    the covariant derivative.

    .. warning::
        This is a low-level utility function. It does **not** validate input shapes or ensure
        coordinate consistency. It assumes that the tensor rank and grid structure are correct.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape ``(F_1, ..., F_m, N, ...)``, where the last `rank` axes
        are the tensor index dimensions. The partial derivatives are computed over the first `m` axes.

        .. hint::

            Because this function is a low-level callable, it does not enforce the density of
            the elements in the trailing dimensions of the field. This can be used in some cases where
            the field is not **technically** a tensor field.

    rank : int
        Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
        the number of identified coordinate axes and therefore determines the shape of the returned array.
    ndim: int
        The number of total dimensions in the relevant coordinate system. This determines the maximum allowed value
        for ``m`` and the number of elements in the trailing dimension of the output.
    *varargs :
        Grid spacing for each spatial axis. Follows the same format as :func:`numpy.gradient`, and can be:

        - A single scalar (applied to all spatial axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `field_axes` is provided, `varargs` must match its length. If `axes` is not provided, then `varargs` should
        be ``tensor_field.ndim - rank`` in length (or be a scalar).
    field_axes : list of int, optional
        The spatial axes over which to compute the component-wise partial derivatives. If `field_axes` is
        not specified, then all ``tensor_field.ndim - rank`` axes are computed.
    output_indices : list of int, optional
        Explicit mapping from each axis listed in `field_axes` to the indices in the output's
        final dimension of size ``ndim``. This parameter allows precise control over the placement
        of computed gradients within the output array.

        If provided, `output_indices` must have the same length as `field_axes`.
        Each computed gradient along the spatial axis ``field_axes[i]`` will be stored at index
        ``output_indices[i]`` in the last dimension of the output.

        If omitted, the default behavior is ``output_indices = field_axes``, i.e., gradients are
        placed in the output slot corresponding to their source axis.

        .. note::
            All positions in the final axis of the output array not listed in ``output_indices``
            are left unmodified (typically zero-filled unless ``out`` was pre-populated).
    edge_order : {1, 2}, optional
        Gradient is calculated using N-th order accurate differences
        at the boundaries. Default: 1.
    out : numpy.ndarray, optional
        Buffer in which to store the output to preserve memory. If provided, `out` must have shape
        ``tensor_field.shape + (ndim,)``. If `out` is not specified, then it is allocated during the
        function call.

    Returns
    -------
    numpy.ndarray

    See Also
    --------
    numpy.gradient: The computational backend for this operation.
    dense_gradient_contravariant_full: Contravariant gradient for full metric.
    dense_gradient_contravariant_diag: Contravariant gradient for diagonal metric.
    dense_gradient: Wrapper for user-facing gradient computations.

    Examples
    --------
    In a simple 1-D case, we can use this function to compute the gradient.

    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_ops import (
    ...     dense_gradient_covariant,
    ... )
    >>>
    >>> x = np.linspace(0, 4, 5)          #  [0, 1, 2, 3, 4]
    >>> f = x**2                          #  [0, 1, 4, 9, 16]
    >>>
    >>> grad_cov = dense_gradient_covariant(f, 0, 1, 1.0)          # dx = 1
    >>> grad_cov
    array([[0.],
           [2.],
           [4.],
           [6.],
           [8.]])

    In a more complicated 2-D case, the call sequence still looks the same.

    .. plot::
        :include-source:

        >>> import numpy as np
        >>> from pymetric.differential_geometry.dense_ops import dense_gradient_covariant
        >>> import matplotlib.pyplot as plt
        >>>
        >>> x = np.linspace(-1,1,100)
        >>> y = np.linspace(-1,1,100)
        >>> X,Y = np.meshgrid(x,y)
        >>> Z = np.sin(X**2 + Y**2)
        >>>
        >>> grad = dense_gradient_covariant(Z,0,2,x,y)
        >>>
        >>> fig,axes = plt.subplots(2,2,sharex=True,sharey=True)
        >>> axes[0,1].set_visible(False)
        >>> _ = axes[0,0].imshow(Z.T, origin='lower', vmin=-1,vmax=1,extent=(-1,1,-1,1))
        >>> _ = axes[1,0].imshow(grad[...,0].T, origin='lower', vmin=-1,vmax=1,extent=(-1,1,-1,1))
        >>> _ = axes[1,1].imshow(grad[...,1].T, origin='lower', vmin=-1,vmax=1,extent=(-1,1,-1,1))
        >>> _ = plt.show()
    """
    # Coerce the axes so that we know how many
    # axes we are actually computing derivatives for.
    # Ensure we don't have an invalid number of axes.
    t_shape, t_ndim = tensor_field.shape, tensor_field.ndim
    if field_axes is None:
        _number_of_axes = t_ndim - rank
        field_axes = tuple(range(t_ndim - rank))
    else:
        _number_of_axes = len(field_axes)

    if _number_of_axes > ndim:
        raise ValueError(
            f"Number of grid axes ({_number_of_axes}) must be <= the "
            f"number of dimensions {ndim}."
        )

    # Setup the output indices.
    if output_indices is None:
        output_indices = np.arange(_number_of_axes)

    # Fix the varargs if the are length 1 so
    # that we never have an issue with broadcasting.
    if len(varargs) == 1:
        varargs *= _number_of_axes

    # Allocate the output array. This should be the
    # same shape as tensor_field but with _number_of_axes elements
    # in an additional
    if out is None:
        out = np.zeros(t_shape + (ndim,), dtype=tensor_field.dtype, order="C")

    # Now iterate through each of the axes and
    # perform the differentiation procedure. We then assign
    # each into the out buffer.
    for _i, (fax, oax) in enumerate(zip(field_axes, output_indices)):
        if t_shape[fax] < edge_order:
            raise ValueError(
                f"Failed to compute a gradient along axis {fax} because its shape was smaller than `edge_order`."
            )
        out[..., oax] = np.gradient(
            tensor_field, varargs[_i], axis=fax, edge_order=edge_order
        )

    return out


def dense_gradient_contravariant_full(
    tensor_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    rank: int,
    ndim: int,
    *varargs,
    field_axes: Optional[Sequence[int]] = None,
    output_indices: Optional[Sequence[int]] = None,
    out: Optional[np.ndarray] = None,
    edge_order: Literal[1, 2] = 2,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the contravariant gradient :math:`\nabla^\mu T^{\dots}` using the inverse metric tensor.

    This function computes the covariant (partial) derivatives of a tensor field and then raises
    the newly introduced index using the inverse metric:

    .. math::

        (\nabla^\mu T^{\dots}) = g^{\mu\nu} \partial_\nu T^{\dots}

    The result has one additional tensor index appended (contravariant index from differentiation).

    .. warning::
        This is a low-level routine and does **not** validate input shapes or metric consistency.
        It assumes all inputs are correctly broadcast and aligned.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape ``(F_1, ..., F_m, N, ...)``, where the last `rank` axes
        are the tensor index dimensions. The partial derivatives are computed over the first `m` axes.

        .. hint::

            Because this function is a low-level callable, it does not enforce the density of
            the elements in the trailing dimensions of the field. This can be used in some cases where
            the field is not **technically** a tensor field.

    inverse_metric_field : numpy.ndarray
        Inverse metric tensor with shape ``(..., ndim, ndim)``,
        where the leading dimensions (denoted here as ``F1, ..., F_n``) must be broadcast-compatible
        with the grid (spatial) portion of `tensor_field`.

        Specifically, if `tensor_field` has shape ``(S1, ..., S_m, I1, ..., I_rank)``,
        then `inverse_metric_field` must be broadcast-compatible with ``(S1, ..., S_m)``.
        The last two dimensions must be exactly ``(ndim, ndim)``, representing the inverse metric
        at each grid point.
    rank : int
        Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
        the number of identified coordinate axes and therefore determines the shape of the returned array.
    ndim: int
        The number of total dimensions in the relevant coordinate system. This determines the maximum allowed value
        for ``m`` and the number of elements in the trailing dimension of the output.
    *varargs :
        Grid spacing for each spatial axis. Follows the same format as :func:`numpy.gradient`, and can be:

        - A single scalar (applied to all spatial axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `field_axes` is provided, `varargs` must match its length. If `axes` is not provided, then `varargs` should
        be ``tensor_field.ndim - rank`` in length (or be a scalar).
    field_axes : list of int, optional
        The spatial axes over which to compute the component-wise partial derivatives. If `field_axes` is
        not specified, then all ``tensor_field.ndim - rank`` axes are computed.
    output_indices : list of int, optional
        Explicit mapping from each axis listed in `field_axes` to the indices in the output's
        final dimension of size ``ndim``. This parameter allows precise control over the placement
        of computed gradients within the output array.

        If provided, `output_indices` must have the same length as `field_axes`.
        Each computed gradient along the spatial axis ``field_axes[i]`` will be stored at index
        ``output_indices[i]`` in the last dimension of the output.

        If omitted, the default behavior is ``output_indices = field_axes``, i.e., gradients are
        placed in the output slot corresponding to their source axis.

        .. note::
            All positions in the final axis of the output array not listed in ``output_indices``
            are left unmodified (typically zero-filled unless ``out`` was pre-populated).
    edge_order : {1, 2}, optional
        Gradient is calculated using N-th order accurate differences
        at the boundaries. Default: 1.
    out : numpy.ndarray, optional
        Buffer in which to store the output to preserve memory. If provided, `out` must have shape
        ``tensor_field.shape + (ndim,)``. If `out` is not specified, then it is allocated during the
        function call.
    **kwargs :
        Additional keyword arguments passed to :func:`numpy.einsum` for the metric contraction.

    Returns
    -------
    numpy.ndarray
        The contravariant gradient of the input tensor field, with shape ``B + (ndim,)``,
        where ``B`` is the broadcasted shape of the spatial portion of `tensor_field`
        and `inverse_metric_field`.

        This result represents the gradient with the differentiation index raised by the inverse metric,
        and the final axis of size ``ndim`` corresponds to the contravariant coordinate direction
        of the derivative at each point.


    See Also
    --------
    numpy.gradient: The computational backend for this operation.
    dense_gradient_covariant: Contravariant gradient for full metric.
    dense_gradient_contravariant_diag: Contravariant gradient for diagonal metric.
    dense_gradient: Wrapper for user-facing gradient computations.

    Notes
    -----
    In effect, this function calls :func:`dense_gradient_covariant` to compute the covariant gradient
    and then contracts the result with the metric.

    Examples
    --------
    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_ops import (
    ...     dense_gradient_contravariant_full,
    ... )
    >>>
    >>> x = np.linspace(0, 4, 5)          #  [0, 1, 2, 3, 4]
    >>> f = x**2                          #  [0, 1, 4, 9, 16]
    >>>
    >>> inv_metric_full = np.ones((f.size, 1, 1))   # shape broadcast‑compatible with grad_cov
    >>> grad_contra_full = dense_gradient_contravariant_full(
    ...     f, inv_metric_full,0 , 1, 1.0)
    >>> grad_contra_full
    array([[0.],
           [2.],
           [4.],
           [6.],
           [8.]])

    As a more advanced example, let's look at the co. versus contra. components of a gradient case:

    .. plot::
        :include-source:

        >>> from scipy.interpolate import RegularGridInterpolator
        >>> from pymetric.differential_geometry.dense_ops import (dense_gradient_contravariant_full,
        ... dense_gradient_covariant)
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Create the coordinates, the field, and the
        >>> # interpolators.
        >>> r = np.linspace(1e-4,1,1000)
        >>> theta = np.linspace(0,np.pi,30)
        >>> R,THETA = np.meshgrid(r,theta,indexing='ij')
        >>> Z = np.sin(10*R) * np.cos(THETA)**2
        >>>
        >>> # Build a 1D interpolator for Z.
        >>> interpZ = RegularGridInterpolator((r,theta),Z,bounds_error=False)
        >>>
        >>> # Compute the gradients of Z
        >>> gradZ = dense_gradient_covariant(Z,0,2,r,theta)
        >>>
        >>> # Create the metric
        >>> metric = np.zeros(R.shape + (2,2))
        >>> metric[:,:,0,0] = 1
        >>> metric[:,:,1,1] = 1/R**2
        >>>
        >>> # Compute the contravariant gradient.
        >>> gradZcontra = dense_gradient_contravariant_full(Z,metric,0,2,r,theta)
        >>>
        >>> # Build interpolators for the 2 gradient components.
        >>> interpZr = RegularGridInterpolator((r,theta),gradZ[...,0],bounds_error=False)
        >>> interpZtheta = RegularGridInterpolator((r,theta),gradZ[...,1],bounds_error=False)
        >>> interpCZr = RegularGridInterpolator((r,theta),gradZcontra[...,0],bounds_error=False)
        >>> interpCZtheta = RegularGridInterpolator((r,theta),gradZcontra[...,1],bounds_error=False)
        >>>
        >>> # Construct an X/Y grid.
        >>> bound = 1/np.sqrt(2)
        >>> x,y = np.linspace(-bound,bound,100),np.linspace(-bound,bound,100)
        >>> X,Y = np.meshgrid(x,y,indexing='ij')
        >>> RG = np.sqrt(X**2+Y**2)
        >>> THETAG = np.arccos(Y/RG)
        >>> grid_points = np.stack([RG.ravel(),THETAG.ravel()],axis=1)
        >>> Zgrid = interpZ(grid_points).reshape(RG.shape)
        >>> Zrgrid = interpZr(grid_points).reshape(RG.shape)
        >>> Zthetagrid = interpZtheta(grid_points).reshape(RG.shape)
        >>> ZCrgrid = interpCZr(grid_points).reshape(RG.shape)
        >>> ZCthetagrid = interpCZtheta(grid_points).reshape(RG.shape)

        >>> # Setup the figure.
        >>> fig,axes = plt.subplots(2,2, sharex=True, sharey=True)
        >>> _ = axes[0,0].imshow(Zrgrid.T    ,extent=[-bound,bound,-bound,bound], vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> _ = axes[0,1].imshow(Zthetagrid.T,extent=[-bound,bound,-bound,bound], vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> _ = axes[1,0].imshow(ZCrgrid.T    ,extent=[-bound,bound,-bound,bound],vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> P = axes[1,1].imshow(ZCthetagrid.T,extent=[-bound,bound,-bound,bound],vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> _ = plt.colorbar(P,ax=axes)
        >>> plt.show()
    """
    # Determine the broadcasted base shape (excluding trailing index dimensions)
    tensor_base_shape = tensor_field.shape
    metric_base_shape = inverse_metric_field.shape[:-2]  # exclude (ndim, ndim)
    try:
        broadcast_shape = np.broadcast_shapes(tensor_base_shape, metric_base_shape)
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast tensor_field.shape={tensor_base_shape} "
            f"with inverse_metric_field.shape={metric_base_shape}."
        ) from e

    # Allocate output buffer if not provided
    if out is None:
        out = np.zeros(broadcast_shape + (ndim,), dtype=tensor_field.dtype)
    else:
        expected_shape = broadcast_shape + (ndim,)
        if out.shape != expected_shape:
            raise ValueError(
                f"Provided `out` has shape {out.shape}, but expected {expected_shape} "
                f"from broadcasting tensor field and inverse metric."
            )

    # Compute the covariant gradient into the broadcasted output buffer
    dense_gradient_covariant(
        tensor_field,
        rank,
        ndim,
        *varargs,
        field_axes=field_axes,
        output_indices=output_indices,
        edge_order=edge_order,
        out=out,
    )

    # Contract the covariant gradient with the inverse metric to raise the index
    return _dense_contract_index_with_metric(
        out,
        inverse_metric_field,
        index=rank,
        rank=rank + 1,
        out=out,
        **kwargs,
    )


def dense_gradient_contravariant_diag(
    tensor_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    rank: int,
    ndim: int,
    *varargs,
    field_axes: Optional[Sequence[int]] = None,
    output_indices: Optional[Sequence[int]] = None,
    out: Optional[np.ndarray] = None,
    edge_order: Literal[1, 2] = 2,
    **_,
) -> np.ndarray:
    r"""
    Compute the contravariant gradient :math:`\nabla^\mu T^{\dots}` of a tensor field
    using a diagonal inverse metric tensor.

    This function computes the covariant partial derivatives over the field dimensions,
    then raises the newly introduced index using the provided diagonal inverse metric:

    .. math::

        \nabla^\mu T^{\dots} = g^{\mu\mu} \partial_\mu T^{\dots}

    No summation is implied since the metric is diagonal.

    .. warning::
        This is a low-level routine and does **not** validate input shapes or consistency.
        It assumes all inputs are correctly broadcast and aligned.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape ``(F_1, ..., F_m, N, ...)``, where the last `rank` axes
        are the tensor index dimensions. The partial derivatives are computed over the first `m` axes.

        .. hint::

            Because this function is a low-level callable, it does not enforce the density of
            the elements in the trailing dimensions of the field. This can be used in some cases where
            the field is not **technically** a tensor field.

    inverse_metric_field : numpy.ndarray
        Inverse metric tensor with shape ``(..., ndim, )``,
        where the leading dimensions (denoted here as ``F1, ..., F_n``) must be broadcast-compatible
        with the grid (spatial) portion of `tensor_field`.

        Specifically, if `tensor_field` has shape ``(S1, ..., S_m, I1, ..., I_rank)``,
        then `inverse_metric_field` must be broadcast-compatible with ``(S1, ..., S_m)``.
        The last dimension must be exactly ``(ndim,)``, representing the inverse metric
        at each grid point.
    rank : int
        Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
        the number of identified coordinate axes and therefore determines the shape of the returned array.
    ndim: int
        The number of total dimensions in the relevant coordinate system. This determines the maximum allowed value
        for ``m`` and the number of elements in the trailing dimension of the output.
    *varargs :
        Grid spacing for each spatial axis. Follows the same format as :func:`numpy.gradient`, and can be:

        - A single scalar (applied to all spatial axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `field_axes` is provided, `varargs` must match its length. If `axes` is not provided, then `varargs` should
        be ``tensor_field.ndim - rank`` in length (or be a scalar).
    field_axes : list of int, optional
        The spatial axes over which to compute the component-wise partial derivatives. If `field_axes` is
        not specified, then all ``tensor_field.ndim - rank`` axes are computed.
    output_indices : list of int, optional
        Explicit mapping from each axis listed in `field_axes` to the indices in the output's
        final dimension of size ``ndim``. This parameter allows precise control over the placement
        of computed gradients within the output array.

        If provided, `output_indices` must have the same length as `field_axes`.
        Each computed gradient along the spatial axis ``field_axes[i]`` will be stored at index
        ``output_indices[i]`` in the last dimension of the output.

        If omitted, the default behavior is ``output_indices = field_axes``, i.e., gradients are
        placed in the output slot corresponding to their source axis.

        .. note::
            All positions in the final axis of the output array not listed in ``output_indices``
            are left unmodified (typically zero-filled unless ``out`` was pre-populated).
    edge_order : {1, 2}, optional
        Gradient is calculated using N-th order accurate differences
        at the boundaries. Default: 1.
    out : numpy.ndarray, optional
        Buffer in which to store the output to preserve memory. If provided, `out` must have shape
        ``tensor_field.shape + (ndim,)``. If `out` is not specified, then it is allocated during the
        function call.

    Returns
    -------
    numpy.ndarray
        The contravariant gradient of the input tensor field, with shape ``B + (ndim,)``,
        where ``B`` is the broadcasted shape of the spatial portion of `tensor_field`
        and `inverse_metric_field`.

        This result represents the gradient with the differentiation index raised by the inverse metric,
        and the final axis of size ``ndim`` corresponds to the contravariant coordinate direction
        of the derivative at each point.

    See Also
    --------
    numpy.gradient: The computational backend for this operation.
    dense_gradient_covariant: Contravariant gradient for full metric.
    dense_gradient_contravariant_full: Contravariant gradient for full metric.
    dense_gradient: Wrapper for user-facing gradient computations.

    Examples
    --------
    As a more advanced example, let's look at the co. versus contra. components of a gradient case:

    .. plot::
        :include-source:

        >>> # ------------------------------------------------------------------
        >>> # 0)  Set up a toy scalar field  f(r) = sin(r/r_0) cos^2(theta)
        >>> # ------------------------------------------------------------------
        >>> from scipy.interpolate import RegularGridInterpolator
        >>> from pymetric.differential_geometry.dense_ops import (dense_gradient_contravariant_diag,
        ... dense_gradient_covariant)
        >>> import matplotlib.pyplot as plt
        >>> r = np.linspace(1e-4,1,1000)
        >>> theta = np.linspace(0,np.pi,30)
        >>> R,THETA = np.meshgrid(r,theta,indexing='ij')
        >>> Z = np.sin(10*R) * np.cos(THETA)**2
        >>>
        >>> # Build a 1D interpolator for Z.
        >>> interpZ = RegularGridInterpolator((r,theta),Z,bounds_error=False)
        >>>
        >>> # Compute the gradients of Z
        >>> gradZ = dense_gradient_covariant(Z,0,2,r,theta)
        >>>
        >>> # Create the metric
        >>> metric = np.zeros(R.shape + (2,))
        >>> metric[:,:,0,] = 1
        >>> metric[:,:,1,] = 1/R**2
        >>>
        >>> # Compute the contravariant gradient.
        >>> gradZcontra = dense_gradient_contravariant_diag(Z,metric,0,2,r,theta)
        >>>
        >>> # Build interpolators for the 2 gradient components.
        >>> interpZr = RegularGridInterpolator((r,theta),gradZ[...,0],bounds_error=False)
        >>> interpZtheta = RegularGridInterpolator((r,theta),gradZ[...,1],bounds_error=False)
        >>> interpCZr = RegularGridInterpolator((r,theta),gradZcontra[...,0],bounds_error=False)
        >>> interpCZtheta = RegularGridInterpolator((r,theta),gradZcontra[...,1],bounds_error=False)
        >>>
        >>> # Construct an X/Y grid.
        >>> bound = 1/np.sqrt(2)
        >>> x,y = np.linspace(-bound,bound,100),np.linspace(-bound,bound,100)
        >>> X,Y = np.meshgrid(x,y,indexing='ij')
        >>> RG = np.sqrt(X**2+Y**2)
        >>> THETAG = np.arccos(Y/RG)
        >>> grid_points = np.stack([RG.ravel(),THETAG.ravel()],axis=1)
        >>> Zgrid = interpZ(grid_points).reshape(RG.shape)
        >>> Zrgrid = interpZr(grid_points).reshape(RG.shape)
        >>> Zthetagrid = interpZtheta(grid_points).reshape(RG.shape)
        >>> ZCrgrid = interpCZr(grid_points).reshape(RG.shape)
        >>> ZCthetagrid = interpCZtheta(grid_points).reshape(RG.shape)

        >>> # Setup the figure.
        >>> fig,axes = plt.subplots(2,2, sharex=True, sharey=True)
        >>> _ = axes[0,0].imshow(Zrgrid.T    ,extent=[-bound,bound,-bound,bound], vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> _ = axes[0,1].imshow(Zthetagrid.T,extent=[-bound,bound,-bound,bound], vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> _ = axes[1,0].imshow(ZCrgrid.T    ,extent=[-bound,bound,-bound,bound],vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> P = axes[1,1].imshow(ZCthetagrid.T,extent=[-bound,bound,-bound,bound],vmin=-3,vmax=3,cmap='seismic',origin='lower')
        >>> _ = plt.colorbar(P,ax=axes)
        >>> plt.show()
    """
    # Determine the broadcasted base shape (excluding trailing index dimensions)
    tensor_base_shape = tensor_field.shape
    metric_base_shape = inverse_metric_field.shape[:-1]  # exclude (ndim, ndim)
    try:
        broadcast_shape = np.broadcast_shapes(tensor_base_shape, metric_base_shape)
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast tensor_field.shape={tensor_base_shape} "
            f"with inverse_metric_field.shape={metric_base_shape}."
        ) from e

    # Allocate output buffer if not provided
    if out is None:
        out = np.zeros(broadcast_shape + (ndim,), dtype=tensor_field.dtype)
    else:
        expected_shape = broadcast_shape + (ndim,)
        if out.shape != expected_shape:
            raise ValueError(
                f"Provided `out` has shape {out.shape}, but expected {expected_shape} "
                f"from broadcasting tensor field and inverse metric."
            )

    # Compute the covariant gradient into the broadcasted output buffer
    dense_gradient_covariant(
        tensor_field,
        rank,
        ndim,
        *varargs,
        field_axes=field_axes,
        output_indices=output_indices,
        edge_order=edge_order,
        out=out,
    )

    # Contract the covariant gradient with the inverse metric to raise the index
    return _dense_contract_index_with_diagonal_metric(
        out,
        inverse_metric_field,
        index=rank,
        rank=rank + 1,
        out=out,
    )


def dense_gradient(
    tensor_field: np.ndarray,
    rank: int,
    ndim: int,
    *varargs,
    basis: Optional[Literal["contravariant", "covariant"]] = "covariant",
    inverse_metric_field: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
    edge_order: Literal[1, 2] = 2,
    field_axes: Optional[Sequence[int]] = None,
    output_indices: Optional[Sequence[int]] = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the gradient of a tensor field in the specified output basis.

    This function computes the component-wise partial derivatives of a tensor field with respect
    to its grid coordinates, and optionally raises the resulting index using a provided inverse metric.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape ``(F_1, ..., F_m, N, ...)``, where the last `rank` axes
        are the tensor index dimensions. The partial derivatives are computed over the first `m` axes.

        .. hint::

            Because this function is a low-level callable, it does not enforce the density of
            the elements in the trailing dimensions of the field. This can be used in some cases where
            the field is not **technically** a tensor field.

    rank : int
        Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
        the number of identified coordinate axes and therefore determines the shape of the returned array.
    ndim: int
        The number of total dimensions in the relevant coordinate system. This determines the maximum allowed value
        for ``m`` and the number of elements in the trailing dimension of the output.
    *varargs :
        Grid spacing for each spatial axis. Follows the same format as :func:`numpy.gradient`, and can be:

        - A single scalar (applied to all spatial axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `field_axes` is provided, `varargs` must match its length. If `axes` is not provided, then `varargs` should
        be ``tensor_field.ndim - rank`` in length (or be a scalar).
    inverse_metric_field : numpy.ndarray
        Inverse metric tensor with shape ``(..., ndim, )`` or ``(..., ndim, ndim)``,
        where the leading dimensions (denoted here as ``F1, ..., F_n``) must be broadcast-compatible
        with the grid (spatial) portion of `tensor_field`.

        Specifically, if `tensor_field` has shape ``(S1, ..., S_m, I1, ..., I_rank)``,
        then `inverse_metric_field` must be broadcast-compatible with ``(S1, ..., S_m)``.
        The last two dimensions must be exactly ``(ndim, ndim)``, representing the inverse metric
        at each grid point.
    basis: {'contravariant', 'covariant'}, optional
        The basis in which to compute the result of the operation.
    field_axes : list of int, optional
        The spatial axes over which to compute the component-wise partial derivatives. If `field_axes` is
        not specified, then all ``tensor_field.ndim - rank`` axes are computed.
    output_indices : list of int, optional
        Explicit mapping from each axis listed in `field_axes` to the indices in the output's
        final dimension of size ``ndim``. This parameter allows precise control over the placement
        of computed gradients within the output array.

        If provided, `output_indices` must have the same length as `field_axes`.
        Each computed gradient along the spatial axis ``field_axes[i]`` will be stored at index
        ``output_indices[i]`` in the last dimension of the output.

        If omitted, the default behavior is ``output_indices = field_axes``, i.e., gradients are
        placed in the output slot corresponding to their source axis.

        .. note::
            All positions in the final axis of the output array not listed in ``output_indices``
            are left unmodified (typically zero-filled unless ``out`` was pre-populated).
    edge_order : {1, 2}, optional
        Gradient is calculated using N-th order accurate differences
        at the boundaries. Default: 1.
    out : numpy.ndarray, optional
        Buffer in which to store the output to preserve memory. If provided, `out` must have shape
        ``tensor_field.shape + (ndim,)``. If `out` is not specified, then it is allocated during the
        function call.
    **kwargs :
        Additional keyword arguments passed to contraction routines.

    Returns
    -------
    ~numpy.ndarray
        Gradient of the tensor field with shape ``tensor_field.shape + (N,)``.

    Raises
    ------
    ValueError
        If input shapes are inconsistent, required metric is missing, or basis is invalid.

    Examples
    --------
    As a demonstration of the difference between covariant and contravariant gradients, let's consider the
    gradient in spherical coordinates using the scalar function:

    .. math::

        f(r, \theta) = r^2 \sin(\theta)

    The gradient is computed in both covariant and contravariant bases. In
    spherical coordinates, the metric tensor is diagonal:

    .. math::

        g_{rr} = 1, \quad g_{\theta\theta} = r^2

    and the inverse metric is:

    .. math::

        g^{rr} = 1, \quad g^{\theta\theta} = \frac{1}{r^2}

    Therefore, the covariant gradient is:

    .. math::

        \nabla_i f = \left( \frac{\partial f}{\partial r}, \frac{\partial f}{\partial \theta} \right)

    and the contravariant gradient is obtained by raising the index:

    .. math::

        \nabla^i f = g^{ij} \nabla_j f =
        \left( \frac{\partial f}{\partial r}, \frac{1}{r^2} \frac{\partial f}{\partial \theta} \right)

    .. plot::
        :include-source: True

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pymetric.differential_geometry.dense_ops import dense_gradient
        >>>
        >>> # Create spherical grid
        >>> r = np.linspace(0.01, 1.0, 100)
        >>> theta = np.linspace(0, np.pi, 100)
        >>> R, THETA = np.meshgrid(r, theta, indexing='ij')
        >>>
        >>> # Define scalar field f(r, theta) = r^2 * sin(theta)
        >>> F = R**2 * np.sin(THETA)
        >>>
        >>> # Define inverse metric for spherical coordinates
        >>> IM = np.zeros(R.shape + (2,))
        >>> IM[..., 0] = 1            # g^rr = 1
        >>> IM[..., 1] = 1 / R**2     # g^thetatheta = 1/r^2
        >>>
        >>> # Compute gradients
        >>> grad_cov = dense_gradient(F, 0, 2, r, theta, basis='covariant', edge_order=2)
        >>> grad_contra = dense_gradient(F, 0, 2, r, theta, basis='contravariant', inverse_metric_field=IM, edge_order=2)
        >>>
        >>> # Visualize theta component (index 1) of both gradients
        >>> fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        >>>
        >>> im0 = axes[0].imshow(grad_cov[..., 1].T, origin='lower', extent=[0.01, 1.0, 0, np.pi], aspect='auto')
        >>> _ = axes[0].set_title(r'Covariant Gradient $(\partial_\theta f)$')
        >>> _ = fig.colorbar(im0, ax=axes[0])
        >>>
        >>> im1 = axes[1].imshow(grad_contra[..., 1].T, origin='lower', extent=[0.01, 1.0, 0, np.pi], aspect='auto')
        >>> _ = axes[1].set_title(r'Contravariant Gradient $(r^{-2} \; \partial_\theta f)$')
        >>> _ = fig.colorbar(im1, ax=axes[1])
        >>>
        >>> for ax in axes:
        ...     _ = ax.set_xlabel("r")
        ...     _ = ax.set_ylabel("theta")
        >>>
        >>> plt.tight_layout()
        >>> plt.show()
    """
    tensor_shape = tensor_field.shape
    tensor_ndim = tensor_field.ndim

    if rank > tensor_ndim:
        raise ValueError(
            "Tensor rank cannot exceed the number of dimensions in the array."
        )

    # Distinguish the basis and proceed to the low-level callable
    # depending on which basis is specified.
    if basis == "covariant":
        try:
            return dense_gradient_covariant(
                tensor_field,
                rank,
                ndim,
                *varargs,
                edge_order=edge_order,
                out=out,
                field_axes=field_axes,
                output_indices=output_indices,
            )
        except Exception as e:
            raise ValueError(f"Failed to compute covariant gradient: {e}") from e
    elif basis == "contravariant":
        # Check that the inverse_metric_field is specified before
        # proceeding.
        if inverse_metric_field is None:
            raise ValueError(
                "`inverse_metric_field` must be provided when `basis='contravariant'`."
            )

        # Determine metric type based on shape compatibility
        spatial_shape = tensor_shape[: tensor_ndim - rank]
        metric_type = infer_metric_type(inverse_metric_field, spatial_shape)

        if metric_type == "full":
            return dense_gradient_contravariant_full(
                tensor_field,
                inverse_metric_field,
                rank,
                ndim,
                *varargs,
                edge_order=edge_order,
                out=out,
                field_axes=field_axes,
                output_indices=output_indices,
                **kwargs,
            )
        elif metric_type == "diagonal":
            return dense_gradient_contravariant_diag(
                tensor_field,
                inverse_metric_field,
                rank,
                ndim,
                *varargs,
                edge_order=edge_order,
                out=out,
                field_axes=field_axes,
                output_indices=output_indices,
                **kwargs,
            )
        else:
            raise ValueError(f"Unrecognized metric type: {metric_type}")

    else:
        raise ValueError(
            f"`basis` must be 'covariant' or 'contravariant', not '{basis}'."
        )


# =================================== #
# Divergence Methods                  #
# =================================== #
# These methods are used to compute the divergence of dense
# tensor fields.
def dense_vector_divergence_contravariant(
    vector_field: np.ndarray,
    Dterm_field: np.ndarray,
    *varargs,
    derivative_field: Optional[np.ndarray] = None,
    field_axes: Optional[Sequence[int]] = None,
    derivative_axes: Optional[Sequence[int]] = None,
    edge_order: Literal[1, 2] = 2,
    out: Optional[np.ndarray] = None,
    **_,
) -> np.ndarray:
    r"""
    Compute the divergence of a contravariant vector field in a general coordinate system using
    the D-terms and raw partial derivatives.

    This implements the formula:

    .. math::

        \nabla_i V^i = D_i V^i + \partial_i V^i

    where :math:`D_i = (\partial_i \rho) / \rho` is the logarithmic derivative of the coordinate density.

    Parameters
    ----------
    vector_field : numpy.ndarray
        Contravariant vector field with shape ``(F_1, ..., F_M, ndim)``, where the final axis corresponds to coordinate
        directions. The first ``m`` axes are spatial grid axes. Must be broadcast-compatible with `Dterm_field`.
    Dterm_field : numpy.ndarray
        D-term array of shape ``(..., ndim)``, where the last axis matches the number of coordinate directions.
        Must broadcast with the spatial axes of `vector_field`.
    *varargs :
        Grid spacing for each axis. Follows the same format as :func:`numpy.gradient`, and can be:

        - A single scalar (applied to all spatial axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `derivative_axes` is provided, then `varargs` must match its shape. Otherwise, there must be ``m`` elements
        in `varargs`.
    derivative_field : numpy.ndarray, optional
        Optional array of precomputed partial derivatives of selected vector components.
        Must have shape broadcast-compatible with the spatial shape of `vector_field`, and a final axis indexing
        the selected derivative components (same length as `derivative_axes`).
    field_axes : list of int, optional
        Maps each grid axis (0 to ``m-1``) to a corresponding component index in the vector field.
        Defaults to identity mapping ``[0, 1, ..., m-1]``.
    derivative_axes : list of int, optional
        Grid axes along which to compute partial derivatives. If not specified, all spatial axes
        listed in `field_axes` are used.
    edge_order : {1, 2}, default=2
        Accuracy order of finite differences used in derivative computation.
    out : numpy.ndarray, optional
        Optional output buffer for storing the result. Must have shape equal to the broadcast of
        the grid (non-component) dimensions of `vector_field` and `Dterm_field`.

    Returns
    -------
    numpy.ndarray
        A scalar field representing the divergence, with shape equal to the broadcasted grid shape
        of `vector_field` and `Dterm_field` (excluding the final component axis).


    Raises
    ------
    ValueError
        If `axes` length does not match the spatial rank of the input.

    Examples
    --------
    In the most basic case, we only need to supply the D-field and the vector field.

    >>> import numpy as np
    >>>
    >>> # Build a field.
    >>> x,y = np.linspace(-1,1,5),np.linspace(-1,1,5)
    >>> X,Y = np.meshgrid(x,y,indexing='ij')
    >>> V = np.stack([np.ones_like(X),Y],axis=-1)
    >>>
    >>> # Create cartesian Dfield.
    >>> D = np.zeros_like(V)
    >>>
    >>> # Compute the divergence.
    >>> DivV = dense_vector_divergence_contravariant(V,D,x,y)
    >>> np.all(DivV == 1.0)
    True

    In some cases, the D term and the vector field might only be broadcastable. This
    is perfectly fine, but the fields must be broadcastable already! For example, this will
    fail:

    >>> import numpy as np
    >>>
    >>> # Build a field.
    >>> x,y = np.linspace(-1,1,6),np.linspace(-1,1,5)
    >>> X,Y = np.meshgrid(x,y,indexing='ij')
    >>> V = np.stack([np.ones_like(y),y],axis=-1)
    >>> V.shape
    (5, 2)
    >>>
    >>> # Create cartesian Dfield.
    >>> D = np.zeros(x.shape + (2,))
    >>>
    >>> # Compute the divergence.
    >>> DivV = dense_vector_divergence_contravariant(V,D,x,y) # doctest: +SKIP
    ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (5,) and arg 1 with shape (6,).

    This will succeed:

    >>> import numpy as np
    >>>
    >>> # Build a field.
    >>> x,y = np.linspace(-1,1,6),np.linspace(-1,1,5)
    >>> X,Y = np.meshgrid(x,y,indexing='ij')
    >>> V = np.stack([np.ones_like(y),y],axis=-1)[None,:,:]
    >>> V.shape
    (1, 5, 2)
    >>>
    >>> # Create cartesian Dfield.
    >>> D = np.zeros(x.shape + (2,))[:,None,:]
    >>> D.shape
    (6, 1, 2)
    >>>
    >>> # Compute the divergence.
    >>> DivV = dense_vector_divergence_contravariant(V,D,y,derivative_axes=[1])
    >>> np.all(DivV == 1.0)
    True

    The ``derivative_axes=[1]`` tells :func:`dense_vector_divergence_contravariant` not to
    compute a derivative over the `x` axis (its singleton).
    """
    # Allocating the output array. This requires computing the broadcasted
    # shape if it is not pre-specified so that we don't try to assign to singletons.
    broadcast_shape = np.broadcast_shapes(
        vector_field.shape[:-1], Dterm_field.shape[:-1]
    )
    grid_dimension = vector_field.ndim - 1

    if out is None:
        out = np.zeros(broadcast_shape, dtype=vector_field.dtype, order="C")

    # Set the field axes and the derivative axes.
    if field_axes is None:
        field_axes = np.arange(grid_dimension)
    if derivative_axes is None:
        derivative_axes = field_axes

    # Correct varargs so that we can treat it as an iterable if a scalar is
    # provided.
    if len(varargs) == 1:
        varargs *= len(field_axes)

    # Begin the computation of the divergence using the
    # two-term approach. We start by computing the contraction over
    # the Dterm and the vector field and placing it in out.
    np.sum(Dterm_field * vector_field, axis=-1, out=out)

    # We now need to complete the second term by computing
    # each of the derivatives and placing them into the buffer.
    # If they are pre-computed, then we can skip over this step.
    if derivative_field is not None:
        # We have precomputed derivatives. We need the broadcasting wrapper
        # because broadcasting doesn't work with inplace arithmetic.
        out += np.broadcast_to(np.sum(derivative_field, axis=-1), broadcast_shape)
    else:
        # We need to compute the derivatives. We have `derivative_axes` specifying
        # which axes we take derivatives of. For each, `field_axes` tells us which element
        # we want.

        # Iterate through each pairing and compute the derivative.
        for i, diff_index in enumerate(derivative_axes):
            field_axis = field_axes[diff_index]
            out += np.broadcast_to(
                np.gradient(
                    vector_field[..., field_axis],
                    varargs[i],
                    axis=diff_index,
                    edge_order=edge_order,
                ),
                broadcast_shape,
            )

    return out


def dense_vector_divergence_covariant_full(
    vector_field: np.ndarray,
    Dterm_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    *varargs,
    field_axes: Optional[Sequence[int]] = None,
    derivative_axes: Optional[Sequence[int]] = None,
    edge_order: Literal[1, 2] = 2,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the divergence of a covariant vector field in a general coordinate system using
    a full inverse metric tensor.

    This function converts the covariant vector field to its contravariant form by contracting
    with the inverse metric, and then computes the divergence using:

    .. math::

        \nabla_i V^i = D_i V^i + \partial_i V^i

    where:

    - :math:`V^i = g^{ij} V_j` is the contravariant form of the input covariant field.
    - :math:`D_i = (\partial_i \rho) / \rho` is the logarithmic derivative of the volume density.
    - :math:`\nabla_i V^i` is the full covariant divergence in curved space.

    Parameters
    ----------
    vector_field : numpy.ndarray
        Contravariant vector field with shape ``(F_1, ..., F_M, ndim)``, where the final axis corresponds to coordinate
        directions. The first ``m`` axes are spatial grid axes. Must be broadcast-compatible with `Dterm_field`.
    Dterm_field : numpy.ndarray
        D-term array of shape ``(..., ndim)``, where the last axis matches the number of coordinate directions.
        Must broadcast with the spatial axes of `vector_field`.
    inverse_metric_field :  numpy.ndarray
        Inverse metric tensor with shape (..., N, N). This is used to raise the index of the covariant field.
    *varargs :
        Grid spacing for each axis. Follows the same format as :func:`numpy.gradient`, and can be:

        - A single scalar (applied to all spatial axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `derivative_axes` is provided, then `varargs` must match its shape. Otherwise, there must be ``M`` elements
        in `varargs`.
    field_axes : list of int, optional
        Maps each grid axis (0 to ``m-1``) to a corresponding component index in the vector field.
        Defaults to identity mapping ``[0, 1, ..., m-1]``.
    derivative_axes : list of int, optional
        Grid axes along which to compute partial derivatives. If not specified, all spatial axes
        listed in `field_axes` are used.
    edge_order : {1, 2}, default=2
        Accuracy order of finite differences used in derivative computation.
    out :  numpy.ndarray, optional
        Optional output buffer into which the result is placed. Specifying `out` can help to conserve
        memory.

        `out` should be specified so that it is the broadcast shape of `vector_field` and `Dterm_field` excluding
        the final (component) dimension of each. Thus, if `vector_field` is ``(A,B,1,3)`` and `Dterm_field` is ``(1,B,C,3)``,
        `out` should be ``(A,B,C)``.
    **kwargs :
        Additional keyword arguments passed to the metric contraction routine.

    Returns
    -------
    ~numpy.ndarray
        Divergence of the covariant vector field, with shape `vector_field.shape[:-1]`.

    See Also
    --------
    dense_vector_divergence_contravariant : Performs divergence on contravariant fields.
    ~differential_geometry.dense_utils.dense_contract_with_metric : Raises the index of the vector field via contraction.
    """
    _contra_vec_field_ = _dense_contract_index_with_metric(
        vector_field, inverse_metric_field, 0, 1, **kwargs
    )
    return dense_vector_divergence_contravariant(
        _contra_vec_field_,
        Dterm_field,
        *varargs,
        derivative_axes=derivative_axes,
        field_axes=field_axes,
        edge_order=edge_order,
        out=out,
    )


def dense_vector_divergence_covariant_diag(
    vector_field: np.ndarray,
    Dterm_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    *varargs,
    field_axes: Optional[Sequence[int]] = None,
    derivative_axes: Optional[Sequence[int]] = None,
    edge_order: Literal[1, 2] = 2,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the divergence of a covariant vector field in a general curvilinear coordinate system
    using a **diagonal** inverse metric tensor.

    This function raises the index of the covariant field using a diagonal inverse metric,
    and then computes the divergence of the resulting contravariant vector field. The method
    implements:

    .. math::

        \nabla_\mu V^\mu = D_\mu V^\mu + \partial_\mu V^\mu

    where:

    - :math:`V^\mu = g^{\mu\mu} V_\mu` is the contravariant vector field obtained by index raising,
    - :math:`D_\mu = \frac{\partial_\mu \rho}{\rho}` is the logarithmic derivative of the volume element,
    - The divergence is computed as the sum of the D-term contraction and the raw partial derivatives.

    This is a low-level routine and assumes input arrays are already correctly aligned and broadcast-compatible.

    Parameters
    ----------
    vector_field : numpy.ndarray
        Covariant vector field of shape ``(F1, ..., Fm, ndim)``, where the final axis indexes vector components,
        and the leading ``m`` axes define the spatial grid dimensions. This field must be broadcast-compatible
        with `Dterm_field` and `inverse_metric_field`.

    Dterm_field : numpy.ndarray
        Log-volume term array of shape ``(..., ndim)``, where the last axis matches the number of coordinate directions.
        Must be broadcast-compatible with the spatial dimensions of `vector_field`.

    inverse_metric_field : numpy.ndarray
        Diagonal inverse metric tensor with shape ``(..., ndim)``, where the final dimension indexes the
        diagonal entries :math:`g^{\mu\mu}`. The leading dimensions must be broadcast-compatible with the
        spatial shape of `vector_field`.

    *varargs :
        Grid spacing for each axis. Accepts:

        - A single scalar (applied to all axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays,
        - A mix of scalars and arrays.

        If `derivative_axes` is specified, the number of elements in `varargs` must match its length.
        Otherwise, `varargs` must match the number of spatial dimensions in `vector_field`.

    field_axes : list of int, optional
        Indices mapping each spatial grid axis to a corresponding component in the final axis of `vector_field`.
        Defaults to ``[0, 1, ..., m-1]`` if not provided.

    derivative_axes : list of int, optional
        Subset of spatial axes over which to compute derivatives. This allows partial divergence over a subset of axes.
        Defaults to `field_axes` if not specified.

    edge_order : {1, 2}, default=2
        Accuracy order for finite difference gradients at the array boundaries.

    out : numpy.ndarray, optional
        Optional output buffer to store the result. Must have shape equal to the broadcasted grid shape
        of `vector_field` and `Dterm_field` (i.e., ``(F1, ..., Fm)``). If not provided, it is allocated internally.

    Returns
    -------
    numpy.ndarray
        Scalar divergence of the covariant vector field after index raising, with shape equal to the broadcasted
        spatial shape of `vector_field` and `Dterm_field`.

    Notes
    -----
    This routine is equivalent to:

    .. code-block:: python

        V_contra = g^{mumu} * V_cov
        div = D_mu * V^mu + diff_mu V^mu

    but optimized for diagonal metrics to avoid unnecessary full tensor contractions.

    No validation is performed on input shapes—use higher-level wrappers for shape checking and metric inference.

    See Also
    --------
    dense_vector_divergence_contravariant : Computes divergence from contravariant vector fields.
    dense_vector_divergence_covariant_full : Version supporting full inverse metric tensors.
    compute_divergence : High-level user-facing wrapper.

    Examples
    --------
    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_ops import dense_vector_divergence_covariant_diag
    >>>
    >>> # Grid
    >>> x = np.linspace(0.01, 1.0, 100)
    >>> y = np.linspace(0.1, np.pi - 0.1, 100)
    >>> X, Y = np.meshgrid(x, y, indexing='ij')
    >>>
    >>> # Covariant field: V_r = x, V_theta = sin(theta)
    >>> V = np.stack([X, np.sin(Y)], axis=-1)
    >>>
    >>> # D-terms (e.g., spherical coords): D_r = 2/x, D_theta = 1/tan(theta)
    >>> Dr = 2 / X
    >>> Dtheta = 1 / np.tan(Y)
    >>> D = np.stack([Dr, Dtheta], axis=-1)
    >>>
    >>> # Diagonal inverse metric: g^rr = 1, g^thetatheta = 1/x^2
    >>> IM_diag = np.stack([np.ones_like(X), 1 / X**2], axis=-1)
    >>>
    >>> # Compute divergence
    >>> div = dense_vector_divergence_covariant_diag(V, D, IM_diag, x, y)
    >>> div.shape
    (100, 100)
    """
    _contra_vec_field_ = _dense_contract_index_with_diagonal_metric(
        vector_field, inverse_metric_field, 0, 1, **kwargs
    )
    return dense_vector_divergence_contravariant(
        _contra_vec_field_,
        Dterm_field,
        *varargs,
        derivative_axes=derivative_axes,
        field_axes=field_axes,
        edge_order=edge_order,
        out=out,
    )


def dense_vector_divergence(
    vector_field: np.ndarray,
    Dterm_field: np.ndarray,
    *varargs,
    basis: Literal["contravariant", "covariant"] = "contravariant",
    inverse_metric_field: Optional[np.ndarray] = None,
    derivative_field: Optional[np.ndarray] = None,
    field_axes: Optional[Sequence[int]] = None,
    derivative_axes: Optional[Sequence[int]] = None,
    out: Optional[np.ndarray] = None,
    edge_order: Literal[1, 2] = 2,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the divergence of a vector field in a general coordinate system.

    This high-level routine supports both **contravariant** and **covariant** vector fields. For covariant fields,
    the divergence is computed by first raising the index using the inverse metric tensor, then applying the
    conservative divergence formula:

    .. math::

        \nabla_\mu V^\mu = D_\mu V^\mu + \partial_\mu V^\mu

    where:

    - :math:`V^\mu` is a contravariant vector field,
    - :math:`D_\mu = (\partial_\mu \rho)/\rho` is the logarithmic derivative of the volume element,
    - The divergence is evaluated as a sum of the D-term contraction and raw partial derivatives.

    The metric can be either full or diagonal, and this function automatically dispatches
    to the appropriate backend implementation.

    Parameters
    ----------
    vector_field : numpy.ndarray
        Input vector field of shape ``(F1, ..., Fm, ndim)``, where the final axis indexes the vector components
        and the first ``m`` axes represent spatial grid dimensions.

        If `basis="covariant"`, this is treated as a covariant field whose index will be raised.
        If `basis="contravariant"`, the field is used directly.

    Dterm_field : numpy.ndarray
        Log-volume term array of shape ``(..., ndim)``, where the final axis matches the number of coordinate axes.
        Must be broadcast-compatible with the spatial dimensions of `vector_field`.

    *varargs :
        Grid spacing for each spatial axis. Accepts:

        - A single scalar (applied to all axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `derivative_axes` is provided, the number of elements in `varargs` must match its length.
        Otherwise, `varargs` must match the number of spatial dimensions in `vector_field`.

    basis : {'contravariant', 'covariant'}, optional
        Specifies the form of the input vector field:

        - ``'contravariant'`` (default): treat the input as a contravariant field.
        - ``'covariant'``: raise the index using the inverse metric before computing divergence.

    inverse_metric_field : numpy.ndarray, optional
        Inverse metric tensor. Required if `basis='covariant'`.

        Accepts:

        - Shape ``(..., ndim)`` for diagonal metric (:math:`g^{\mu\mu}`),
        - Shape ``(..., ndim, ndim)`` for full metric (:math:`g^{\mu\nu}`).

        Must be broadcast-compatible with the spatial shape of `vector_field`.

    derivative_field : numpy.ndarray, optional
        Optional precomputed partial derivatives of the vector field components.
        If provided, must have shape ``(..., k)``, where `k` is the number of derivatives being taken
        (i.e., length of `derivative_axes`).

    field_axes : list of int, optional
        Maps each spatial axis to the corresponding component index in `vector_field`.
        Defaults to ``[0, 1, ..., m-1]`` if not specified.

    derivative_axes : list of int, optional
        Axes over which to compute partial derivatives. Defaults to `field_axes` if not provided.

    out : numpy.ndarray, optional
        Output buffer for storing the divergence result. Must have shape equal to the broadcasted
        grid shape of `vector_field` and `Dterm_field` (excluding the final axis). If not provided,
        the buffer is allocated automatically.

    edge_order : {1, 2}, default=2
        Order of accuracy for finite differencing used by `numpy.gradient`.

    **kwargs :
        Additional keyword arguments forwarded to internal contraction routines.

    Returns
    -------
    numpy.ndarray
        Scalar field representing the divergence of the input vector field.
        Shape is the broadcasted grid shape of `vector_field` and `Dterm_field`.

    Raises
    ------
    ValueError
        If required arguments are missing, incompatible with the basis, or have shape mismatches.

    See Also
    --------
    dense_gradient : Computes the gradient of a tensor field.
    dense_vector_divergence_contravariant : Low-level divergence for contravariant fields.
    dense_vector_divergence_covariant_full : Divergence for covariant fields with full metrics.
    dense_vector_divergence_covariant_diag : Divergence for covariant fields with diagonal metrics.

    Examples
    --------
    Compute the divergence of a vector field in 2D **spherical coordinates** :math:`(r, \theta)`:

    .. math::

        \vec{V}(r, \theta) =
        \begin{bmatrix}
            r \
            \cos(k\theta)
        \end{bmatrix}

    with known volume density:

    .. math::

        \rho(r, \theta) = r^2 \sin(\theta)

    which gives:

    .. math::

        D_r = \frac{2}{r}, \quad D_\theta = \frac{1}{\tan(\theta)},
        \quad g^{rr} = 1, \quad g^{\theta\theta} = \frac{1}{r^2}

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pymetric.differential_geometry.dense_ops import compute_divergence
    >>>
    >>> # Create coordinate grid
    >>> r = np.linspace(0.01, 1.0, 100)
    >>> theta = np.linspace(0.1, np.pi - 0.1, 100)  # avoid singularities
    >>> R, THETA = np.meshgrid(r, theta, indexing='ij')
    >>>
    >>> # Define vector field: V^r = r, V^theta = cos(theta)
    >>> Vr = R
    >>> Vtheta = np.cos(THETA)
    >>> vector = np.stack([Vr, Vtheta], axis=-1)
    >>>
    >>> # D-terms
    >>> D_r = 2 / R
    >>> D_theta = 1 / np.tan(THETA)
    >>> Dterm = np.stack([D_r, D_theta], axis=-1)
    >>>
    >>> # Inverse metric (diagonal)
    >>> g_inv = np.stack([np.ones_like(R), 1 / R**2], axis=-1)
    >>>
    >>> # Compute divergence (contravariant)
    >>> div = compute_divergence(vector, Dterm, r, theta, inverse_metric_field=g_inv, basis="contravariant")
    >>>
    >>> # Visualize
    >>> _ = plt.imshow(div.T, extent=[0.01, 1.0, 0.1, np.pi-0.1], origin='lower', aspect='auto', cmap='RdBu')
    >>> _ = plt.xlabel("r")
    >>> _ = plt.ylabel(r"$\theta$")
    >>> _ = plt.title(r"Divergence of $V = [r, \cos(\theta)]$ in Spherical Coordinates")
    >>> _ = plt.colorbar(label="Divergence")
    >>> _ = plt.tight_layout()
    >>> _ = plt.show()
    """
    if basis == "contravariant":
        return dense_vector_divergence_contravariant(
            vector_field,
            Dterm_field,
            *varargs,
            derivative_field=derivative_field,
            field_axes=field_axes,
            derivative_axes=derivative_axes,
            edge_order=edge_order,
            out=out,
            **kwargs,
        )
    elif basis == "covariant":
        if inverse_metric_field is None:
            raise ValueError(
                "`inverse_metric_field` must be provided for covariant divergence."
            )

        spatial_shape = vector_field.shape[:-1]
        metric_type = infer_metric_type(inverse_metric_field, spatial_shape)

        if metric_type == "full":
            return dense_vector_divergence_covariant_full(
                vector_field,
                Dterm_field,
                inverse_metric_field,
                *varargs,
                field_axes=field_axes,
                derivative_axes=derivative_axes,
                edge_order=edge_order,
                out=out,
                **kwargs,
            )
        elif metric_type == "diagonal":
            return dense_vector_divergence_covariant_diag(
                vector_field,
                Dterm_field,
                inverse_metric_field,
                *varargs,
                field_axes=field_axes,
                derivative_axes=derivative_axes,
                edge_order=edge_order,
                out=out,
                **kwargs,
            )
        else:
            raise ValueError(f"Unrecognized metric type: {metric_type}")
    else:
        raise ValueError(
            f"`basis` must be 'contravariant' or 'covariant', not '{basis}'."
        )


# =================================== #
# Laplacian Methods                   #
# =================================== #
# These methods are used to compute the Laplacian
# over some set of relevant tensor classes.
def dense_scalar_laplacian_diag(
    tensor_field: np.ndarray,
    Lterm_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    rank: int,
    ndim: int,
    *varargs,
    field_axes: Optional[np.ndarray] = None,
    derivative_axes: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
    first_derivative_field: Optional[np.ndarray] = None,
    second_derivative_field: Optional[np.ndarray] = None,
    edge_order: Literal[1, 2] = 2,
    **_,
):
    r"""
    Compute the element-wise Laplacian of a densely represented tensor field in an orthogonal coordinate system.

    This method computes the Laplace-Beltrami operator for a for :math:`T`:

    .. math::

        \Delta T = F^\mu \, \partial_\mu T + g^{\mu\mu} \, \partial^2_\mu T,

    where:

    - :math:`F^\mu = \frac{1}{\sqrt{|g|}} \, \partial_\mu \sqrt{|g|}` is the log-derivative of the volume element.
    - :math:`g^{\mu\mu}` is **symmetric** the inverse metric tensor.
    - The first term represents the contraction of the F-term and the gradient.
    - The second term is the contraction of the inverse metric with the Hessian (second derivatives).

    .. note::

        This operation is an **element-wise** operation!

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape ``(F_1, ..., F_m, ndim, ...)``, where the last `rank` axes
        are the tensor index dimensions.

        .. hint::

            Because this function is a low-level callable, it does not enforce the density of
            the elements in the trailing dimensions of the field. This can be used in some cases where
            the field is not **technically** a tensor field.
    Lterm_field : numpy.ndarray
        The F-term field of shape ``(..., ndim)``, where the first ``m`` dimensions are
        broadcast compatible with those of ``tensor_field`` and ``inverse_metric_field``.
    inverse_metric_field : numpy.ndarray
        The inverse metric field of shape ``(..., ndim)``, where the first ``m`` dimensions are
        broadcast compatible with those of ``tensor_field`` and ``Fterm_field``.
    rank : int
        Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
        the number of identified coordinate axes and therefore determines the shape of the returned array.
    ndim: int
        The number of total dimensions in the relevant coordinate system. This determines the maximum allowed value
        for ``m`` and the number of elements in the trailing dimension of the output.
    *varargs :
        Grid spacing for each spatial axis. Accepts:

        - A single scalar (applied to all axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `derivative_axes` is provided, the number of elements in `varargs` must match its length.
        Otherwise, `varargs` must match the number of spatial dimensions in `tensor_field` (``m``).
    field_axes : list of int, optional
        Mapping between each spatial dimension of `tensor_field` and the corresponding coordinate axis
        it represents (and therefore the component it represents). `field_axes` should be a length ``m``
        sequence of integers between ``0`` and ``ndim-1``. If not specified, then `field_axes` is simply
        ``0, 1, ..., m-1``.

        Specifying `field_axes` is critical when working with fields which are incomplete over their
        spatial domain (missing axes) as `field_axes` determines how contraction occurs with derivative
        terms.
    derivative_axes : list of int, optional
        The axes of the `tensor_field` to perform derivatives over. By default, all ``m`` axes are
        used when computing derivatives.

        `derivative_axes` should be used when certain axes of `tensor_field` are constant and should
        be excluded from numerical differentiation. It can also be used when `tensor_field` has been
        broadcast to a new set of axes (and therefore has singleton axes) on which differentiation
        would fail.
    first_derivative_field :  numpy.ndarray, optional
        Precomputed first derivatives of the `tensor_field` with shape ``tensor_field.shape + (q,)``, where
        ``q`` is the number of axes in `derivative_axes` or (if the former is not provided), the number
        of spatial dimensions in `tensor_field` (``m``).

        Specifying `first_derivative_field` can be used to avoid having to compute the
        relevant derivatives numerically, which can improve efficiency and accuracy.
    second_derivative_field :  numpy.ndarray, optional
        Precomputed second derivatives of the `tensor_field` with shape ``tensor_field.shape + (q,q)``, where
        ``q`` is the number of axes in `derivative_axes` or (if the former is not provided), the number
        of spatial dimensions in `tensor_field` (``m``).

        Specifying `first_derivative_field` can be used to avoid having to compute the
        relevant derivatives numerically, which can improve efficiency and accuracy.
    edge_order : {1, 2}, optional
        Order of accuracy for boundary differences. Default is 2.
    out: numpy.ndarray, optional
        An output buffer into which the result should be written. This should be an ``(..., ndim, ...)`` array
        where the leading indices are the broadcasted shape of the spatial components of the `tensor_field`, `Fterm_field`,
        and the `inverse_metric_field`. The trailing indices must match the non-spatial shape of the `tensor_field`.
        If `out` is not specified, then a new buffer will be created with the correct shape.


    Returns
    -------
    ~numpy.ndarray
        Laplacian of `tensor_field` with shape ``tensor_field.shape``.

    Notes
    -----
    This function assumes a full inverse metric (not diagonal) and uses upper-triangular evaluation of
    second derivatives to reduce computation using symmetry.

    See Also
    --------
    dense_gradient
    dense_scalar_laplacian_full

    Examples
    --------
    This example demonstrates computing the Laplace-Beltrami operator for a scalar field
    in 2D **spherical coordinates** :math:`(r, \theta)`.

    We define the scalar field:

    .. math::

        \phi(r, \theta) = r^2 \cos(\theta)

    The Laplace-Beltrami operator in spherical coordinates is given by:

    .. math::

        \Delta \phi =
            \frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial \phi}{\partial r} \right)
            + \frac{1}{r^2 \sin\theta} \frac{\partial}{\partial \theta} \left( \sin\theta \frac{\partial \phi}{\partial \theta} \right)

    which expands to:

    .. math::

        \Delta \phi =
            F^\mu \, \partial_\mu \phi + g^{\mu\nu} \, \partial_\mu \partial_\nu \phi

    where:

    - :math:`F^r = \frac{2}{r}`, the derivative of :math:`\log r^2`
    - :math:`F^\theta = \cot(\theta)`, the derivative of :math:`\log \sin(\theta)`
    - :math:`g^{rr} = 1`, :math:`g^{\theta\theta} = \frac{1}{r^2}` are the components of the inverse metric

    .. plot::
        :include-source: True

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pymetric.differential_geometry.dense_ops import dense_scalar_laplacian_diag
        >>>
        >>> # --- Grid in spherical coordinates --- #
        >>> r = np.linspace(0.01, 1.0, 100)
        >>> theta = np.linspace(0.1, np.pi - 0.1, 100)  # avoid tan(theta)=0
        >>> R, THETA = np.meshgrid(r, theta, indexing='ij')
        >>>
        >>> # --- Scalar field phi(r, theta) = r^2 * cos(theta) --- #
        >>> phi = R**2 * np.cos(THETA)
        >>>
        >>> # --- F-term: [2/r, cot(theta)] --- #
        >>> Fterm = np.zeros(R.shape + (2,))
        >>> Fterm[:,:,0] = 2 / R
        >>> Fterm[:,:,1] = 1 / (R**2 * np.tan(THETA))
        >>>
        >>> # --- Inverse metric --- #
        >>> IM = np.zeros(R.shape + (2,))
        >>> IM[..., 0] = 1
        >>> IM[..., 1] = 1 / R**2
        >>>
        >>> # --- Compute Laplacian --- #
        >>> lap = dense_scalar_laplacian_diag(phi, Fterm, IM, 0,2,r,theta)
        >>>
        >>> # --- Plot --- #
        >>> _ = plt.imshow(lap.T, origin='lower', extent=[0.01, 1.0, 0.1, np.pi - 0.1], aspect='auto', cmap='viridis')
        >>> _ = plt.colorbar(label=r"Laplacian $\Delta \phi$")
        >>> _ = plt.title(r"Laplacian of $\phi(r, \theta) = r^2 \cos(\theta)$")
        >>> _ = plt.xlabel("r")
        >>> _ = plt.ylabel(r"$\theta$")
        >>> _ = plt.tight_layout()
        >>> plt.show()
    """
    # --- Relevant Constants --- #
    # Extract the shapes from the arrays so that we
    # can work with them.
    field_ndim, fterm_ndim = tensor_field.ndim, Lterm_field.ndim
    tf_shape, imf_shape, ftf_shape = (
        tensor_field.shape,
        inverse_metric_field.shape,
        Lterm_field.shape,
    )
    _buffer_shape_ = np.broadcast_shapes(
        tf_shape[: field_ndim - rank],  # spatial component of TF
        imf_shape[:-1],  # spatial component of IMF
        ftf_shape[: fterm_ndim - (rank + 1)],  # spatial component of the Fterm field.
    )

    # Propagate the varargs out in case they are
    # scalar.
    if len(varargs) == 1:
        varargs *= field_ndim

    # Fill in missing information.
    if field_axes is None:
        field_axes = np.arange(field_ndim)
    if derivative_axes is None:
        derivative_axes = field_axes

    derivative_axes = np.asarray(derivative_axes, dtype=int)
    field_axes = np.asarray(field_axes, dtype=int)

    # --- Generate the output buffer --- #
    # the output buffer needs to be broadcast compatible
    # with the various fields.
    if out is None:
        out = np.zeros(_buffer_shape_, dtype=tensor_field.dtype, order="C")

    # --- Compute Term 1 --- #
    # The first term is the contraction between the F-terms and the
    # first derivatives.
    if first_derivative_field is None:
        # The first derivatives needs to be computed. We either want to
        # do this and allocate for it (if we need them later) or just
        # pass through and dump (if we don't need them later).
        if second_derivative_field is None:
            # We will need to allocate the second derivatives so we
            # need to hold onto the 1st derivatives.
            first_derivative_field = np.zeros(
                tf_shape + (ndim,), dtype=np.float64, order="C"
            )
            dense_gradient_covariant(
                tensor_field,
                rank,
                ndim,
                *varargs,
                field_axes=derivative_axes,
                output_indices=field_axes[derivative_axes],
                out=first_derivative_field,
                edge_order=edge_order,
            )

            # Dump the contraction result to `out`. We have all `ndim` elements in `Fterm`.
            np.sum(
                np.broadcast_to(
                    Lterm_field * first_derivative_field, _buffer_shape_ + (ndim,)
                ),
                axis=-1,
                out=out,
            )

        else:
            # We don't need to store the derivatives in memory because we only use them
            # once so we can just go through and compute the derivatives.
            for diff_index, diff_axis in enumerate(derivative_axes):
                comp_index = field_axes[diff_index]
                out[...] += np.broadcast_to(
                    Lterm_field[..., comp_index]
                    * np.gradient(
                        tensor_field,
                        varargs[diff_index],
                        edge_order=edge_order,
                        axis=diff_axis,
                    ),
                    _buffer_shape_,
                )

    # --- Compute Term 2 --- #
    # At this stage, we take advantage of Schwarz' Theorem to cut down on the number
    # of derivatives we actually have to compute. We just use the upper triangular segment
    # of the second derivative tensor.
    if second_derivative_field is not None:
        # We are already given the second derivative field, so we just need to ensure
        # that the summation occurs properly.
        out[...] += np.broadcast_to(
            np.sum(second_derivative_field * inverse_metric_field, axis=(-1, -2)),
            _buffer_shape_,
        )
    else:
        for diff_index, diff_axis in enumerate(derivative_axes):
            comp_index = field_axes[diff_index]
            out[...] += np.broadcast_to(
                inverse_metric_field[..., comp_index]
                * np.gradient(
                    first_derivative_field[..., comp_index],
                    varargs[diff_index],
                    edge_order=edge_order,
                    axis=diff_axis,
                ),
                _buffer_shape_,
            )

    return out


def dense_scalar_laplacian_full(
    tensor_field: np.ndarray,
    Lterm_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    rank: int,
    ndim: int,
    *varargs,
    field_axes: Optional[np.ndarray] = None,
    derivative_axes: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
    first_derivative_field: Optional[np.ndarray] = None,
    second_derivative_field: Optional[np.ndarray] = None,
    edge_order: Literal[1, 2] = 2,
    **_,
):
    r"""
    Compute the element-wise Laplacian of a densely represented tensor field.

    This method computes the Laplace-Beltrami operator for a for :math:`T`:

    .. math::

        \Delta T = F^\mu \, \partial_\mu T + g^{\mu\nu} \, \partial_\mu \partial_\nu T,

    where:

    - :math:`F^\mu = \frac{1}{\sqrt{|g|}} \, \partial_\mu \sqrt{|g|}` is the log-derivative of the volume element.
    - :math:`g^{\mu\nu}` is the inverse metric tensor.
    - The first term represents the contraction of the F-term and the gradient.
    - The second term is the contraction of the inverse metric with the Hessian (second derivatives).

    .. note::

        This operation is an **element-wise** operation!

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape ``(F_1, ..., F_m, ndim, ...)``, where the last `rank` axes
        are the tensor index dimensions.

        .. hint::

            Because this function is a low-level callable, it does not enforce the density of
            the elements in the trailing dimensions of the field. This can be used in some cases where
            the field is not **technically** a tensor field.
    Lterm_field : numpy.ndarray
        The F-term field of shape ``(..., ndim)``, where the first ``m`` dimensions are
        broadcast compatible with those of ``tensor_field`` and ``inverse_metric_field``.
    inverse_metric_field : numpy.ndarray
        The inverse metric field of shape ``(..., ndim, ndim)``, where the first ``m`` dimensions are
        broadcast compatible with those of ``tensor_field`` and ``Fterm_field``.
    rank : int
        Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
        the number of identified coordinate axes and therefore determines the shape of the returned array.
    ndim: int
        The number of total dimensions in the relevant coordinate system. This determines the maximum allowed value
        for ``m`` and the number of elements in the trailing dimension of the output.
    *varargs :
        Grid spacing for each spatial axis. Accepts:

        - A single scalar (applied to all axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `derivative_axes` is provided, the number of elements in `varargs` must match its length.
        Otherwise, `varargs` must match the number of spatial dimensions in `tensor_field` (``m``).
    field_axes : list of int, optional
        Mapping between each spatial dimension of `tensor_field` and the corresponding coordinate axis
        it represents (and therefore the component it represents). `field_axes` should be a length ``m``
        sequence of integers between ``0`` and ``ndim-1``. If not specified, then `field_axes` is simply
        ``0, 1, ..., m-1``.

        Specifying `field_axes` is critical when working with fields which are incomplete over their
        spatial domain (missing axes) as `field_axes` determines how contraction occurs with derivative
        terms.
    derivative_axes : list of int, optional
        The axes of the `tensor_field` to perform derivatives over. By default, all ``m`` axes are
        used when computing derivatives.

        `derivative_axes` should be used when certain axes of `tensor_field` are constant and should
        be excluded from numerical differentiation. It can also be used when `tensor_field` has been
        broadcast to a new set of axes (and therefore has singleton axes) on which differentiation
        would fail.
    first_derivative_field :  numpy.ndarray, optional
        Precomputed first derivatives of the `tensor_field` with shape ``tensor_field.shape + (q,)``, where
        ``q`` is the number of axes in `derivative_axes` or (if the former is not provided), the number
        of spatial dimensions in `tensor_field` (``m``).

        Specifying `first_derivative_field` can be used to avoid having to compute the
        relevant derivatives numerically, which can improve efficiency and accuracy.
    second_derivative_field :  numpy.ndarray, optional
        Precomputed second derivatives of the `tensor_field` with shape ``tensor_field.shape + (q,q)``, where
        ``q`` is the number of axes in `derivative_axes` or (if the former is not provided), the number
        of spatial dimensions in `tensor_field` (``m``).

        Specifying `first_derivative_field` can be used to avoid having to compute the
        relevant derivatives numerically, which can improve efficiency and accuracy.
    edge_order : {1, 2}, optional
        Order of accuracy for boundary differences. Default is 2.
    out: numpy.ndarray, optional
        An output buffer into which the result should be written. This should be an ``(..., ndim, ...)`` array
        where the leading indices are the broadcasted shape of the spatial components of the `tensor_field`, `Fterm_field`,
        and the `inverse_metric_field`. The trailing indices must match the non-spatial shape of the `tensor_field`.
        If `out` is not specified, then a new buffer will be created with the correct shape.


    Returns
    -------
    ~numpy.ndarray
        Laplacian of `tensor_field` with shape ``tensor_field.shape``.

    Notes
    -----
    This function assumes a full inverse metric (not diagonal) and uses upper-triangular evaluation of
    second derivatives to reduce computation using symmetry.

    See Also
    --------
    dense_gradient_covariant
    dense_scalar_laplacian_diag

    Examples
    --------
    This example demonstrates computing the Laplace-Beltrami operator for a scalar field
    in 2D **spherical coordinates** :math:`(r, \theta)`.

    We define the scalar field:

    .. math::

        \phi(r, \theta) = r^2 \cos(\theta)

    The Laplace-Beltrami operator in spherical coordinates is given by:

    .. math::

        \Delta \phi =
            \frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial \phi}{\partial r} \right)
            + \frac{1}{r^2 \sin\theta} \frac{\partial}{\partial \theta} \left( \sin\theta \frac{\partial \phi}{\partial \theta} \right)

    which expands to:

    .. math::

        \Delta \phi =
            F^\mu \, \partial_\mu \phi + g^{\mu\nu} \, \partial_\mu \partial_\nu \phi

    where:

    - :math:`F^r = \frac{2}{r}`, the derivative of :math:`\log r^2`
    - :math:`F^\theta = \cot(\theta)`, the derivative of :math:`\log \sin(\theta)`
    - :math:`g^{rr} = 1`, :math:`g^{\theta\theta} = \frac{1}{r^2}` are the components of the inverse metric

    .. plot::
        :include-source: True

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pymetric.differential_geometry.dense_ops import dense_scalar_laplacian_full
        >>>
        >>> # --- Grid in spherical coordinates --- #
        >>> r = np.linspace(0.01, 1.0, 100)
        >>> theta = np.linspace(0.1, np.pi - 0.1, 100)  # avoid tan(theta)=0
        >>> R, THETA = np.meshgrid(r, theta, indexing='ij')
        >>>
        >>> # --- Scalar field phi(r, theta) = r^2 * cos(theta) --- #
        >>> phi = R**2 * np.cos(THETA)
        >>>
        >>> # --- F-term: [2/r, cot(theta)] --- #
        >>> Fterm = np.zeros(R.shape + (2,))
        >>> Fterm[:,:,0] = 2 / R
        >>> Fterm[:,:,1] = 1 / (R**2 * np.tan(THETA))
        >>>
        >>> # --- Inverse metric --- #
        >>> IM = np.zeros(R.shape + (2,2))
        >>> IM[..., 0,0] = 1
        >>> IM[..., 1,1] = 1 / R**2
        >>>
        >>> # --- Compute Laplacian --- #
        >>> lap = dense_scalar_laplacian_full(phi, Fterm, IM, 0,2,r,theta)
        >>>
        >>> # --- Plot --- #
        >>> _ = plt.imshow(lap.T, origin='lower', extent=[0.01, 1.0, 0.1, np.pi - 0.1], aspect='auto', cmap='viridis')
        >>> _ = plt.colorbar(label=r"Laplacian $\Delta \phi$")
        >>> _ = plt.title(r"Laplacian of $\phi(r, \theta) = r^2 \cos(\theta)$")
        >>> _ = plt.xlabel("r")
        >>> _ = plt.ylabel(r"$\theta$")
        >>> _ = plt.tight_layout()
        >>> plt.show()
    """
    # --- Relevant Constants --- #
    # Extract the shapes from the arrays so that we
    # can work with them.
    field_ndim, fterm_ndim = tensor_field.ndim, Lterm_field.ndim
    tf_shape, imf_shape, ftf_shape = (
        tensor_field.shape,
        inverse_metric_field.shape,
        Lterm_field.shape,
    )
    _buffer_shape_ = np.broadcast_shapes(
        tf_shape[: field_ndim - rank],  # spatial component of TF
        imf_shape[:-2],  # spatial component of IMF
        ftf_shape[: fterm_ndim - (rank + 1)],  # spatial component of the Fterm field.
    )

    # Propagate the varargs out in case they are
    # scalar.
    if len(varargs) == 1:
        varargs *= field_ndim

    # Fill in missing information.
    if field_axes is None:
        field_axes = np.arange(field_ndim)
    if derivative_axes is None:
        derivative_axes = field_axes

    derivative_axes = np.asarray(derivative_axes, dtype=int)
    field_axes = np.asarray(field_axes, dtype=int)

    # --- Generate the output buffer --- #
    # the output buffer needs to be broadcast compatible
    # with the various fields.
    if out is None:
        out = np.zeros(_buffer_shape_, dtype=tensor_field.dtype, order="C")

    # --- Compute Term 1 --- #
    # The first term is the contraction between the F-terms and the
    # first derivatives.
    if first_derivative_field is None:
        # The first derivatives needs to be computed. We either want to
        # do this and allocate for it (if we need them later) or just
        # pass through and dump (if we don't need them later).
        if second_derivative_field is None:
            # We will need to allocate the second derivatives so we
            # need to hold onto the 1st derivatives.
            first_derivative_field = np.zeros(
                tf_shape + (ndim,), dtype=np.float64, order="C"
            )
            dense_gradient_covariant(
                tensor_field,
                rank,
                ndim,
                *varargs,
                field_axes=derivative_axes,
                output_indices=field_axes[derivative_axes],
                out=first_derivative_field,
                edge_order=edge_order,
            )

            # Dump the contraction result to `out`. We have all `ndim` elements in `Fterm`.
            np.sum(
                np.broadcast_to(
                    Lterm_field * first_derivative_field, _buffer_shape_ + (ndim,)
                ),
                axis=-1,
                out=out,
            )

        else:
            # We don't need to store the derivatives in memory because we only use them
            # once so we can just go through and compute the derivatives.
            for diff_index, diff_axis in enumerate(derivative_axes):
                comp_index = field_axes[diff_index]
                out[...] += np.broadcast_to(
                    Lterm_field[..., comp_index]
                    * np.gradient(
                        tensor_field,
                        varargs[diff_index],
                        edge_order=edge_order,
                        axis=diff_axis,
                    ),
                    _buffer_shape_,
                )

    # --- Compute Term 2 --- #
    # At this stage, we take advantage of Schwarz' Theorem to cut down on the number
    # of derivatives we actually have to compute. We just use the upper triangular segment
    # of the second derivative tensor.
    if second_derivative_field is not None:
        # We are already given the second derivative field, so we just need to ensure
        # that the summation occurs properly.
        out[...] += np.broadcast_to(
            np.sum(second_derivative_field * inverse_metric_field, axis=(-1, -2)),
            _buffer_shape_,
        )
    else:
        # We don't already have the derivatives so we need to perform the computation.
        # For this, we're going to cycle through all upper-triangular pairs of the derivative
        # axes.
        number_of_derivative_axes = len(derivative_axes)
        indices = np.triu_indices(number_of_derivative_axes)

        for a_index, b_index in zip(*indices):
            # We first need to take the a/b indices and determine the field axes
            # and then the component axes.
            a_field_index, b_field_index = (
                derivative_axes[a_index],
                derivative_axes[b_index],
            )
            a_comp_index, b_comp_index = (
                field_axes[a_field_index],
                field_axes[b_field_index],
            )

            # We now need to compute the second derivative. This will be the derivative of
            # the first_derivative_field[..., a_comp_index] with respect to the b_index vararg and
            # on the b_field_index axis.
            factor = 1 if a_index == b_index else 2
            out += np.broadcast_to(
                factor
                * inverse_metric_field[..., a_comp_index, b_comp_index]
                * np.gradient(
                    first_derivative_field[..., a_comp_index],
                    varargs[b_index],
                    axis=b_field_index,
                    edge_order=edge_order,
                ),
                _buffer_shape_,
            )

    return out


def dense_scalar_laplacian(
    tensor_field: np.ndarray,
    Lterm_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    rank: int,
    ndim: int,
    *varargs,
    field_axes: Optional[Sequence[int]] = None,
    derivative_axes: Optional[Sequence[int]] = None,
    out: Optional[np.ndarray] = None,
    first_derivative_field: Optional[np.ndarray] = None,
    second_derivative_field: Optional[np.ndarray] = None,
    edge_order: Literal[1, 2] = 2,
) -> np.ndarray:
    r"""
    Compute the Laplacian (Laplace-Beltrami operator) of a tensor field in a general
    curvilinear coordinate system, using either a full or diagonal inverse metric.

    This function implements:

    .. math::

        \Delta T = F^\mu \, \partial_\mu T + g^{\mu\nu} \, \partial_\mu \partial_\nu T

    and dispatches to the appropriate low-level method depending on the shape of the inverse metric.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Input tensor field of shape ``(F1, ..., Fm, ...)``, where the final axes (if any) correspond
        to tensor indices, and the first ``m`` axes are spatial.
    Lterm_field : numpy.ndarray
        Log-volume term array of shape ``(..., ndim)``, where the last axis matches the number of coordinate directions.
    inverse_metric_field : numpy.ndarray
        Either a diagonal inverse metric (shape ``(..., ndim)``) or a full tensor (shape ``(..., ndim, ndim)``).
    rank : int
        Number of trailing axes of `tensor_field` corresponding to its tensor rank.
    ndim: int
        The number of total dimensions in the relevant coordinate system. This determines the maximum allowed value
        for ``m`` and the number of elements in the trailing dimension of the output.
    *varargs :
        Grid spacing along each axis. Must match number of spatial axes or derivative axes.
    field_axes : list of int, optional
        Maps each spatial axis of `tensor_field` to a corresponding coordinate axis.
        Defaults to `[0, 1, ..., m-1]`.
    derivative_axes : list of int, optional
        Subset of axes over which derivatives are taken. Defaults to `field_axes`.
    out : numpy.ndarray, optional
        Output buffer to store result. Must be broadcast-compatible with spatial shape of inputs.
    first_derivative_field : numpy.ndarray, optional
        Optional precomputed first derivative field of shape ``tensor_field.shape + (ndim,)``.
    second_derivative_field : numpy.ndarray, optional
        Optional precomputed second derivative field of shape ``tensor_field.shape + (ndim, ndim)``.
    edge_order : {1, 2}, default=2
        Accuracy order of numerical finite differences.

    Returns
    -------
    numpy.ndarray
        Laplacian of the input tensor field, with shape ``tensor_field.shape``.

    Raises
    ------
    ValueError
        If the metric type is not recognized.
    """
    spatial_shape = tensor_field.shape[: tensor_field.ndim - rank]
    metric_type = infer_metric_type(inverse_metric_field, spatial_shape)

    kwargs = dict(
        field_axes=field_axes,
        derivative_axes=derivative_axes,
        out=out,
        first_derivative_field=first_derivative_field,
        second_derivative_field=second_derivative_field,
        edge_order=edge_order,
    )

    if metric_type == "full":
        return dense_scalar_laplacian_full(
            tensor_field,
            Lterm_field,
            inverse_metric_field,
            rank,
            ndim,
            *varargs,
            **kwargs,
        )
    elif metric_type == "diagonal":
        return dense_scalar_laplacian_diag(
            tensor_field,
            Lterm_field,
            inverse_metric_field,
            rank,
            ndim,
            *varargs,
            **kwargs,
        )
    else:
        raise ValueError(f"Unrecognized metric type: {metric_type}")
