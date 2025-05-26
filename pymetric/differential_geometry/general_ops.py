"""
General operations for performing differential geometry operations.

In most cases, the functions defined in this module are intended to apply simplified operations
to structures which are not, strictly speaking, tensors.
"""
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .dense_ops import dense_scalar_laplacian


def dense_element_wise_partial_derivatives(
    field: np.ndarray,
    rank: int,
    *varargs,
    field_axes: Optional[Sequence[int]] = None,
    output_indices: Optional[Sequence[int]] = None,
    edge_order: Literal[1, 2] = 2,
    out: Optional[ArrayLike] = None,
    **_,
) -> np.ndarray:
    r"""
    Compute the element-wise covariant gradient of an array-valued field.

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
    field : numpy.ndarray
        Field of shape ``(F_1, ..., F_m, N, ...)``, where the last `rank` axes
        are the tensor index dimensions. The partial derivatives are computed over the first `m` axes.
    rank : int
        Number of trailing axes that represent tensor indices (i.e., tensor rank). The `rank` determines
        the number of identified coordinate axes and therefore determines the shape of the returned array.
    *varargs :
        Grid spacing for each spatial axis. Follows the same format as :func:`numpy.gradient`, and can be:

        - A single scalar (applied to all spatial axes),
        - A list of scalars (one per axis),
        - A list of coordinate arrays (one per axis),
        - A mix of scalars and arrays (broadcast-compatible).

        If `field_axes` is provided, `varargs` must match its length. If `axes` is not provided, then `varargs` should
        be ``field.ndim - rank`` in length (or be a scalar).
    field_axes : list of int, optional
        The spatial axes over which to compute the component-wise partial derivatives. If `field_axes` is
        not specified, then all ``field.ndim - rank`` axes are computed.
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
        ``field.shape + (ndim,)``. If `out` is not specified, then it is allocated during the
        function call.

    Returns
    -------
    numpy.ndarray

    See Also
    --------
    numpy.gradient: The computational backend for this operation.
    ~differential_geometry.dense_ops.dense_gradient_contravariant_full: Contravariant gradient for full metric.
    ~differential_geometry.dense_ops.dense_gradient_contravariant_diag: Contravariant gradient for diagonal metric.
    ~differential_geometry.dense_ops.dense_gradient: Wrapper for user-facing gradient computations.
    """
    # Coerce the axes so that we know how many
    # axes we are actually computing derivatives for.
    # Ensure we don't have an invalid number of axes.
    t_shape, t_ndim = field.shape, field.ndim
    if field_axes is None:
        _number_of_axes = t_ndim - rank
        field_axes = tuple(range(t_ndim - rank))
    else:
        _number_of_axes = len(field_axes)

    # Setup the output indices.
    if output_indices is None:
        output_indices = np.arange(_number_of_axes)

    # Fix the varargs if the are length 1 so
    # that we never have an issue with broadcasting.
    if len(varargs) == 1:
        varargs *= _number_of_axes

    # Allocate the output array. This should be the
    # same shape as field but with _number_of_axes elements
    # in an additional
    if out is None:
        out = np.zeros(t_shape + (_number_of_axes,), dtype=field.dtype, order="C")

    # Now iterate through each of the axes and
    # perform the differentiation procedure. We then assign
    # each into the out buffer.
    for _i, (fax, oax) in enumerate(zip(field_axes, output_indices)):
        if t_shape[fax] < edge_order:
            raise ValueError(
                f"Failed to compute a gradient along axis {fax} because its shape was smaller than `edge_order`."
            )
        out[..., oax] = np.gradient(field, varargs[_i], axis=fax, edge_order=edge_order)

    return out


def dense_element_wise_laplacian(
    field: np.ndarray,
    Lterm_field: np.ndarray,
    inverse_metric_field: np.ndarray,
    rank: int,
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

    .. note::

        This is a *very thin wrapper* around :func:`~differential_geometry.dense_ops.dense_scalar_laplacian` with
        the sole intent of providing a "general" alias for the operation. This function does have a slightly different
        signature as it allows for inference of ``ndim``.

    Parameters
    ----------
    field : numpy.ndarray
        Input tensor field of shape ``(F1, ..., Fm, ...)``, where the final axes (if any) correspond
        to tensor indices, and the first ``m`` axes are spatial.
    Lterm_field : numpy.ndarray
        Log-volume term array of shape ``(..., ndim)``, where the last axis matches the number of coordinate directions.
    inverse_metric_field : numpy.ndarray
        Either a diagonal inverse metric (shape ``(..., ndim)``) or a full tensor (shape ``(..., ndim, ndim)``).
    rank : int
        Number of trailing axes of `field` corresponding to its tensor rank.
    *varargs :
        Grid spacing along each axis. Must match number of spatial axes or derivative axes.
    field_axes : list of int, optional
        Maps each spatial axis of `field` to a corresponding coordinate axis.
        Defaults to `[0, 1, ..., m-1]`.
    derivative_axes : list of int, optional
        Subset of axes over which derivatives are taken. Defaults to `field_axes`.
    out : numpy.ndarray, optional
        Output buffer to store result. Must be broadcast-compatible with spatial shape of inputs.
    first_derivative_field : numpy.ndarray, optional
        Optional precomputed first derivative field of shape ``field.shape + (ndim,)``.
    second_derivative_field : numpy.ndarray, optional
        Optional precomputed second derivative field of shape ``field.shape + (ndim, ndim)``.
    edge_order : {1, 2}, default=2
        Accuracy order of numerical finite differences.

    Returns
    -------
    numpy.ndarray
        Laplacian of the input tensor field, with shape ``field.shape``.

    Raises
    ------
    ValueError
        If the metric type is not recognized.
    """
    ndim = Lterm_field.shape[-1]
    return dense_scalar_laplacian(
        field,
        Lterm_field,
        inverse_metric_field,
        rank,
        ndim,
        *varargs,
        field_axes=field_axes,
        derivative_axes=derivative_axes,
        out=out,
        first_derivative_field=first_derivative_field,
        second_derivative_field=second_derivative_field,
        edge_order=edge_order,
    )
