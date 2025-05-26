"""
Utility functions for performing basic tensor manipulations including index raising and lowering.
"""
import string
from typing import Any, List, Literal, Optional, Tuple

import numpy as np


# ------------------------------------------ #
# General Utilities                          #
# ------------------------------------------ #
# These are low-level tensor field utilities with little to
# no protection in place to ensure correct behavior of the
# inputs.
def _dense_contract_index_with_metric(
    tensor_field: np.ndarray,
    metric_field: np.ndarray,
    index: int,
    rank: int,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Contract a tensor index with a metric (or inverse metric) using :py:func:`numpy.einsum`.

    This operation performs the index contraction:

    .. math::

        T^{\\ldots\\mu\\ldots} = g^{\\mu\\nu} T^{\\ldots}_{\\ldots\\nu\\ldots}

    or

    .. math::

        T_{\\ldots\\mu\\ldots} = g_{\\mu\\nu} T_{\\ldots}^{\\ldots\\nu\\ldots}

    depending on context and whether the provided metric is the metric tensor or its inverse.
    The contraction is performed over a single tensor index (specified by `index`), replacing it
    with the contracted result.

    This function assumes that the tensor rank occupies the last `rank` axes of `tensor_field`,
    and that the metric has shape ``(..., N, N)``, where `N` matches the size of the contracted index.

    .. note::

        This function performs a full batched matrix multiplication via :py:func:`numpy.einsum`.
        If your metric is diagonal or otherwise structured (e.g., orthogonal or identity),
        specialized routines may offer better performance.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        A dense tensor field with shape ``(..., i_1, ..., i_rank)``, where the last `rank` axes
        represent tensor indices.

        .. hint::

            Dense tensors are technically ``(..., Ndim, ..., Ndim)``; however, :func:`dense_contract_index_with_metric`
            has slightly looser behavior allowing for any array shape ``(*grid_shape,*element_shape)`` so long as
            ``element_shape[index]`` matches the shape of the `metric_field`.

    metric_field : numpy.ndarray
        The metric (or inverse metric) tensor used for contraction.
        Must have shape ``(..., N, N)``, broadcastable with the leading dimensions of `tensor_field`.
    index : int
        The index among the trailing `rank` tensor indices to contract with the metric.
        Must satisfy ``0 <= index < rank``.
    rank : int
        The number of trailing axes in `tensor_field` that represent tensor indices.
    out : numpy.ndarray, optional
        Optional output array to store the result. If provided, must have the same shape as the expected
        output: identical to `tensor_field` but with the contracted index replaced by the new one.

        This allows for memory reuse and avoids allocating a new array, which can improve performance
        in tight loops or high-throughput workflows. If not provided, a new array is returned.

        The dtype and shape must be compatible with the result of the einsum contraction.
    **kwargs :
        Additional keyword arguments passed to :py:func:`numpy.einsum`.
        For example, use `optimize=True` to enable internal optimization of the contraction path.

    Returns
    -------
    numpy.ndarray
        The resulting tensor field after contraction. The output shape is broadcasted over
        the leading dimensions of both `tensor_field` and `metric_field`, and the contracted
        index is replaced by the output index from the metric.

        Specifically, if `tensor_field` has shape ``(..., I₁, ..., I_rank)`` and the contraction
        is performed over index `i` with size `N`, and `metric_field` has shape ``(..., N, N)``,
        then the resulting array has shape:

        .. code-block:: python

            broadcast(... from both inputs, ..., I₁, ..., I_{i-1}, N, I_{i+1}, ..., I_rank)

        where the contracted index axis (of size `N`) is replaced by the output metric index.

        This shape supports broadcasting of mismatched dimensions, such as contraction of a
        field with shape ``(A, 1, C)`` and a metric of shape ``(A, B, C, C)`` resulting in output
        shape ``(A, B, C)``.

    Examples
    --------

    Contract a random tensor with an identity metric:

    >>> import numpy as np
    >>> T = np.random.rand(10, 10, 3)  # scalar field with a vector index (rank-1)
    >>> ginv = np.eye(3)[np.newaxis,np.newaxis,:,:] * np.ones((10,10,1,1))              # inverse metric (identity in this case)
    >>> _dense_contract_index_with_metric(T, ginv, index=0, rank=1).shape
    (10, 10, 3)

    :func:`dense_contract_index_with_metric` can also broadcast for inconsistent
    tensor and metric shapes. In this case, we'll contract a tensor of shape ``(10, 1, 3)``
    with a metric ``(10, 10, 3, 3)``.

    >>> import numpy as np
    >>> T = np.random.rand(10, 1, 3)  # scalar field with a vector index (rank-1)
    >>> ginv = np.eye(3)[np.newaxis,np.newaxis,:,:] * np.ones((10,10,3,3))              # inverse metric (identity in this case)
    >>> _dense_contract_index_with_metric(T, ginv, index=0, rank=1).shape
    (10, 10, 3)

    Arrays can be implicitly broadcastable as well:

    >>> import numpy as np
    >>> T = np.random.rand(10, 3)  # Inconsistent field shape
    >>> ginv = np.eye(3)[np.newaxis,np.newaxis,:,:] * np.ones((10,10,3,3))              # inverse metric (identity in this case)
    >>> _dense_contract_index_with_metric(T, ginv, index=0, rank=1).shape
    (10, 10, 3)

    But direct inconsistency will lead to an error:

    >>> import numpy as np
    >>> T = np.random.rand(12, 3)  # VERY Inconsistent field shape
    >>> ginv = np.eye(3)[np.newaxis,np.newaxis,:,:] * np.ones((10,10,3,3))              # inverse metric (identity in this case)
    >>> _dense_contract_index_with_metric(T, ginv, index=0, rank=1).shape # doctest: +SKIP
    ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (12,3)->(12,newaxis,3) (10,10,3,3)->(10,10,3,3)

    See Also
    --------
    numpy.einsum : Generalized Einstein summation in NumPy.
    """
    # Extract any relevant metadata from the tensor field to
    # ensure that this operation can be performed effectively.
    ndim = tensor_field.ndim

    # Construct the index strings. This is faster than moving around
    # axes of the resulting array. We'll create 3 sets (field, tensor, and metric).
    __letters__ = string.ascii_lowercase
    tensor_indices = __letters__[ndim - rank : ndim]

    # Fix the tensor index to have a matching summation.
    tensor_indices = tensor_indices[:index] + "I" + tensor_indices[index + 1 :]
    output_indices = tensor_indices[:index] + "J" + tensor_indices[index + 1 :]

    # Write the expression string.
    __expression__ = f"...{tensor_indices},...IJ->...{output_indices}"

    # Perform the computation.
    return np.einsum(__expression__, tensor_field, metric_field, out=out, **kwargs)


def _dense_contract_index_with_diagonal_metric(
    tensor_field: np.ndarray,
    metric_field: np.ndarray,
    index: int,
    rank: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Contract a tensor index with a diagonal metric (or inverse metric) using scalar multiplication.

    This operation performs the contraction:

    .. math::

        T^{\ldots\mu\ldots} = g^{\mu\mu} T^{\ldots}_{\ldots\mu\ldots}

    or

    .. math::

        T_{\ldots\mu\ldots} = g_{\mu\mu} T_{\ldots}^{\ldots\mu\ldots}

    depending on context and whether the provided metric is the metric tensor or its inverse.
    The contraction is performed by directly scaling the tensor along a single index axis using
    the diagonal entries of the metric.

    This function assumes that the metric is diagonal and only requires the diagonal entries
    as a field with shape ``(..., N)``, where `N` is the size of the tensor index being contracted.
    The contraction is carried out by broadcasting the diagonal metric values and multiplying them
    along the appropriate axis.

    Unlike full metric contractions, no summation is performed; this function is significantly more
    efficient and numerically stable when the metric is known to be diagonal.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        A dense tensor field with shape ``(..., i_1, ..., i_rank)``, where the last `rank` axes
        represent tensor indices.

        .. hint::

            This function expects `tensor_field` to have shape ``(*grid_shape, *tensor_shape)``, and
            `tensor_shape[index]` must match the size of the final axis in `metric_field`.

    metric_field : numpy.ndarray
        A diagonal metric (or inverse metric) represented as an array of shape ``(..., N)``,
        where the leading dimensions must be broadcast-compatible with the grid dimensions
        of `tensor_field`.

    index : int
        The index among the trailing `rank` tensor indices to contract with the metric.
        Must satisfy ``0 <= index < rank``.

    rank : int
        The number of trailing axes in `tensor_field` that represent tensor indices.

    out : numpy.ndarray, optional
        Optional array into which the result will be stored. If provided, must have the same
        shape as `tensor_field`. If not provided, a new array is returned.

    Returns
    -------
    numpy.ndarray
        The resulting tensor field after contraction. The shape is identical to `tensor_field`,
        with the contracted index scaled (not removed or reduced).

        Broadcasting allows, for example, a contraction of shape ``(A, 1, C)`` against a metric
        of shape ``(A, C)`` to produce a result of shape ``(A, 1, C)``.

    Examples
    --------

    Contract a vector field with a diagonal inverse metric tensor:

    >>> import numpy as np
    >>> T = np.random.rand(10, 10, 3)  # Inconsistent field shape
    >>> ginv = np.ones((10,10,3))              # inverse metric (identity in this case)
    >>> dense_contract_index_with_diagonal_metric(T, ginv, index=0, rank=1).shape
    (10, 10, 3)

    With broadcasting:

    >>> import numpy as np
    >>> T = np.random.rand(10, 3)  # Inconsistent field shape
    >>> ginv = np.ones((10,10,3))              # inverse metric (identity in this case)
    >>> _dense_contract_index_with_diagonal_metric(T, ginv, index=0, rank=1).shape
    (10, 10, 3)


    See Also
    --------
    dense_contract_index_with_metric : Full contraction using general metric tensors.
    numpy.multiply : Element-wise multiplication in NumPy.

    Notes
    -----
    Technically, neither ``index`` or ``rank`` is actually used in this function. This function
    is just a wrapper on :func:`numpy.multiply`.
    """
    ndim = tensor_field.ndim
    field_ndim = ndim - rank
    axis = field_ndim + index  # actual axis in tensor_field

    broadcast_shape = list(metric_field.shape[:field_ndim]) + [1] * rank
    broadcast_shape[axis] = metric_field.shape[-1]

    metric_broadcast = metric_field.reshape(broadcast_shape)

    if out is None:
        return tensor_field * metric_broadcast
    else:
        np.multiply(tensor_field, metric_broadcast, out=out)
        return out


def _dense_compute_tensor_trace(
    tensor_field: np.ndarray,
    indices: Tuple[int, int],
    rank: int,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Perform a trace over two tensor indices of a rank-`r` tensor field.

    This function contracts two specified tensor axes in the field by summing over
    matching values, effectively reducing the rank of the tensor by 2. The indices
    must refer to distinct axes within the trailing `rank` tensor slots.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        A dense tensor field with shape ``(..., i₁, ..., i_r)``, where the last `rank` axes
        represent the tensor indices and the leading axes represent the field (grid) dimensions.
    indices : tuple of int
        A pair of indices (i, j) among the trailing `rank` dimensions to contract.
        Must satisfy ``0 <= i, j < rank`` and ``i != j``.
    rank : int
        The number of trailing tensor dimensions of `tensor_field`.
    out : numpy.ndarray, optional
        Optional output array to store the result. If provided, must have the same shape
        as the expected output: the shape of `tensor_field` with the specified `indices`
        removed (i.e., `rank - 2` fewer trailing axes).

        Enables memory reuse and avoids additional allocations, which is beneficial in
        high-performance or memory-sensitive workflows. Must be broadcast-compatible with
        the output of `np.trace`.
    **kwargs :
        Additional keyword arguments passed to :py:func:`numpy.trace`, such as `dtype` or `out`.

    Returns
    -------
    numpy.ndarray
        Tensor field with the specified pair of tensor indices traced (contracted).
        The result has the same field dimensions and `rank - 2` tensor dimensions.

    Raises
    ------
    ValueError
        If indices are invalid or refer to the same axis.

    Examples
    --------
    >>> import numpy as np
    >>> T = np.zeros((4, 4, 3, 3))  # 2nd-rank tensor field over 4x4 grid
    >>> np.fill_diagonal(T[0, 0], 1)
    >>> _dense_compute_tensor_trace(T, indices=(0, 1), rank=2)[0, 0]
    3.0
    """
    # Determine the number of field dimensions.
    field_ndim = tensor_field.ndim - rank

    # Now perform and return the trace.
    return np.trace(
        tensor_field,
        axis1=indices[0] + field_ndim,
        axis2=indices[1] + field_ndim,
        out=out,
        **kwargs,
    )


def infer_metric_type(
    metric_field: np.ndarray,
    field_shape: Tuple[int, ...],
) -> Literal["diagonal", "full"]:
    """
    Infer whether the given metric (or inverse metric) tensor is diagonal or full.

    The function compares the shape of the metric tensor against the field's spatial shape.
    A metric is considered:

    - "diagonal" if it has shape broadcastable to ``field_shape + (N,)``
    - "full"     if it has shape broadcastable to ``field_shape + (N, N)``

    Parameters
    ----------
    metric_field : numpy.ndarray
        Metric or inverse metric tensor. Shape must be either (..., N) for diagonal,
        or (..., N, N) for full metrics.

    field_shape : tuple of int
        Shape of the spatial (non-tensor) portion of the tensor field the metric will apply to.

    Returns
    -------
    {"diagonal", "full"}
        A string indicating the type of metric tensor.

    Raises
    ------
    ValueError
        If the metric shape is not compatible with the field shape, or cannot be interpreted.
    """
    mshape = metric_field.shape
    ndim_field = len(field_shape)

    if metric_field.ndim == ndim_field + 1:
        # Possibly diagonal: check broadcastability
        try:
            np.broadcast_shapes(field_shape, mshape[:-1])
        except ValueError:
            raise ValueError(
                f"Diagonal metric shape {mshape} is not broadcast-compatible with field shape {field_shape}."
            )
        return "diagonal"

    elif metric_field.ndim == ndim_field + 2:
        if mshape[-1] != mshape[-2]:
            raise ValueError(
                f"Expected square trailing dimensions for full metric, but got {mshape[-2:]}."
            )

        try:
            np.broadcast_shapes(field_shape, mshape[:-2])
        except ValueError:
            raise ValueError(
                f"Full metric shape {mshape} is not broadcast-compatible with field shape {field_shape}."
            )
        return "full"

    else:
        raise ValueError(
            f"Metric shape {mshape} is incompatible with field shape {field_shape}. "
            "Expected shape ending in (N,) for diagonal or (N, N) for full metric."
        )


def _dense_adjust_tensor_signature(
    tensor_field: np.ndarray,
    indices: List[int],
    tensor_signature: np.ndarray,
    metric_field: Optional[np.ndarray] = None,
    inverse_metric_field: Optional[np.ndarray] = None,
    out=None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusts multiple tensor indices (simultaneously) using einsum with appropriate metric or inverse metric.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        The tensor field with shape ``(..., I₁, ..., I_r)``, where the last `r` axes represent tensor indices
        and the leading axes represent spatial/grid dimensions.
    indices : List[int]
        Indices among the last `rank` axes to be modified.
    tensor_signature : numpy.ndarray
        Array of integers, +1 for contravariant, -1 for covariant index.
    metric_field : numpy.ndarray, optional
        Metric tensor of shape (..., N, N).
    inverse_metric_field : numpy.ndarray, optional
        Inverse metric tensor of shape (..., N, N).
    out : numpy.ndarray, optional
        Optional array to store the result of the einsum contraction. Must have the same
        shape as the expected output. If provided, will be used in-place by `np.einsum`.
    Returns
    -------
    result : numpy.ndarray
        Tensor field with modified indices.
    new_signature : numpy.ndarray
        Tensor signature after modifications. Same shape as `tensor_signature`.
    """
    # Determine the rank and dimensionality
    # of the tensor field.
    rank = tensor_signature.size
    ndim = tensor_field.ndim

    # Start building the indices arrays from
    # strings. This is the core of the function and is
    # the cumulatively passed to numpy.einsum.
    letters = string.ascii_lowercase
    caps = string.ascii_uppercase

    # Field (non-tensor) indices
    tensor_indices = list(
        letters[ndim - rank : ndim]
    )  # We need to manipulate later -> list.
    cap_tensor_indices = list(
        caps[ndim - rank : ndim]
    )  # We need to manipulate later -> list.
    output_indices = tensor_indices.copy()

    # Start building the flags that are going to contain
    # the operation information for this function. We're
    # going to utilize these to store the strings and optimize
    # checks.
    __operands__ = [tensor_field]
    __lhs__: Any = [
        "..." + "".join(tensor_indices),
    ]
    __rhs__: Any = output_indices

    # Copy and update the signature
    new_signature = tensor_signature.copy()

    # Pass through each of the tensor's indices and
    # check if we need to alter it based on the passed through
    # indices. We always alter to the opposite variance.
    for index, variance in enumerate(tensor_signature):
        # Check if we are actually making any modifications to
        # this tensor.
        if index not in indices:
            continue

        # A modification is required - check that the
        # metric / inverse metric was provided.
        if variance == 1 and metric_field is None:
            raise ValueError(
                "Lowering a contravariant index requires the metric tensor."
            )
        elif variance == -1 and inverse_metric_field is None:
            raise ValueError(
                "Raising a covariant index requires the inverse metric tensor."
            )

        # Add the correct metric to the __operands__ list.
        __operands__.append(metric_field if variance == 1 else inverse_metric_field)

        # Every operation looks like abcd...,aA,bB,..., ABcDe... so we pull out
        # the capitalized and lowercase version of the current index and
        # add a metric element with that signature.
        __lhs__.append(f"...{tensor_indices[index]}{cap_tensor_indices[index]}")
        __rhs__[index] = __rhs__[index].upper()
        new_signature[index] *= -1  # Flip variance

    # Now we construct the full einsum expression. We keep in
    # mind that EVERYTHING gets the same set of field indices.
    __rhs__ = "..." + "".join(__rhs__)
    __lhs__ = ",".join(__lhs__)
    einsum_expr = __lhs__ + "->" + __rhs__
    return np.einsum(einsum_expr, *__operands__, out=out, **kwargs), new_signature


def _dense_adjust_tensor_signature_diagonal_metric(
    tensor_field: np.ndarray,
    indices: List[int],
    tensor_signature: np.ndarray,
    metric_field: Optional[np.ndarray] = None,
    inverse_metric_field: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusts specified tensor indices by scaling with diagonal metric components (or inverse).

    This function applies index raising or lowering on the specified tensor indices
    assuming a diagonal metric. It avoids full matrix multiplication and instead uses
    elementwise multiplication or division, with optional in-place behavior.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        The input tensor field to be transformed. Must have shape ``(..., I₁, ..., I_r)`` with
        `r` trailing tensor index axes.

        The array should have shape ``(F₁, ..., F_m, I₁, ..., I_r)``, where:

        - ``(F₁, ..., F_m)`` are the field (spatial or grid) dimensions,
        - ``(I₁, ..., I_r)`` are the tensor index dimensions, and
        - `r` is the tensor rank (i.e., the number of tensor indices, inferred from `tensor_signature`).
    indices : List[int]
        The indices among the trailing `rank` axes to modify.
    tensor_signature : numpy.ndarray
        Array of integers: +1 for contravariant, -1 for covariant indices.
    metric_field : numpy.ndarray, optional
        The diagonal metric entries (e.g., (..., N)).
    inverse_metric_field : numpy.ndarray, optional
        The diagonal inverse metric entries (e.g., (..., N)).
    out : numpy.ndarray, optional
        Output array. If provided, will be modified in-place.

    Returns
    -------
    result : numpy.ndarray
        Tensor field with modified indices.
    new_signature : numpy.ndarray
        Tensor signature after modifications. Same shape as `tensor_signature`.

    Examples
    --------

    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_utils import dense_adjust_tensor_signature
    >>>
    >>> # Coordinate values
    >>> r = 2.0
    >>> theta = np.pi / 4
    >>> sin2 = np.sin(theta) ** 2
    >>>
    >>> # Define a rank-2 tensor with a single nonzero component T^{theta,phi}
    >>> T = np.zeros((3, 3))
    >>> T[1, 2] = 1.0
    >>>
    >>> # Diagonal metric and inverse metric
    >>> g = np.array([1.0, r**2, r**2 * sin2])
    >>> g_inv = np.array([1.0, 1/r**2, 1/(r**2 * sin2)])
    >>>
    >>> # Lower both indices (contravariant → covariant)
    >>> _dense_adjust_tensor_signature_diagonal_metric(
    ...     tensor_field=T,
    ...     indices=[0, 1],
    ...     tensor_signature=np.array([+1, +1]),
    ...     metric_field=g
    ... )
    (array([[0., 0., 0.],
           [0., 0., 8.],
           [0., 0., 0.]]), array([-1, -1]))

    >>> # Raise both indices back (covariant → contravariant)
    >>> T_lowered = T
    >>> _dense_adjust_tensor_signature_diagonal_metric(
    ...     tensor_field=T_lowered,
    ...     indices=[0, 1],
    ...     tensor_signature=np.array([-1, -1]),
    ...     inverse_metric_field=g_inv
    ... )
    (array([[0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.125],
           [0.   , 0.   , 0.   ]]), array([1, 1]))
    """
    # Determine the rank and dimensionality
    # of the tensor field.
    rank = tensor_signature.size
    ndim = tensor_field.ndim

    # Manage the "out" array. If it hasn't been created we need to
    # create it.
    if out is None:
        out = np.empty_like(tensor_field, **kwargs)

    if out is not tensor_field:
        out[...] = tensor_field

    # Pass through each of the indices in the
    # tensor field and check if we are altering it. If we are, we need to
    # simply scale all elements by the values of the correct metric.
    new_signature = tensor_signature.copy()
    for index, variance in enumerate(tensor_signature):
        # Check if we are actually making any modifications to
        # this tensor.
        if index not in indices:
            continue

        # Create the broadcast shape for the metric tensor
        # to compy with for this index.
        reshape = (
            list(tensor_field.shape[: ndim - rank])
            + [
                1,
            ]
            * rank
        )
        reshape[ndim - rank + index] = -1

        if variance == 1:
            # We are going to try to lower the index. This is either
            # a multiplication with the metric or a division with the
            # inverse metric.
            if metric_field is not None:
                out *= metric_field.reshape(reshape)
            elif inverse_metric_field is not None:
                out /= inverse_metric_field.reshape(reshape)
            else:
                raise ValueError(
                    "Cannot lower contravariant index: no metric or inverse metric provided."
                )

        elif variance == -1:
            # We are going to try to lower the index. This is either
            # a multiplication with the metric or a division with the
            # inverse metric.
            if inverse_metric_field is not None:
                out *= inverse_metric_field.reshape(reshape)
            elif metric_field is not None:
                out /= metric_field.reshape(reshape)
            else:
                raise ValueError(
                    "Cannot raise covariant index: no metric or inverse metric provided."
                )

        # Flip the new variance
        new_signature[index] *= -1

    return out, new_signature


def _dense_transform_tensor(
    tensor_field: np.ndarray,
    tensor_signature: np.ndarray,
    jacobian_field: Optional[np.ndarray] = None,
    inverse_jacobian_field: Optional[np.ndarray] = None,
    out=None,
    **kwargs,
) -> np.ndarray:
    """
    Apply a coordinate transformation to a tensor field using the Jacobian and/or its inverse.

    This function transforms each tensor index of the field according to its variance type:

        - Contravariant indices (denoted +1 in `tensor_signature`) transform using the Jacobian,
        - Covariant indices (denoted -1) transform using the inverse Jacobian.

    The transformation is performed via Einstein summation (einsum). The coordinate transformation
    matrices may be functions of space and should be broadcast-compatible with the grid dimensions.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        The input tensor field of shape (..., I₁, ..., I_r), where the last `r` axes
        represent the tensor indices.
    tensor_signature : numpy.ndarray
        Array of shape `(rank,)` with entries:
            - +1 for contravariant indices (upper indices),
            - -1 for covariant indices (lower indices).
    jacobian_field : numpy.ndarray, optional
        The Jacobian matrix ∂x_old / ∂x_new of shape (..., N, N), used to transform contravariant indices.
    inverse_jacobian_field : numpy.ndarray, optional
        The inverse Jacobian ∂x_new / ∂x_old of shape (..., N, N), used to transform covariant indices.
    **kwargs :
        Additional keyword arguments passed to :py:func:`numpy.einsum` (e.g., optimize=True).

    Returns
    -------
    numpy.ndarray
        The transformed tensor field in the new coordinate system. Shape is identical to `tensor_field`.

    Raises
    ------
    ValueError
        If a required transformation matrix is missing for the given tensor signature.

    Examples
    --------
    >>> import numpy as np
    >>> T = np.random.rand(10, 3)  # Vector field (rank-1)
    >>> sig = np.array([+1])       # Contravariant
    >>> J = np.eye(3)[None, ...]   # Identity Jacobian
    >>> _dense_transform_tensor(T, sig, jacobian=J).shape
    (10, 3)
    """
    rank = tensor_signature.size
    ndim = tensor_field.ndim

    if rank == 0:
        return tensor_field  # Scalar field, no transformation needed

    # Set up einsum notation
    letters = string.ascii_lowercase
    caps = string.ascii_uppercase

    field_ndim = ndim - rank
    field_indices = letters[:field_ndim]
    tensor_indices = list(letters[field_ndim:ndim])
    transformed_indices = list(tensor_indices)
    capital_indices = list(caps[field_ndim:ndim])

    lhs_terms = [field_indices + "".join(tensor_indices)]
    operands = [tensor_field]

    for i, variance in enumerate(tensor_signature):
        if variance == +1:
            if jacobian_field is None:
                raise ValueError(
                    "Jacobian is required to transform contravariant indices."
                )
            matrix = jacobian_field
        elif variance == -1:
            if inverse_jacobian_field is None:
                raise ValueError(
                    "Inverse Jacobian is required to transform covariant indices."
                )
            matrix = inverse_jacobian_field
        else:
            raise ValueError(
                f"Invalid tensor signature: expected +1 or -1, got {variance} at index {i}"
            )

        lhs_terms.append(f"{field_indices}{capital_indices[i]}{tensor_indices[i]}")
        transformed_indices[i] = capital_indices[i]
        operands.append(matrix)

    # Construct and evaluate einsum
    result_expr = (
        f"{','.join(lhs_terms)}->{field_indices}{''.join(transformed_indices)}"
    )
    return np.einsum(result_expr, *operands, out=out, **kwargs)


# ------------------------------------------- #
# User level utilities                        #
# ------------------------------------------- #
# These are wrapper methods to ensure that the private methods
# above are correctly handled in terms of error handling and validation.
def dense_contract_with_metric(
    tensor_field: np.ndarray,
    metric_field: np.ndarray,
    index: int,
    rank: int,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Contract a tensor index with the provided metric tensor.

    This function contracts one of the tensor indices of the input tensor field
    with the supplied metric. If the metric is diagonal (1D or (..., N)), an optimized
    elementwise contraction is used.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        The tensor field whose index signature is to be adjusted.
        The array should have shape ``(F₁, ..., F_m, I₁, ..., I_r)``, where:

        - ``(F₁, ..., F_m)`` are the field (spatial or grid) dimensions,
        - ``(I₁, ..., I_r)`` are the tensor index dimensions, and
        - `r` is the tensor rank (i.e., the number of tensor indices, inferred from `tensor_signature`).
    metric_field : numpy.ndarray
        Metric tensor. Must be either:

        - Full matrix of shape ``(..., N, N)``, or
        - Diagonal-only array of shape ``(..., N)``.

    index : int
        Index (among the trailing `rank` tensor indices) to contract.
    rank : int
        The number of trailing axes in `tensor_field` that represent tensor indices.
    out : numpy.ndarray, optional
        Optional output array to store the result. If provided, must have the same shape and dtype
        as the expected output, and will be used for in-place storage.
    **kwargs :
        Additional keyword arguments forwarded to low-level routines (e.g., ``optimize=True`` for :func:`numpy.einsum`).

    Returns
    -------
    numpy.ndarray
        A new tensor field with the specified index contracted, replacing that axis with the contracted result.
        The shape of the output reflects broadcasting between `tensor_field` and `metric_field`.


    Raises
    ------
    ValueError
        If shapes are incompatible or inputs are invalid.

    Examples
    --------

    Contract a rank-1 tensor field with a full metric (identity):

    >>> import numpy as np
    >>> T = np.random.rand(5, 5, 3)  # shape: (grid_x, grid_y, vector_index)
    >>> g = np.eye(3)[np.newaxis, np.newaxis, :, :] * np.ones((5, 5, 1, 1))  # shape: (5, 5, 3, 3)
    >>> result = dense_contract_with_metric(T, g, index=0, rank=1)
    >>> result.shape
    (5, 5, 3)

    Contract with a diagonal metric:

    >>> g_diag = np.array([1.0, 2.0, 3.0])[np.newaxis, np.newaxis, :] * np.ones((5, 5, 1))  # shape: (5, 5, 3)
    >>> result = dense_contract_with_metric(T, g_diag, index=0, rank=1)
    >>> result.shape
    (5, 5, 3)

    Use broadcasting between mismatched shapes:

    >>> T = np.random.rand(5, 1, 3)                 # shape: (5, 3)
    >>> g = np.ones((5, 7, 3, 3))                # shape: (5, 7, 3, 3)
    >>> result = dense_contract_with_metric(T, g, index=0, rank=1)
    >>> result.shape
    (5, 7, 3)

    >>> g_diag = np.ones((5, 7, 3))              # diagonal version
    >>> result = dense_contract_with_metric(T, g_diag, index=0, rank=1)
    >>> result.shape
    (5, 7, 3)

    Notes
    -----
    Internally, this function dispatches to one of two lower-level routines:

    - :func:`_dense_contract_index_with_metric`: used for full metric tensors of shape ``(..., N, N)``.
      This uses a batched contraction implemented via `np.einsum` with automatic broadcasting.

    - :func:`_dense_contract_index_with_diagonal_metric`: used for diagonal metrics with shape ``(..., N)``.
      This uses elementwise multiplication and is significantly faster and lighter on memory.

    These helper functions follow NumPy-style broadcasting and allow tensor fields and metric tensors
    to differ in shape along non-contracted dimensions, as long as they are broadcast-compatible.
    No shape validation is performed beyond what NumPy itself requires.

    See Also
    --------
    numpy.einsum : Batched Einstein summation (used internally).
    numpy.multiply : Elementwise multiplication (used for diagonal metric contraction).
    """
    if rank > tensor_field.ndim:
        raise ValueError("Rank exceeds number of tensor field dimensions.")
    if not (0 <= index < rank):
        raise ValueError("Index must be within the range [0, rank).")

    # Determine grid shape and index size
    grid_shape = tensor_field.shape[:-rank]
    axis_size = tensor_field.shape[-rank + index]

    # Handle full metric case (..., N, N)
    if metric_field.ndim >= 2 and metric_field.shape[-2:] == (axis_size, axis_size):
        try:
            np.broadcast(np.empty(grid_shape), np.empty(metric_field.shape[:-2]))
        except ValueError:
            raise ValueError(
                f"Full metric tensor shape is not broadcast-compatible with tensor field grid shape:\n"
                f"grid_shape: {grid_shape}, metric shape: {metric_field.shape}"
            )
        return _dense_contract_index_with_metric(
            tensor_field, metric_field, index, rank, out=out, **kwargs
        )

    # Handle diagonal metric case (..., N)
    elif metric_field.ndim >= 1 and metric_field.shape[-1] == axis_size:
        try:
            np.broadcast(np.empty(grid_shape), np.empty(metric_field.shape[:-1]))
        except ValueError:
            raise ValueError(
                f"Diagonal metric tensor shape is not broadcast-compatible with tensor field grid shape:\n"
                f"grid_shape: {grid_shape}, metric shape: {metric_field.shape}"
            )
        return _dense_contract_index_with_diagonal_metric(
            tensor_field, metric_field, index, rank, out=out
        )

    # Not compatible with either expected form
    else:
        raise ValueError(
            "Metric field must be either (..., N, N) or (..., N), with N matching the contracted axis size."
        )


def dense_raise_index(
    tensor_field: np.ndarray,
    index: int,
    rank: int,
    inverse_metric_field: np.ndarray,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Raises a specified index of a tensor field using the inverse metric tensor.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        The tensor field whose index signature is to be adjusted.
        The array should have shape ``(F₁, ..., F_m, I₁, ..., I_r)``, where:

        - ``(F₁, ..., F_m)`` are the field (spatial or grid) dimensions,
        - ``(I₁, ..., I_r)`` are the tensor index dimensions, and
        - `r` is the tensor rank (i.e., the number of tensor indices, inferred from `tensor_signature`).
    index : int
        The index to lower, ranging from ``0`` to ``rank-1``.
    rank : int
        The tensor rank (number of tensor indices, not including grid dimensions).
    inverse_metric_field : numpy.ndarray, optional
        The inverse metric tensor used to raise covariant indices. This can be either:

        - A full inverse metric of shape (..., N, N), or
        - A diagonal inverse metric of shape (..., N).

        Must match the metric type (diagonal vs full) and be broadcast-compatible with `tensor_field`.
    out : numpy.ndarray, optional
        Optional output array to store the result. If provided, must have the same shape and dtype
        as the expected output, and will be used for in-place storage.

    Returns
    -------
    numpy.ndarray
        A tensor field with the specified index raised. Has the same shape as `tensor_field`.

    See Also
    --------
    dense_lower_index
    dense_adjust_tensor_signature

    Raises
    ------
    ValueError
        If the input shapes or indices are invalid.

    Examples
    --------
    In spherical coordinates, if you have a covariant vector:

    .. math::

        {\bf v} = r {\bf e}^\theta

    Then the contravariant version is:

    .. math::

        v^\theta = g^{\theta \mu} v_\mu = g^{\theta \theta} v_\theta = \frac{1}{r^2} v_{\theta} = \frac{1}{r}.

    Let's see this work in practice:

    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_utils import dense_raise_index
    >>>
    >>> # Construct the vector field at a point.
    >>> # We'll need the metric (inverse) and the vector field at the point.
    >>> r,theta = 2,np.pi/4
    >>> v_cov = np.asarray([0,r,0])
    >>>
    >>> # Construct the metric tensor.
    >>> g_inv = np.diag([1, 1 / r**2, 1 / (r**2 * np.sin(theta)**2)])
    >>>
    >>> # Now we can use the inverse metric to raise the tensor index.
    >>> dense_raise_index(v_cov, index=0, rank=1, inverse_metric_field=g_inv)
    array([0. , 0.5, 0. ])

    A 2D example on a spherical grid:

    .. plot::
        :include-source: True

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pymetric.differential_geometry.dense_utils import dense_raise_index
        >>>
        >>> # Define 2D spherical grid
        >>> r = np.linspace(1, 2, 100)
        >>> theta = np.linspace(0.1, np.pi - 0.1, 100)
        >>> R, THETA = np.meshgrid(r, theta, indexing="ij")
        >>>
        >>> # Define a covariant vector field: v_θ = R
        >>> v_cov = np.zeros(R.shape + (2,))
        >>> v_cov[..., 1] = R  # non-zero only in theta direction
        >>>
        >>> # Define the inverse metric: g^{rr} = 1, g^{θθ} = 1 / r^2
        >>> g_inv = np.zeros(R.shape + (2,))
        >>> g_inv[..., 0] = 1
        >>> g_inv[..., 1] = 1 / R**2
        >>>
        >>> # Raise the index
        >>> v_contra = dense_raise_index(v_cov, index=0, rank=1, inverse_metric_field=g_inv)
        >>>
        >>> # Plot the raised θ-component
        >>> _ = plt.figure(figsize=(6, 4))
        >>> im = plt.imshow(v_contra[..., 1].T, extent=[1, 2, 0.1, np.pi - 0.1], aspect="auto", origin="lower")
        >>> _ = plt.colorbar(im, label="Raised $v^\theta$")
        >>> _ = plt.xlabel("r")
        >>> _ = plt.ylabel(r"$\theta$")
        >>> _ = plt.title(r"Contravariant Component $v^\theta = r$")
        >>> _ = plt.tight_layout()
        >>> _ = plt.show()

    """
    if rank > tensor_field.ndim:
        raise ValueError(
            "Rank must be less than or equal to the number of tensor_field dimensions."
        )
    if not (0 <= index < rank):
        raise ValueError("Index must be in the range [0, rank).")

    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]
    index_size = tensor_shape[index]

    # Dispatch based on whether the metric is full or diagonal
    if inverse_metric_field.shape == grid_shape + (index_size, index_size):
        return _dense_contract_index_with_metric(
            tensor_field, inverse_metric_field, index, rank, out=out, **kwargs
        )
    elif inverse_metric_field.shape == grid_shape + (index_size,):
        return _dense_contract_index_with_diagonal_metric(
            tensor_field, inverse_metric_field, index, rank, out=out
        )
    else:
        raise ValueError(
            f"Metric shape {inverse_metric_field.shape} is incompatible with tensor shape {tensor_field.shape} "
            f"and expected index size {index_size}."
        )


def dense_lower_index(
    tensor_field: np.ndarray,
    index: int,
    rank: int,
    metric_field: np.ndarray,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Lowers a specified index of a tensor field using the metric tensor.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        The tensor field whose index signature is to be adjusted.
        The array should have shape ``(F₁, ..., F_m, I₁, ..., I_r)``, where:

        - ``(F₁, ..., F_m)`` are the field (spatial or grid) dimensions,
        - ``(I₁, ..., I_r)`` are the tensor index dimensions, and
        - `r` is the tensor rank (i.e., the number of tensor indices, inferred from `tensor_signature`).
    index : int
        The index to lower, ranging from ``0`` to ``rank-1``.
    rank : int
        The tensor rank (number of tensor indices, not including grid dimensions).
    metric_field : numpy.ndarray, optional
        The metric tensor used to lower contravariant indices. This can be either:

        - A full metric of shape (..., N, N), where N is the size of each tensor slot.
        - A diagonal metric of shape (..., N), representing only the diagonal components.

        Must be broadcast-compatible with the grid shape of `tensor_field`.
    out : numpy.ndarray, optional
        Optional output array to store the result. If provided, must have the same shape and dtype
        as the expected output, and will be used for in-place storage.


    Returns
    -------
    numpy.ndarray
        A tensor field with the specified index lowered. Has the same shape as `tensor_field`.

    See Also
    --------
    raise_index
    adjust_tensor_signature

    Raises
    ------
    ValueError
        If the input shapes or indices are invalid.

    Examples
    --------
    In spherical coordinates, the metric tensor is diagonal:

    .. math::

        g_{\mu\nu} = \mathrm{diag}(1, r^2, r^2 \sin^2\theta)

    If you have a contravariant vector with components:

    .. math::

        v^\mu = [0,\ r,\ 0]

    then the covariant version (with one index lowered) is given by:

    .. math::

        v_\mu = g_{\mu\nu} v^\nu = [0,\ r^3,\ 0]

    Let's see this work in practice:

    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_utils import dense_lower_index
    >>>
    >>> # Construct the contravariant vector field at a point.
    >>> r, theta = 2.0, np.pi / 4
    >>> v_contra = np.array([0.0, r, 0.0])  # v^theta = r
    >>>
    >>> # Construct the spherical coordinate metric (diagonal)
    >>> g_diag = np.array([1.0, r**2, r**2 * np.sin(theta)**2])  # g_{rr}, g_{θθ}, g_{φφ}
    >>>
    >>> # Lower the index
    >>> v_cov = dense_lower_index(v_contra, index=0, rank=1, metric_field=g_diag)
    >>> v_cov
    array([0., 8., 0.])
    """
    if rank > tensor_field.ndim:
        raise ValueError(
            "Rank must be less than or equal to the number of tensor_field dimensions."
        )
    if not (0 <= index < rank):
        raise ValueError("Index must be in the range [0, rank).")

    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]
    index_size = tensor_shape[index]

    # Dispatch based on whether the metric is full or diagonal
    if metric_field.shape == grid_shape + (index_size, index_size):
        return _dense_contract_index_with_metric(
            tensor_field, metric_field, index, rank, out=out, **kwargs
        )
    elif metric_field.shape == grid_shape + (index_size,):
        return _dense_contract_index_with_diagonal_metric(
            tensor_field, metric_field, index, rank, out=out
        )
    else:
        raise ValueError(
            f"Metric shape {metric_field.shape} is incompatible with tensor shape {tensor_field.shape} "
            f"and expected index size {index_size}."
        )


def dense_adjust_tensor_signature(
    tensor_field: np.ndarray,
    indices: List[int],
    tensor_signature: np.ndarray,
    metric_field: Optional[np.ndarray] = None,
    inverse_metric_field: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Adjusts multiple indices of a tensor field by raising or lowering them using the metric or inverse metric.

    This function modifies the variance (covariant vs. contravariant) of selected tensor indices
    by contracting them with either the metric tensor or its inverse. The transformation can be
    performed efficiently for both full (2D) and diagonal (1D) metric representations.

    The adjustment is specified through a signature array, where each element corresponds to the
    current variance of a tensor index:

        - ``+1``: Contravariant (upper index) → lowering uses the metric
        - ``-1``: Covariant (lower index) → raising uses the inverse metric

    Only the indices listed in `indices` will be modified; others are left untouched.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        The tensor field whose index signature is to be adjusted.
        The array should have shape ``(F₁, ..., F_m, I₁, ..., I_r)``, where:

        - ``(F₁, ..., F_m)`` are the field (spatial or grid) dimensions,
        - ``(I₁, ..., I_r)`` are the tensor index dimensions, and
        - `r` is the tensor rank (i.e., the number of tensor indices, inferred from `tensor_signature`).
    indices : list of int
        List of indices (from 0 to `rank - 1`) specifying which tensor slots to modify.
        These refer to the positions within the tensor portion of the array (i.e., the last `rank` axes).
        Indices may appear in any order. If the same index appears multiple times, the corresponding
        variance transformations will be applied sequentially, potentially resulting in no net change.
    tensor_signature : numpy.ndarray
        A 1D array of integers of shape ``(rank,)`` specifying the current variance of each tensor index.
        Each entry must be either ``+1`` (indicating a contravariant index) or ``-1`` (indicating a covariant index).
        The order of entries corresponds to the last ``rank`` axes of ``tensor_field``, and defines how each
        tensor slot is currently positioned with respect to the coordinate basis.
    metric_field : numpy.ndarray, optional
        The metric tensor used to lower contravariant indices. This can be either:

        - A full metric of shape (..., N, N), where N is the size of each tensor slot.
        - A diagonal metric of shape (..., N), representing only the diagonal components.

        Must be broadcast-compatible with the grid shape of `tensor_field`.

    inverse_metric_field : numpy.ndarray, optional
        The inverse metric tensor used to raise covariant indices. This can be either:

        - A full inverse metric of shape (..., N, N), or
        - A diagonal inverse metric of shape (..., N).

        Must match the metric type (diagonal vs full) and be broadcast-compatible with `tensor_field`.
    out : numpy.ndarray, optional
        Optional output array to store the result. If provided, must have the same shape and dtype
        as the expected output, and will be used for in-place storage.
    **kwargs :
        Additional keyword arguments forwarded to the underlying einsum or broadcasting routines.
        These may include:

        - ``out`` : Optional output array to hold the result. If provided, the operation may be performed in-place.
        - Array creation keywords such as ``dtype`` or ``order`` when a new output array is allocated (e.g., via `np.empty`).
        - Einsum-specific options such as ``optimize=True`` for path optimization.

        The accepted keywords depend on whether a diagonal or full metric is used, as different implementations
        (`np.einsum` vs broadcasting and `np.empty`) are internally dispatched.

    Returns
    -------
    numpy.ndarray
        The resulting tensor field with modified index variances. Shape is identical to the input.

    Raises
    ------
    ValueError
        If input shapes are inconsistent or if necessary metrics are missing.

    See Also
    --------
    raise_index
    lower_index

    Examples
    --------

    Consider a tensor defined in spherical coordinates :math:`(r,\theta,\phi)` with a rank-2 structure.
    The metric tensor in these coordinates is diagonal:

    .. math::

        g_{ij} = \text{diag}(1, r^2, r^2 \sin^2 \theta)

    The inverse metric is:

    .. math::

        g^{ij} = \text{diag}(1, 1/r^2, 1/(r^2 \sin^2 \theta))

    We can lower or raise indices of a tensor as follows:

    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_utils import dense_adjust_tensor_signature
    >>>
    >>> # Coordinate values
    >>> r = 2.0
    >>> theta = np.pi / 4
    >>> sin2 = np.sin(theta) ** 2
    >>>
    >>> # Define a rank-2 tensor with a single nonzero component T^{theta,phi}
    >>> T = np.zeros((3, 3))
    >>> T[1, 2] = 1.0
    >>>
    >>> # Diagonal metric and inverse metric
    >>> g = np.array([1.0, r**2, r**2 * sin2])
    >>> g_inv = np.array([1.0, 1/r**2, 1/(r**2 * sin2)])
    >>>
    >>> # Lower both indices (contravariant → covariant)
    >>> dense_adjust_tensor_signature(
    ...     tensor_field=T,
    ...     indices=[0, 1],
    ...     tensor_signature=np.array([+1, +1]),
    ...     metric_field=g
    ... )
    (array([[0., 0., 0.],
           [0., 0., 8.],
           [0., 0., 0.]]), array([-1, -1]))

    >>> # Raise both indices back (covariant → contravariant)
    >>> T_lowered = T
    >>> dense_adjust_tensor_signature(
    ...     tensor_field=T_lowered,
    ...     indices=[0, 1],
    ...     tensor_signature=np.array([-1, -1]),
    ...     inverse_metric_field=g_inv
    ... )
    (array([[0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.125],
           [0.   , 0.   , 0.   ]]), array([1, 1]))
    """
    # Perform shape validation. The tensor signature will imply a rank which
    # must be larger than the field dimensions. We also check that the metric / inverse metric
    # are correctly shaped given the rank and number of dimensions and that
    # the indices never exceed the rank.
    rank = tensor_signature.size
    if rank == 0:
        raise ValueError("Tensor must have positive rank.")

    field_ndim = tensor_field.ndim - rank

    # Check that the field ndim is non-negative and then check the metric and inverse metric.
    if field_ndim < 0:
        raise ValueError(
            f"Tensor of rank {rank} requires at least {rank} dimensions in input array."
        )

    grid_shape = tensor_field.shape[:field_ndim]
    tensor_shape = tensor_field.shape[field_ndim:]

    # Perform the metric tensor validation process. We need to check that, whichever one
    # we got, it gets triaged into the correct call pattern. If we have a type mismatch,
    # we need to raise an error.
    _metric_type, _inv_metric_type = None, None
    if metric_field is not None:
        if metric_field.shape == grid_shape + (tensor_shape[0], tensor_shape[0]):
            _metric_type = "full"
        elif metric_field.shape == grid_shape + (tensor_shape[0],):
            _metric_type = "diag"
        else:
            raise ValueError(
                f"Metric tensor shape ({metric_field.shape})"
                f" does not match field grid and tensor index dimensions."
            )

    if inverse_metric_field is not None:
        if inverse_metric_field.shape == grid_shape + (
            tensor_shape[0],
            tensor_shape[0],
        ):
            _inv_metric_type = "full"
        elif inverse_metric_field.shape == grid_shape + (tensor_shape[0],):
            _inv_metric_type = "diag"
        else:
            raise ValueError(
                "Inverse metric tensor shape does not match field grid and tensor index dimensions."
            )

    # Ensure metric and inverse metric agree in type
    if _metric_type and _inv_metric_type and _metric_type != _inv_metric_type:
        raise ValueError(
            "Metric and inverse metric types (diagonal vs full) must match."
        )

    # Determine final type if only one metric is provided
    metric_type = _metric_type or _inv_metric_type
    if metric_type is None:
        raise ValueError(
            "At least one of `metric_field` or `inverse_metric_field` must be provided."
        )

        # Validate that all indices are within rank range
    if any(i < 0 or i >= rank for i in indices):
        raise ValueError(
            f"All indices must be in the range [0, {rank}). Got: {indices}"
        )

    # Dispatch
    if metric_type == "diag":
        return _dense_adjust_tensor_signature_diagonal_metric(
            tensor_field,
            indices,
            tensor_signature,
            metric_field=metric_field,
            inverse_metric_field=inverse_metric_field,
            out=out,
            **kwargs,
        )
    else:
        return _dense_adjust_tensor_signature(
            tensor_field,
            indices,
            tensor_signature,
            metric_field=metric_field,
            inverse_metric_field=inverse_metric_field,
            out=out,
            **kwargs,
        )


def dense_transform_tensor_field(
    tensor_field: np.ndarray,
    indices: List[int],
    tensor_signature: np.ndarray,
    jacobian: Optional[np.ndarray] = None,
    inverse_jacobian: Optional[np.ndarray] = None,
    out=None,
    **kwargs,
) -> np.ndarray:
    """
    Apply a coordinate transformation to a tensor field based on its signature and selected indices.

    This function transforms selected indices of a tensor field using the Jacobian or its inverse,
    depending on the variance (contravariant/covariant) of each index as specified by `tensor_signature`.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape (..., I₁, ..., I_r), where the last `r` axes are tensor indices.
    indices : List[int]
        Indices to be transformed (among the last `rank` axes). Each must be in [0, rank).
    tensor_signature : numpy.ndarray
        Array of +1 (contravariant) or -1 (covariant) values of shape `(rank,)`.
    jacobian : numpy.ndarray, optional
        Jacobian matrix ∂x_old/∂x_new with shape (..., N, N), required for transforming contravariant indices.
    inverse_jacobian : numpy.ndarray, optional
        Inverse Jacobian ∂x_new/∂x_old with shape (..., N, N), required for covariant indices.
    out : numpy.ndarray, optional
        Optional output array to store the result. If provided, must have the same shape and dtype
        as the expected output, and will be used for in-place storage.
    **kwargs :
        Additional arguments passed to the underlying einsum transformation.

    Returns
    -------
    numpy.ndarray
        The transformed tensor field.

    Raises
    ------
    ValueError
        If tensor signature or Jacobians are incompatible with input dimensions.

    See Also
    --------
    adjust_tensor_signature
    _transform_tensor_signature
    """
    rank = tensor_signature.size
    if rank == 0:
        return tensor_field
    if any(i < 0 or i >= rank for i in indices):
        raise ValueError(
            f"All indices must be in the range [0, {rank}). Got: {indices}"
        )
    if tensor_field.ndim < rank:
        raise ValueError(
            f"Tensor field has {tensor_field.ndim} dims but signature implies rank {rank}."
        )

    # Validate that Jacobians have appropriate shape
    field_ndim = tensor_field.ndim - rank
    grid_shape = tensor_field.shape[:field_ndim]
    tensor_shape = tensor_field.shape[field_ndim:]

    jacobian_type = None
    if jacobian is not None:
        if jacobian.shape == grid_shape + (tensor_shape[0], tensor_shape[0]):
            jacobian_type = "full"
        else:
            raise ValueError(
                "Jacobian shape must be (..., N, N), matching field grid dimensions."
            )

    inv_jacobian_type = None
    if inverse_jacobian is not None:
        if inverse_jacobian.shape == grid_shape + (tensor_shape[0], tensor_shape[0]):
            inv_jacobian_type = "full"
        else:
            raise ValueError(
                "Inverse Jacobian shape must be (..., N, N), matching field grid dimensions."
            )

    if jacobian_type and inv_jacobian_type and jacobian_type != inv_jacobian_type:
        raise ValueError(
            "Jacobian and inverse Jacobian must match in structure (both full or both diag)."
        )

    if not (jacobian or inverse_jacobian):
        raise ValueError(
            "At least one of `jacobian` or `inverse_jacobian` must be provided."
        )

    # Mask signature for selected indices only
    effective_signature = tensor_signature.copy()
    for i in range(rank):
        if i not in indices:
            effective_signature[i] = 0

    return _dense_transform_tensor(
        tensor_field=tensor_field,
        tensor_signature=effective_signature,
        jacobian=jacobian,
        inverse_jacobian=inverse_jacobian,
        out=out,
        **kwargs,
    )


def dense_permute_tensor_indices(
    tensor_field: np.ndarray, permutation: List[int], rank: int
) -> np.ndarray:
    """
    Permutes the order of the tensor indices in the last `rank` dimensions of the tensor field.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Input tensor with shape (..., i₁, ..., i_r), where the last `rank` axes are tensor indices.
    permutation : list of int
        Permutation to apply to the tensor indices. Must be of length `rank` and contain a permutation of [0, ..., rank-1].
    rank : int
        The rank (number of tensor axes) of the tensor.

    Returns
    -------
    numpy.ndarray
        Tensor field with permuted tensor indices.

    Raises
    ------
    ValueError
        If permutation is not valid.

    Examples
    --------
    >>> import numpy as np
    >>> T = np.random.rand(4, 4, 3, 3)
    >>> permute_tensor_indices(T, [1, 0], rank=2).shape
    (4, 4, 3, 3)
    """
    if sorted(permutation) != list(range(rank)):
        raise ValueError(
            f"Invalid permutation: expected a permutation of [0, ..., {rank - 1}]"
        )

    ndim = tensor_field.ndim
    field_ndim = ndim - rank
    full_permutation = list(range(field_ndim)) + [field_ndim + i for i in permutation]
    return tensor_field.transpose(full_permutation)


def dense_tensor_product(
    tensor_a: np.ndarray,
    type_a: tuple[int, int],
    tensor_b: np.ndarray,
    type_b: tuple[int, int],
) -> np.ndarray:
    """
    Computes the tensor product of two tensor fields with specified (p, q) types,
    ensuring that the result has contravariant indices first and covariant indices last.

    Parameters
    ----------
    tensor_a : numpy.ndarray
        First tensor field of shape (..., i₁, ..., i_p, j₁, ..., j_q).
    type_a : tuple[int, int]
        Tuple (p, q) specifying number of contravariant and covariant indices of `tensor_a`.
    tensor_b : numpy.ndarray
        Second tensor field of shape (..., k₁, ..., k_r, l₁, ..., l_t).
    type_b : tuple[int, int]
        Tuple (r, t) specifying number of contravariant and covariant indices of `tensor_b`.

    Returns
    -------
    numpy.ndarray
        The tensor product with type (p + r, q + t), shape (..., i₁, ..., i_p, k₁, ..., k_r, j₁, ..., j_q, l₁, ..., l_t).

    Raises
    ------
    ValueError
        If tensor shapes are incompatible or types are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.random.rand(4, 3)     # (1, 0) vector
    >>> B = np.random.rand(4, 3)     # (0, 1) covector
    >>> tensor_product(A, (1, 0), B, (0, 1)).shape
    (4, 3, 3)
    """
    # Parse tensor types
    p, q = type_a
    r, t = type_b

    # Get raw dimensions
    ndim_a = tensor_a.ndim
    ndim_b = tensor_b.ndim
    rank_a = p + q
    rank_b = r + t

    field_shape_a = tensor_a.shape[: ndim_a - rank_a]
    field_shape_b = tensor_b.shape[: ndim_b - rank_b]

    # Validate broadcastability of the field parts
    try:
        broadcast_shape = np.broadcast_shapes(field_shape_a, field_shape_b)
    except ValueError:
        raise ValueError(
            f"Field dimensions not broadcast-compatible: {field_shape_a} vs {field_shape_b}"
        )

    # Build einsum subscripts
    letters = string.ascii_lowercase
    caps = string.ascii_uppercase

    # Field dims
    fdim = len(broadcast_shape)
    field_indices = letters[:fdim]

    # Build index labels
    a_contra = letters[fdim : fdim + p]
    a_co = letters[fdim + p : fdim + p + q]
    b_contra = caps[fdim : fdim + r]
    b_co = caps[fdim + r : fdim + r + t]

    # Full index lists
    a_indices = field_indices + "".join(a_contra + a_co)
    b_indices = field_indices + "".join(b_contra + b_co)
    result_indices = field_indices + "".join(a_contra + b_contra + a_co + b_co)

    einsum_expr = f"{a_indices},{b_indices}->{result_indices}"

    return np.einsum(einsum_expr, tensor_a, tensor_b)


def dense_compute_tensor_trace(
    tensor_field: np.ndarray,
    indices: Tuple[int, int],
    tensor_signature: np.ndarray,
    metric_field: Optional[np.ndarray] = None,
    inverse_metric_field: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Compute the trace over a pair of tensor indices, adjusting their variances if needed.

    This function traces over two tensor slots in a tensor field by summing over their
    diagonal components. If the indices are both covariant or both contravariant, the appropriate
    metric (or inverse metric) is used to first raise or lower one of them to enable contraction.

    Parameters
    ----------
    tensor_field : numpy.ndarray
        Tensor field of shape ``(..., I₁, ..., I_r)``, where the last `r` axes represent tensor indices.
    indices : Tuple[int, int]
        The pair of indices to trace over. Indices must be distinct and in the range [0, rank).
    tensor_signature : numpy.ndarray
        Array of shape `(rank,)` with +1 for contravariant indices and -1 for covariant ones.
    metric_field : numpy.ndarray, optional
        Metric tensor to lower contravariant indices. Required if both traced indices are contravariant.
    inverse_metric_field : numpy.ndarray, optional
        Inverse metric tensor to raise covariant indices. Required if both traced indices are covariant.
    **kwargs :
        Additional keyword arguments passed to the low-level `numpy.trace` call.

    Returns
    -------
    numpy.ndarray
        Tensor field with the two specified indices traced over. Output rank is `rank - 2`.

    Raises
    ------
    ValueError
        If input validation fails or required metrics are not provided.

    Examples
    --------
    >>> import numpy as np
    >>> from pymetric.differential_geometry.dense_utils import compute_tensor_trace
    >>>
    >>> T = np.eye(3)[None, None, :, :]  # Rank-2 (1,1)-tensor over a 1x1 grid
    >>> sig = np.array([+1, -1])
    >>> compute_tensor_trace(T, indices=(0, 1), tensor_signature=sig)
    array([[3.]])
    """
    # Validate the indices for the contraction of
    # the trace.
    i, j = indices
    if i == j:
        raise ValueError("Cannot trace over the same index twice.")
    rank = tensor_signature.size
    if not (0 <= i < rank and 0 <= j < rank):
        raise ValueError(f"indices {indices} must lie in the range [0, {rank})")

    # Check variance compatibility
    vi, vj = tensor_signature[i], tensor_signature[j]

    if vi == vj:
        # Need to raise or lower one index
        if vi == +1:
            tensor_field, tensor_signature = dense_adjust_tensor_signature(
                tensor_field,
                [j],
                tensor_signature,
                metric_field=metric_field,
                inverse_metric_field=inverse_metric_field,
            )
        elif vi == -1:
            tensor_field, tensor_signature = dense_adjust_tensor_signature(
                tensor_field,
                [j],
                tensor_signature,
                inverse_metric_field=inverse_metric_field,
                metric_field=metric_field,
            )
    # else, one is +1 and one is -1 → already suitable for contraction

    return _dense_compute_tensor_trace(
        tensor_field, indices=(i, j), rank=rank, **kwargs
    )


def dense_compute_volume_element(
    metric_field: np.ndarray, metric_type: str
) -> np.ndarray:
    """
    Compute the volume element (√|det(g)|) from a metric tensor field.

    Parameters
    ----------
    metric_field : numpy.ndarray
        The metric tensor field. Depending on `metric_type`, expected shape is:

            - "full": shape (..., L, L), representing a full rank-2 metric tensor.
            - "diag": shape (..., L), representing a diagonal metric tensor.

    metric_type : str
        One of {"full", "diag"}. Specifies whether the metric is full or diagonal.

    Returns
    -------
    numpy.ndarray
        A scalar field of shape (...), representing the volume element √|det(g)| at each point.

    Raises
    ------
    ValueError
        If the input does not match the specified metric type or shape is invalid.
    """
    if metric_type == "full":
        if metric_field.ndim < 2 or metric_field.shape[-1] != metric_field.shape[-2]:
            raise ValueError(
                "Full metric tensor must have shape (..., L, L) with square last two dimensions."
            )
        det = np.linalg.det(metric_field)
    elif metric_type == "diag":
        if metric_field.ndim < 1:
            raise ValueError("Diagonal metric tensor must have shape (..., L).")
        det = np.prod(metric_field, axis=-1)
    else:
        raise ValueError(
            f"Unknown metric_type '{metric_type}'. Expected 'full' or 'diag'."
        )

    return np.sqrt(np.abs(det))
