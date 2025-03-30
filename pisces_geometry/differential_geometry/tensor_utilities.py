"""
Utility functions for performing basic tensor manipulations including index raising and lowering.
"""
from typing import List, Literal, Optional
import numpy as np

# ------------------------------------------ #
# General Utilities                          #
# ------------------------------------------ #
def contract_index_with_metric(
        tensor: np.ndarray,
        metric: np.ndarray,
        index: int,
        rank: int
) -> np.ndarray:
    r"""
    Contracts a single tensor index with a metric tensor (or inverse metric),
    adjusting axis positions as needed.

    Parameters
    ----------
    tensor : :py:class:`numpy.ndarray`
        The tensor field to transform. Last `rank` axes are tensor indices.
    metric : :py:class:`numpy.ndarray`
        The metric or inverse metric tensor to contract with, shape (..., k, k).
    index : int
        The index position (0 to rank-1) to contract.
    rank : int
        Rank of the tensor (i.e., number of trailing tensor index axes).

    Returns
    -------
    numpy.ndarray
        The transformed tensor with the specified index raised or lowered.
    """
    tensor = np.moveaxis(tensor, -rank + index, -1)
    tensor = np.einsum("...ij,...j->...i", metric, tensor)
    tensor = np.moveaxis(tensor, -1, -rank + index)
    return tensor

def contract_index_with_metric_orthogonal(
        tensor: np.ndarray,
        metric: np.ndarray,
        index: int,
        rank: int
) -> np.ndarray:
    r"""
    Contracts a single tensor index with a metric tensor (or inverse metric),
    adjusting axis positions as needed.

    Parameters
    ----------
    tensor : :py:class:`numpy.ndarray`
        The tensor field to transform. Last `rank` axes are tensor indices.
    metric : :py:class:`numpy.ndarray`
        The diagonal components of the metric tensor with shape `(..., k)` matching the grid dimensions.
    index : int
        The index position (0 to rank-1) to contract.
    rank : int
        Rank of the tensor (i.e., number of trailing tensor index axes).

    Returns
    -------
    numpy.ndarray
        The transformed tensor with the specified index raised or lowered.
    """
    # Construct the metric shape so that things behave as intended when
    # performing the operation under broadcasting.
    __metric_shape__ = list(tensor.shape[:-rank] + (1,)*rank)
    __metric_shape__[-rank+index] = tensor.shape[-rank+index]

    # Now perform the broadcast operation on the tensor.
    tensor *= np.reshape(metric,__metric_shape__)
    return tensor

# ------------------------------------------ #
# Raising and Lowering Indices               #
# ------------------------------------------ #
def raise_index(
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        inverse_metric: np.ndarray,
        inplace: bool = False
) -> np.ndarray:
    r"""
    Raises a specified index of a tensor field using the inverse metric tensor.

    Parameters
    ----------
    tensor_field : :py:class:`numpy.ndarray`
        The input tensor field defined on a grid. The last `rank` dimensions are assumed to be tensor indices.
    index : int
        The index to raise, ranging from 0 to rank-1.
    rank : int
        The tensor rank (number of tensor indices, not including grid dimensions).
    inverse_metric : :py:class:`numpy.ndarray`
        The inverse metric tensor with shape `(..., k, k)` matching the grid dimensions.
    inplace : bool, optional
        If True, modifies the input tensor in place. Default is False.

    Returns
    -------
    numpy.ndarray
        A tensor field with the specified index raised. Has the same shape as `tensor_field`.

    See Also
    --------
    lower_index
    adjust_tensor_signature

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
    >>> from pisces_geometry.differential_geometry.tensor_utilities import raise_index
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
    >>> raise_index(v_cov, index=0, rank=1, inverse_metric=g_inv)
    array([0. , 0.5, 0. ])

    """
    if rank > tensor_field.ndim:
        raise ValueError("Rank must be less than or equal to the number of tensor_field dimensions.")
    if index < 0 or index >= rank:
        raise ValueError("Index must be in the range [0, rank).")

    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]

    if inverse_metric.shape[:-2] != grid_shape:
        raise ValueError("Inverse metric and tensor field have different grid shapes.")
    if inverse_metric.shape[-2:] != (tensor_shape[index], tensor_shape[index]):
        raise ValueError("Inverse metric shape is incompatible with the contraction index.")

    working_tensor = tensor_field if inplace else np.copy(tensor_field)
    return contract_index_with_metric(working_tensor, inverse_metric, index, rank)


def lower_index(
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        metric: np.ndarray,
        inplace: bool = False
) -> np.ndarray:
    r"""
    Lowers a specified index of a tensor field using the metric tensor.

    Parameters
    ----------
    tensor_field : :py:class:`numpy.ndarray`
        The input tensor field defined on a grid. The last `rank` dimensions are assumed to be tensor indices.
    index : int
        The index to lower, ranging from 0 to rank-1.
    rank : int
        The tensor rank (number of tensor indices, not including grid dimensions).
    metric : :py:class:`numpy.ndarray`
        The metric tensor with shape `(..., k, k)` matching the grid dimensions.
    inplace : bool, optional
        If True, modifies the input tensor in place. Default is False.

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
    In spherical coordinates, if you have a covariant vector:

    .. math::

        {\bf v} = r {\bf e}^\theta

    Then the contravariant version is:

    .. math::

        v^\theta = g^{\theta \mu} v_\mu = g^{\theta \theta} v_\theta = \frac{1}{r^2} v_{\theta} = \frac{1}{r}.

    Let's see this work in practice:

    >>> import numpy as np
    >>> from pisces_geometry.differential_geometry.tensor_utilities import raise_index
    >>>
    >>> # Construct the vector field at a point.
    >>> # We'll need the metric and the vector field at the point.
    >>> r,theta = 2,np.pi/4
    >>> v_cont = np.asarray([0,r,0])
    >>>
    >>> # Construct the metric tensor.
    >>> g = np.diag([1,  r**2, (r**2 * np.sin(theta)**2)])
    >>>
    >>> # Now we can use the inverse metric to raise the tensor index.
    >>> lower_index(v_cont, index=0, rank=1, metric=g)
    array([0., 8., 0.])
    """
    if rank > tensor_field.ndim:
        raise ValueError("Rank must be less than or equal to the number of tensor_field dimensions.")
    if index < 0 or index >= rank:
        raise ValueError("Index must be in the range [0, rank).")

    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]

    if metric.shape[:-2] != grid_shape:
        raise ValueError("Metric and tensor field have different grid shapes.")
    if metric.shape[-2:] != (tensor_shape[index], tensor_shape[index]):
        raise ValueError("Metric shape is incompatible with the contraction index.")

    working_tensor = tensor_field if inplace else np.copy(tensor_field)
    return contract_index_with_metric(working_tensor, metric, index, rank)


def adjust_tensor_signature(
        tensor_field: np.ndarray,
        indices: List[int],
        modes: List[Literal["raise", "lower"]],
        rank: int,
        metric: Optional[np.ndarray] = None,
        inverse_metric: Optional[np.ndarray] = None,
        component_masks: Optional[List[np.ndarray]] = None,
        inplace: bool = False
) -> np.ndarray:
    r"""
    Adjusts the signature of a tensor field by raising or lowering specified indices.

    Parameters
    ----------
    tensor_field : :py:class:`numpy.ndarray`
        The input tensor field of shape (..., tensor indices...).
    indices : List[int]
        List of tensor index positions to modify.
    modes : List[str]
        List of operations for each index: either 'raise' (raise) or 'lower'.
    rank : int
        Rank of the tensor (number of trailing tensor indices).
    metric : :py:class:`numpy.ndarray`, optional
        Metric tensor of shape (..., k, k). Required for lowering indices.
    inverse_metric : :py:class:`numpy.ndarray`, optional
        Inverse metric tensor of shape (..., k, k). Required for raising indices.
    component_masks : List[:py:class:`numpy.ndarray`], optional
        Optional masks to restrict which components are active for each index.
        Each mask should be a boolean array of shape (k,) where k = size of tensor axis.
    inplace : bool
        Whether to modify the tensor in-place. Default is False.

    Returns
    -------
    numpy.ndarray
        Tensor field with adjusted index signature.

    Examples
    --------
    Consider the type (1, 1) tensor constructed as

    .. math::

        T(\omega,V) = \omega \otimes V.

    This tensor has components :math:`T_\mu^\nu`. If we then switch the tensor so that we have

    .. math::

        T^\mu_\nu = g^{\mu \alpha} g_{\nu \beta} T^{\beta}_\alpha,

    in spherical coordinates, this becomes

    .. math::

        T^\mu_\nu = g^{\mu \mu} g_{\nu \nu} T^{\nu}_\mu,

    If the original tensor has only a :math:`T^{\theta}_{\theta}` term, then

    .. math::

        T^{\theta}_{\theta} = g^{\theta \theta} g_{\theta \theta} T^{\theta}_\theta = T^\theta_\theta.

    If instead, the tensor has two up indices, then

    .. math::

        T_{\theta\theta} = g_{\theta \theta} g_{\theta \theta} T^{\theta\theta} = \frac{1}{r^4} T^{\theta \theta}.

    Let's see how this plays out programmatically:

    >>> import numpy as np
    >>> from pisces_geometry.differential_geometry.tensor_utilities import adjust_tensor_signature
    >>>
    >>> # Start by setting up the tensor.
    >>> r, theta = 2.0, np.pi / 4
    >>> tensor = np.zeros((3,3))
    >>> tensor[1,1] = 1
    >>>
    >>> # Now construct both the metric and its inverse.
    >>> g = np.diag([1, r**2, r**2 * np.sin(theta)**2])
    >>> g_inv = np.diag([1, 1/r**2, 1/(r**2 * np.sin(theta)**2)])
    >>>
    >>> # Case 1: We have a 1-1 tensor and expect to have no change in the tensor
    >>> # output because the metric terms cancel one another.
    >>> adjust_tensor_signature(
    ...     tensor,
    ...     indices=[0,1],
    ...     modes=["raise","lower"],
    ...     rank=2,
    ...     metric=g,
    ...     inverse_metric=g_inv)
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    >>>
    >>> # Case 2: We have a 0-2 tensor and expect to have no change in the tensor
    >>> adjust_tensor_signature(
    ...     tensor,
    ...     indices=[0,1],
    ...     modes=["raise","raise"],
    ...     rank=2,
    ...     metric=g,
    ...     inverse_metric=g_inv)
    array([[0.    , 0.    , 0.    ],
           [0.    , 0.0625, 0.    ],
           [0.    , 0.    , 0.    ]])
    """
    # Validate that the indices and the modes are mutually valid. Then check
    # that the component masks are valid and start iterating through.
    if len(indices) != len(modes):
        raise ValueError("Each index must have a corresponding mode ('raise' or 'lower').")
    if component_masks and len(component_masks) != len(indices):
        raise ValueError("If component_masks is provided, it must have the same length as indices.")

    # Set up the working tensor and pull out the relevant shape variables.
    working_tensor = tensor_field if inplace else np.copy(tensor_field)
    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]

    # Iterate through the various indices and modes.
    for i, (index, mode) in enumerate(zip(indices, modes)):
        # Check the validity of the index against the rank.
        if not (0 <= index < rank):
            raise ValueError(f"Index {index} is out of bounds for rank {rank}.")
        axis_size = tensor_shape[index]

        # Select the appropriate metric and slice it if needed
        mask = component_masks[i] if component_masks else slice(None)
        if mode == "lower":
            if metric is None:
                raise ValueError("Metric tensor is required to lower indices.")
            current_metric = metric[..., mask]
        elif mode == "raise":
            if inverse_metric is None:
                raise ValueError("Inverse metric tensor is required to raise indices.")
            current_metric = inverse_metric[..., mask]
        else:
            raise ValueError(f"Invalid mode '{mode}' for index {index}. Use 'raise' or 'lower'.")

        # Validate metric shape
        if current_metric.shape[:-2] != grid_shape:
            raise ValueError("Metric and tensor field have different grid shapes.")
        if current_metric.shape[-1] != axis_size:
            raise ValueError("Metric shape is incompatible with tensor axis size.")

        # Move axis to the end for contraction
        working_tensor = contract_index_with_metric(working_tensor, current_metric, index, rank)

    return working_tensor

def raise_index_orth(
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        metric: np.ndarray,
        inplace: bool = False
) -> np.ndarray:
    r"""
    Raises a specified index of a tensor field using the inverse metric tensor.

    Parameters
    ----------
    tensor_field : :py:class:`numpy.ndarray`
        The input tensor field defined on a grid. The last `rank` dimensions are assumed to be tensor indices.
    index : int
        The index to raise, ranging from 0 to rank-1.
    rank : int
        The tensor rank (number of tensor indices, not including grid dimensions).
    metric : :py:class:`numpy.ndarray`
        The diagonal components of the metric tensor with shape `(..., k)` matching the grid dimensions.
    inplace : bool, optional
        If True, modifies the input tensor in place. Default is False.

    Returns
    -------
    numpy.ndarray
        A tensor field with the specified index raised. Has the same shape as `tensor_field`.

    See Also
    --------
    lower_index
    adjust_tensor_signature

    Raises
    ------
    ValueError
        If the input shapes or indices are invalid.
    """
    if rank > tensor_field.ndim:
        raise ValueError("Rank must be less than or equal to the number of tensor_field dimensions.")
    if index < 0 or index >= rank:
        raise ValueError("Index must be in the range [0, rank).")

    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]

    if metric.shape[:-1] != grid_shape:
        raise ValueError("Inverse metric and tensor field have different grid shapes.")
    if metric.shape[-1:] != (tensor_shape[index],):
        raise ValueError("Inverse metric shape is incompatible with the contraction index.")

    working_tensor = tensor_field if inplace else np.copy(tensor_field)
    return contract_index_with_metric_orthogonal(working_tensor,metric,index,rank)


def lower_index_orth(
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        metric: np.ndarray,
        inplace: bool = False
) -> np.ndarray:
    r"""
    Lowers a specified index of a tensor field using the metric tensor.

    Parameters
    ----------
    tensor_field : :py:class:`numpy.ndarray`
        The input tensor field defined on a grid. The last `rank` dimensions are assumed to be tensor indices.
    index : int
        The index to lower, ranging from 0 to rank-1.
    rank : int
        The tensor rank (number of tensor indices, not including grid dimensions).
    metric : :py:class:`numpy.ndarray`
        The diagonal components of the metric tensor with shape `(..., k)` matching the grid dimensions.
    inplace : bool, optional
        If True, modifies the input tensor in place. Default is False.

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

    """
    if rank > tensor_field.ndim:
        raise ValueError("Rank must be less than or equal to the number of tensor_field dimensions.")
    if index < 0 or index >= rank:
        raise ValueError("Index must be in the range [0, rank).")

    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]

    if metric.shape[:-1] != grid_shape:
        raise ValueError("Metric and tensor field have different grid shapes.")
    if metric.shape[-1:] != (tensor_shape[index], ):
        raise ValueError("Metric shape is incompatible with the contraction index.")

    working_tensor = tensor_field if inplace else np.copy(tensor_field)
    return contract_index_with_metric_orthogonal(working_tensor,metric,index,rank)

def adjust_tensor_signature_orth(
        tensor_field: np.ndarray,
        indices: List[int],
        modes: List[Literal["raise", "lower"]],
        rank: int,
        metric: Optional[np.ndarray] = None,
        component_masks: Optional[List[np.ndarray]] = None,
        inplace: bool = False
) -> np.ndarray:
    r"""
    Adjusts the signature of a tensor field by raising or lowering specified indices.

    Parameters
    ----------
    tensor_field : :py:class:`numpy.ndarray`
        The input tensor field of shape (..., tensor indices...).
    indices : List[int]
        List of tensor index positions to modify.
    modes : List[str]
        List of operations for each index: either 'raise' (raise) or 'lower'.
    rank : int
        Rank of the tensor (number of trailing tensor indices).
    metric : :py:class:`numpy.ndarray`, optional
        The **diagnonal** elements of the metric tensor with shape ``(...,k)``. Required for lowering indices.
    component_masks : List[:py:class:`numpy.ndarray`], optional
        Optional masks to restrict which components are active for each index.
        Each mask should be a boolean array of shape (k,) where k = size of tensor axis.
    inplace : bool
        Whether to modify the tensor in-place. Default is False.

    Returns
    -------
    numpy.ndarray
        Tensor field with adjusted index signature.

    Examples
    --------
    Consider the type (1, 1) tensor constructed as

    .. math::

        T(\omega,V) = \omega \otimes V.

    This tensor has components :math:`T_\mu^\nu`. If we then switch the tensor so that we have

    .. math::

        T^\mu_\nu = g^{\mu \alpha} g_{\nu \beta} T^{\beta}_\alpha,

    in spherical coordinates, this becomes

    .. math::

        T^\mu_\nu = g^{\mu \mu} g_{\nu \nu} T^{\nu}_\mu,

    If the original tensor has only a :math:`T^{\theta}_{\theta}` term, then

    .. math::

        T^{\theta}_{\theta} = g^{\theta \theta} g_{\theta \theta} T^{\theta}_\theta = T^\theta_\theta.

    If instead, the tensor has two up indices, then

    .. math::

        T_{\theta\theta} = g_{\theta \theta} g_{\theta \theta} T^{\theta\theta} = \frac{1}{r^4} T^{\theta \theta}.

    Let's see how this plays out programmatically:

    >>> import numpy as np
    >>> from pisces_geometry.differential_geometry.tensor_utilities import adjust_tensor_signature_orth
    >>>
    >>> # Start by setting up the tensor.
    >>> r, theta = 2.0, np.pi / 4
    >>> tensor = np.zeros((3,3))
    >>> tensor[1,1] = 1
    >>>
    >>> # Now construct both the metric and its inverse.
    >>> g = np.asarray([1, r**2, r**2 * np.sin(theta)**2])
    >>>
    >>> # Case 1: We have a 1-1 tensor and expect to have no change in the tensor
    >>> # output because the metric terms cancel one another.
    >>> adjust_tensor_signature_orth(
    ...     tensor,
    ...     indices=[0,1],
    ...     modes=["raise","lower"],
    ...     rank=2,
    ...     metric=g)
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    >>>
    >>> # Case 2: We have a 0-2 tensor and expect to have no change in the tensor
    >>> adjust_tensor_signature_orth(
    ...     tensor,
    ...     indices=[0,1],
    ...     modes=["raise","raise"],
    ...     rank=2,
    ...     metric=g)
    array([[0.    , 0.    , 0.    ],
           [0.    , 0.0625, 0.    ],
           [0.    , 0.    , 0.    ]])

    """
    # Validate that the indices and the modes are mutually valid. Then check
    # that the component masks are valid and start iterating through.
    if len(indices) != len(modes):
        raise ValueError("Each index must have a corresponding mode ('raise' or 'lower').")
    if component_masks and len(component_masks) != len(indices):
        raise ValueError("If component_masks is provided, it must have the same length as indices.")

    # Set up the working tensor and pull out the relevant shape variables.
    working_tensor = tensor_field if inplace else np.copy(tensor_field)
    grid_shape = tensor_field.shape[:-rank]
    tensor_shape = tensor_field.shape[-rank:]

    # Iterate through the various indices and modes.
    for i, (index, mode) in enumerate(zip(indices, modes)):
        # Check the validity of the index against the rank.
        if not (0 <= index < rank):
            raise ValueError(f"Index {index} is out of bounds for rank {rank}.")
        axis_size = tensor_shape[index]

        # Select the appropriate metric and slice it if needed
        mask = component_masks[i] if component_masks else slice(None)
        current_metric = metric[..., mask]

        # Validate metric shape
        if current_metric.shape[:-1] != grid_shape:
            raise ValueError("Metric and tensor field have different grid shapes.")
        if current_metric.shape[-1] != axis_size:
            raise ValueError("Metric shape is incompatible with tensor axis size.")

        # Move axis to the end for contraction
        if mode == 'raise':
            working_tensor = contract_index_with_metric_orthogonal(working_tensor, 1/current_metric, index, rank)
        else:
            working_tensor = contract_index_with_metric_orthogonal(working_tensor, current_metric, index, rank)

    return working_tensor