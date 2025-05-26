"""
Differential geometry operations and tensor calculus utilities.

This module provides core tools for performing differential geometry calculations over structured coordinate systems,
with or without the overhead of higher-level abstractions like fields or grids. These low-level utilities are designed to be
flexible and composable, and are used internally by many high-level features of the library.

The functions in this module generally require explicit specification of coordinate systems, metrics, and tensor index conventions,
and are intended for use in advanced workflows where full control is required.

Tensor operations are supported in both **dense** and **sparse** representation formats:

- **Dense representation** expects all tensor components to be packed into a single array of shape
  ``(*field_shape, N_dim, N_dim, ...)`` where the trailing axes correspond to tensor indices. This layout is suitable for
  vector and tensor fields where full component information is available.

- **Sparse representation** accepts a dictionary of the form ``{(i, j, ...): array}``, where each key is a multi-index
  identifying a single component of the tensor, and the value is the corresponding array of scalar values over a set of
  field axes. This layout is useful when working with tensors where many components are zero.

Examples of supported operations include index raising/lowering, covariant differentiation, divergence, and the Laplacian,
all of which are compatible with symbolic and numerical coordinate systems.
"""
__all__ = [
    "dense_tensor_product",
    "dense_compute_tensor_trace",
    "dense_compute_volume_element",
    "dense_lower_index",
    "dense_raise_index",
    "dense_permute_tensor_indices",
    "dense_transform_tensor_field",
    "dense_permute_tensor_indices",
    "dense_adjust_tensor_signature",
    "dense_contract_with_metric",
    "compute_Dterm",
    "compute_Lterm",
    "compute_gradient",
    "compute_laplacian",
    "compute_divergence",
    "compute_metric_density",
    "raise_index",
    "lower_index",
    "invert_metric",
    "dense_element_wise_partial_derivatives",
    "DenseDependenceObject",
    "DenseTensorDependence",
]
from .dense_utils import (
    dense_adjust_tensor_signature,
    dense_compute_tensor_trace,
    dense_compute_volume_element,
    dense_contract_with_metric,
    dense_lower_index,
    dense_permute_tensor_indices,
    dense_raise_index,
    dense_tensor_product,
    dense_transform_tensor_field,
)
from .dependence import DenseDependenceObject, DenseTensorDependence
from .general_ops import dense_element_wise_partial_derivatives
from .symbolic import (
    compute_divergence,
    compute_Dterm,
    compute_gradient,
    compute_laplacian,
    compute_Lterm,
    compute_metric_density,
    invert_metric,
    lower_index,
    raise_index,
)
