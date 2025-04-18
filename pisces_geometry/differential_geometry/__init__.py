"""
Basic tools for performing differential geometry calculations.

This module provides low-level access to much of the differential geometry tooling that is implemented via fields, grids,
and coordinate systems elsewhere in the package. Many of the functions in this module take a large number of inputs which would
be auto-generated in other access patterns.
"""
__all__ = [
    "compute_laplacian",
    "compute_gradient",
    "compute_divergence",
    "get_laplacian_dependence",
    "get_gradient_dependence",
    "get_divergence_dependence",
    "raise_index_orth",
    "lower_index_orth",
    "compute_Dterm",
    "compute_Lterm",
    "compute_metric_density",
    "raise_index_orth",
    "lower_index_orth",
    "adjust_tensor_signature_orth",
    "contract_index_with_metric",
    "raise_index",
    "lower_index",
    "adjust_tensor_signature",
    "contract_index_with_metric_orthogonal",
    "ggrad_cl_covariant_component",
    "ggrad_cl_covariant",
    "ggrad_cl_contravariant",
    "gdiv_cl_covariant",
    "gdiv_cl_contravariant",
    "glap_cl",
    "ggrad_orth_covariant_component",
    "ggrad_orth_covariant",
    "ggrad_orth_contravariant",
    "gdiv_orth_covariant",
    "gdiv_orth_contravariant",
    "glap_orth",
    "get_raising_dependence",
    "get_lowering_dependence",
]

from pisces_geometry.differential_geometry.operators import (
    gdiv_cl_contravariant,
    gdiv_cl_covariant,
    gdiv_orth_contravariant,
    gdiv_orth_covariant,
    ggrad_cl_contravariant,
    ggrad_cl_covariant,
    ggrad_cl_covariant_component,
    ggrad_orth_contravariant,
    ggrad_orth_covariant,
    ggrad_orth_covariant_component,
    glap_cl,
    glap_orth,
)
from pisces_geometry.differential_geometry.symbolic import (
    compute_divergence,
    compute_Dterm,
    compute_gradient,
    compute_laplacian,
    compute_Lterm,
    compute_metric_density,
    get_divergence_dependence,
    get_gradient_dependence,
    get_laplacian_dependence,
    get_lowering_dependence,
    get_raising_dependence,
    lower_index,
    raise_index,
)
from pisces_geometry.differential_geometry.tensor_utilities import (
    adjust_tensor_signature,
    adjust_tensor_signature_orth,
    contract_index_with_metric,
    contract_index_with_metric_orthogonal,
    lower_index,
    lower_index_orth,
    raise_index,
    raise_index_orth,
)
