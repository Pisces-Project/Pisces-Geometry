"""
Symbolic manipulations for differential geometry operations.

This module provides symbolic tools for computing geometric quantities
such as gradients, divergences, Laplacians, and metric-related operations
in general curvilinear coordinates using SymPy.

These operations are essential in symbolic tensor calculus, particularly
in the context of differential geometry, general relativity, and
coordinate transformations in physics and applied mathematics.

These functions are largely integrated into Pisces-Geometry in their ability to distinguish
the dependence of particular operations and by constructing specialized functions like the D and L terms.

Key Features
------------

- Raise or lower tensor indices using metric or inverse metric tensors
- Compute symbolic gradient, divergence, and Laplacian operators
- Automatically determine variable dependencies for geometric quantities
- Support for covariant and contravariant bases
- Support for metric density, L-terms, and D-terms in non-Cartesian systems
"""
import string
from typing import Sequence, Any
import numpy as np
import sympy as sp
from sympy.tensor.array import tensorcontraction, tensorproduct, permutedims
from pisces_geometry._typing._generic import BasisAlias

def invert_metric(metric: sp.Matrix) -> sp.Matrix:
    r"""
    Compute the inverse of the metric :math:`g_{\mu \nu}`.

    Parameters
    ----------
    metric: :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The metric to invert.


    Returns
    -------
    :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The inverted metric.

    See Also
    --------
    raise_index
    lower_index
    compute_metric_density
    compute_Dterm
    compute_Lterm

    Examples
    --------
    To invert the metric for spherical coordinates, one need only do the following:

    .. code-block:: python

        >>> from pisces_geometry.differential_geometry.symbolic import invert_metric
        >>> import sympy as sp
        >>>
        >>> # Construct the symbols and the metric to
        >>> # pass into the function.
        >>> r,theta,phi = sp.symbols('r,theta,phi')
        >>> metric = sp.Matrix([
        ...     [1,0,0],
        ...     [0,r**2,0],
        ...     [0,0,(r*sp.sin(theta))**2]
        ...     ])
        >>>
        >>> # Now compute the inverse metric.
        >>> print(invert_metric(metric))
        Matrix([[1, 0, 0], [0, r**(-2), 0], [0, 0, 1/(r**2*sin(theta)**2)]])

    """
    return metric.inv()

def compute_metric_density(metric: sp.Matrix) -> sp.Basic:
    r"""
    Compute the metric density function :math:`\sqrt{{\rm Det}(g)}`.

    Parameters
    ----------
    metric: :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The metric tensor (:math:`g_{\mu \nu}`) of which to compute the metric density
        field.

    Returns
    -------
    :py:class:`~sympy.core.basic.Basic`
        The metric density function.

    Examples
    --------
    For the spherical metric,

    .. code-block:: python

        >>> from pisces_geometry.differential_geometry.symbolic import compute_metric_density
        >>> import sympy as sp
        >>>
        >>> # Construct the symbols and the metric that need to be converted to a metric density function.
        >>> r = sp.Symbol('r',positive=True)
        >>> theta,phi = sp.symbols("theta, phi")
        >>> metric = sp.Matrix([[1,0,0],[0,r**2,0],[0,0,(r**2*sp.sin(theta)**2)]])
        >>>
        >>> # Now compute the metric density function.
        >>> print(compute_metric_density(metric))
        r**2*sqrt(sin(theta)**2)

    """
    return sp.simplify(sp.sqrt(metric.det()))

def compute_Dterm(metric_density: sp.Basic, axes: Sequence[sp.Symbol]) -> sp.Array:
    r"""
    Compute the **D-term** components for a particular coordinate system from the
    metric density function.

    In a general, curvilinear coordinate system, the divergence is

    .. math::

        \nabla \cdot {\bf F} = \frac{1}{\rho} \partial_\mu(\rho F^\mu) = D_\mu F^\mu + \partial_\mu F^\mu,

    where

    .. math::

        D_\mu = \frac{1}{\rho} \partial_\mu \rho.

    This function therefore computes each of the :math:`D_\mu` components.

    Parameters
    ----------
    metric_density: :py:class:`~sympy.core.basic.Basic`
        The metric density function :math:`\sqrt{{\bf Det} \; g}`.
    axes: list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate axes symbols on which to compute the D-terms. There will be ``len(axes)`` resulting
        elements in the output array each corresponding to the :math:`D_{x^i}` component of the D-terms.


    Returns
    -------
    :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        The **D-term** components.

    See Also
    --------
    compute_Lterm
    compute_metric_density

    Examples
    --------
    To compute the :math:`D_\mu` components for a spherical coordinate system, we can do the following:

    >>> from pisces_geometry.differential_geometry.symbolic import compute_Dterm
    >>> import sympy as sp
    >>> r,theta,phi = sp.symbols('r,theta,phi')
    >>> metric_density = r**2 * sp.sin(theta)
    >>> print(compute_Dterm(metric_density, axes=[r,theta,phi]))
    [2/r, 1/tan(theta), 0]
    """
    # For each axis, compute the differential of the metric density with the specific axes.
    _derivatives = sp.Array([sp.simplify(sp.diff(metric_density, __symb__) / metric_density)
                             for __symb__ in axes])
    return _derivatives

def compute_Lterm(inverse_metric: sp.Matrix, metric_density: Any, axes: Sequence[sp.Symbol]) -> sp.Matrix:
    r"""
    Compute the **L-term** components for a particular coordinate system from the metric density and the metric.

    In a general, curvilinear coordinate system, the Laplacian is

    .. math::

        \nabla^2 \phi = \frac{1}{\rho}\partial_\mu\left(\rho g^{\mu\nu} \partial_\nu \phi\right) = L^\mu \partial_\mu \phi + g^{\mu\nu}\partial^2_{\mu\nu}\phi.

    The coefficients on the first term are called the **L-term** coefficients and defined as

    .. math::

        L^\nu = \frac{1}{\rho} \partial_\mu \left(\rho g^{\mu\nu}\right).

    Parameters
    ----------
    inverse_metric: :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The inverse metric :math:`g^{\mu\nu}`, with shape (mu, nu).
    metric_density: :py:class:`~sympy.core.basic.Basic`
        The metric density function :math:`\rho`, typically :math:`\sqrt{\det g}`.
    axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate axes :math:`x^\mu`, used for partial derivatives.

    Returns
    -------
    :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        A 1D array of L-term components :math:`L^\nu`.

    See Also
    --------
    compute_Dterm
    compute_metric_density

    Examples
    --------
    In spherical coordinates, the metric takes the form

    .. math::

        g_{\mu\nu} = \begin{bmatrix}1&0&0\\0&r^2&0\\0&0&r^2\sin^2\theta\end{bmatrix},

    so the inverse metric is simply

    .. math::

        g^{\mu\nu} = \begin{bmatrix}1&0&0\\0&r^{-2}&0\\0&0&r^{-2}\sin^{-2}\theta\end{bmatrix},

    and the resulting L-terms are

    .. math::

        L_\nu = \frac{1}{\rho} \partial_\mu \left(\rho g^{\mu\nu}\right) = \frac{1}{\rho} \partial_\nu \left(\rho g^{\nu\nu}\right).

    Thus,

    .. math::

        \begin{aligned}
        L_r &=2/r\\
        L_\theta &= \frac{1}{r^2\tan\theta}\\
        L_\phi &= 0.
        \end{aligned}

    To see this computationally,

    >>> from pisces_geometry.differential_geometry.symbolic import compute_Lterm
    >>> import sympy as sp
    >>> r,theta,phi = sp.symbols('r,theta,phi')
    >>> metric_density = r**2 * sp.sin(theta)
    >>> inv_metric = sp.Matrix([[1,0,0],[0,r**2,0],[0,0,1/(r**2*sp.sin(theta)**2)]]).inv()
    >>> print(compute_Lterm(inv_metric, metric_density, axes=[r,theta,phi]))
    [2/r, 1/(r**2*tan(theta)), 0]
    """
    # Ensure the metric inverse is the correct shape. It can have any nu shape but
    # must have the same mu shape as the number of axes specified.
    if inverse_metric.shape[0] != len(axes):
        raise ValueError(
            f"Incompatible shapes: inverse_metric has {inverse_metric.shape[0]} rows, "
            f"but {len(axes)} axes were provided. They must match."
        )

    L_terms = []
    for nu in range(inverse_metric.shape[1]):
        term_sum = sum(
            sp.diff(metric_density * inverse_metric[mu, nu], axes[mu])
            for mu in range(inverse_metric.shape[0])
        )
        L_nu = sp.simplify(term_sum / metric_density)
        L_terms.append(L_nu)

    return sp.Array(L_terms)

def compute_Lterm_orthogonal(metric: sp.Array, metric_density: Any, axes: Sequence[sp.Symbol]) -> sp.Matrix:
    r"""
    Compute the **L-term** components for a particular (orthogonal) coordinate system from the metric density and the metric.

    In a general, curvilinear coordinate system, the Laplacian is

    .. math::

        \nabla^2 \phi = \frac{1}{\rho}\partial_\mu\left(\rho g^{\mu\nu} \partial_\nu \phi\right) = L^\mu \partial_\mu \phi + g^{\mu\nu}\partial^2_{\mu\nu}\phi.

    The coefficients on the first term are called the **L-term** coefficients and defined as

    .. math::

        L^\nu = \frac{1}{\rho} \partial_\mu \left(\rho g^{\mu\nu}\right).

    In the special case of a fully **diagonal** metric tensor, this is simply

    .. math::

        L^\nu = \frac{1}{\rho} \partial_\nu \left(\frac{\rho}{g_{\mu\nu}} \right).

    Parameters
    ----------
    metric: :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        The inverse metric :math:`g^{\mu}`, with shape ``(mu,)``.
    metric_density: :py:class:`~sympy.core.basic.Basic`
        The metric density function :math:`\rho`, typically :math:`\sqrt{\det g}`.
    axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate axes :math:`x^\mu`, used for partial derivatives.

    Returns
    -------
    :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        A 1D array of L-term components :math:`L^\nu`.

    See Also
    --------
    compute_Dterm
    compute_Lterm
    compute_metric_density

    Examples
    --------
    In spherical coordinates, the metric takes the form

    .. math::

        g_{\mu\nu} = \begin{bmatrix}1&0&0\\0&r^2&0\\0&0&r^2\sin^2\theta\end{bmatrix},

    so the inverse metric is simply

    .. math::

        g^{\mu\nu} = \begin{bmatrix}1&0&0\\0&r^{-2}&0\\0&0&r^{-2}\sin^{-2}\theta\end{bmatrix},

    and the resulting L-terms are

    .. math::

        L_\nu = \frac{1}{\rho} \partial_\mu \left(\rho g^{\mu\nu}\right) = \frac{1}{\rho} \partial_\nu \left(\rho g^{\nu\nu}\right).

    Thus,

    .. math::

        \begin{aligned}
        L_r &=2/r\\
        L_\theta &= \frac{1}{r^2\tan\theta}\\
        L_\phi &= 0.
        \end{aligned}

    To see this computationally,

    >>> from pisces_geometry.differential_geometry.symbolic import compute_Lterm
    >>> import sympy as sp
    >>> r,theta,phi = sp.symbols('r,theta,phi')
    >>> metric_density = r**2 * sp.sin(theta)
    >>> inv_metric = sp.Matrix([[1,0,0],[0,r**2,0],[0,0,1/(r**2*sp.sin(theta)**2)]]).inv()
    >>> print(compute_Lterm(inv_metric, metric_density, axes=[r,theta,phi]))
    [2/r, 1/(r**2*tan(theta)), 0]
    """
    # Ensure the metric inverse is the correct shape. It can have any nu shape but
    # must have the same mu shape as the number of axes specified.
    if metric.shape[0] != len(axes):
        raise ValueError(
            f"Incompatible shapes: inverse_metric has {metric.shape[0]} rows, "
            f"but {len(axes)} axes were provided. They must match."
        )

    # Generate the L-terms in order.
    L_terms = []
    for nu in range(metric.shape[0]):
        term = sp.diff(metric_density / metric[nu], axes[nu])
        L_nu = sp.simplify(term / metric_density)
        L_terms.append(L_nu)

    return sp.Array(L_terms)

def raise_index(
        tensor: sp.Array,
        inverse_metric: sp.Matrix,
        axis: int,
) -> sp.Array:
    r"""
    Raise a single index of a generic tensor using the provided inverse metric. Mathematically,
    this is the tensor contraction

    .. math::

        T^{\ldots\mu\ldots}_{\ldots} = T^{\ldots}_{\ldots \nu\ldots} g^{\mu \nu}.

    Parameters
    ----------
    tensor:  :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        A symbolic tensor of any rank.
    inverse_metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The inverse metric tensor :math:`g^{\mu\nu}` used to raise the index.
    axis : int
        The axis (index position) of the tensor to raise.

    Returns
    -------
    :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        A new tensor with the specified index raised.

    See Also
    --------
    lower_index

    Examples
    --------
    The following example shows the raising of a generic tensor in polar coordinates:

    .. code-block:: python

        >>> # import the necessary functions.
        >>> import sympy as sp
        >>>
        >>> # Construct the symbols, the inverse metric,
        >>> # and the tensor.
        >>> r, theta = sp.symbols('r theta')
        >>> ginv = sp.Matrix([[1, 0], [0, 1/r**2]])
        >>> T = sp.Array([[sp.Function("T0")(r, theta), sp.Function("T1")(r, theta)],
        ...                            [sp.Function("T2")(r, theta), sp.Function("T3")(r, theta)]])
        >>>
        >>> # Raise the index.
        >>> raise_index(T, ginv, axis=1)
        [[T0(r, theta), T1(r, theta)/r**2], [T2(r, theta), T3(r, theta)/r**2]]
    """
    # Validate that the tensor field is a valid tensor field
    # and that the axis is within the number of available dimensions.
    ndim = tensor.rank()
    if not (0 <= axis < ndim):
        raise ValueError(f"Axis {axis} out of bounds for tensor of rank {ndim}.")

    # Construct index labels for each of the axes of the
    # tensor.
    index_labels = list(string.ascii_lowercase[:ndim])
    metric_labels = ('A', index_labels[axis])  # g^{A i}
    result_labels = list(index_labels)
    result_labels[axis] = metric_labels[0]  # replace i with A

    # Compute tensor product and contract over shared index
    tp = tensorproduct(inverse_metric, tensor)  # shape: (ndim, ndim, ...)
    contracted = tensorcontraction(tp, (1, axis + 2))
    perm = list(range(1, axis + 1)) + [0] + list(range(axis + 1, ndim))
    result = permutedims(contracted, perm)

    return result

# noinspection DuplicatedCode
def lower_index(
        tensor: sp.Array,
        metric: sp.Matrix,
        axis: int,
) -> sp.Array:
    r"""
    Lower a single index of a generic tensor using the provided inverse metric. Mathematically,
    this is the tensor contraction

    .. math::

        T^{\ldots\mu\ldots}_{\ldots}g_{\mu \nu} = T^{\ldots}_{\ldots \nu\ldots}.

    Parameters
    ----------
    tensor:  :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        A symbolic tensor of any rank.
    metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The metric tensor :math:`g_{\mu\nu}` used to lower the index.
    axis : int
        The axis (index position) of the tensor to lower.

    Returns
    -------
    :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        A new tensor with the specified index lower.

    See Also
    --------
    raise_index

    Examples
    --------
    The following example shows the raising of a generic tensor in polar coordinates:

    .. code-block:: python

        >>> # import the necessary functions.
        >>> import sympy as sp
        >>>
        >>> # Construct the symbols, the inverse metric,
        >>> # and the tensor.
        >>> r, theta = sp.symbols('r theta')
        >>> g = sp.Matrix([[1, 0], [0, r**2]])
        >>> T = sp.Array([[sp.Function("T0")(r, theta), sp.Function("T1")(r, theta)],
        ...                            [sp.Function("T2")(r, theta), sp.Function("T3")(r, theta)]])
        >>>
        >>> # Raise the index.
        >>> lower_index(T, g, axis=1)
        [[T0(r, theta), r**2*T1(r, theta)], [T2(r, theta), r**2*T3(r, theta)]]

    """
    # Validate that the tensor field is a valid tensor field
    # and that the axis is within the number of available dimensions.
    ndim = tensor.rank()
    if not (0 <= axis < ndim):
        raise ValueError(f"Axis {axis} out of bounds for tensor of rank {ndim}.")

    # Construct index labels for each of the axes of the
    # tensor.
    index_labels = list(string.ascii_lowercase[:ndim])
    metric_labels = ('A', index_labels[axis])  # g^{A i}
    result_labels = list(index_labels)
    result_labels[axis] = metric_labels[0]  # replace i with A

    # Compute tensor product and contract over shared index
    tp = tensorproduct(metric, tensor)  # shape: (ndim, ndim, ...)
    contracted = tensorcontraction(tp, (1, axis + 2))
    perm = list(range(1, axis + 1)) + [0] + list(range(axis + 1, ndim))
    result = permutedims(contracted, perm)

    return result

def compute_gradient(
        scalar_field: sp.Basic,
        coordinate_axes: Sequence[sp.Symbol],
        basis: BasisAlias = 'covariant',
        inverse_metric: sp.Matrix = None,
) -> sp.Array:
    r"""
    Compute the symbolic gradient of a scalar field :math:`\phi` in either covariant or contravariant basis.

    Parameters
    ----------
    scalar_field : :py:class:`~sympy.core.basic.Basic`
        The scalar field :math:`\phi` to differentiate. This should be any valid sympy expression dependent
        on the ``coordinate_axes`` and any other relevant symbols.
    coordinate_axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate axes (variables) with respect to which to compute the gradient. This should be the full
        list of the coordinate axes for the relevant coordinate system.
    basis : 'covariant' or 'contravariant', optional
        The basis in which to return the gradient. Defaults to 'covariant'.

        .. note::

            if ``basis != 'covariant'``, the index must be raised and the ``inverse_metric`` will be used
            for contraction. If ``inverse_metric`` is not specified, an error results.

    inverse_metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`, optional
        The inverse metric :math:`g^{\mu\nu}` used to raise the index if ``basis='contravariant'``.

    Returns
    -------
    :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        The components of the gradient of :math:`\phi`, in the chosen basis.

    See Also
    --------
    compute_divergence
    compute_laplacian

    Examples
    --------
    Compute the gradient of the scalar field :math:`\phi(r,\theta) = r^2 \sin(\theta)`.

    >>> # Import the necessary functions.
    >>> import sympy as sp
    >>> from pisces_geometry.differential_geometry.symbolic import compute_gradient
    >>>
    >>> # Create the symbols.
    >>> r, theta, phi = sp.symbols('r theta phi')
    >>> Phi = (r**2)*sp.sin(theta)
    >>> inv_metric = sp.Matrix([[1,0,0],[0,r**2,0],[0,0,r**2*sp.sin(theta)]]).inv()
    >>>
    >>> # Compute the covariant gradient.
    >>> compute_gradient(Phi, [r,theta, phi])
    [2*r*sin(theta), r**2*cos(theta), 0]
    >>>
    >>> # Compute the contravariant gradient.
    >>> compute_gradient(Phi, [r, theta, phi],basis='contravariant',inverse_metric=inv_metric)
    [2*r*sin(theta), cos(theta), 0]

    """
    # Begin by computing each of the relevant derivatives of the scalar field.
    _field_derivatives_ = sp.Array([
        sp.diff(scalar_field, __axis_symbol__) for __axis_symbol__ in coordinate_axes
    ])

    # If contravariant basis is requested, raise the index using the inverse metric.
    if basis == 'contravariant':
        if inverse_metric is None:
            raise ValueError("An inverse_metric is required for contravariant gradient computation.")
        _field_derivatives_ = raise_index(_field_derivatives_, inverse_metric, axis=0)
    elif basis != 'covariant':
        raise ValueError("`basis` must be either 'covariant' or 'contravariant'.")

    return _field_derivatives_

def compute_divergence(
        vector_field: sp.Array,
        coordinate_axes: Sequence[sp.Symbol],
        d_term: sp.Array = None,
        basis: BasisAlias = 'contravariant',
        inverse_metric: sp.Matrix = None,
        metric_density: sp.Basic = None,
) -> sp.Basic:
    r"""
    Compute the divergence :math:`\nabla \cdot {\bf F}` of a vector field symbolically.

    In general curvilinear coordinates, the divergence of a vector field :math:`{\bf F}` is

    .. math::

        \nabla \cdot {\bf F} = \frac{1}{\rho} \partial_\mu \left(\rho F^\mu\right),

    where :math:`\rho = \sqrt{{\rm Det} \; g}`.

    Parameters
    ----------
    vector_field : :py:class:`~sympy.tensor.array.MutableDenseNDimArray`
        The vector field components, assumed to be contravariant unless otherwise specified.
    coordinate_axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate symbols associated with each axis.
    d_term : :py:class:`~sympy.tensor.array.MutableDenseNDimArray`, optional
        The D-term components, used to account for the geometry (can be derived from metric_density).
    basis : {'covariant', 'contravariant'}, optional
        The basis in which the input vector field is expressed. Defaults to 'contravariant'.
    inverse_metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`, optional
        The inverse metric :math:`g^{\mu\nu}` to raise indices if `basis='covariant'`.
    metric_density : :py:class:`~sympy.core.basic.Basic`, optional
        The metric density :math:`\rho`, used to compute the D-term if it is not provided.

    Returns
    -------
    :py:class:`~sympy.core.basic.Basic`
        The symbolic expression for the divergence of the vector field.

    See Also
    --------
    compute_gradient
    compute_laplacian

    Examples
    --------
    In spherical coordinates, then vector field :math:`{\bf F} = r \hat{\bf e}_\theta` has a divergence

    .. math::

        \nabla \cdot {\bf F} = \frac{1}{r^2\sin\theta} \partial_\theta \left[r^3\sin \theta \right] = \frac{r}{\tan \theta}.

    To perform this operation in Pisces-Geometry,

    >>> import sympy as sp
    >>> from pisces_geometry.differential_geometry.symbolic import (
    ...     compute_divergence, compute_Dterm
    ... )
    >>>
    >>> # Define coordinate symbols and metric
    >>> r, theta, phi = sp.symbols('r theta phi', positive=True)
    >>> coords = [r, theta, phi]
    >>> metric_density = r**2 * sp.sin(theta)
    >>>
    >>> # Define the vector field.
    >>> V = sp.Array([0, r, 0])
    >>>
    >>> # Compute divergence
    >>> compute_divergence(V, coords, metric_density=metric_density)
    r/tan(theta)

    """
    # Validation steps. Ensure that the vector field has the correct number of dimensions
    # and that the necessary components are derived to proceed with the computation.
    ndim = len(coordinate_axes)
    if vector_field.shape != (ndim,):
        raise ValueError(f"Expected vector field of shape ({ndim},), got {vector_field.shape}")

    # check the d-term. We may need to construct it and then we need to ensure that
    # it has the intended shape.
    if d_term is None:
        # We need to derive the d_term.
        if metric_density is None:
            raise ValueError("Either d_term or metric_density must be provided.")
        d_term = compute_Dterm(metric_density, coordinate_axes)
    if d_term.shape != (ndim,):
        raise ValueError(f"Expected d_term of shape ({ndim},), got {d_term.shape}")

    # Ensure that the vector field is correctly cast in the contravariant basis so that
    # we can perform the necessary operations. If it is not, then we need to raise the index.
    if basis == 'covariant':
        if inverse_metric is None:
            raise ValueError("inverse_metric is required to raise a covariant vector field.")
        vector_field = raise_index(vector_field, inverse_metric, axis=0)
    elif basis != 'contravariant':
        raise ValueError("`basis` must be either 'covariant' or 'contravariant'.")

    # Perform the sums to get the desired behavior.
    divergence = sum(
        d_term[i] * vector_field[i] + sp.diff(vector_field[i], coordinate_axes[i])
        for i in range(ndim)
    )

    return sp.simplify(divergence)

def compute_laplacian(
        scalar_field: sp.Basic,
        coordinate_axes: Sequence[sp.Symbol],
        inverse_metric: sp.Matrix,
        l_term: sp.Array = None,
        metric_density: sp.Basic = None,
) -> sp.Basic:
    r"""
    Compute the Laplacian :math:`\nabla^2 \phi` of a scalar field in curvilinear coordinates.

    In general curvilinear coordinates, the Laplacian of a scalar field :math:`\phi` takes the form

    .. math::

        \nabla^2\phi = \nabla \cdot \nabla \phi = \frac{1}{\rho} \partial_\mu \left( \rho g^{\mu \nu} \partial_\nu \phi \right),

    where :math:`\rho = \sqrt{\det g}` is the metric density, and :math:`g^{\mu \nu}` is the inverse metric.

    This function computes the symbolic expression for this Laplacian. Internally, it expands the divergence term into:

    .. math::

        \nabla^2\phi = L^\nu \partial_\nu \phi + g^{\mu\nu} \partial^2_{\mu\nu} \phi,

    where the **L-term** is defined as

    .. math::

        L^\nu = \frac{1}{\rho} \partial_\mu \left( \rho g^{\mu \nu} \right).

    You may either pass the ``l_term`` explicitly if it is precomputed, or let this function compute it automatically by providing the ``metric_density`` and ``inverse_metric``.

    Parameters
    ----------
    scalar_field : :py:class:`~sympy.core.basic.Basic`
        The scalar field :math:`\phi` whose Laplacian is to be computed.

    coordinate_axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate variables with respect to which differentiation occurs. This should be in order (matching
        the ordering in ``l_term`` or ``metric_density``) which each element containing the symbol for that particular
        axis of the coordinate system.

    inverse_metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The inverse metric tensor :math:`g^{\mu \nu}`, used in both the second derivative term
        and to compute the L-terms if not provided. This should be an ``(ndim,ndim)`` array where ``ndim`` is the
        number of ``coordinate_axes``.

    l_term : :py:class:`~sympy.tensor.array.MutableDenseNDimArray`, optional
        Precomputed L-term array :math:`L^\nu`. If this is provided, the function will skip
        computing L-terms using `metric_density`.

    metric_density : :py:class:`~sympy.core.basic.Basic`, optional
        The metric density :math:`\rho = \sqrt{\det g}`. This is only required if ``l_term`` is not given,
        as it is used to compute the L-terms.

    Returns
    -------
    :py:class:`~sympy.core.basic.Basic`
        The symbolic expression for the Laplacian of the scalar field.

    See Also
    --------
    compute_divergence : Symbolic divergence of a vector field in curvilinear coordinates.
    compute_gradient : Symbolic gradient of a scalar field in different bases.

    Examples
    --------
    Compute the Laplacian of the scalar field :math:`\phi(r,\theta) = r^2 \sin(\theta)` in spherical coordinates.

    Analytically, the spherical Laplacian will be

    .. math::

        \nabla^2\phi = 6\sin(\theta) + \frac{\cos^2\theta - \sin^2\theta}{\sin \theta} = 4\sin(\theta) + \rm{csc}(\theta)

    Computationally, we can obtain the same result with the following code:

    >>> import sympy as sp
    >>> from pisces_geometry.differential_geometry.symbolic import (
    ...     compute_laplacian, compute_Lterm
    ... )
    >>>
    >>> # Coordinates
    >>> r, theta, phi = sp.symbols('r theta phi', positive=True)
    >>> coords = [r, theta, phi]
    >>>
    >>> # Define scalar field: φ(r,θ) = r² * sin(θ)
    >>> phi_field = r**2 * sp.sin(theta)
    >>>
    >>> # Inverse metric for spherical coordinates
    >>> g_inv = sp.Matrix([
    ...     [1, 0, 0],
    ...     [0, 1/r**2, 0],
    ...     [0, 0, 1/(r**2 * sp.sin(theta)**2)]
    ... ])
    >>>
    >>> # Metric density
    >>> metric_density = r**2 * sp.sin(theta)
    >>>
    >>> # Compute Laplacian with automatic L-term computation
    >>> compute_laplacian(phi_field, coords, g_inv, metric_density=metric_density)
    4*sin(theta) + 1/sin(theta)

    """

    # Validation steps. Ensure that the vector field has the correct number of dimensions
    # and that the necessary components are derived to proceed with the computation.
    ndim = len(coordinate_axes)

    # check the l-term. We may need to construct it and then we need to ensure that
    # it has the intended shape.
    if l_term is None:
        # We need to derive the d_term.
        if (metric_density is None) or (inverse_metric is None):
            raise ValueError("Either `l_term` or `metric_density` and `inverse_metric` must be provided.")
        l_term = compute_Lterm(inverse_metric, metric_density, coordinate_axes)
    if l_term.shape != (ndim,):
        raise ValueError(f"Expected l_term of shape ({ndim},), got {l_term.shape}")

    # Step 2: Construct the Laplacian
    # ∇²φ = L^μ ∂_μ φ + g^{μν} ∂²_{μν} φ
    gradient_terms = [l_term[i] * sp.diff(scalar_field, coordinate_axes[i]) for i in range(ndim)]
    second_deriv_terms = [
        inverse_metric[i, j] * sp.diff(scalar_field, coordinate_axes[i], coordinate_axes[j])
        for i in range(ndim) for j in range(ndim)
    ]
    laplacian = sum(gradient_terms) + sum(second_deriv_terms)
    return sp.simplify(laplacian)

def get_gradient_dependence(
        scalar_field_dependence: Sequence[sp.Symbol],
        coordinate_axes: Sequence[sp.Symbol],
        basis: BasisAlias = 'covariant',
        inverse_metric: sp.Matrix = None,
) -> np.ndarray:
    r"""
    Determine the symbolic variable dependencies of each component of the gradient of a scalar field.

    Formally, for a coordinate system :math:`q^1,\ldots,q^N`, and a scalar field :math:`\phi(q^{k_1},\ldots,q^{k_N})`,
    this function determines which coordinates appear in the resulting gradient of :math:`\phi`.

    Parameters
    ----------
    scalar_field_dependence : list of :py:class:`~sympy.core.symbol.Symbol`
        The variables the scalar field depends on. This should be a subset (or all) of `coordinate_axes` and any additional
        (non-coordinate) variables that the field depends on.
    coordinate_axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The full set of coordinate symbols used for computing the gradient.
    basis : {'covariant', 'contravariant'}, optional
        The basis in which to compute the gradient. Defaults to 'covariant'. If metric manipulations are required to ensure
        that the basis is the one specified, additional dependencies will appear.
    inverse_metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`, optional
        The inverse metric used to raise indices if `basis='contravariant'`.

    Returns
    -------
    :py:class:`numpy.ndarray`
        An array (of shape ``(len(coordinate_axes),)``) containing the symbolic dependence of each component of the gradient.
        Each element of the array is either:

        - a :py:class:`set` containing the symbols on which that component depends, or
        - ``0``, indicating that this element of the gradient is everywhere 0.

    See Also
    --------
    compute_gradient : Compute the full gradient tensor.
    raise_index : Used internally for contravariant basis.

    Notes
    -----
    If a particular gradient component is identically zero, the result will be `0`.

    Examples
    --------
    Analyze the variable dependencies of the gradient of :math:`\phi(r,\theta) = r^2 \sin(\theta)`:

    .. code-block:: python

        import sympy as sp
        from pisces_geometry.differential_geometry.symbolic import get_gradient_dependence
        r, theta, phi = sp.symbols('r theta phi', positive=True)
        get_gradient_dependence([r, theta], [r, theta, phi])
        >>> array([{theta, r}, {theta, r}, 0], dtype=object)

    """
    scalar_field = sp.Function("T")(*scalar_field_dependence)
    gradient = compute_gradient(
        scalar_field=scalar_field,
        coordinate_axes=coordinate_axes,
        basis=basis,
        inverse_metric=inverse_metric,
    )
    dependencies = np.empty_like(gradient, dtype=object)
    for i, expr in enumerate(gradient):
        simplified = sp.simplify(expr)
        if simplified == 0:
            dependencies[i] = 0
        else:
            syms = simplified.free_symbols
            dependencies[i] = set() if not syms else syms
    return dependencies

def get_divergence_dependence(
        vector_field_dependence: Sequence[Sequence[sp.Symbol]],
        d_term: Sequence[sp.Basic],
        coordinate_axes: Sequence[sp.Symbol],
        inverse_metric: sp.Matrix = None,
        basis: BasisAlias = 'contravariant',
):
    r"""
    Determine the symbolic variable dependencies of the divergence of a vector field.

    Parameters
    ----------
    vector_field_dependence : sequence of sequences of :py:class:`~sympy.core.symbol.Symbol`
        The list of variables each component of the vector field depends on.
    d_term : list of :py:class:`~sympy.core.basic.Basic`
        Precomputed D-term components. Must match the length of `coordinate_axes`.
    coordinate_axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate variables with respect to which the divergence is computed.
    inverse_metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`, optional
        The inverse metric used to raise the vector field if it's in covariant form.
    basis : {'covariant', 'contravariant'}, optional
        Basis in which the vector field is defined. Defaults to 'contravariant'.

    Returns
    -------
    set or 0
        The set of symbolic variables that the divergence expression depends on,
        or 0 if the divergence is identically zero.

    See Also
    --------
    compute_divergence : Compute the divergence of a vector field.
    compute_Dterm : Derive D-term from metric density.

    Examples
    --------
    Determine the divergence dependencies of a vector field :math:`\vec{V} = [0, r, 0]` in spherical coordinates:

    >>> import sympy as sp
    >>> from pisces_geometry.differential_geometry.symbolic import get_divergence_dependence
    >>> r, theta, phi = sp.symbols('r theta phi', positive=True)
    >>> d_term = [2/r, 1/sp.tan(theta), 0]
    >>> vector_field = sp.Array([0,r,0])
    >>> get_divergence_dependence(vector_field, d_term, [r, theta, phi])
    {r, theta}
    """
    # Construct the vector field from the dependence arrays. Depending on the behavior, we need
    # to be a little bit careful about how different types are managed.
    _vector_field_generator = []
    for i,element in enumerate(vector_field_dependence):
        if isinstance(element, Sequence):
            _vector_field_generator.append(
                sp.Function(f"V{i}")(*element)
            )
        elif isinstance(element, sp.Symbol):
            _vector_field_generator.append(
                sp.Function(f"V{i}")(element)
            )
        elif isinstance(element, sp.Basic):
            _vector_field_generator.append(
                element
            )
        else:
            pass

    vector_field = sp.tensor.Array(_vector_field_generator)

    d_term = sp.tensor.Array(d_term)
    divergence = compute_divergence(
        vector_field=vector_field,
        coordinate_axes=coordinate_axes,
        d_term=d_term,
        inverse_metric=inverse_metric,
        basis=basis,
    )
    simplified = sp.simplify(divergence)
    if simplified == 0:
        return 0
    syms = simplified.free_symbols
    return set() if not syms else syms

def get_laplacian_dependence(
        scalar_field_dependence: Sequence[sp.Symbol],
        coordinate_axes: Sequence[sp.Symbol],
        inverse_metric: sp.Matrix,
        l_term: Sequence[sp.Basic] = None,
        metric_density: sp.Basic = None,
):
    r"""
    Determine the symbolic variable dependencies of the Laplacian of a scalar field.

    Parameters
    ----------
    scalar_field_dependence : list of :py:class:`~sympy.core.symbol.Symbol`
        The variables on which the scalar field depends.
    coordinate_axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The coordinate axes used for computing the Laplacian.
    inverse_metric : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The inverse metric tensor used in the Laplacian computation.
    l_term : list of :py:class:`~sympy.core.basic.Basic`, optional
        Precomputed L-term components. If not provided, will be derived from `metric_density`.
    metric_density : :py:class:`~sympy.core.basic.Basic`, optional
        Metric density used for computing L-terms, required if `l_term` is not provided.

    Returns
    -------
    set or 0
        A set of symbols the Laplacian depends on, or 0 if identically zero.

    See Also
    --------
    compute_laplacian : Compute the symbolic Laplacian.
    compute_Lterm : Derive L-terms from metric density and inverse metric.

    Notes
    -----
    This is useful in symbolic PDE analysis and geometric simplifications where you only need
    to know which variables affect a Laplacian term.

    Examples
    --------
    For a scalar field :math:`\phi(r, \theta) = r^2 \sin(\theta)` in spherical coordinates:

    >>> import sympy as sp
    >>> from pisces_geometry.differential_geometry.symbolic import get_laplacian_dependence
    >>> r, theta, phi = sp.symbols('r theta phi', positive=True)
    >>> metric_density = r**2 * sp.sin(theta)
    >>> ginv = sp.Matrix([
    ...     [1, 0, 0],
    ...     [0, 1/r**2, 0],
    ...     [0, 0, 1/(r**2 * sp.sin(theta)**2)]
    ... ])
    >>> get_laplacian_dependence([r, theta], [r, theta, phi], ginv, metric_density=metric_density)
    set([r, theta])
    """
    scalar_field = sp.Function("T")(*scalar_field_dependence)
    laplacian = compute_laplacian(
        scalar_field=scalar_field,
        coordinate_axes=coordinate_axes,
        inverse_metric=inverse_metric,
        l_term=l_term,
        metric_density=metric_density,
    )
    simplified = sp.simplify(laplacian)
    if simplified == 0:
        return 0
    syms = simplified.free_symbols
    return set() if not syms else syms
