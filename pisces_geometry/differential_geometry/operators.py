"""
Functions for performing differential geometry calculations.

These are not placed directly in the :py:mod:`pisces_geometry.coordinates.base` module to maintain better readability; however,
the most natural way to execute most of these operations is through a valid :py:class:`~pisces_geometry.coordinates.base._CoordinateSystemBase` or
subclass.

Functions in this module are executed in their most native basis - raising and lowering operations can easily be chained
with a given operation to manage basis changes on either inputs or outputs.
"""
from typing import Sequence, Union

import numpy as np

from pisces_geometry.differential_geometry.tensor_utilities import (
    raise_index,
    raise_index_orth,
)


# ------------------------------------------ #
# Gradient Computation Functions             #
# ------------------------------------------ #
# Functions defined in this section of the file are used to complete various aspects of the gradient computation
# in different coordinate systems and / or different bases.
# Extended coordinate system types (beyond curvilinear / orthogonal) might require more sophisticated machinery
# to be added to this section.
def ggrad_cl_covariant_component(
    scalar_field: np.ndarray,
    component: int,
    /,
    spacing: Union[float, np.ndarray] = None,
    derivative_field: np.ndarray = None,
    **kwargs,
):
    r"""
    Compute the covariant gradient component :math:`\partial_\mu \phi` for a scalar field along one axis (:math:`\mu`).

    Parameters
    ----------
    scalar_field : :py:class:`numpy.ndarray`
        The scalar field of shape ``(...)``, representing a scalar function over the grid.
    component : int
        Index of the axis (0-based) along which to compute the gradient.
    spacing : float or numpy.ndarray, optional
        Grid spacing along the selected axis. Required if `derivative_field` is not provided. If a ``float`` is
        provided, then it is interpreted as a uniform spacing parameter. If an array is provided, then it is assumed
        to be the grid spacing in a non-uniform setting.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivative values of shape ``(..., 1)``. If provided, used directly.
    **kwargs :
        Additional arguments passed to `np.gradient`.

    Returns
    -------
    numpy.ndarray
        The gradient component with shape ``(..., 1)``.

    See Also
    --------
    ggrad_cl_covariant: Compute to covariant gradient.
    ggrad_cl_contravariant: Compute to contravariant gradient.
    gdiv_cl_covariant: Compute to covariant divergence.
    gdiv_cl_contravariant: Compute to contravariant divergence.
    glap_cl: Compute the laplacian.

    Examples
    --------
    In spherical coordinates, the function :math:`\phi({\bf x}) = r` has gradient

    .. math::

        \nabla_r \phi = \partial_r \phi = 1.

    To see this computationally, we can simply generate a coordinate grid and call :py:func:`ggrad_cl_covariant_component`.

    .. code-block:: python

        from pisces_geometry.differential_geometry.operators import ggrad_cl_covariant_component
        import numpy as np
        r = np.linspace(0,1,100)
        phi = r
        ggrad_cl_covariant_component(phi,0,spacing=r[1]-r[0])

    """
    # Compute the actual gradient.
    if (derivative_field is None) and (spacing is None):
        raise ValueError(
            "Either ``spacing`` or ``derivative_field`` must be specified."
        )
    if derivative_field is None:
        # This requires a reshape because there is only one axis.
        return np.gradient(scalar_field, spacing, axis=component, **kwargs).reshape(
            (*scalar_field.shape, 1)
        )
    else:
        # Check that the derivative field has to correct shape.
        if derivative_field != scalar_field.shape + (1,):
            raise ValueError(
                f"Derivative field should have shape {scalar_field.shape} + ({1},) but has shape {derivative_field.shape}."
            )

        return derivative_field


def ggrad_orth_covariant_component(
    scalar_field: np.ndarray,
    component: int,
    /,
    spacing: Union[float, np.ndarray] = None,
    derivative_field: np.ndarray = None,
    **kwargs,
):
    r"""
    Compute the covariant gradient component :math:`\partial_\mu \phi` for a scalar field along one axis (:math:`\mu`) in orthogonal coordinates.

    Parameters
    ----------
    scalar_field : :py:class:`numpy.ndarray`
        The scalar field of shape ``(...)``, representing a scalar function over the grid.
    component : int
        Index of the axis (0-based) along which to compute the gradient.
    spacing : float or numpy.ndarray, optional
        Grid spacing along the selected axis. Required if `derivative_field` is not provided. If a ``float`` is
        provided, then it is interpreted as a uniform spacing parameter. If an array is provided, then it is assumed
        to be the grid spacing in a non-uniform setting.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivative values of shape ``(..., 1)``. If provided, used directly.
    **kwargs :
        Additional arguments passed to `np.gradient`.

    Returns
    -------
    numpy.ndarray
        The gradient component with shape ``(..., 1)``.

    See Also
    --------
    ggrad_cl_covariant: Compute to covariant gradient.
    ggrad_cl_contravariant: Compute to contravariant gradient.
    gdiv_cl_covariant: Compute to covariant divergence.
    gdiv_cl_contravariant: Compute to contravariant divergence.
    glap_cl: Compute the laplacian.

    Examples
    --------
    In spherical coordinates, the function :math:`\phi({\bf x}) = r` has gradient

    .. math::

        \nabla_r \phi = \partial_r \phi = 1.

    To see this computationally, we can simply generate a coordinate grid and call :py:func:`ggrad_cl_covariant_component`.

    .. code-block:: python

        from pisces_geometry.differential_geometry.operators import ggrad_cl_covariant_component
        import numpy as np
        r = np.linspace(0,1,100)
        phi = r
        ggrad_cl_covariant_component(phi,0,spacing=r[1]-r[0])

    Notes
    -----
    This function is a direct alias for :py:func:`ggrad_cl_covariant_component`.

    """
    return ggrad_cl_covariant_component(
        scalar_field,
        component,
        spacing=spacing,
        derivative_field=derivative_field,
        **kwargs,
    )


def ggrad_cl_covariant(
    scalar_field: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the covariant gradient :math:`\partial_\mu \phi` of a scalar field (:math:`\phi`).

    In the covariant basis, the components of the gradient :math:`\nabla_\mu \phi` are

    .. math::

        \nabla_\mu \phi = \partial_\mu \phi.

    Parameters
    ----------
    scalar_field : :py:class:`numpy.ndarray`
        The scalar field :math:`\phi` discretized on a grid with shape ``(...)``.
    spacing : list of float or numpy.ndarray, optional
        The grid spacing along each of the axes of the ``scalar_field``. This should be
        an array-like object with ``scalar_field.ndim`` elements corresponding to the grid spacing
        along each of the ``ndim`` axes of the grid. Either ``spacing`` or ``derivative_field`` is required.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivatives (:math:`\partial_\mu \phi`) of shape ``(...,scalar_field.ndim)``. If provided,
        the ``derivative_field`` is simply returned as the output for the operation. Either ``spacing`` or ``derivative_field`` is required.
    **kwargs :
        Additional keyword arguments for :py:func:`numpy.gradient`.

    Returns
    -------
    numpy.ndarray
        The covariant gradient components :math:`\partial_\mu \phi` of shape ``(...,scalar_field.ndim)``.

    See Also
    --------
    ggrad_cl_covariant_component
    ggrad_cl_contravariant
    """
    # Determine the number of axes that are present in the scalar field that is being
    # passed into the function. Use this to construct the free dimensions.
    free_dimensions = scalar_field.ndim
    axes = np.arange(free_dimensions, dtype=int)

    # Compute the actual gradient.
    if (derivative_field is None) and (spacing is None):
        raise ValueError(
            "Either ``spacing`` or ``derivative_field`` must be specified."
        )
    if derivative_field is None:
        # Check that the spacing has the correct shape.
        if len(spacing) != free_dimensions:
            raise ValueError(
                f"`spacing` did not match the number of grid axes ({len(spacing)},{free_dimensions})."
            )

        # Compute the derivatives.
        if len(axes) > 1:
            # These can be automatically stacked without reshaping.
            return np.stack(
                np.gradient(scalar_field, *spacing, axis=axes, **kwargs), axis=-1
            )
        else:
            # This requires a reshape because there is only one axis.
            return np.gradient(scalar_field, *spacing, axis=axes, **kwargs).reshape(
                (*scalar_field.shape, 1)
            )
    else:
        # Check that the derivative field has to correct shape.
        if derivative_field != scalar_field.shape + (free_dimensions,):
            raise ValueError(
                f"Derivative field should have shape {scalar_field.shape} + ({free_dimensions},) but has shape {derivative_field.shape}."
            )

        return derivative_field


def ggrad_cl_contravariant(
    scalar_field: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]],
    inverse_metric: np.ndarray,
    derivative_field: np.ndarray = None,
    **kwargs,
):
    r"""
    Compute the contravariant gradient :math:`g^{\nu \mu} \partial_\mu \phi` of a scalar field (:math:`\phi`).

    In the contravariant basis, the components of the gradient :math:`\nabla^\mu \phi` are

    .. math::

        \nabla^\mu \phi = g^{\mu \nu} \nabla_\nu \phi = g^{\mu \nu} \partial_\nu \phi.

    Parameters
    ----------
    scalar_field : :py:class:`numpy.ndarray`
        The scalar field :math:`\phi` discretized on a grid with shape ``(...)``.
    inverse_metric : :py:class:`numpy.ndarray`
        The inverse metric tensor field :math:`g^{\mu \nu}` at each point on the grid. This should be provided as a
        ``(..., M, scalar_field.ndim)`` array where ``M`` is an arbitrary integer representing the output components of
        interest. The metric is then contracted against the covariant gradient to compute the result.
    spacing : list of float or numpy.ndarray, optional
        The grid spacing along each of the axes of the ``scalar_field``. This should be
        an array-like object with ``scalar_field.ndim`` elements corresponding to the grid spacing
        along each of the ``ndim`` axes of the grid. Either ``spacing`` or ``derivative_field`` is required.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivatives (:math:`\partial_\mu \phi`) of shape ``(...,scalar_field.ndim)``. If provided,
        the ``derivative_field`` is simply returned as the output for the operation. Either ``spacing`` or ``derivative_field`` is required.
    **kwargs :
        Additional keyword arguments for :py:func:`numpy.gradient`.

    Returns
    -------
    numpy.ndarray
        The covariant gradient components :math:`\partial_\mu \phi` of shape ``(...,M)``.

    See Also
    --------
    ggrad_cl_covariant_component
    ggrad_cl_covariant

    Notes
    -----
    This function first constructs the covariant gradient via :py:func:`ggrad_cl_covariant` and then
    proceeds to compute the contravariant components by raising the index using :py:func:`~pisces_geometry.differential_geometry.tensor_utilities.raise_index`.

    """
    # Start by computing the covariant gradient.
    _gradient_field = ggrad_cl_covariant(
        scalar_field, spacing, derivative_field, **kwargs
    )

    # Start performing the raising computation. We first ensure
    # that the metric has a valid shape and then proceed with the rest
    # of the computation.
    _gradient_field = raise_index(_gradient_field, 0, 1, inverse_metric, inplace=True)
    return _gradient_field


def ggrad_orth_covariant(
    scalar_field: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the covariant gradient :math:`\partial_\mu \phi` of a scalar field (:math:`\phi`) in orthogonal coordinates.

    In the covariant basis, the components of the gradient :math:`\nabla_\mu \phi` are

    .. math::

        \nabla_\mu \phi = \partial_\mu \phi.

    Parameters
    ----------
    scalar_field : :py:class:`numpy.ndarray`
        The scalar field :math:`\phi` discretized on a grid with shape ``(...)``.
    spacing : list of float or numpy.ndarray, optional
        The grid spacing along each of the axes of the ``scalar_field``. This should be
        an array-like object with ``scalar_field.ndim`` elements corresponding to the grid spacing
        along each of the ``ndim`` axes of the grid. Either ``spacing`` or ``derivative_field`` is required.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivatives (:math:`\partial_\mu \phi`) of shape ``(...,scalar_field.ndim)``. If provided,
        the ``derivative_field`` is simply returned as the output for the operation. Either ``spacing`` or ``derivative_field`` is required.
    **kwargs :
        Additional keyword arguments for :py:func:`numpy.gradient`.

    Returns
    -------
    numpy.ndarray
        The covariant gradient components :math:`\partial_\mu \phi` of shape ``(...,scalar_field.ndim)``.

    See Also
    --------
    ggrad_cl_covariant_component
    ggrad_cl_contravariant

    Notes
    -----
    This function is a direct alias for :py:func:`ggrad_cl_covariant`.
    """
    return ggrad_cl_covariant(
        scalar_field, spacing=spacing, derivative_field=derivative_field, **kwargs
    )


def ggrad_orth_contravariant(
    scalar_field: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]],
    inverse_metric: np.ndarray,
    derivative_field: np.ndarray = None,
    **kwargs,
):
    r"""
    Compute the contravariant gradient :math:`g^{\nu \mu} \partial_\mu \phi` of a scalar field (:math:`\phi`) in orthogonal coordinates.

    In the contravariant basis, the components of the gradient :math:`\nabla^\mu \phi` are

    .. math::

        \nabla^\mu \phi = g^{\mu \mu} \nabla_\mu \phi = g^{\mu \mu} \partial_\mu \phi.

    This is simplified in **orthogonal coordinates** because

    .. math::

        g^{\mu\nu} = g^{\mu\mu} \delta_\mu^\nu,

    so the contraction becomes simply a scaling:

    .. math::

        \nabla^\mu \phi = g^{\mu \nu} \nabla_\nu \phi = g^{\mu \nu} \partial_\nu \phi.

    Parameters
    ----------
    scalar_field : :py:class:`numpy.ndarray`
        The scalar field :math:`\phi` discretized on a grid with shape ``(...)``.
    inverse_metric : :py:class:`numpy.ndarray`
        The inverse metric tensor's diagonal component field :math:`g^{\mu \mu}` at each point on the grid. This should be provided as a
        ``(..., scalar_field.ndim)`` array. The metric is then contracted against the covariant gradient to compute the result.
    spacing : list of float or numpy.ndarray, optional
        The grid spacing along each of the axes of the ``scalar_field``. This should be
        an array-like object with ``scalar_field.ndim`` elements corresponding to the grid spacing
        along each of the ``ndim`` axes of the grid. Either ``spacing`` or ``derivative_field`` is required.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivatives (:math:`\partial_\mu \phi`) of shape ``(...,scalar_field.ndim)``. If provided,
        the ``derivative_field`` is simply returned as the output for the operation. Either ``spacing`` or ``derivative_field`` is required.
    **kwargs :
        Additional keyword arguments for :py:func:`numpy.gradient`.

    Returns
    -------
    numpy.ndarray
        The covariant gradient components :math:`\partial_\mu \phi` of shape ``(...,M)``.

    See Also
    --------
    ggrad_cl_covariant_component
    ggrad_cl_covariant

    Notes
    -----
    This function first constructs the covariant gradient via :py:func:`ggrad_cl_covariant` and then
    proceeds to compute the contravariant components by raising the index using :py:func:`~pisces_geometry.differential_geometry.tensor_utilities.raise_index`.

    """
    # Start by computing the covariant gradient.
    _gradient_field = ggrad_orth_covariant(
        scalar_field, spacing, derivative_field, **kwargs
    )

    # Start performing the raising computation. We first ensure
    # that the metric has a valid shape and then proceed with the rest
    # of the computation.
    _gradient_field = raise_index_orth(
        _gradient_field, 0, 1, inverse_metric, inplace=True
    )
    return _gradient_field


# ------------------------------------------ #
# Divergence Computation Functions           #
# ------------------------------------------ #
# Functions defined in this section of the file are used to complete various aspects of the divergence computation
# in different coordinate systems and / or different bases.
# Extended coordinate system types (beyond curvilinear / orthogonal) might require more sophisticated machinery
# to be added to this section.
def gdiv_cl_contravariant(
    vector_field: np.ndarray,
    dterm_field: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    axis_connector: Sequence[int] = None,
    **kwargs,
):
    r"""
    Compute the divergence of a contravariant vector field in a curvilinear coordinate system.

    This uses the identity:

    .. math::

        \nabla_\mu V^\mu = \frac{1}{\sqrt{g}} \partial_\mu (\sqrt{g} V^\mu)
                         = D_\mu V^\mu + \partial_\mu V^\mu

    where the **D-term** and derivative term are provided by the user. The D-term has the form

    .. math::

        D_\mu = \frac{1}{\sqrt{g}} \partial_\mu(\sqrt{g}).

    Parameters
    ----------
    vector_field : :py:class:`numpy.ndarray`
        The contravariant vector field of shape ``(..., k)`` to take the divergence of. The shape make include up to
        ``ndim`` components in the final slot; however, if ``k!=ndim``, then ``axis_connector`` is required to ensure that
        the function knows how to associate different components of the vector field with their correct spatial component of
        the grid.
    dterm_field : :py:class:`numpy.ndarray`
        The precomputed D-term (:math:`D_\mu`) of same shape as `vector_field`. In most applications, the D-term can be
        furnished directly from the coordinate system in which the computation is being performed.
    spacing : list of float or numpy.ndarray, optional
        Grid spacing for each axis, used to compute derivatives. This should be provided as a ``(k,)`` array or array-like
        object with ``k`` elements corresponding to the grid spacing along each component axis.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivative of the vector field along grid directions, shape ``(..., C)``. In this case, ``C`` should be
        the number of components of the vector field which are non-zero and which also have corresponding axes in the grid.

        .. note::

            Internally, the ``derivative_field`` is calculated using the ``axis_connector`` and the ``vector_field``; however,
            if an axis (``i``) is not in the grid or is not a component of ``vector_field`` then it can be excluded. As such, the
            ``derivative_field`` can be number of elements, but **should** be only as large as the number of non-zero derivatives.

    axis_connector : list of int, optional
        Maps each component of the vector to a grid axis (used if vector has fewer components than dimensions). Should be
        provided as a list of integers or ``None``. For each index ``i`` of the ``axis_connector``, ``axis_connector[i]`` is
        the corresponding axis of the grid corresponding to that variable. If ``axis_connector[i]`` is ``None``, then there
        is no grid component for that particular axis.
    **kwargs :
        Additional options for :py:func:`numpy.gradient`.

    Returns
    -------
    numpy.ndarray
        The scalar divergence field of shape ``(...,)``.

    Examples
    --------
    >>> from pisces_geometry.coordinates.coordinate_systems import CylindricalCoordinateSystem
    >>> import numpy as np
    >>> coords = CylindricalCoordinateSystem()
    >>> r, theta, z = np.linspace(1, 2, 50), np.linspace(0, 2*np.pi, 50), np.zeros(50)
    >>> R, THETA, Z = np.meshgrid(r, theta, z, indexing='ij')
    >>> field = np.zeros(R.shape + (3,))
    >>> field[..., 0] = R
    >>> grid = np.stack([R, THETA, Z], axis=-1)
    >>> dterm = coords.compute_expression_on_grid('Dterm', grid)
    >>> div = gdiv_cl_contravariant(field, dterm, spacing=[r[1]-r[0], theta[1]-theta[0], 1])
    """
    # Determine the number of components and grid axes present in the vector field.
    # These are critical for performing standardization tasks.
    num_grid_axes = vector_field.ndim - 1
    num_comp_axes = vector_field.shape[-1]

    # Check and coerce the Dterm field.
    if dterm_field.shape != vector_field.shape:
        raise ValueError(
            f"Dterm and vector fields must have the same shape. Had {dterm_field.shape} and {vector_field.shape}."
        )

    # Compute the derivative term if it is not already available.
    if (derivative_field is None) and (spacing is None):
        raise ValueError(
            "Either ``spacing`` or ``derivative_field`` must be specified."
        )
    if derivative_field is None:
        # We will compute the derivative field using np.gradient in a sequence. We need to ensure
        # that the axes are connected properly and that spacing has the correct shape.
        if (num_grid_axes != num_comp_axes) and (axis_connector is None):
            raise ValueError(
                f"The `vector_field` has {num_grid_axes} grid axes and {num_comp_axes} components, which are not equal.\n"
                "To allow an incomplete vector field, you must provide ``axis_connector`` to specify the correspondence between"
                "grid axis and components."
            )
        elif (num_grid_axes != num_comp_axes) and (axis_connector is not None):
            # Check that the axis connector is legitimate for the computation.
            if len(axis_connector) != num_comp_axes:
                raise ValueError(
                    "Not all component axes were specified in the axis connector."
                )
        else:
            axis_connector = np.arange(num_comp_axes)

        # Ensure that spacing has the correct length.
        if len(spacing) != num_grid_axes:
            raise ValueError(
                "`spacing` must match the number of vector field components."
            )

        # Now we can actually compute the derivatives.
        derivative_field = np.stack(
            [
                np.gradient(
                    vector_field[..., __comp_index__],
                    spacing[__comp_index__],
                    axis=__grid_index__,
                    **kwargs,
                )
                for __comp_index__, __grid_index__ in enumerate(axis_connector)
                if __grid_index__ is not None
            ],
            axis=-1,
        )
    else:
        # The derivative field is provided, we just need to ensure that it has exactly the correct number
        # of non-degenerate axes necessary.
        if axis_connector is None:
            axis_connector = np.arange(num_comp_axes)
        non_degenerate_axes = len([_ax for _ax in axis_connector if _ax is not None])

        if derivative_field.shape != vector_field.shape[:-1] + (non_degenerate_axes,):
            raise ValueError(
                "Based on the number of components in `vector_field` and the `axis_connector`, derivative field"
                f"should only have {non_degenerate_axes} components."
            )

    # Construct the two independent terms of the divergence.
    _div_term_1 = np.sum(dterm_field * vector_field, axis=-1)
    _div_term_2 = np.sum(derivative_field, axis=-1)
    return _div_term_1 + _div_term_2


def gdiv_cl_covariant(
    vector_field: np.ndarray,
    dterm_field: np.ndarray,
    inverse_metric: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    axis_connector: Sequence[int] = None,
    **kwargs,
):
    """
    Compute the divergence of a covariant vector field in a curvilinear coordinate system.

    This function first raises the index of the covariant field to obtain a contravariant field,
    then calls `gdiv_cl_contravariant`.

    Parameters
    ----------
    vector_field : np.ndarray
        The covariant vector field of shape (..., ndim).
    dterm_field : np.ndarray
        The D-term expression, same shape as `vector_field`.
    inverse_metric : np.ndarray
        The inverse metric tensor used to raise indices, shape (..., ndim, ndim).
    spacing : list of float or numpy.ndarray, optional
        Grid spacing.
    derivative_field : np.ndarray, optional
        Precomputed partial derivatives.
    axis_connector : list of int, optional
        Maps components to grid axes, used if vector is not full-rank.
    **kwargs :
        Additional keyword arguments for numerical differentiation.

    Returns
    -------
    np.ndarray
        The scalar divergence field.

    See Also
    --------
    gdiv_cl_contravariant

    Examples
    --------
    >>> from pisces_geometry.coordinates.coordinate_systems import SphericalCoordinateSystem
    >>> coords = SphericalCoordinateSystem()
    >>> r, theta = np.linspace(1, 2, 100), np.linspace(0.1, np.pi-0.1, 100)
    >>> R, THETA = np.meshgrid(r, theta, indexing='ij')
    >>> V = np.zeros(R.shape + (3,))
    >>> V[..., 1] = np.sin(THETA)
    >>> G = np.stack([R, THETA, np.zeros_like(R)], axis=-1)
    >>> D = coords.compute_expression_on_grid("Dterm", G)
    >>> ginv = coords.compute_metric_on_grid(G, inverse=True)
    >>> result = gdiv_cl_covariant(V, D, ginv, spacing=[r[1]-r[0], theta[1]-theta[0], 1])
    """
    # Begin by converting the vector field from the covariant to the contravariant
    # basis so that computations can be fed to the gdiv_cl_contravariant function.
    _contra_vector_field = raise_index(
        vector_field, 0, 1, inverse_metric, inplace=False
    )

    # Pass the contravariant vector field into the gdiv_cl_contravariant method.
    return gdiv_cl_contravariant(
        _contra_vector_field,
        dterm_field,
        spacing=spacing,
        derivative_field=derivative_field,
        axis_connector=axis_connector,
        **kwargs,
    )


def gdiv_orth_contravariant(
    vector_field: np.ndarray,
    dterm_field: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    axis_connector: Sequence[int] = None,
    **kwargs,
):
    r"""
    Compute the divergence of a contravariant vector field in an orthogonal coordinate system.

    This uses the identity:

    .. math::

        \nabla_\mu V^\mu = \frac{1}{\sqrt{g}} \partial_\mu (\sqrt{g} V^\mu)
                         = D_\mu V^\mu + \partial_\mu V^\mu

    where the **D-term** and derivative term are provided by the user. The D-term has the form

    .. math::

        D_\mu = \frac{1}{\sqrt{g}} \partial_\mu(\sqrt{g}).

    Parameters
    ----------
    vector_field : :py:class:`numpy.ndarray`
        The contravariant vector field of shape ``(..., k)`` to take the divergence of. The shape make include up to
        ``ndim`` components in the final slot; however, if ``k!=ndim``, then ``axis_connector`` is required to ensure that
        the function knows how to associate different components of the vector field with their correct spatial component of
        the grid.
    dterm_field : :py:class:`numpy.ndarray`
        The precomputed D-term (:math:`D_\mu`) of same shape as `vector_field`. In most applications, the D-term can be
        furnished directly from the coordinate system in which the computation is being performed.
    spacing : list of float or numpy.ndarray, optional
        Grid spacing for each axis, used to compute derivatives. This should be provided as a ``(k,)`` array or array-like
        object with ``k`` elements corresponding to the grid spacing along each component axis.
    derivative_field : :py:class:`numpy.ndarray`, optional
        Precomputed derivative of the vector field along grid directions, shape ``(..., C)``. In this case, ``C`` should be
        the number of components of the vector field which are non-zero and which also have corresponding axes in the grid.

        .. note::

            Internally, the ``derivative_field`` is calculated using the ``axis_connector`` and the ``vector_field``; however,
            if an axis (``i``) is not in the grid or is not a component of ``vector_field`` then it can be excluded. As such, the
            ``derivative_field`` can be number of elements, but **should** be only as large as the number of non-zero derivatives.

    axis_connector : list of int, optional
        Maps each component of the vector to a grid axis (used if vector has fewer components than dimensions). Should be
        provided as a list of integers or ``None``. For each index ``i`` of the ``axis_connector``, ``axis_connector[i]`` is
        the corresponding axis of the grid corresponding to that variable. If ``axis_connector[i]`` is ``None``, then there
        is no grid component for that particular axis.
    **kwargs :
        Additional options for :py:func:`numpy.gradient`.

    Returns
    -------
    numpy.ndarray
        The scalar divergence field of shape ``(...,)``.

    """
    return gdiv_cl_contravariant(
        vector_field,
        dterm_field,
        spacing=spacing,
        derivative_field=derivative_field,
        axis_connector=axis_connector,
        **kwargs,
    )


def gdiv_orth_covariant(
    vector_field: np.ndarray,
    dterm_field: np.ndarray,
    inverse_metric: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    axis_connector: Sequence[int] = None,
    **kwargs,
):
    """
    Compute the divergence of a covariant vector field in an orthogonal coordinate system.

    This function first raises the index of the covariant field to obtain a contravariant field,
    then calls `gdiv_orth_contravariant`.

    Parameters
    ----------
    vector_field : np.ndarray
        The covariant vector field of shape (..., ndim).
    dterm_field : np.ndarray
        The D-term expression, same shape as `vector_field`.
    inverse_metric : :py:class:`numpy.ndarray`
        The inverse metric tensor's diagonal component field :math:`g^{\mu \mu}` at each point on the grid. This should be provided as a
        ``(..., scalar_field.ndim)`` array. The metric is then contracted against the covariant gradient to compute the result.
    spacing : list of float or numpy.ndarray, optional
        Grid spacing.
    derivative_field : np.ndarray, optional
        Precomputed partial derivatives.
    axis_connector : list of int, optional
        Maps components to grid axes, used if vector is not full-rank.
    **kwargs :
        Additional keyword arguments for numerical differentiation.

    Returns
    -------
    np.ndarray
        The scalar divergence field.

    See Also
    --------
    gdiv_cl_contravariant

    """
    # Begin by converting the vector field from the covariant to the contravariant
    # basis so that computations can be fed to the gdiv_cl_contravariant function.
    _contra_vector_field = raise_index_orth(
        vector_field, 0, 1, inverse_metric, inplace=False
    )

    # Pass the contravariant vector field into the gdiv_cl_contravariant method.
    return gdiv_cl_contravariant(
        _contra_vector_field,
        dterm_field,
        spacing=spacing,
        derivative_field=derivative_field,
        axis_connector=axis_connector,
        **kwargs,
    )


# ------------------------------------- #
# Laplacian Computation Functions       #
# ------------------------------------- #
# The Laplacian is computed in terms of the L-term and the derivatives of the scalar field.
def glap_cl(
    scalar_field: np.ndarray,
    lterm_field: np.ndarray,
    inv_metric: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    second_derivative_field: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the Laplacian of a scalar field in a curvilinear coordinate system.

    This function evaluates the Laplacian using the formula:

    .. math::

        \Delta \phi = L^\mu \partial_\mu \phi + g^{\mu \nu} \partial_\mu \partial_\nu \phi

    where :math:`L^\mu` is the so-called L-term (from the Jacobian of the coordinate system),
    and :math:`g^{\mu \nu}` is the inverse metric tensor.

    Parameters
    ----------
    scalar_field : np.ndarray
        The scalar field :math:`\phi` discretized on a grid with shape ``(...)``.
    lterm_field : np.ndarray
        Field of L-term values with shape ``(..., ndim)``, where ``ndim = scalar_field.ndim``.
        This can usually be computed from the coordinate system.
    inv_metric : np.ndarray
        Inverse metric tensor :math:`g^{\mu \nu}`, with shape ``(..., ndim, ndim)``.
    spacing : list of float or numpy.ndarray, optional
        Grid spacing along each axis. Required if ``derivative_field`` and ``second_derivative_field`` are not provided.
    derivative_field : np.ndarray, optional
        Precomputed first derivatives of shape ``(..., ndim)``. If provided, numerical derivatives are skipped.
    second_derivative_field : np.ndarray, optional
        Precomputed second derivatives of shape ``(..., ndim, ndim)``. If provided, second-order gradients are not computed.
    **kwargs :
        Additional keyword arguments passed to :py:func:`numpy.gradient`.

    Returns
    -------
    np.ndarray
        The Laplacian of the scalar field evaluated at all grid points. Shape ``(...)``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or required parameters are missing.

    Examples
    --------

    .. plot::
        :include-source:

        >>> # Import modules
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces_geometry.differential_geometry.operators import glap_cl
        >>>
        >>> # Construct the coordinates and the coordinate grid.
        >>> # For this example, we just use R and THETA (phi held constant)
        >>> r,theta = np.linspace(0.1,2,100),np.linspace(np.pi/4,np.pi*(3/4),100)
        >>> R,THETA = np.meshgrid(r,theta,indexing='ij')
        >>> C = np.stack([R,THETA],axis=-1)
        >>>
        >>> # Define the scalar field sin(theta) on the grid.
        >>> F = np.sin(THETA)
        >>>
        >>> # Construct the inverse-metric tensor field on the grid
        >>> # so that it can be used when performing operations.
        >>> g_inv = np.zeros((100,100,2,2))
        >>> g_inv[...,0,0] = 1
        >>> g_inv[...,1,1] = R**-2
        >>>
        >>> # Construct the 2 L-terms for this case.
        >>> l_terms = np.zeros((100,100,2))
        >>> l_terms[...,0] = 2/R
        >>> l_terms[...,1] = 1/(R**2 * np.tan(THETA))
        >>>
        >>> # Compute the laplacian and the "true" (theoretical) laplacian.
        >>> L = glap_cl(F,l_terms,g_inv,spacing=[(2-0.1)/100,(np.pi/2)/100])
        >>> Ltrue = np.cos(2*THETA)/(np.sin(THETA) * R**2)
        >>>
        >>> # Plotting data
        >>> fig,axes = plt.subplots(2,1,figsize=(6,8),sharex=True)
        >>> extent = [0.1,2,np.pi/4,np.pi*(3/4)]
        >>>
        >>> # Plot the computed case:
        >>> Q = axes[0].imshow(L.T,vmin=-20,vmax=1,origin='lower',extent=extent)
        >>> _ = plt.colorbar(Q)
        >>> P = axes[1].imshow(Ltrue.T,vmin=-20,vmax=1,origin='lower',extent=extent)
        >>> _ = plt.colorbar(P)
        >>> _ = axes[1].set_xlabel('r')
        >>> _ = axes[0].set_ylabel('theta')
        >>> _ = axes[1].set_ylabel('theta')
        >>>
        >>> plt.show()

    """
    # Extract the field shape so that we can run the validation procedures
    # on the various other components.
    grid_shape = scalar_field.shape

    # Validate the Lterm-field first. This MUST have the same number of components as derivatives
    # we can take.
    if lterm_field.shape != scalar_field.shape + (scalar_field.ndim,):
        raise ValueError(
            f"L-term field must have shape (grid_shape, ndim) [in this case {grid_shape + (scalar_field.ndim,)}], but had shape {lterm_field.shape}."
        )

    # Construct the first order derivative field from the input data.
    # These are just the grid component gradients of the field.
    if (derivative_field is None) and (spacing is None):
        raise ValueError(
            "Either ``spacing`` or ``derivative_field`` must be specified."
        )
    elif derivative_field is not None:
        # The derivative field is provided. We just need to validate that it has the correct
        # shape.
        if derivative_field.shape != scalar_field.shape + (scalar_field.ndim,):
            raise ValueError(
                f"Derivative field had shape {derivative_field.shape} but was expected to have shape {grid_shape + (scalar_field.ndim,)}."
            )
    else:
        # We need to compute the derivative field using the spacing array.
        if len(spacing) != scalar_field.ndim:
            raise ValueError("`spacing` must match the number of grid dimensions.")

        # Compute the first derivative field.
        axes = np.arange(scalar_field.ndim)
        if len(spacing) > 1:
            # These can be automatically stacked without reshaping.
            derivative_field = np.stack(
                np.gradient(scalar_field, *spacing, axis=axes, **kwargs), axis=-1
            )
        else:
            # This requires a reshape because there is only one axis.
            derivative_field = np.gradient(
                scalar_field, *spacing, axis=axes, **kwargs
            ).reshape((*scalar_field.shape, 1))

    # Construct the second derivative grid from the first derivatives.
    _sdf_expected_shape = scalar_field.shape + (scalar_field.ndim, scalar_field.ndim)
    if second_derivative_field is not None:
        # This has been provided, we just need to validate that the shape is correct and
        # behaves the way we expect it to.
        if second_derivative_field.shape != _sdf_expected_shape:
            raise ValueError(
                f"Expected `second_derivative_field` to have shape {_sdf_expected_shape} not {derivative_field.shape}."
            )
    elif spacing is None:
        raise ValueError(
            "Either ``spacing`` or `second_derivative_field`` must be specified."
        )
    else:
        # Compute the second derivatives.
        second_derivative_field = np.zeros(_sdf_expected_shape)
        for i in range(scalar_field.ndim):
            for j in range(scalar_field.ndim):
                second_derivative_field[..., i, j] = np.gradient(
                    derivative_field[..., i], spacing[j], axis=j, **kwargs
                )

    # Now contract against the metric to form the standardized value.

    if inv_metric.shape != second_derivative_field.shape:
        raise ValueError(
            f"Shape mismatch between `inv_metric` and `second_derivative_field`: {inv_metric.shape} != {second_derivative_field.shape}."
        )

    _laplacian_term_2 = np.sum(inv_metric * second_derivative_field, axis=(-1, -2))
    _laplacian_term_1 = np.sum(lterm_field * derivative_field, axis=-1)

    return _laplacian_term_1 + _laplacian_term_2


def glap_orth(
    scalar_field: np.ndarray,
    lterm_field: np.ndarray,
    inv_metric: np.ndarray,
    spacing: Sequence[Union[float, np.ndarray]] = None,
    derivative_field: np.ndarray = None,
    second_derivative_field: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the Laplacian of a scalar field in an orthogonal coordinate system.

    This function evaluates the Laplacian using the formula:

    .. math::

        \Delta \phi = L^\mu \partial_\mu \phi + g^{\mu \mu} \partial^2_\mu \phi

    where :math:`L^\mu` is the so-called L-term (from the Jacobian of the coordinate system),
    and :math:`g^{\mu \mu}` is the inverse metric tensor.

    Parameters
    ----------
    scalar_field : np.ndarray
        The scalar field :math:`\phi` discretized on a grid with shape ``(...)``.
    lterm_field : np.ndarray
        Field of L-term values with shape ``(..., ndim)``, where ``ndim = scalar_field.ndim``.
        This can usually be computed from the coordinate system.
    inv_metric : :py:class:`numpy.ndarray`
        The inverse metric tensor's diagonal component field :math:`g^{\mu \mu}` at each point on the grid. This should be provided as a
        ``(..., scalar_field.ndim)`` array. The metric is then contracted against the covariant gradient to compute the result.
    spacing : list of float or numpy.ndarray, optional
        Grid spacing along each axis. Required if ``derivative_field`` and ``second_derivative_field`` are not provided.
    derivative_field : np.ndarray, optional
        Precomputed first derivatives of shape ``(..., ndim)``. If provided, numerical derivatives are skipped.
    second_derivative_field : np.ndarray, optional
        Precomputed second derivatives of shape ``(..., ndim, ndim)``. If provided, second-order gradients are not computed.
    **kwargs :
        Additional keyword arguments passed to :py:func:`numpy.gradient`.

    Returns
    -------
    np.ndarray
        The Laplacian of the scalar field evaluated at all grid points. Shape ``(...)``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or required parameters are missing.

    Examples
    --------

    .. plot::
        :include-source:

        >>> # Import modules
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces_geometry.differential_geometry.operators import glap_cl
        >>>
        >>> # Construct the coordinates and the coordinate grid.
        >>> # For this example, we just use R and THETA (phi held constant)
        >>> r,theta = np.linspace(0.1,2,100),np.linspace(np.pi/4,np.pi*(3/4),100)
        >>> R,THETA = np.meshgrid(r,theta,indexing='ij')
        >>> C = np.stack([R,THETA],axis=-1)
        >>>
        >>> # Define the scalar field sin(theta) on the grid.
        >>> F = np.sin(THETA)
        >>>
        >>> # Construct the inverse-metric tensor field on the grid
        >>> # so that it can be used when performing operations.
        >>> g_inv = np.zeros((100,100,2))
        >>> g_inv[...,0] = 1
        >>> g_inv[...,1] = R**-2
        >>>
        >>> # Construct the 2 L-terms for this case.
        >>> l_terms = np.zeros((100,100,2))
        >>> l_terms[...,0] = 2/R
        >>> l_terms[...,1] = 1/(R**2 * np.tan(THETA))
        >>>
        >>> # Compute the laplacian and the "true" (theoretical) laplacian.
        >>> L = glap_orth(F,l_terms,g_inv,spacing=[(2-0.1)/100,(np.pi/2)/100])
        >>> Ltrue = np.cos(2*THETA)/(np.sin(THETA) * R**2)
        >>>
        >>> # Plotting data
        >>> fig,axes = plt.subplots(2,1,figsize=(6,8),sharex=True)
        >>> extent = [0.1,2,np.pi/4,np.pi*(3/4)]
        >>>
        >>> # Plot the computed case:
        >>> Q = axes[0].imshow(L.T,vmin=-20,vmax=1,origin='lower',extent=extent)
        >>> _ = plt.colorbar(Q)
        >>> P = axes[1].imshow(Ltrue.T,vmin=-20,vmax=1,origin='lower',extent=extent)
        >>> _ = plt.colorbar(P)
        >>> _ = axes[1].set_xlabel('r')
        >>> _ = axes[0].set_ylabel('theta')
        >>> _ = axes[1].set_ylabel('theta')
        >>>
        >>> plt.show()

    """
    # Extract the field shape so that we can run the validation procedures
    # on the various other components.
    grid_shape = scalar_field.shape

    # Validate the Lterm-field first. This MUST have the same number of components as derivatives
    # we can take.
    if lterm_field.shape != scalar_field.shape + (scalar_field.ndim,):
        raise ValueError(
            f"L-term field must have shape (grid_shape, ndim) [in this case {grid_shape + (scalar_field.ndim,)}], but had shape {lterm_field.shape}."
        )

    # Construct the first order derivative field from the input data.
    # These are just the grid component gradients of the field.
    if (derivative_field is None) and (spacing is None):
        raise ValueError(
            "Either ``spacing`` or ``derivative_field`` must be specified."
        )
    elif derivative_field is not None:
        # The derivative field is provided. We just need to validate that it has the correct
        # shape.
        if derivative_field.shape != scalar_field.shape + (scalar_field.ndim,):
            raise ValueError(
                f"Derivative field had shape {derivative_field.shape} but was expected to have shape {grid_shape + (scalar_field.ndim,)}."
            )
    else:
        # We need to compute the derivative field using the spacing array.
        if len(spacing) != scalar_field.ndim:
            raise ValueError("`spacing` must match the number of grid dimensions.")

        # Compute the first derivative field.
        axes = np.arange(scalar_field.ndim)
        if len(spacing) > 1:
            # These can be automatically stacked without reshaping.
            derivative_field = np.stack(
                np.gradient(scalar_field, *spacing, axis=axes, **kwargs), axis=-1
            )
        else:
            # This requires a reshape because there is only one axis.
            derivative_field = np.gradient(
                scalar_field, *spacing, axis=axes, **kwargs
            ).reshape((*scalar_field.shape, 1))

    # Construct the second derivative grid from the first derivatives.
    _sdf_expected_shape = scalar_field.shape + (scalar_field.ndim,)
    if second_derivative_field is not None:
        # This has been provided, we just need to validate that the shape is correct and
        # behaves the way we expect it to.
        if second_derivative_field.shape != _sdf_expected_shape:
            raise ValueError(
                f"Expected `second_derivative_field` to have shape {_sdf_expected_shape} not {derivative_field.shape}."
            )
    elif spacing is None:
        raise ValueError(
            "Either ``spacing`` or `second_derivative_field`` must be specified."
        )
    else:
        # Compute the second derivatives.
        second_derivative_field = np.zeros(_sdf_expected_shape)
        for i in range(scalar_field.ndim):
            second_derivative_field[..., i] = np.gradient(
                derivative_field[..., i], spacing[i], axis=i, **kwargs
            )

    # Now contract against the metric to form the standardized value.
    if inv_metric.shape != second_derivative_field.shape:
        raise ValueError(
            f"Shape mismatch between `inv_metric` and `second_derivative_field`: {inv_metric.shape} != {second_derivative_field.shape}."
        )

    _laplacian_term_2 = np.sum(inv_metric * second_derivative_field, axis=-1)
    _laplacian_term_1 = np.sum(lterm_field * derivative_field, axis=-1)

    return _laplacian_term_1 + _laplacian_term_2
