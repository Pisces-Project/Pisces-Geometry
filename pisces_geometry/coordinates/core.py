"""
Core coordinate system classes for user / developer subclassing.

This module (as opposed to :py:mod:`~pisces_geometry.coordinates.base`) provides stub classes for defining
custom coordinate systems that fall into a few standard types:

1. **Curvilinear Coordinate Systems**: should be descended from :py:class:`CurvilinearCoordinateSystem`.
2. **Orthogonal Coordinate Systems**: should be descended from :py:class:`OrthogonalCoordinateSystem`.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Callable, Sequence, Optional

import numpy as np
import sympy as sp

from pisces_geometry.coordinates.base import _CoordinateSystemBase, class_expression, _get_grid_spacing
from pisces_geometry.differential_geometry import (
    raise_index_orth,
    lower_index_orth,
    contract_index_with_metric_orthogonal,
    ggrad_orth_covariant,
    ggrad_orth_contravariant,
    gdiv_orth_covariant,
    gdiv_orth_contravariant,
    compute_Dterm,
    compute_Lterm,
    compute_Lterm_orthogonal,
    glap_orth
)
from pisces_geometry.utilities.general import find_in_subclasses


class CurvilinearCoordinateSystem(_CoordinateSystemBase, ABC):
    """
    Base class for curvilinear coordinate systems.
    """
    pass

class OrthogonalCoordinateSystem(_CoordinateSystemBase, ABC):
    """
    # TODO

    Attributes
    ----------
    __is_abstract__ : bool
        Indicates whether the class is abstract (not directly instantiable). For developers subclassing this class, this
        flag should be set to ``False`` if the coordinate system is actually intended for use. Behind the scenes, this flag
        is checked by the metaclass to ensure that it does not attempt to validate or create symbols for abstract classes.
    __setup_point__ : 'init' or 'import'
        Determines when the class should perform symbolic processing. If ``import``, then the class will create its symbols
        and its metric function as soon as the class is loaded (the metaclass performs this). If ``'init'``, then the symbolic
        processing is delayed until a user instantiates the class for the first time.

        .. admonition:: Developer Standard

            In general, there is no reason to use anything other than ``__setup_point__ = 'init'``. Using ``'import'`` can
            significantly slow down the loading process because it requires processing many coordinate systems which may not
            end up getting used at all.

    __is_setup__ : bool
        Tracks whether the class has been set up. **This should not be changed**.
    __AXES__ : :py:class:`list` of str
        A list of the coordinate system's axes. These are then used to create the symbolic versions of the axes which
        are used in expressions. Subclasses should fill ``__AXES__`` with the intended list of axes in the intended axis
        order.
    __PARAMETERS__ : :py:class:`dict` of str, float
        Dictionary of system parameters with default values. Each entry should be the name of the parameter and each value
        should correspond to the default value. These are then provided by the user as ``**kwargs`` during ``__init__``.
    __axes_symbols__ : :py:class:`list` of :py:class:`~sympy.core.symbol.Symbol`
        Symbolic representations of each coordinate axis. **Do not alter**.
    __parameter_symbols__ : :py:class:`dict` of str, :py:class:`~sympy.core.symbol.Symbol`
        Symbolic representations of parameters in the system. **Do not alter**.
    __class_expressions__ : dict
        Dictionary of symbolic expressions associated with the system. **Do not alter**.
    __NDIM__ : int
        Number of dimensions in the coordinate system. **Do not alter**.

    """
    # @@ CLASS FLAGS @@ #
    # CoordinateSystem flags are used to indicate to the metaclass whether
    # certain procedures should be executed on the class.
    __is_abstract__: bool = True  # Marks this class as abstract - no symbolic processing (unusable)
    __setup_point__: Literal['init', 'import'] = 'init'  # Determines when symbolic processing should occur.
    __is_setup__: bool = False  # Used to check if the class has already been set up.

    # @@ CLASS ATTRIBUTES @@ #
    # The CoordinateSystem class attributes provide some of the core attributes
    # for all coordinate systems and should be adjusted in all subclasses to initialize
    # the correct axes, dimensionality, etc.
    __AXES__: List[str] = None
    """list of str: The axes (coordinate variables) in this coordinate system.
    This is one of the class-level attributes which is specified in all coordinate systems to determine
    the names and symbols for the axes. The length of this attribute also determines how many dimensions 
    the coordinate system has.
    """
    __PARAMETERS__: Dict[str, Any] = dict()
    """ dict of str, Any: The parameters for this coordinate system and their default values.

    Each of the parameters in :py:attr:`~pisces.geometry.base.CoordinateSystem.PARAMETERS` may be provided as
    a ``kwarg`` when creating a new instance of this class.
    """

    # @@ CLASS BUILDING PROCEDURES @@ #
    # During either import or init, the class needs to build its symbolic attributes in order to
    # be usable. The class attributes and relevant class methods are defined in this section
    # of the class object.
    __axes_symbols__: List[sp.Symbol] = None  # The symbolic representations of each axis.
    __parameter_symbols__: Dict[str, sp.Symbol] = None  # The symbolic representation of each of the parameters.
    __class_expressions__: Dict[
        str, Any] = {}  # The expressions that are generated for this class.
    __NDIM__: int = None  # The number of dimensions that this coordinate system has.

    @classmethod
    def __construct_explicit_class_expressions__(cls):
        """
        Construct the symbolic metric and inverse metric tensors along with any other critical
        symbolic attributes for operations.

        This method calls:
        - `__construct_metric_tensor_symbol__`
        - `__construct_inverse_metric_tensor_symbol__`

        It stores the results in:
        - `__class_metric_tensor__`
        - `__class_inverse_metric_tensor__`
        - `__metric_determinant_expression__`

        Notes
        -----
        This method is typically overridden in `_OrthogonalCoordinateSystemBase` to avoid computing the inverse directly.
        """
        # Derive the metric, inverse metric, and the metric density. We call to the
        # __construct_metric_tensor_symbol__ and then take the inverse and the determinant of
        # the matrices.
        cls.__class_expressions__['metric_tensor'] = cls.__construct_metric_tensor_symbol__(*cls.__axes_symbols__,
                                                                                            **cls.__parameter_symbols__)
        cls.__class_expressions__['inverse_metric_tensor'] = sp.Array([1/_element for _element in cls.__class_expressions__['metric_tensor']])
        cls.__class_expressions__['metric_density'] = sp.sqrt(sp.prod(cls.__class_expressions__['metric_tensor']))

        # Any additional core expressions can be added here. The ones above can also be modified as
        # needed.

    @property
    def metric_tensor_symbol(self) -> Any:
        """
        The symbolic array representing the metric tensor. This is an ``(ndim,)`` array of symbolic expressions
        each representing the diagonal elements of the metric tensor.
        """
        return super().metric_tensor_symbol

    @property
    def metric_tensor(self) -> Callable:
        """
        Returns the callable function for the metric tensor of the coordinate system.

        The (diagonal components of the) metric tensor :math:`g_{ii}` defines the inner product structure of the coordinate system.
        It is used for measuring distances, computing derivatives, and raising/lowering indices.
        This function returns the precomputed metric tensor as a callable function, which can be
        evaluated at specific coordinates.

        Returns
        -------
        Callable
            A function that computes the metric tensor :math:`g_{ii}` when evaluated at specific coordinates.
            The returned function takes numerical coordinate values as inputs and outputs a NumPy array
            of shape ``(ndim, )``.

        Example
        -------
        .. code-block:: python

            cs = MyCoordinateSystem()
            g_ii = cs.metric_tensor(x=1, y=2, z=3)  # Evaluates the metric at (1,2,3)
            print(g_ii.shape)  # Output: (ndim, )

        """
        return super().metric_tensor

    @property
    def inverse_metric_tensor(self) -> Callable:
        """
        Returns the callable function for the inverse metric tensor of the coordinate system.

        The inverse metric tensor :math:`g^{ii}` is the inverse of :math:`g_{ii}` and is used to raise indices,
        compute dual bases, and perform coordinate transformations. This function returns a callable
        representation of :math:`g^{ii}`, allowing evaluation at specific coordinate points.

        Returns
        -------
        Callable
            A function that computes the inverse metric tensor :math:`g^{ii}` when evaluated at specific coordinates.
            The returned function takes numerical coordinate values as inputs and outputs a NumPy array
            of shape ``(ndim,)``.

        Example
        -------
        .. code-block:: python

            cs = MyCoordinateSystem()
            g_inv = cs.inverse_metric_tensor(x=1, y=2, z=3)  # Evaluates g^{ij} at (1,2,3)
            print(g_inv.shape)  # Output: (ndim,)

        """
        return super().inverse_metric_tensor


    # @@ COORDINATE METHODS @@ #
    # These methods dictate the behavior of the coordinate system including how
    # coordinate conversions behave and how the coordinate system handles differential
    # operations.
    @staticmethod
    @abstractmethod
    def __construct_metric_tensor_symbol__(*args, **kwargs) -> sp.Array:
        r"""
        Constructs the metric tensor for the coordinate system.

        The metric tensor defines the way distances and angles are measured in the given coordinate system.
        It is used extensively in differential geometry and tensor calculus, particularly in transformations
        between coordinate systems.

        This method must be implemented by subclasses to specify how the metric tensor is computed.
        The returned array should contain symbolic expressions that define the metric's DIAGONAL components.

        Parameters
        ----------
        *args : tuple of sympy.Symbol
            The symbolic representations of each coordinate axis.
        **kwargs : dict of sympy.Symbol
            The symbolic representations of the coordinate system parameters.

        Returns
        -------
        sp.Array
            A symbolic ``NDIM `` matrix representing the metric tensor's diagonal components.

        Notes
        -----
        - This method is abstract and must be overridden in derived classes.
        - The metric tensor is used to compute distances, gradients, and other differential operations.
        - In orthogonal coordinate systems, the metric tensor is diagonal.
        """
        pass


    # @@ Conversion Functions @@ #
    # Perform conversions to / from cartesian coordinates.
    @abstractmethod
    def _convert_native_to_cartesian(self, *args):
        pass

    @abstractmethod
    def _convert_cartesian_to_native(self, *args):
        pass

    # @@ Mathematical Operations @@ #
    # These should only be changed in fundamental subclasses where
    # new mathematical approaches become available.
    def raise_index(self,
                    tensor_field: np.ndarray,
                    index: int,
                    rank: int,
                    metric: np.ndarray = None,
                    coordinate_grid: np.ndarray = None,
                    fixed_axes: Dict[str, float] = None,
                    **kwargs) -> np.ndarray:
        r"""
        Raise a single index of a tensor field using the metric of this coordinate system.

        Parameters
        ----------
        tensor_field : numpy.ndarray
            The input tensor field. The tensor field should be an array of shape ``(..., ndim, ndim, ...)``, where
            each of the later axes is a tensor component axis for a particular rank.
        index : int
            Index (0-based, relative to tensor rank) to raise.
        rank : int
            Number of tensor dimensions (excluding grid).
        metric : numpy.ndarray, optional
            Metric tensor :math:`g_{ii}` of shape ``(..., ndim)``. If not provided,
            it will be computed from ``coordinate_grid``.
        coordinate_grid : numpy.ndarray, optional
            Coordinate grid of shape ``(..., k)``. Required if ``inverse_metric`` is not provided.
        fixed_axes : dict, optional
            Dictionary of axis values to fix during metric evaluation.
        **kwargs :
            Passed through to the core :py:func:`~pisces_geometry.differential_geometry.tensor_utilities.raise_index` utility.

        Returns
        -------
        numpy.ndarray
            Tensor field with the specified index raised.

        Raises
        ------
        ValueError
            If both inverse_metric and coordinate_grid are None.

        Notes
        -----
        This operation performs:

        .. math::

            T^{a} = g^{aa} T_a

        The index position to be raised is specified by `index`.

        """
        metric = self.compute_metric_on_grid(coordinate_grid, inverse=False, fixed_axes=fixed_axes,
                                                     value=metric)
        return raise_index_orth(tensor_field, index, rank, metric, **kwargs)

    def lower_index(self,
                    tensor_field: np.ndarray,
                    index: int,
                    rank: int,
                    metric: np.ndarray = None,
                    coordinate_grid: np.ndarray = None,
                    fixed_axes: Dict[str, float] = None,
                    **kwargs) -> np.ndarray:
        r"""
        Lower a single index of a tensor field using the metric of this coordinate system.

        Parameters
        ----------
        tensor_field : numpy.ndarray
            The input tensor field. The tensor field should be an array of shape ``(..., ndim, ndim, ...)``, where
            each of the later axes is a tensor component axis for a particular rank.
        index : int
            Index (0-based, relative to tensor rank) to raise.
        rank : int
            Number of tensor dimensions (excluding grid).
        metric : numpy.ndarray, optional
            Metric tensor :math:`g_{ij}` of shape ``(..., ndim, ndim)``. If not provided,
            it will be computed from ``coordinate_grid``.
        coordinate_grid : numpy.ndarray, optional
            Coordinate grid of shape ``(..., k)``. Required if ``inverse_metric`` is not provided.
        fixed_axes : dict, optional
            Dictionary of axis values to fix during metric evaluation.
        **kwargs :
            Passed through to the core :py:func:`~pisces_geometry.differential_geometry.tensor_utilities.lower_index` utility.

        Returns
        -------
        numpy.ndarray
            Tensor field with the specified index raised.

        Raises
        ------
        ValueError
            If both inverse_metric and coordinate_grid are None.

        Notes
        -----
        This operation performs:

        .. math::

            T^{a} = g^{aa} T_a

        The index position to be raised is specified by `index`.

        """
        metric = self.compute_metric_on_grid(coordinate_grid, inverse=False, fixed_axes=fixed_axes, value=metric)
        return lower_index_orth(tensor_field, index, rank, metric, **kwargs)

    # noinspection PyMethodOverriding
    def adjust_tensor_signature(self,
                                tensor_field: np.ndarray,
                                indices: List[int],
                                modes: List[Literal["upper", "lower"]],
                                rank: int,
                                metric: np.ndarray = None,
                                coordinate_grid: np.ndarray = None,
                                component_masks: Optional[List[np.ndarray]] = None,
                                inplace: bool = False,
                                fixed_axes: Dict[str, float] = None) -> np.ndarray:
        """
        Adjust the tensor signature by raising or lowering specified indices.

        This method allows selective index raising or lowering using the metric or
        inverse metric tensor of the coordinate system. It is useful when converting between
        covariant and contravariant representations of tensors.

        Parameters
        ----------
        tensor_field : numpy.ndarray
            The input tensor field. Shape must be (..., i1, ..., iN) where the last `rank`
            axes are the tensor indices.
        indices : list of int
            The indices (0-based, relative to the tensor rank) to adjust.
        modes : list of {"upper", "lower"}
            Modes corresponding to each index in `indices`. Use "upper" to raise and
            "lower" to lower an index.
        rank : int
            The total number of tensor indices in the last dimensions of `tensor_field`.
        metric : numpy.ndarray, optional
            The full metric tensor g_{ii}. Required unless `coordinate_grid`
            is provided.
        coordinate_grid : numpy.ndarray, optional
            Coordinate grid of shape (..., k), where k is the number of free axes.
            Used to evaluate metric/inverse_metric if they are not directly provided.
        component_masks : list of numpy.ndarray, optional
            Masks to restrict which components are raised/lowered along specific axes.
            Each mask should match the dimensionality of the tensor index being adjusted.
        inplace : bool, optional
            If True, modifies the tensor in-place. Defaults to False (returns a copy).
        fixed_axes : dict of str, float, optional
            Mapping of axis names to fixed values, used for evaluating the metric from
            `coordinate_grid`.

        Returns
        -------
        numpy.ndarray
            Tensor with specified indices raised or lowered.

        Raises
        ------
        ValueError
            If the number of `indices` and `modes` do not match.
            If the number of `component_masks` does not match `indices`.
            If neither a metric nor coordinate grid is available to compute the required contraction.

        Notes
        -----
        - The operation contracts the specified index with the metric (for lowering)
          or inverse metric (for raising).
        - Supports selective component manipulation through `component_masks`.

        """
        if len(indices) != len(modes):
            raise ValueError("Each index must have a corresponding mode.")
        if component_masks and len(component_masks) != len(indices):
            raise ValueError("If masks are provided, must match length of indices.")

        grid_shape = tensor_field.shape[:-rank]
        tensor_shape = tensor_field.shape[-rank:]

        working_tensor = np.copy(tensor_field) if not inplace else tensor_field
        for i, (idx, mode) in enumerate(zip(indices, modes)):

            # Apply the element mask.
            mask = component_masks[i] if component_masks else slice(None)
            metric = self.compute_metric_on_grid(coordinate_grid, inverse=False, fixed_axes=fixed_axes,
                                                 value=metric)
            current_metric = metric[..., mask] if isinstance(mask, np.ndarray) else metric
            if mode == "lower":
                pass
            elif mode == "upper":
                current_metric = 1/current_metric
            else:
                raise ValueError(f"Invalid mode '{mode}' for index {idx}")

            working_tensor = contract_index_with_metric_orthogonal(working_tensor, current_metric, idx, rank)

        return working_tensor

    @class_expression(name='Lterm')
    @classmethod
    def __compute_Lterm__(cls, *args, **kwargs):
        r"""
        Computes the D-term :math:`(1/\rho)\partial_\mu \rho` for use in
        computing the divergence numerically.
        """
        _metric_density = cls.__class_expressions__['metric_density']
        _metric_tensor = cls.__class_expressions__['metric_tensor']
        _axes = cls.__axes_symbols__

        return compute_Lterm_orthogonal(_metric_tensor, _metric_density, _axes)

    def compute_gradient(self,
                         scalar_field: np.ndarray,
                         /,
                         spacing: Sequence[int] = None,
                         coordinate_grid: np.ndarray = None,
                         derivative_field: np.ndarray = None,
                         metric: np.ndarray = None,
                         *,
                         basis: Literal['covariant', 'contravariant'] = 'covariant',
                         fixed_axes: Dict[str, float] = None,
                         is_uniform: bool = False,
                         **kwargs):
        r"""
        Compute the gradient of a scalar field in either covariant or contravariant basis.

        In a general coordinate system, the gradient is

        .. math::

            \nabla \phi = \partial_\mu \phi {\bf e}^\mu = g^{\mu \nu} \partial_\nu \phi {\bf e}_\mu.

        Parameters
        ----------
        scalar_field: numpy.ndarray
            The scalar field (:math:`\phi`) to compute the gradient for. This should be provided as an array
            of shape ``(...,)`` where ``(...)`` is a uniformly spaced grid stencil.
        spacing : list of float, optional
            Grid spacing along each axis. If None, inferred from ``coordinate_grid``.
        coordinate_grid : numpy.ndarray
            Grid of coordinates of shape ``(..., ndim)``, containing the **free axes**. ``coordinate_grid`` does not
            need to include all coordinates, just those that appear in the grid. Fixed coordinates (i.e. ``z`` in an ``x,y`` slice)
            can be specified in ``fixed_axes``.
        derivative_field : numpy.ndarray
            Field of derivative values of shape ``(..., n_free)`` containing the derivatives along each of the grid axes.
            If provided, this will circumvent the numerical evaluation of the derivatives.
        metric : numpy.ndarray, optional
            The metric tensor :math:`g^{ii}` which is used when evaluating the contravariant components. If it is
            not explicitly specified, then it will be computed from other parameters when it is needed.
        basis : {'covariant', 'contravariant'}
            The basis in which the gradient should be computed.
        fixed_axes : dict, optional
            Dictionary of fixed axis values for axes not included in the coordinate grid.
        is_uniform: bool, optional
            Whether the coordinate system is uniform or not. This dictates how spacing is computed for
            derivatives. Defaults to ``False``.
        **kwargs :
            Additional keyword arguments passed to :py:func:`numpy.gradient` when computing numerical
            derivatives. This can include options like ``edge_order`` or ``axis`` if needed for customized
            differentiation.

        Returns
        -------
        numpy.ndarray
            Gradient of the scalar field in the requested basis, shape (..., len(components)).

        Raises
        ------
        ValueError
            If there are shape mismatches or inconsistent axis definitions.

        """
        # Determine the number of axes in the grid.
        grid_ndim = scalar_field.ndim

        # Determine the grid spacing if it is necessary (derivative field not provided). Either the
        # spacing has been provided or we need to get it from the coordinate grid.
        if (derivative_field is None) and (spacing is None) and (coordinate_grid is None):
            raise ValueError()
        elif derivative_field is not None:
            # We have a derivative field which massively simplifies things because we no longer need
            # the spacing.
            pass
        elif spacing is not None:
            # We have the spacing. We just need to validate that it matches the number of
            # dimensions in the grid and then it'll be ready to use.
            if len(spacing) != grid_ndim:
                raise ValueError()
        elif coordinate_grid is not None:
            # The coordinate grid is not none, which means we need to get the spacing from
            # the coordinate grid.

            spacing = _get_grid_spacing(coordinate_grid, is_uniform=is_uniform)

        # The spacing or the derivative field is now available, we have everything we need
        # to at least compute the covariant case. For the contravariant case, we'll still need to
        # establish a metric.
        if basis == 'covariant':
            return ggrad_orth_covariant(scalar_field,
                                        spacing=spacing,
                                        derivative_field=derivative_field,
                                        **kwargs)
        if basis == 'contravariant':
            # The contravariant approach will require an inverse metric to be established
            # on the coordinate grid. If we don't have it, then we need to build it from scratch
            # using the coordinate system internals.
            inverse_metric = self.compute_metric_on_grid(coordinate_grid, inverse=True, fixed_axes=fixed_axes,
                                                         value=metric)

            # Now the inverse metric is assuredly available and we can pass to the lower
            # level callable.
            return ggrad_orth_contravariant(scalar_field,
                                            spacing,
                                            inverse_metric,
                                            derivative_field=derivative_field,
                                            **kwargs)
        else:
            raise ValueError(f"Unknown basis {basis}.")

    def compute_divergence(self,
                           vector_field: np.ndarray,
                           /,
                           dterm_field: np.ndarray = None,
                           coordinate_grid: np.ndarray = None,
                           spacing: np.ndarray = None,
                           metric: np.ndarray = None,
                           derivative_field: np.ndarray = None,
                           *,
                           basis: str = 'contravariant',
                           fixed_axes: Dict[str, float] = None,
                           components: List[str] = None,
                           is_uniform: bool = False,
                           **kwargs):
        r"""
        Compute the divergence of a vector field in a specified basis.

        The divergence of a vector field :math:`{\bf v}` is defined as:

        .. math::

            \nabla \cdot {\bf v} = \frac{1}{\rho}\partial_\mu \left(\rho v^\mu\right),

        where :math:`\rho` is the metric density. In the covariant basis (``basis='covariant'``),

        .. math::

            \nabla \cdot {\bf v} = \frac{1}{\rho}\partial_\mu \left(\rho g^{\mu\nu} v_\nu\right).

        Parameters
        ----------
        vector_field : numpy.ndarray
            The input vector field of shape ``(..., k)``, where ``k`` is the number of components. The leading components
            (the ``...``) should be the grid stencil for the field. The values of the ``vector_field`` array should be the
            component values of the field in the basis specified by the ``basis`` kwarg.
        dterm_field : numpy.ndarray, optional
            Precomputed values for the D-term field (:math:`D_\mu`). If provided, the ``dterm_field`` should be an array
            with shape matching the ``vector_field``.

            If these are provided, they won't be calculated based on the coordinate system and the ``coordinate_grid``.
            If they are not provided, then ``coordinate_grid`` is required to compute the D-term on the grid.

            .. hint::

                The **D-term** is the coordinate-system specific term

                .. math::

                    D_\mu = \frac{1}{\rho}\partial_\mu \rho.

                Coordinate system classes know how to compute these on their own, but providing values can speed up
                computations.

        coordinate_grid : numpy.ndarray, optional
            The coordinate grid of shape ``(..., ndim)``. Used to infer spacing and compute metric/D-term if not provided. This
            argument is required if any of ``inverse_metric``, ``dterm_field``, or ``spacing`` are not provided.
        spacing : numpy.ndarray, optional
            Grid spacing along each axis. If ``None``, inferred from ``coordinate_grid``.
        metric : numpy.ndarray, optional
            The metric tensor (:math:`g^{\mu\mu}`), used only for covariant basis computations. If ``coordinate_grid`` is not
            provided, and ``basis`` is ``covariant`` then this is a required argument. Otherwise it will be computed if necessary.
        derivative_field : numpy.ndarray, optional
            Optional precomputed derivatives of the vector field. The ``derivative_field`` should be an array of shape ``(..., k)``,
            where ``k`` are the axes of the grid which also have components in the ``vector_field``. Specifying this argument will
            skip numerical derivative computations.

            .. note::

                The divergence may be written as

                .. math::

                    \nabla \cdot {\bf F} = D_\mu F^\mu + \partial_\mu F^\mu,

                For the derivative term, the only components that are necessary are those for which both :math:`F^\mu \neq 0`,
                and the grid also spans :math:`x^\mu`. These components are determined internally based on ``fixed_axes``, and
                ``derivative_field`` (if specified) must match the number of such axes in its size.

            .. warning::

                If ``basis='covariant'``, then the derivative terms are

                .. math::

                    \partial_\mu g^{\mu \nu} F_\nu.

                and must be provided as such to get correct results.

        basis : {'contravariant', 'covariant'}, default 'contravariant'
            The basis in which ``vector_field`` is represented when it is passed to the method.
        fixed_axes : dict, optional
            Dictionary of axis names and values held constant (for slices) in the grid. If not specified,
            then the grid must contain all of the coordinate axes.

            .. hint::

                This allows for slices of the coordinate system to be treated self-consistently by
                adding the invariant axis as a ``fixed_axes`` element.

        components : list of str, optional
            The names of the components in ``vector_field``. If not specified, then ``vector_field`` must provide
            **all** of the components for each of the coordinate system axes. If specified, then ``components`` should be a
            list of axes names matching the number of components specified in ``vector_field``.
        is_uniform: bool, optional
            Whether the coordinate system is uniform or not. This dictates how spacing is computed for
            derivatives. Defaults to ``False``.
        **kwargs :
            Additional keyword arguments passed to :py:func:`numpy.gradient` when computing numerical
            derivatives. This can include options like ``edge_order`` or ``axis`` if needed for customized
            differentiation.

        Returns
        -------
        numpy.ndarray
            The computed divergence of the vector field, shape ``(...,)``.

        Raises
        ------
        ValueError
            If required inputs are missing or mismatched in shape.
        """
        # Determine the number of grid dimensions and the number
        # of vector components.
        grid_ndim = vector_field.ndim - 1
        comp_ndim = vector_field.shape[-1]
        free_axes, _ = self.get_free_fixed_axes(grid_ndim, fixed_axes=fixed_axes)

        # Validate the input for the components and generate it.
        if (components is None) and (comp_ndim != self.ndim):
            raise ValueError()
        elif components is not None:
            # we have the components already available.
            pass
        elif comp_ndim == self.ndim:
            components = self.axes

        # We can now construct the axis connector.
        axis_connector = self.get_axis_connector(free_axes, components)

        # Determine the grid spacing if it is necessary (derivative field not provided). Either the
        # spacing has been provided or we need to get it from the coordinate grid.
        if (derivative_field is None) and (spacing is None) and (coordinate_grid is None):
            raise ValueError()
        elif derivative_field is not None:
            # We have a derivative field which massively simplifies things because we no longer need
            # the spacing.
            pass
        elif spacing is not None:
            # We have the spacing. We just need to validate that it matches the number of
            # dimensions in the grid and then it'll be ready to use.
            if len(spacing) != grid_ndim:
                raise ValueError()
        elif coordinate_grid is not None:
            # The coordinate grid is not none, which means we need to get the spacing from
            # the coordinate grid.
            spacing = _get_grid_spacing(coordinate_grid, is_uniform=is_uniform)

        # Compute the D-term fields if they are not already available. Validate the shape of the dterms.
        dterm_field = self.compute_expression_on_grid('Dterm', coordinate_grid, fixed_axes=fixed_axes,
                                                      value=dterm_field)
        if dterm_field.shape == (comp_ndim,) + vector_field.shape[:-1]:
            dterm_field = np.moveaxis(dterm_field, 0, -1)

        # The spacing or the derivative field is now available, we have everything we need
        # to at least compute the covariant case. For the contravariant case, we'll still need to
        # establish a metric.
        if basis == 'contravariant':
            # We can plug into the low level `gdiv_cl_contravariant` which doesn't need us
            # to provide a metric tensor because this is the "natural" basis.
            return gdiv_orth_contravariant(vector_field,
                                         dterm_field,
                                         spacing=spacing,
                                         derivative_field=derivative_field,
                                         axis_connector=axis_connector, )
        elif basis == 'covariant':
            # We do need to use the metric tensor in `gdiv_cl_covariant`, which may require
            # computing the metric tensor. We'll follow the same procedure as in compute_gradient to
            # establish the correct inverse metric tensor.
            inverse_metric = self.compute_metric_on_grid(coordinate_grid, inverse=True, fixed_axes=fixed_axes,
                                                         value=metric)

            # Hand off the computation to the covariant solver.
            return gdiv_orth_covariant(vector_field,
                                     dterm_field,
                                     inverse_metric,
                                     spacing=spacing,
                                     derivative_field=None,
                                     axis_connector=axis_connector)

        else:
            raise ValueError(f"Unknown basis {basis}.")

    def compute_laplacian(self,
                          scalar_field: np.ndarray,
                          /,
                          lterm_field: np.ndarray = None,
                          coordinate_grid: np.ndarray = None,
                          spacing: Sequence[float] = None,
                          metric: np.ndarray = None,
                          derivative_field: np.ndarray = None,
                          second_derivative_field: np.ndarray = None,
                          fixed_axes: Dict[str, float] = None,
                          is_uniform: bool = False,
                          **kwargs) -> np.ndarray:
        r"""
        Compute the Laplacian of a scalar field in the coordinate system.

        The Laplacian is computed as:

        .. math::

            \Delta \phi = L^\mu \partial_\mu \phi + g^{\mu \nu} \partial_\mu \partial_\nu \phi

        where :math:`L^\mu` is the L-term field and :math:`g^{\mu \nu}` is the inverse metric tensor.

        Parameters
        ----------
        scalar_field : numpy.ndarray
            The scalar field :math:`\phi` whose Laplacian is to be computed. This should be a NumPy array
            of shape ``(...,)``, where the leading dimensions correspond to the discretized spatial grid.

        lterm_field : numpy.ndarray, optional
            Optional precomputed values for the L-term, which arises in the Laplace-Beltrami operator in curved spaces:

            .. math::

                L_\mu = \frac{1}{\sqrt{g}} \partial_\mu \left( \sqrt{g} \right),

            where :math:`g` is the determinant of the metric tensor. This should be provided as an array of shape
            ``(..., ndim)``. If not provided, it will be computed using the coordinate system's metric density expression.

            .. hint::

                Providing this field manually can improve performance when reused across multiple evaluations.

        coordinate_grid : numpy.ndarray, optional
            The coordinate grid over which the scalar field is defined. Should have shape ``(..., k)``, where ``k``
            is the number of varying axes (free axes) in the computation. Required if spacing, inverse metric, or
            L-term need to be computed and not provided directly.

            .. note::

                If the coordinate grid is a slice of a higher-dimensional system, fixed coordinate values must be
                provided via ``fixed_axes``.

        spacing : list of float, optional
            List of spacing values for each of the grid axes. Should be the same length as the number of free axes
            in ``coordinate_grid``. Required if numerical derivatives need to be computed and ``derivative_field`` or
            ``second_derivative_field`` are not provided.

            .. tip::

                Spacing can be inferred from the coordinate grid if not explicitly provided, though this may be less efficient.

        metric : numpy.ndarray, optional
            Optional precomputed values of the inverse metric tensor :math:`g^{\mu\mu}` on the grid. This should have
            shape ``(..., ndim)`` or ``(..., k)`` depending on whether a full or sliced grid is used.

            If not provided, it will be computed automatically from the coordinate system using the grid and
            any fixed axes.

        derivative_field : numpy.ndarray, optional
            Optional precomputed first-order partial derivatives of the scalar field. Should have shape
            ``(..., k)``, where ``k`` is the number of free axes. If not provided, these will be computed
            numerically using finite differences based on ``spacing``.

            .. warning::

                Must match the grid stencil exactly. Providing incorrect derivative shapes will raise an error.

        second_derivative_field : numpy.ndarray, optional
            Optional precomputed second-order partial derivatives of the scalar field. Should have shape
            ``(..., k, k)``. If not provided, these are computed from the first derivative field (or
            from ``scalar_field`` directly if no derivative fields are given).

        fixed_axes : dict, optional
            Dictionary mapping any fixed (non-varying) coordinate axes to their constant values. This is used to
            reconstruct the full coordinate system when working with sliced grids (e.g., a 2D slice of a 3D space).

            .. hint::

                Use this to treat slices of higher-dimensional coordinate systems correctly, especially in cylindrical
                or spherical coordinates.
        is_uniform: bool, optional
            Whether the coordinate system is uniform or not. This dictates how spacing is computed for
            derivatives. Defaults to ``False``.
        **kwargs :
            Additional keyword arguments passed to :py:func:`numpy.gradient` when computing numerical
            derivatives. This can include options like ``edge_order`` or ``axis`` if needed for customized
            differentiation.
        Returns
        -------
        numpy.ndarray
            Laplacian of the scalar field.
        """
        # Identify the grid dimension and construct the free / fixed axes so that we can
        # use them for shape compliance and other construction procedures.
        grid_ndim = scalar_field.ndim
        free_axes, _ = self.get_free_fixed_axes(grid_ndim, fixed_axes=fixed_axes)
        free_mask = self.build_axes_mask(free_axes)

        # Determine the grid spacing if it is necessary (derivative field not provided). Either the
        # spacing has been provided or we need to get it from the coordinate grid.
        if ((derivative_field is None) or (second_derivative_field is None)) and (spacing is None) and (
                coordinate_grid is None):
            raise ValueError()
        elif derivative_field is not None:
            # We have a derivative field which massively simplifies things because we no longer need
            # the spacing.
            pass
        elif spacing is not None:
            # We have the spacing. We just need to validate that it matches the number of
            # dimensions in the grid and then it'll be ready to use.
            if len(spacing) != grid_ndim:
                raise ValueError()
        elif coordinate_grid is not None:
            # The coordinate grid is not none, which means we need to get the spacing from
            # the coordinate grid.
            spacing = _get_grid_spacing(coordinate_grid, is_uniform=is_uniform)

        # Compute the inverse metric and check its shape. We are either given this as an argument or
        # the coordinate system will try to compute it from the coordinate grid. The output may be in
        # some set of odd array shapes that need to be corrected.
        metric = self.compute_metric_on_grid(coordinate_grid, inverse=True, fixed_axes=fixed_axes,
                                             value=metric)
        # Correct the possibly incorrect inverse_metric shapes.
        if metric.shape == scalar_field.shape + (self.ndim, self.ndim,):
            # The inverse metric was returned for the full coordinate system but we
            # only need the relevant grid axes (in BOTH indices).
            metric = metric[..., free_mask]

        # Compute the L-term field and check its shape. We are either given this as an argument or
        # the coordinate system will try to compute it from the coordinate grid. The output may be in
        # some set of odd array shapes that need to be corrected.
        lterm_field = self.compute_expression_on_grid('Lterm', coordinate_grid, fixed_axes=fixed_axes,
                                                      value=lterm_field)
        # Correct the shape issues if they arise.
        if lterm_field.shape == (self.ndim,) + scalar_field.shape:
            lterm_field = np.moveaxis(lterm_field, 0, -1)[..., free_mask]
        elif lterm_field.shape == (grid_ndim,) + scalar_field.shape:
            lterm_field = np.moveaxis(lterm_field, 0, -1)

        # Compute the Laplacian using the core differential geometry operator
        return glap_orth(scalar_field,
                       lterm_field,
                       metric,
                       spacing=spacing,
                       derivative_field=derivative_field,
                       second_derivative_field=second_derivative_field,
                       **kwargs)