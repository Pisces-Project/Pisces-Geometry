"""
Field support for dense mathematical operations.
"""
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

import numpy as np
from numpy.typing import ArrayLike

from pymetric.fields.buffers import buffer_from_array
from pymetric.fields.components import FieldComponent

# -------------------------------- #
# Type Checking Support / IDE      #
# -------------------------------- #
# This section provides mypy support and ensures
# that IDE inspection is correct for development.
if TYPE_CHECKING:
    from pymetric.fields.buffers import BufferRegistry
    from pymetric.fields.buffers.base import BufferBase

    # noinspection PyUnresolvedReferences
    from pymetric.fields.mixins._typing import (
        _SupportsDFieldDMathOps,
        _SupportsDTFieldDMathOps,
    )

# Type variables for generics
_SupDFDMOs = TypeVar("_SupDFDMOs", bound="_SupportsDFieldDMathOps")
_SupDTFDMOs = TypeVar("_SupDTFDMOs", bound="_SupportsDTFieldDMathOps")


# -------------------------------- #
# DMOs for DenseFields             #
# -------------------------------- #
# Support for DMOs in general dense fields.
class DenseFieldDMOMixin(Generic[_SupDFDMOs]):
    """
    Mixin class adding DMOs to :class:`fields.base.DenseField`.
    """

    # --- Utility Methods --- #
    def determine_op_dependence(self: _SupDFDMOs, opname: str, *args, **kwargs):
        """
        Infer the symbolic coordinate dependence resulting from a differential operation.

        This utility method performs symbolic dependence tracking by delegating
        to the internal dependence object (``self.dependence``) and extracting the
        resulting coordinate axis dependencies after applying the specified operation.

        It is commonly used by high-level field and component operations to
        automatically annotate new objects (e.g., gradients, divergences, Laplacians)
        with their correct symbolic dependence.

        Parameters
        ----------
        opname : str
            The name of the symbolic operation to perform (e.g., ``"gradient"``,
            ``"laplacian"``, ``"divergence"``, ``"raise_index"``, etc.).
            This must be a method of ``self.dependence``.
        *args : tuple
            Positional arguments passed directly to the method being invoked.
        **kwargs : dict
            Keyword arguments passed directly to the method being invoked.

        Returns
        -------
        List[str]
            A list of coordinate axis names (e.g., ``["r", "theta"]``) on which
            the resulting field depends after the symbolic operation.

        Raises
        ------
        ValueError
            If the given operation name is not a valid method of ``self.dependence``.

        Examples
        --------
        >>> field.determine_op_dependence("gradient", basis="covariant")
        ['r', 'theta']

        >>> field.determine_op_dependence("laplacian")
        ['r', 'theta']

        Notes
        -----
        This method is purely symbolic and does **not** perform any numerical evaluation.
        It is used for metadata propagation and symbolic analysis of coordinate dependence.
        """
        # Ensure that the method exists.
        if not hasattr(self.dependence, opname):
            raise ValueError(f"Dependence object cannot resolve {opname}.")

        # Perform the dependence operation.
        result = getattr(self.dependence, opname)(*args, **kwargs)

        # return the dependent axes.
        return self.__grid__.standardize_axes(result.dependent_axes)

    def _process_output_to_field_or_array(
        self: _SupDFDMOs,
        output: Any,
        *args,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        output_axes: Optional[Sequence[Any]] = None,
        **kwargs,
    ):
        """
        This helper method allows the output of a grid-level operation to
        be cast upward to a field on its own. It incorporates the various buffer
        arguments and keywords as well as the as_array argument.
        """
        # Check if the as_array option is True. If it is,
        # we just need to dump to raw result as an array.
        if as_array:
            return np.asarray(output)

        # Otherwise, we need to proceed to interpreting
        # the output as a buffer.
        buffer_args = buffer_args or ()
        buffer_kwargs = buffer_kwargs or {}
        buffer = buffer_from_array(
            output,
            *buffer_args,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            **buffer_kwargs,
        )

        # With the buffer constructed, we now
        # proceed to determine the output axes, construct
        # the component, and then pass through any additional
        # args or kwargs.
        if output_axes is None:
            output_axes = self.axes
        else:
            pass

        # standardize the output axes.
        output_axes = self.grid.standardize_axes(output_axes)

        # Finally, construct the field component and the
        # resulting field.
        component = FieldComponent(self.grid, buffer, output_axes)
        return self.__class__(self.grid, component, *args, **kwargs)

    # ======================================= #
    # General Dense Ops                       #
    # ======================================= #
    # recasting of functions from differential geometry's
    # general_ops module.
    # noinspection PyIncorrectDocstring
    def element_wise_partial_derivatives(
        self: _SupDFDMOs,
        out: Optional[ArrayLike] = None,
        output_axes: Optional[Sequence[str]] = None,
        *,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> "_SupDFDMOs":
        r"""
        Compute the element-wise partial derivatives of the field.

        This method computes the coordinate-wise partial derivatives of the field,
        treating each scalar component independently. It returns the covariant gradient
        of each element of the field with respect to the grid's coordinate directions.

        .. math::

            \phi \mapsto \left( \frac{\partial \phi}{\partial x^1}, \dots, \frac{\partial \phi}{\partial x^n} \right)

        For a vector or tensor field, the derivative is applied independently to each
        component in the field's trailing axes.

        .. hint::

            This wraps the grid-level method :meth:`~grids.base.GridBase.dense_element_wise_partial_derivatives`,
            which in turn calls the low-level routine
            :func:`~differential_geometry.general_ops.dense_element_wise_partial_derivatives`.


        Parameters
        ----------
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, the dependence is calculated internally.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            :attr:`axes` of the calling field.
        **kwargs
            Additional keyword arguments passed to the low-level operation
            :func:`~differential_geometry.general_ops.dense_element_wise_partial_derivatives`.

        Other Parameters
        ----------------
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when ``in_chunks=True``.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        as_array : bool, optional
            If ``True``, then no post-processing occurs and an array is returned.
        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            Defaults to buffer resolution to determine a fitting buffer for the resulting
            array result.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Optional registry to use when resolving the buffer class. This is eventually passed on
            to :func:`fields.buffers.base.buffer_from_array`.
        buffer_args, buffer_kwargs : optional
            arguments to pass to :func:`fields.buffers.base.buffer_from_array`.

        Returns
        -------
        ~fields.base.DenseField or ~numpy.ndarray
            The partial derivatives of the field, computed along each spatial coordinate direction.

            - If ``as_array=False`` (default), returns a new :class:`~fields.base.DenseField`
              whose element shape is extended by an additional dimension of size `ndim`,
              corresponding to the partial derivatives :math:`\partial_\mu`.

            - If ``as_array=True``, returns a raw NumPy array with shape
              ``(grid_shape over output_axes, ..., element_shape, ndim)``
              where `element_shape` refers to the shape of each scalar/vector/tensor component in the field.

        Notes
        -----
        **Input and Output Shapes**

        The input field must have shape ``(grid_shape, ..., element_shape)``

        where:

        - `grid_shape` is the shape of the domain over the field’s spatial axes.
        - `element_shape` represents any trailing axes used to encode vector or tensor components.

        The output will append a new axis of length `ndim`, corresponding to the partial derivative
        with respect to each spatial coordinate. The resulting shape is:
        ``(grid_shape over `output_axes`, ..., element_shape, ndim)``

        Examples:

        - A scalar field on a 2D grid → shape ``(Nx, Ny, 2)``
        - A 3-vector field on a 3D grid → shape ``(Nx, Ny, Nz, 3, 3)``

        **Chunking Behavior**

        If ``in_chunks=True``, the operation is evaluated over grid chunks using a streaming
        strategy. Each chunk is automatically extended with a 1-cell halo in every direction to
        preserve stencil accuracy for finite differences.

        Chunked evaluation is particularly useful when:

        - Working with large or out-of-core fields (e.g., HDF5-backed)
        - Minimizing peak memory use
        - Avoiding full-domain materialization for intermediate results

        **Buffer Class Customization**

        If ``as_array=False``, the output is wrapped in a new :class:`~fields.base.DenseField`
        using the specified `buffer_class`, `buffer_registry`, and buffer construction arguments.

        If ``as_array=True``, no wrapping occurs, and a raw NumPy array is returned. In this case,
        buffer configuration options are ignored.

        See Also
        --------
        ~grids.base.GridBase.dense_element_wise_partial_derivatives :
            Grid-level implementation this method wraps.
        ~differential_geometry.general_ops.dense_element_wise_partial_derivatives :
            Lower-level numerical routine.
        """
        # Determine the output axes based on the relevant
        # operation and the specified output axes.
        #
        # This operation fixes the basis to the covariant basis, so we
        # need to fix it here too.
        if output_axes is None:
            output_axes = self.determine_op_dependence(
                "element_wise_gradient", basis="covariant"
            )

        # Now compute the result by pass-through.
        result = self.__grid__.dense_element_wise_partial_derivatives(
            self.__component__.__buffer__,
            self.__component__.__axes__,
            out=out,
            output_axes=output_axes,
            **kwargs,
        )

        # Now cast the result.
        return self._process_output_to_field_or_array(
            result,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
        )

    # noinspection PyIncorrectDocstring
    def element_wise_laplacian(
        self: _SupDFDMOs,
        out: Optional[ArrayLike] = None,
        Lterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        second_derivative_field: Optional[ArrayLike] = None,
        *,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> "_SupDFDMOs":
        r"""
        Compute the element-wise Laplacian of the field.

        This method evaluates the Laplacian operator for each scalar component of the field
        using the generalized curved-space expression:

        .. math::

            \nabla^2 \phi = \frac{1}{\rho} \partial_\mu \left( \rho g^{\mu\nu} \partial_\nu \phi \right)

        which, for numerical stability, is implemented in expanded form as:

        .. math::

            \nabla^2 \phi = \left( \frac{1}{\rho} \partial_\mu \left[ g^{\mu\nu} \rho \right] \right) \partial_\nu \phi
                          + g^{\mu\nu} \partial_\mu \partial_\nu \phi
                          = L^\nu \partial_\nu \phi + g^{\mu\nu} \partial_\mu \partial_\nu \phi

        where :math:`L^\nu` is the "log-volume derivative" term for the coordinate system.

        This method wraps the corresponding grid-level Laplacian operator and performs chunked or
        full-domain computation as needed. It supports optional auxiliary fields to bypass
        internal derivative or metric computations for improved efficiency or accuracy.

        .. hint::

            Internally wraps :meth:`~grids.base.GridBase.dense_element_wise_laplacian`, which delegates to
            :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_diag` or
            :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_full`.

        Parameters
        ----------
        out : array-like, optional
            Optional buffer to store the result. Must match the grid shape over the
            evaluation domain (see `output_axes`) and the element shape of the field.
            If not provided, a new buffer is allocated.
        Lterm_field : array-like, optional
            Precomputed log-volume term:

            .. math::

                L^\nu = \frac{1}{\rho} \partial_\mu \left( g^{\mu\nu} \rho \right)

            Used to reduce numerical error and avoid redundant computation. Must have shape ``(*spatial_shape, ndim)``,
            where ``spatial_shape`` is this field's :attr:`spatial_shape`.

            If not supplied, it is computed automatically from the coordinate system.
        inverse_metric_field : array-like, optional
            The inverse metric tensor :math:`g^{\mu\nu}`, used for curved-coordinate contraction.

            - For diagonal metrics: shape ``(*spatial_shape, ndim)``
            - For full metrics: shape ``(*spatial_shape, ndim, ndim)``

            where ``spatial_shape`` is this field's :attr:`spatial_shape`.

            If not provided, it is derived from the coordinate system.
        derivative_field : array-like, optional
            Precomputed first derivatives :math:`\partial_\nu \phi` for each scalar component.
            This avoids re-computation of first-order gradients.

            Must have shape ``(*spatial_shape, ..., ndim)`` where ``spatial_shape``
            is this field's :attr:`spatial_shape`, and the trailing `ndim` axis
            corresponds to derivatives with respect to each coordinate direction.

        second_derivative_field : array-like, optional
            Precomputed second derivatives :math:`\partial_\mu \partial_\nu \phi`.

            Required shape depends on the metric type:

            - Diagonal metric: ``(*spatial_shape, ..., ndim)``
            - Full metric: ``(*spatial_shape, ..., ndim, ndim)``

            where ``spatial_shape``
            is this field's :attr:`spatial_shape`, and the trailing `ndim` axis
            corresponds to derivatives with respect to each coordinate direction.

        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, the dependence is calculated internally.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            :attr:`axes` of the calling field.


        Other Parameters
        ----------------
        in_chunks : bool, default False
            Whether to compute the result chunk-by-chunk using the grid's chunking logic.
            This can reduce memory usage for large domains or lazy-loaded buffers.

        edge_order : {1, 2}, default 2
            The order of the finite difference scheme to use for numerical derivatives.
            Passed to :func:`numpy.gradient`.

        pbar : bool, default True
            If `in_chunks=True`, whether to display a progress bar during processing.

        pbar_kwargs : dict, optional
            Additional keyword arguments forwarded to the progress bar utility (:mod:`tqdm`).

        as_array : bool, default False
            If True, return the raw NumPy array result.
            If False (default), wrap the result in a new :class:`~fields.base.DenseField` instance.

        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            Defaults to buffer resolution to determine a fitting buffer for the resulting
            array result.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry to use when resolving the buffer type.

        buffer_args, buffer_kwargs : optional
            Arguments forwarded to :func:`fields.buffers.base.buffer_from_array` when constructing the buffer.

        **kwargs
            Additional arguments forwarded to the underlying numerical Laplacian routines.

        Returns
        -------
        ~fields.base.DenseField or ~numpy.ndarray
            The computed Laplacian. Shape matches the field's spatial domain and element shape.

            - If ``as_array=True``, returns a NumPy array with shape ``(grid_shape, ..., element_shape)``.
            - If ``as_array=False``, returns a new :class:`~fields.base.DenseField` instance.

        Notes
        -----
        **Element-wise Behavior**

        The Laplacian is applied independently to each scalar component of the field.
        This includes each entry of a vector or tensor field. No automatic raising/lowering of indices is performed.

        **Shape Compatibility**

        The field must be defined over a structured grid. Its shape should follow:
        ``(grid_shape, ..., element_shape)``

        where `grid_shape` matches the dimensions of the field’s domain,
        and `element_shape` represents any trailing vector/tensor axes.

        The result will have the same shape as the input field, unless `output_axes` introduces broadcasting.

        **Auxiliary Field Requirements**

        When provided, auxiliary fields must be shape-compatible with the grid and derivatives:

        - `Lterm_field`: shape ``(..., ndim)``
        - `inverse_metric_field`: shape ``(..., ndim)`` or ``(..., ndim, ndim)``
        - `derivative_field`: shape ``(..., ndim)``
        - `second_derivative_field`: shape ``(..., ndim)`` or ``(..., ndim, ndim)`` depending on metric type

        **Chunked Execution**

        When `in_chunks=True`, each chunk of the domain is processed independently.
        Ghost cells (1-cell halo) are included automatically to preserve stencil accuracy
        during finite difference operations.

        See Also
        --------
        ~grids.base.GridBase.dense_element_wise_laplacian :
            Grid-level implementation wrapped by this method.
        ~differential_geometry.dense_ops.dense_scalar_laplacian_full :
            Full-metric low-level Laplacian kernel.
        ~differential_geometry.dense_ops.dense_scalar_laplacian_diag :
            Diagonal-metric version used in orthogonal systems.
        """
        # Determine the output axes based on the relevant
        # operation and the specified output axes.
        #
        # This operation fixes the basis to the covariant basis, so we
        # need to fix it here too.
        if output_axes is None:
            output_axes = self.determine_op_dependence("element_wise_laplacian")

        # Perform the operation at the grid level using the
        # single component buffer of the field.
        result = self.__grid__.dense_element_wise_laplacian(
            self.__component__.__buffer__,
            self.__component__.__axes__,
            out=out,
            Lterm_field=Lterm_field,
            inverse_metric_field=inverse_metric_field,
            derivative_field=derivative_field,
            second_derivative_field=second_derivative_field,
            output_axes=output_axes,
            **kwargs,
        )

        return self._process_output_to_field_or_array(
            result,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
        )


# -------------------------------- #
# DMOs for DenseTensorFields       #
# -------------------------------- #
# Support for DMOs in tensor dense fields.
class DenseTensorFieldDMOMixin(DenseFieldDMOMixin, Generic[_SupDTFDMOs]):
    """
    Mixin class adding DMOs to :class:`fields.tensors.DenseTensorField`.
    """

    # ======================================= #
    # General Dense Ops                       #
    # ======================================= #
    # recasting of functions from differential geometry's
    # general_ops module.
    # noinspection PyIncorrectDocstring
    def element_wise_partial_derivatives(
        self: _SupDTFDMOs,
        out: Optional[ArrayLike] = None,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> "_SupDTFDMOs":
        result = super().element_wise_partial_derivatives(
            out=out,
            output_axes=output_axes,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
        )

        # Fix the signature before returning the output.
        result.__signature__ = tuple(self.__signature__) + (-1,)

    # noinspection PyIncorrectDocstring
    def element_wise_laplacian(
        self: _SupDFDMOs,
        out: Optional[ArrayLike] = None,
        Lterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        second_derivative_field: Optional[ArrayLike] = None,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> "_SupDFDMOs":
        result = super().element_wise_laplacian(
            out=out,
            Lterm_field=Lterm_field,
            inverse_metric_field=inverse_metric_field,
            derivative_field=derivative_field,
            second_derivative_field=second_derivative_field,
            output_axes=output_axes,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            **kwargs,
        )

        # Fix the signature before returning the output.
        result.__signature__ = tuple(self.__signature__)

    # ======================================= #
    # Dense Ops                               #
    # ======================================= #
    # These methods allow wrapping for methods
    # in `dense_utils`.
    # noinspection PyIncorrectDocstring
    def contract_with_metric(
        self: _SupDTFDMOs,
        index: int,
        mode: str = "lower",
        out: Optional[ArrayLike] = None,
        metric_field: Optional[ArrayLike] = None,
        *,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        r"""
        Contract this tensor field with the metric or inverse metric tensor.

        This method raises or lowers a single index of the field by contracting it with
        either the metric :math:`g_{\mu\nu}` or its inverse :math:`g^{\mu\nu}`, depending on the
        selected `mode`. The contraction modifies one of the field's trailing tensor slots,
        changing its variance (covariant or contravariant).

        .. hint::

            This method wraps the grid-level :meth:`~grids.base.GridBase.dense_contract_with_metric`,
            which wraps the low-level :func:`~differential_geometry.dense_utils.dense_contract_with_metric`.

        Parameters
        ----------
        index : int
            The tensor slot (i.e., axis in the field's trailing element shape) to operate
            on.

            This refers to the position among the field’s value axes, not its spatial grid axes.
            For example, in a rank-2 tensor field with shape ``(Nx, Ny, 3, 3)``, ``index=0`` contracts
            the first value index (e.g., the row), while ``index=1`` contracts the second (e.g., the column).
        mode : {'raise', 'lower'}, optional
            Whether to raise or lower the index being operated on. This determines
            which metric is used in the contraction and (therefore) which metric field
            is expected if manually supplied by `metric_field`.

            By default, ``mode='lower'``.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        metric_field : array-like, optional
            Override metric to use for the contraction:

            - For ``mode='raise'`` this must be the **inverse** metric
              :math:`g^{\mu\nu}`.
            - For ``mode='lower'`` this must be the metric
              :math:`g_{\mu\nu}`.

            The field can be rank-2 (full) or rank-1 (diagonal); if not
            supplied it is generated chunk-by-chunk (or once globally)
            from the coordinate system.
            See `Notes` for more details on the shape requirements.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, the dependence is calculated internally.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            :attr:`axes` of the calling field.
        **kwargs
            Additional keyword arguments passed to the low-level operation
            :func:`~differential_geometry.dense_utils.dense_contract_with_metric`.

        Other Parameters
        ----------------
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        pbar : bool, optional
            Whether to display a progress bar when ``in_chunks=True``.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        as_array : bool, optional
            If ``True``, then no post-processing occurs and an array is returned.
        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            Defaults to buffer resolution to determine a fitting buffer for the resulting
            array result.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Optional registry to use when resolving the buffer class. This is eventually passed on
            to :func:`fields.buffers.base.buffer_from_array`.
        buffer_args, buffer_kwargs : optional
            arguments to pass to :func:`fields.buffers.base.buffer_from_array`.

        Returns
        -------
        ~fields.tensors.DenseTensorField or numpy.ndarray
            The result of contracting one index of the tensor field with the metric or inverse metric.

            - If ``as_array=False`` (default), returns a new :class:`~fields.base.DenseField` instance
              with the specified slot adjusted (lowered or raised).
            - If ``as_array=True``, returns a raw NumPy array with shape:

              .. code:: python

                  (grid_shape over output_axes, ..., element_shape)

            where ``element_shape`` reflects the adjusted tensor signature after contraction.

        Notes
        -----
        **Contraction Behavior**

        This operation modifies the variance of a single tensor index by contracting it with
        either the metric :math:`g_{\mu\nu}` or its inverse :math:`g^{\mu\nu}`. The result has
        the same rank but with the specified slot changed from covariant to contravariant or vice versa.

        **Index Semantics**

        The `index` parameter refers to the position within the field's element shape.
        For example, a rank-2 tensor field with shape ``(Nx, Ny, 3, 3)`` uses `index=0` for the first
        component axis and `index=1` for the second.

        **Metric Field Requirements**

        The shape of `metric_field` depends on the metric type:

        - **Diagonal metric (orthogonal coordinates):** ``(*spatial_shape, ndim)``
        - **Full metric (non-orthogonal):** ``(*spatial_shape, ndim, ndim)``

        If not supplied, the appropriate metric is computed automatically from the coordinate system
        (either once globally or per chunk).

        **Chunking Behavior**

        If ``in_chunks=True``, the operation is performed chunk-by-chunk across the grid. This is
        particularly helpful when working with large, memory-mapped, or HDF5-backed fields.

        Each chunk is evaluated with a 1-cell halo to preserve accuracy during intermediate gradient computations.

        **Backend Buffer Customization**

        If ``as_array=False``, the result is wrapped in a new :class:`DenseField`. You may control
        the buffer backend (NumPy, Unyt, HDF5, etc.) using `buffer_class`, `buffer_registry`,
        `buffer_args`, and `buffer_kwargs`.

        If ``as_array=True``, these options are ignored and a raw NumPy array is returned.
        """
        # Perform preliminary checks. We don't permit
        # this to proceed if the index and mode aren't
        # consistent with the current signature.
        index_variance = self.signature[index]

        if mode == "lower" and index_variance == -1:
            raise ValueError("Cannot lower a covariant index.")
        elif mode == "raise" and index_variance == 1:
            raise ValueError("Cannot raise a contravariant index.")
        elif mode not in ("raise", "lower"):
            raise ValueError("Invalid mode.")
        else:
            pass

        # Create the new signature.
        new_sig = (
            self.signature[:index] + (-index_variance,) + self.signature[index + 1 :]
        )

        # Determine the dependence.
        if output_axes is None:
            if mode == "raise":
                output_axes = self.determine_op_dependence("raise_index", index)
            else:
                output_axes = self.determine_op_dependence("lower_index", index)

        # Perform the operation at the grid level using the
        # single component buffer of the field.
        result = self.__grid__.dense_contract_with_metric(
            self.__component__.__buffer__,
            self.__component__.__axes__,
            index,
            mode=mode,
            out=out,
            metric_field=metric_field,
            output_axes=output_axes,
            **kwargs,
        )

        return self._process_output_to_field_or_array(
            result,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
            signature=new_sig,
        )

    # noinspection PyIncorrectDocstring
    def raise_index(
        self: _SupDTFDMOs,
        index,
        inverse_metric_field: Optional[ArrayLike] = None,
        out: Optional[ArrayLike] = None,
        *,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        r"""
        Raise a single covariant index of this tensor field.

        This method contracts one of the tensor's value axes with the inverse metric
        :math:`g^{\mu\nu}` to convert a covariant component into a contravariant one.
        The operation modifies only the specified slot and leaves all other indices unchanged.

        .. hint::

            This is a convenience wrapper around :meth:`~grids.base.GridBase.dense_raise_index`,
            which itself wraps :func:`~differential_geometry.dense_utils.dense_contract_with_metric`.

        Parameters
        ----------
        index : int
            The tensor slot (i.e., axis in the field's trailing element shape) to operate
            on.

            This refers to the position among the field’s value axes, not its spatial grid axes.
            For example, in a rank-2 tensor field with shape ``(Nx, Ny, 3, 3)``, ``index=0`` contracts
            the first value index (e.g., the row), while ``index=1`` contracts the second (e.g., the column).
        inverse_metric_field : array-like, optional
            The inverse metric tensor :math:`g^{\mu\nu}`, used for curved-coordinate contraction.

            - For diagonal metrics: shape ``(*spatial_shape, ndim)``
            - For full metrics: shape ``(*spatial_shape, ndim, ndim)``

            where ``spatial_shape`` is this field's :attr:`spatial_shape`.

            If not provided, it is derived from the coordinate system.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, the dependence is calculated internally.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            :attr:`axes` of the calling field.
        **kwargs
            Passed straight to the low-level metric-contraction kernels
            (e.g. `where=` masks).

        Other Parameters
        ----------------
        as_array : bool, default False
            If True, return the raw array representing the result.
            If False, return a new symbolic field with the raised index.
        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            Defaults to buffer resolution to determine a fitting buffer for the resulting
            array result.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry to use when resolving the buffer class. Required for advanced dispatch.
        buffer_args : Sequence, optional
            Positional arguments forwarded to the buffer constructor.
        buffer_kwargs : dict, optional
            Keyword arguments forwarded to the buffer constructor.
        in_chunks : bool, default False
            If True, compute the result in streaming fashion over grid chunks to reduce
            memory usage.
        pbar : bool, default True
            Show a progress bar during chunked computation.
        pbar_kwargs : dict, optional
            Extra keyword arguments passed to the progress bar (e.g., `desc` or `position`).

        Returns
        -------
        ~fields.tensors.DenseTensorField or numpy.ndarray
            The resulting tensor field with the specified index raised.

            - If ``as_array=False`` (default), returns a new field of the same type as this one
              (e.g., :class:`~fields.base.DenseField`) with the indicated slot raised to contravariant form.
            - If ``as_array=True``, returns a raw NumPy array with shape:

              .. code:: python

                  (grid_shape over output_axes, ..., element_shape)

              where the `element_shape` reflects the updated tensor signature.

        Notes
        -----
        **Index Contraction**

        This operation raises a single index of the field by contracting it with the inverse metric
        :math:`g^{\mu\nu}`. The index position is given by `index`, which refers to one of the
        trailing component axes in the field’s element shape (not spatial grid axes).

        The rank of the tensor remains the same; only the variance (covariant → contravariant)
        of the specified slot is updated.

        **Metric Handling**

        If `inverse_metric_field` is not supplied, it is computed from the grid's coordinate system:

        - For orthogonal systems (diagonal metric): shape ``(*spatial_shape, ndim)``
        - For general systems (full metric): shape ``(*spatial_shape, ndim, ndim)``

        You may precompute this expression using
        :meth:`~coordinates.base.CoordinateSystem.compute_expression_from_coordinates`
        to avoid repeated work.

        **Chunked Evaluation**

        If `in_chunks=True`, the contraction is performed chunk-by-chunk across the domain.
        This can reduce memory pressure for large or lazily loaded fields.

        **Buffer Customization**

        If ``as_array=False``, the resulting array is wrapped in a new field using the specified
        `buffer_class`, `buffer_registry`, and optional constructor arguments.

        If ``as_array=True``, those options are ignored and the raw array is returned directly.
        """
        # Ensure that the specified index is allowed to
        # be raised.
        if self.signature[index] != -1:
            raise ValueError(
                f"Cannot raise index {index} of {self}. It is already contravariant."
            )

        # Determine the relevant output axes.
        if output_axes is None:
            output_axes = self.determine_op_dependence("raise_index")

        # Perform the operation at the grid level using the
        # single component buffer of the field.
        result_field = self.__grid__.dense_raise_index(
            self.__component__.__buffer__,
            self.axes,
            index,
            out=out,
            inverse_metric_field=inverse_metric_field,
            output_axes=output_axes,
            as_array=as_array,
            **kwargs,
        )

        # Coerce the output to the desired typing.
        # This is where we use the various buffer info.
        new_signature = [s if i != index else -s for i, s in enumerate(self.signature)]
        return self._cast_result_to_field_or_array(
            result_field,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
            signature=new_signature,
        )

    # noinspection PyIncorrectDocstring
    def lower_index(
        self: _SupDTFDMOs,
        index,
        metric_field: Optional[ArrayLike] = None,
        out: Optional[ArrayLike] = None,
        *,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        r"""
        Lower a single contravariant index of this tensor field.

        This method contracts one of the tensor's value axes with the metric
        :math:`g_{\mu\nu}` to convert a contravariant component into a covariant one.
        The operation modifies only the specified slot and leaves all other indices unchanged.

        .. hint::

            This is a convenience wrapper around :meth:`~grids.base.GridBase.dense_lower_index`,
            which itself wraps :func:`~differential_geometry.dense_utils.dense_contract_with_metric`.

        Parameters
        ----------
        index : int
            The tensor slot (i.e., axis in the field's trailing element shape) to operate
            on.

            This refers to the position among the field’s value axes, not its spatial grid axes.
            For example, in a rank-2 tensor field with shape ``(Nx, Ny, 3, 3)``, ``index=0`` contracts
            the first value index (e.g., the row), while ``index=1`` contracts the second (e.g., the column).

        metric_field : array-like, optional
            The metric tensor :math:`g_{\mu\nu}`, used to perform the contraction.

            - For diagonal metrics: shape ``(*spatial_shape, ndim)``
            - For full metrics: shape ``(*spatial_shape, ndim, ndim)``

            where ``spatial_shape`` is this field's :attr:`spatial_shape`.

            If not provided, it is derived from the coordinate system.

        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.

        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, the dependence is calculated internally.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            :attr:`axes` of the calling field.

        **kwargs
            Passed straight to the low-level metric-contraction kernels
            (e.g. `where=` masks).

        Other Parameters
        ----------------
        as_array : bool, default False
            If True, return the raw array representing the result.
            If False, return a new field with the lowered index.

        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            If not provided, the buffer type is inferred automatically from the result.

        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry to use when resolving the buffer class. Required for advanced dispatch.

        buffer_args : Sequence, optional
            Positional arguments forwarded to the buffer constructor.

        buffer_kwargs : dict, optional
            Keyword arguments forwarded to the buffer constructor.

        in_chunks : bool, default False
            If True, compute the result in streaming fashion over grid chunks to reduce
            memory usage.

        pbar : bool, default True
            Show a progress bar during chunked computation.

        pbar_kwargs : dict, optional
            Extra keyword arguments passed to the progress bar (e.g., `desc` or `position`).

        Returns
        -------
        ~fields.tensors.DenseTensorField or numpy.ndarray
            The resulting tensor field with the specified index lowered.

            - If ``as_array=False`` (default), returns a new field of the same type as this one
              (e.g., :class:`~fields.base.DenseField`) with the indicated slot converted to covariant form.
            - If ``as_array=True``, returns a raw NumPy array with shape:

              .. code:: python

                  (grid_shape over output_axes, ..., element_shape)

              where the `element_shape` reflects the updated tensor signature.

        Notes
        -----
        **Index Contraction**

        This operation lowers a single index of the field by contracting it with the metric
        :math:`g_{\mu\nu}`. The index position is given by `index`, which refers to one of the
        trailing component axes in the field’s element shape (not spatial grid axes).

        The rank of the tensor remains the same; only the variance (contravariant → covariant)
        of the specified slot is updated.

        **Metric Handling**

        If `metric_field` is not supplied, it is computed from the grid's coordinate system:

        - For orthogonal systems (diagonal metric): shape ``(*spatial_shape, ndim)``
        - For general systems (full metric): shape ``(*spatial_shape, ndim, ndim)``

        You may precompute this expression using
        :meth:`~coordinates.base.CoordinateSystem.compute_expression_from_coordinates`
        to avoid repeated work.

        **Chunked Evaluation**

        If `in_chunks=True`, the contraction is performed chunk-by-chunk across the domain.
        This can reduce memory pressure for large or lazily loaded fields.

        **Buffer Customization**

        If ``as_array=False``, the resulting array is wrapped in a new field using the specified
        `buffer_class`, `buffer_registry`, and optional constructor arguments.

        If ``as_array=True``, those options are ignored and the raw array is returned directly.
        """
        if self.signature[index] != 1:
            raise ValueError(
                f"Cannot lower index {index} of {self}. It is already covariant."
            )

        # Determine the relevant output axes.
        if output_axes is None:
            output_axes = self.determine_op_dependence("lower_index")

        # Perform the operation at the grid level using the
        # single component buffer of the field.
        result_field = self.__grid__.dense_lower_index(
            self.__component__.__buffer__,
            self.axes,
            index,
            out=out,
            metric_field=metric_field,
            output_axes=output_axes,
            as_array=as_array,
            **kwargs,
        )

        # Coerce the output to the desired typing.
        # This is where we use the various buffer info.
        new_signature = [s if i != index else -s for i, s in enumerate(self.signature)]
        return self._cast_result_to_field_or_array(
            result_field,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
            signature=new_signature,
        )

    # noinspection PyIncorrectDocstring
    def adjust_tensor_signature(
        self: _SupDTFDMOs,
        indices,
        metric_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        out: Optional[ArrayLike] = None,
        *,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        r"""
        Adjust the variance (covariant or contravariant) of multiple tensor indices.

        This method applies metric contractions to raise or lower selected slots of the
        field's element shape, transforming them between covariant and contravariant form.
        It modifies only the specified indices and leaves all other slots unchanged.

        Under the hood, this wraps :meth:`~grids.base.GridBase.dense_adjust_tensor_signature`,
        which applies one or more contractions using either the metric :math:`g_{\mu\nu}` or its inverse
        :math:`g^{\mu\nu}`, depending on the target signature.

        Parameters
        ----------
        indices : list of int
            The tensor slots (i.e., axes in the field's trailing element shape) to operate
            on.

            This refers to the position among the field’s value axes, not its spatial grid axes.
            For example, in a rank-2 tensor field with shape ``(Nx, Ny, 3, 3)``, ``indices=[0]`` contracts
            the first value index (e.g., the row), while ``indices=[1]`` contracts the second (e.g., the column).
        metric_field, inverse_metric_field : array-like, optional
            Optional precomputed metric and inverse metric tensors used during contraction.

            - `metric_field` is used to **lower** covariant indices and must correspond to :math:`g_{\mu\nu}`
            - `inverse_metric_field` is used to **raise** contravariant indices and must correspond to :math:`g^{\mu\nu}`

            You may provide one or both, depending on which indices are being modified. Any missing tensor
            will be computed automatically from the grid’s coordinate system.

            Expected shapes:

            - Diagonal (orthogonal) systems: ``(*spatial_shape, ndim)``
            - Full (non-orthogonal) systems: ``(*spatial_shape, ndim, ndim)``

            Here, ``spatial_shape`` refers to the shape of the field over its domain grid axes.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, the dependence is calculated internally.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            :attr:`axes` of the calling field.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        **kwargs
            Passed straight to the low-level metric-contraction kernels
            (e.g. `where=` masks).

        Other Parameters
        ----------------
        in_chunks : bool, default False
            If True, compute the result in streaming fashion over grid chunks to reduce
            memory usage.
        pbar : bool, default True
            Show a progress bar during chunked computation.
        pbar_kwargs : dict, optional
            Extra keyword arguments passed to the progress bar (e.g., `desc` or `position`).
        as_array : bool, default False
            If True, return the raw array representing the result.
            If False, return a new symbolic field with the raised index.
        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            Defaults to buffer resolution to determine a fitting buffer for the resulting
            array result.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry to use when resolving the buffer class. Required for advanced dispatch.
        buffer_args : Sequence, optional
            Positional arguments forwarded to the buffer constructor.
        buffer_kwargs : dict, optional
            Keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        ~fields.tensors.DenseTensorField or numpy.ndarray
            The tensor field with updated variance on the specified slots. It will match the
            shape of the calling field.

        Notes
        -----
        **Variance Adjustment Logic**

        Each entry in `indices` is matched against the corresponding component in the
        current tensor signature (stored in `self.signature`). Based on the desired
        transformation, the following logic is applied:

        - If the current slot is covariant (−1) and the target is +1 → **raise**
        - If the current slot is contravariant (+1) and the target is −1 → **lower**
        - If the slot already matches the target signature, no action is taken.

        **Metric Field Requirements**

        - Orthogonal metrics (diagonal): shape ``(*spatial_shape, ndim)``
        - Non-orthogonal metrics (full): shape ``(*spatial_shape, ndim, ndim)``

        **Chunking Behavior**

        If ``in_chunks=True``, each grid chunk is processed independently with 1-cell halo padding
        for stencil accuracy. This mode is especially useful with HDF5-backed or large fields.

        **Buffer Handling**

        If ``as_array=False``, the result is wrapped using the buffer system. The buffer type may be
        explicitly controlled via `buffer_class`, `buffer_registry`, and related args.

        If ``as_array=True``, the raw NumPy array is returned, and no wrapping occurs.
        """
        # Determine the relevant output axes. To do so, we need to
        # determine the current and output variance.
        current_variance, new_variance = self.signature[:], self.signature[:]
        for index in indices:
            new_variance[index] *= -1

        if output_axes is None:
            output_axes = self.determine_op_dependence(
                "lower_index", current_variance, new_variance
            )

        # Perform the operation at the grid level using the
        # single component buffer of the field.
        result_field = self.__grid__.dense_adjust_tensor_signature(
            self.__component__.__buffer__,
            self.axes,
            indices,
            self.signature,
            inverse_metric_field=inverse_metric_field,
            metric_field=metric_field,
            out=out,
            output_axes=output_axes,
            as_array=as_array,
            **kwargs,
        )

        # Coerce the output to the desired typing.
        # This is where we use the various buffer info.
        return self._cast_result_to_field_or_array(
            result_field,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
            signature=new_variance,
        )

    # --- Differential Operators --- #
    # These methods support differential
    # operations.

    # noinspection PyIncorrectDocstring
    def gradient(
        self: _SupDFDMOs,
        basis: Optional[Literal["contravariant", "covariant"]] = "covariant",
        inverse_metric_field: Optional[ArrayLike] = None,
        out: Optional[ArrayLike] = None,
        *,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        r"""
        Compute the covariant or contravariant gradient of this field.

        This method computes the coordinate-wise gradient of the field, applied
        independently to each scalar component. The result is a new tensor field
        whose rank is one greater than that of the input, with the final axis
        encoding the derivative direction (covariant or contravariant).

        .. math::

            \phi \mapsto \nabla_\mu \phi \quad \text{or} \quad \nabla^\mu \phi

        The gradient basis is selected using the `basis` argument. In covariant form,
        the gradient consists of raw partial derivatives. In contravariant form, the
        derivatives are raised using the inverse metric tensor :math:`g^{\mu\nu}`.

        .. hint::

            This is a wrapper around :meth:`~grids.base.GridBase.dense_gradient`, which in turn
            dispatches to either :func:`~differential_geometry.dense_ops.dense_gradient_covariant`
            or :func:`~differential_geometry.dense_ops.dense_gradient_contravariant_full`.

        Parameters
        ----------
        basis : {'covariant', 'contravariant'}, optional
            The basis in which to compute the gradient. Defaults to ``'covariant'``.

            - If ``'covariant'``: returns raw partial derivatives :math:`\partial_\mu \phi`
            - If ``'contravariant'``: raises the result using :math:`g^{\mu\nu}` to produce :math:`\nabla^\mu \phi`
        inverse_metric_field : array-like, optional
            The inverse metric tensor :math:`g^{\mu\nu}` required for computing the contravariant gradient.

            - For diagonal metrics: shape ``(*spatial_shape, ndim)``
            - For full metrics: shape ``(*spatial_shape, ndim, ndim)``

            If not provided, it is computed automatically from the coordinate system.
            Ignored when ``basis='covariant'``.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, the dependence is calculated internally.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            :attr:`axes` of the calling field.

        Other Parameters
        ----------------
        in_chunks : bool, default False
            If True, compute the result in streaming fashion over grid chunks to reduce
            memory usage.
        pbar : bool, default True
            Show a progress bar during chunked computation.
        pbar_kwargs : dict, optional
            Extra keyword arguments passed to the progress bar (e.g., `desc` or `position`).
        as_array : bool, default False
            If True, return the raw array representing the result.
            If False, return a new symbolic field with the raised index.
        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            Defaults to buffer resolution to determine a fitting buffer for the resulting
            array result.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry to use when resolving the buffer class. Required for advanced dispatch.
        buffer_args : Sequence, optional
            Positional arguments forwarded to the buffer constructor.
        buffer_kwargs : dict, optional
            Keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        ~fields.base.DenseTensorField or ~numpy.ndarray
            The gradient of the field. The output shape matches the input shape with
            one additional trailing axis of size `ndim`.

            - If ``as_array=False`` (default), returns a new :class:`DenseTensorField` instance.
            - If ``as_array=True``, returns a NumPy array of shape:

              .. code:: python

                  (grid_shape over output_axes, ..., element_shape, ndim)

        Notes
        -----
        **Output Shape and Signature**

        The gradient adds one additional index to the tensor field:

        - For a scalar field, the output becomes a vector field.
        - For a vector field, the output becomes a rank-2 tensor.

        The updated element shape is the original field’s `element_shape` extended by `(ndim,)`.

        The gradient basis determines the signature of this final axis:

        - ``'covariant'`` → covariant index (e.g., ∂μ)
        - ``'contravariant'`` → contravariant index (e.g., ∇^μ)

        **Metric Requirements**

        The inverse metric field is only required when ``basis='contravariant'``.
        If not provided, it is evaluated from the coordinate system automatically.

        **Chunked Evaluation**

        When ``in_chunks=True``, the operation is evaluated block-by-block.
        Ghost cells (1-cell halo) are added to preserve the accuracy of finite-difference stencils.

        This is especially useful for HDF5-backed buffers or streaming workloads.

        See Also
        --------
        ~grids.base.GridBase.dense_gradient :
            Grid-level implementation used here.
        ~differential_geometry.dense_ops.dense_gradient_covariant :
            Covariant gradient kernel (raw partial derivatives).
        ~differential_geometry.dense_ops.dense_gradient_contravariant_full :
            Contravariant gradient kernel (full metric).
        ~differential_geometry.dense_ops.dense_gradient_contravariant_diag :
            Contravariant gradient kernel (diagonal metric).
        """
        # Adjust the signature based on the basis
        # we got given.
        new_signature = tuple(self.signature)
        new_signature += (1,) if basis == "covariant" else (-1,)

        # Determine the relevant output axes.
        if output_axes is None:
            output_axes = self.determine_op_dependence("gradient", basis=basis)

        # Perform the operation at the grid level using the
        # single component buffer of the field.
        result_field = self.__grid__.dense_gradient(
            self.__component__.__buffer__,
            self.axes,
            out=out,
            inverse_metric_field=inverse_metric_field,
            basis=basis,
            output_axes=output_axes,
            as_array=as_array,
            **kwargs,
        )

        # Coerce the output to the desired typing.
        # This is where we use the various buffer info.
        return self._cast_result_to_field_or_array(
            result_field,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
            signature=new_signature,
        )

    # noinspection PyIncorrectDocstring
    def vector_divergence(
        self: _SupDFDMOs,
        Dterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        out: Optional[ArrayLike] = None,
        *,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        r"""
        Compute the divergence of a vector field.

        This method applies the divergence operator to a rank-1 tensor field (i.e., a vector field)
        by contracting the gradient with the metric or its inverse, depending on the field's basis.
        For a vector field :math:`V^\mu` or covector field :math:`V_\mu`, the divergence is:

        .. math::

            \nabla \cdot V = \frac{1}{\rho} \partial_\mu (\rho V^\mu)
                           = \partial_\mu V^\mu + \frac{1}{\rho} V^\mu \partial_\mu \rho

        for contravariant vectors, or

        .. math::

            \nabla \cdot V = \frac{1}{\rho} \partial_\mu (\rho g^{\mu\nu} V_\nu)

        for covariant vectors.

        The appropriate form is automatically selected based on the tensor signature.

        .. hint::

            This is a wrapper around :meth:`~grids.base.GridBase.dense_vector_divergence`, which delegates to
            either :func:`~differential_geometry.dense_ops.dense_vector_divergence_contravariant` or
            :func:`~differential_geometry.dense_ops.dense_vector_divergence_covariant`.

        Parameters
        ----------
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        Dterm_field : array-like, optional
            The D-term field :math:`D_\mu = \frac{\partial_\mu \rho}{\rho}`, where :math:`\rho = \sqrt{|g|}`
            is the metric volume element of the coordinate system.

            This term arises in expressions for the divergence and Laplacian in curved coordinates, and accounts
            for coordinate-dependent geometric volume effects.

            - If not provided, the D-term is automatically computed from the coordinate system.
            - If supplied, this can reduce redundant computation or improve accuracy if an analytical
              expression is known.

        inverse_metric_field : array-like, optional
            The inverse metric tensor :math:`g^{\mu\nu}`, used to raise covariant vector components
            when the field's basis (as inferred from its tensor signature) is covariant.

            - Required **only if** the field is covariant (i.e., its signature is ``(-1,)``).
            - Ignored when the field is contravariant (i.e., its signature is ``(+1,)``).

            If not provided, the inverse metric is automatically computed from the coordinate system.

        derivative_field : array-like, optional
            Precomputed partial derivatives of the input vector field: :math:`\partial_\mu V^\nu`.

            This array is only used if the field is **contravariant**, and will be ignored if the field is covariant.

            Providing this can reduce runtime and improve numerical accuracy when the derivatives are analytically known
            or already computed elsewhere. Otherwise, they are approximated using finite differences.

        Other Parameters
        ----------------
        in_chunks : bool, default False
            If True, compute the result in streaming fashion over grid chunks to reduce
            memory usage.
        pbar : bool, default True
            Show a progress bar during chunked computation.
        pbar_kwargs : dict, optional
            Extra keyword arguments passed to the progress bar (e.g., `desc` or `position`).
        as_array : bool, default False
            If True, return the raw array representing the result.
            If False, return a new symbolic field with the raised index.
        buffer_class : type, optional
            Class to use when wrapping the result buffer, e.g., :class:`fields.buffers.core.ArrayBuffer`
            or :class:`fields.buffers.core.UnytArrayBuffer`.

            Defaults to buffer resolution to determine a fitting buffer for the resulting
            array result.
        buffer_registry : ~fields.buffers.registry.BufferRegistry, optional
            Registry to use when resolving the buffer class. Required for advanced dispatch.
        buffer_args : Sequence, optional
            Positional arguments forwarded to the buffer constructor.
        buffer_kwargs : dict, optional
            Keyword arguments forwarded to the buffer constructor.

        Returns
        -------
        ~fields.tensors.DenseTenseField or ~numpy.ndarray
            The scalar divergence of the field.

            - If ``as_array=False`` (default), returns a new :class:`DenseTensorField` instance representing the divergence
              as a scalar field defined over the appropriate grid axes.
            - If ``as_array=True``, returns a raw NumPy array with shape:

              .. code:: python

                  (grid_shape over output_axes)

            This result contains one scalar value per spatial grid point and is rank-0 in element shape.

        Notes
        -----
        **Field Signature Requirements**

        This operation is valid **only** for rank-1 tensor fields with one of the following signatures:

        - ``(+1,)``: a contravariant vector field
        - ``(-1,)``: a covariant vector field

        An error is raised for all other signatures.

        **Output Shape**

        The result is a scalar field with the same spatial shape as the input, over the `output_axes`.

        - For example, if the input field spans ``(Nx, Ny, Nz)`` over `['x', 'y', 'z']`, the output will have shape
          ``(Nx, Ny, Nz)``.
        - No trailing element axes are retained in the result.

        **Auxiliary Inputs**

        - `Dterm_field` must have shape ``(*spatial_shape, ndim)``, where the last axis indexes coordinate directions.
        - `inverse_metric_field` must have shape:
            - ``(*spatial_shape, ndim)`` for diagonal metrics
            - ``(*spatial_shape, ndim, ndim)`` for full metrics
        - `derivative_field` must have shape ``(*spatial_shape, ndim)`` if supplied, and is ignored for covariant fields.

        **Chunked Evaluation**

        When ``in_chunks=True``, the divergence is computed block-by-block. A 1-cell halo is added around each chunk
        to preserve stencil accuracy in derivative operations. This is especially useful when:

        - Working with memory-mapped or HDF5-backed buffers
        - Reducing memory overhead for large grids

        **Coordinate System Handling**

        All required geometric quantities—including the metric, inverse metric, volume factor, and D-terms—are computed
        from the grid’s associated coordinate system unless manually overridden.
        """
        # Ensure that this is really a vector and that
        # we can properly take the divergence.
        if tuple(self.signature) not in [(1,), (-1,)]:
            raise ValueError("Field is not a vector field.")

        # Adjust the signature based on the basis
        # we got given.
        new_signature = ()

        # Determine the relevant output axes.
        basis = "covariant" if tuple(self.signature) == (-1,) else "contravariant"
        if output_axes is None:
            output_axes = self.determine_op_dependence("divergence", basis=basis)

        # Perform the operation at the grid level using the
        # single component buffer of the field.
        result_field = self.__grid__.dense_vector_divergence(
            self.__component__.__buffer__,
            self.axes,
            derivative_field=derivative_field,
            out=out,
            Dterm_field=Dterm_field,
            inverse_metric_field=inverse_metric_field,
            basis=basis,
            output_axes=output_axes,
            as_array=as_array,
            **kwargs,
        )

        # Coerce the output to the desired typing.
        # This is where we use the various buffer info.
        return self._cast_result_to_field_or_array(
            result_field,
            as_array=as_array,
            buffer_class=buffer_class,
            buffer_registry=buffer_registry,
            buffer_args=buffer_args,
            buffer_kwargs=buffer_kwargs,
            output_axes=output_axes,
            signature=new_signature,
        )

    # noinspection PyIncorrectDocstring
    def scalar_laplacian(
        self: _SupDFDMOs,
        *args,
        **kwargs,
    ):
        r"""
        Alias for :meth:`element_wise_laplacian`.
        """
        return self.element_wise_laplacian(*args, **kwargs)
