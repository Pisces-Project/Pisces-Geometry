"""
Grid operator mixins for differential geometry and field computations.

This module provides mixin classes that add high-level mathematical operations
to grid classes. These operations include gradient, divergence, Laplacian, and
tensor contractions, and are implemented as thin wrappers over lower-level
kernels from the :mod:`differential_geometry` package.

The mixins handle tasks such as:

- Automatically computing coordinate-based expressions from the grid’s coordinate system.
- Performing operations in memory-efficient chunks using the grid’s chunking interface.
- Broadcasting input fields and auxiliary buffers to the appropriate grid shape.
- Managing metric and volume element dependencies implicitly.

These mixins are intended to be inherited by structured grid classes (e.g., :class:`~grids.core.UniformGrid`)
to provide a clean and consistent API for field-based differential operators.
"""
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
from numpy.typing import ArrayLike

from pymetric.differential_geometry.dense_ops import (
    dense_gradient_contravariant_diag,
    dense_gradient_contravariant_full,
    dense_gradient_covariant,
    dense_scalar_laplacian_diag,
    dense_scalar_laplacian_full,
    dense_vector_divergence_contravariant,
    dense_vector_divergence_covariant_diag,
    dense_vector_divergence_covariant_full,
)

# noinspection PyProtectedMember
from pymetric.differential_geometry.dense_utils import (
    _dense_adjust_tensor_signature,
    _dense_adjust_tensor_signature_diagonal_metric,
    _dense_contract_index_with_diagonal_metric,
    _dense_contract_index_with_metric,
)
from pymetric.differential_geometry.general_ops import (
    dense_element_wise_partial_derivatives,
)

# =================================== #
# Type Annotations                    #
# =================================== #
# These type annotations are used for compatibility
# with static type checkers like mypy.
if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pymetric.grids.mixins._typing import _SupportsDenseGridMathOps

_SupDGMO = TypeVar("_SupDGMO", bound="_SupportsDenseGridMathOps")


# ================================ #
# Dense MathOperations Mixin       #
# ================================ #
# This class provides support for differential geometry on tensor
# fields with dense representations.
class DenseMathOpsMixin(Generic[_SupDGMO]):
    r"""
    Mixin class for dense differential geometry operations on structured grids.

    This class provides high-level numerical implementations of common differential
    geometry operators—such as gradients, divergence, and Laplacians—on fields defined over
    structured grids with curvilinear coordinate systems.

    It serves as a convenience interface to the low-level functions in
    :mod:`differential_geometry`, managing:

    - Axis labeling and broadcasting between tensor fields and grid domains.
    - Automatic handling of coordinate expressions like metric tensors and volume terms.
    - Support for lazy-loading or memory-intensive buffers through optional chunked execution.
    - Compatibility with ghost zones and progress bars.
    """
    # TODO: This can probably be unified... Priority: low

    # ======================================= #
    # Utility Functions                       #
    # ======================================= #
    # These are utility methods that can be called during the
    # execution sequence for operational clarity.
    def _prepare_output_buffer(
        self: _SupDGMO,
        axes: Sequence[str],
        *,
        out: Optional[np.ndarray] = None,
        output_element_shape: Optional[Tuple[int, ...]] = (),
        **kwargs,
    ) -> ArrayLike:
        """
        Allocate or validate an output buffer for grid-based tensor computations.

        This method either validates a provided `out` buffer or allocates a new one using
        the grid’s `empty()` method. It is intended for internal use in operations such as
        gradient, divergence, and Laplacian calculations, where output shape and alignment
        with logical axes are critical.

        Parameters
        ----------
        axes : Sequence[str]
            Logical axes over which the field is defined (e.g. ``["x", "y"]``). These define
            the grid space that the output buffer must align with.
        out : np.ndarray, optional
            Optional pre-allocated output array. If provided, its shape is checked for
            compatibility with the grid shape along `axes` and with the specified
            `output_element_shape`.
        output_element_shape : Tuple[int], optional
            Shape of each individual field element. For scalar fields, this is `()`. For vector
            or tensor fields, this may be `(ndim,)`, `(ndim, ndim)`, etc. If `out` is not provided,
            this will be used to construct the new output buffer.
        **kwargs
            Additional keyword arguments passed to `self.empty()` when allocating a new buffer.
            Common options include:
            - `dtype`: Data type of the new array.
            - `order`: Memory layout order ('C' or 'F').
            - `include_ghosts`: Whether to include ghost zones.

        Returns
        -------
        ArrayLike
            A NumPy-compatible array that can be written to by the calling operation.

        Raises
        ------
        ValueError
            If `out` is provided but does not match the required shape.

        Notes
        -----
        - This method is typically used at the beginning of a differential operator to ensure
          the output buffer is valid for writing results.
        - It supports both scalar and tensor-valued field outputs and is compatible with
          broadcasting and ghost zone inclusion.
        """
        if out is not None:
            self.check_field_shape(
                out.shape, axes=axes, field_shape=output_element_shape
            )
        else:
            out = self.empty(axes=axes, element_shape=output_element_shape, **kwargs)
        return out

    def _set_input_output_axes(self: _SupDGMO, field_axes, output_axes=None):
        # Standardize the field axes and the output axes.
        # convert them both to indices.
        field_axes = self.standardize_axes(field_axes)
        output_axes = (
            self.standardize_axes(output_axes)
            if output_axes is not None
            else field_axes
        )
        fidx, oidx = self.__cs__.convert_axes_to_indices(
            field_axes
        ), self.__cs__.convert_axes_to_indices(output_axes)

        # Ensure axes are a subset of the output.
        if not self.__cs__.is_axes_subset(field_axes, output_axes):
            raise ValueError(f"Axes {field_axes} are not a subset of {output_axes}.")

        return field_axes, output_axes, fidx, oidx

    def _compute_fixed_axes_and_values(
        self: _SupDGMO, free_axes: Sequence[str]
    ) -> Tuple[Sequence[str], dict]:
        """
        Compute the fixed axes and corresponding fill values for a set of free axes.

        This utility method identifies all axes not included in `free_axes` (i.e., the complement
        of the coordinate system's axes) and returns a dictionary mapping those fixed axes
        to their default fill values. This is useful when computing expressions that depend on
        a subset of the coordinate system's axes (e.g., chunked computations with broadcasting).

        Parameters
        ----------
        free_axes : Sequence[str]
            The list of logical axes over which the operation is being performed.

        Returns
        -------
        fixed_axes : Sequence[str]
            Axes not in `free_axes`. These are assumed to be fixed during computation.
        fixed_values : dict
            Dictionary mapping each fixed axis to its default value as specified in
            `self.fill_values`.

        Examples
        --------
        >>> fixed_axes, fixed_values = self.compute_fixed_axes_and_values(["x", "y"])
        >>> print(fixed_axes)      # ['z'] (for example)
        >>> print(fixed_values)    # {'z': 0.0}
        """
        fixed_axes = self.__cs__.axes_complement(free_axes)
        fixed_values = {k: v for k, v in self.fill_values.items() if k in fixed_axes}
        return fixed_axes, fixed_values

    # noinspection PyDefaultArgument
    def _make_expression_chunk_fetcher(
        self: _SupDGMO,
        expr_name: str,
        fixed_axes: dict,
        value: Optional[np.ndarray] = None,
    ):
        """
        Returns a chunk-aware getter function for a grid expression.

        If `value` is None, computes the expression on-the-fly for each chunk using the coordinate system.
        Otherwise, slices into the precomputed array.

        Parameters
        ----------
        expr_name : str
            Name of the expression to compute (e.g., 'inverse_metric_tensor').
        fixed_axes : dict
            Dictionary of fixed axes for the coordinate evaluation.
        value : Optional[np.ndarray]
            Precomputed field array to use if provided.

        Returns
        -------
        Callable
            A function (chunk_slices, coordinates) -> array for this chunk.
        """
        if value is None:
            # Capture `expr_name` and `fixed_axes` at function definition time using default args
            return (
                lambda chunk_slices, coordinates, _e=expr_name, _f=fixed_axes.copy(): (
                    self.__cs__.compute_expression_from_coordinates(
                        _e, coordinates, fixed_axes=_f
                    )
                )
            )
        else:
            # Return a simple slicing function
            return lambda chunk_slices, coordinates: value[(*chunk_slices, ...)]

    # ======================================= #
    # General Math Ops                        #
    # ======================================= #
    def compute_function_on_grid(
        self: _SupDGMO,
        func: Callable,
        /,
        result_shape: Optional[Sequence[int]] = None,
        out: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        output_axes: Optional[Sequence[str]] = None,
        func_type: Literal["all", "axes"] = "axes",
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Evaluate a coordinate-based function over the grid domain.

        This method computes the values of a user-supplied callable `func` at each point on the grid
        spanned by `output_axes`, using the physical (coordinate system-defined) coordinates at those points.
        It supports both global (in-memory) and streaming (chunked) evaluation strategies.

        This is useful for constructing scalar, vector, or tensor fields defined analytically in terms
        of coordinates, and can be applied efficiently even for large grids.

        Parameters
        ----------
        func : callable
            A Python function that returns values defined over physical coordinates. Its signature must match
            the coordinate mode:

            - If ``func_type="axes"`` (default), the function takes only the coordinates of `output_axes`
              (in canonical coordinate order).
            - If ``func_type="all"``, the function must take all coordinate axes of the system in canonical order.

            The return value should be an array with shape `result_shape` per grid point.

        result_shape : sequence of int, optional
            The shape of the result at each grid point (e.g., `()` for scalar fields, `(3,)` for 3-vectors).
            Defaults to `()` if unspecified.

        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        pbar : bool, optional
            Whether to show a progress bar during chunked evaluation.
        pbar_kwargs : dict, optional
            Extra keyword arguments passed to `tqdm` when `pbar=True`.
        func_type : {"axes", "all"}, optional
                    Specifies whether `func` accepts all coordinate axes or only those listed in `output_axes`.
                    By default, ``"axes"``.
        **kwargs :
            Additional keyword arguments forwarded to `func`.

        Returns
        -------
        np.ndarray
            A NumPy array with shape `(grid_shape over output_axes) + result_shape`, containing the
            evaluated function values at each grid point.

        Notes
        -----
        This method evaluates the function:

        .. math::

            f(x^1, x^2, ..., x^n)

        where :math:`x^i` are the physical coordinates (e.g., `r`, `\theta`, `z`, etc.) defined
        by the coordinate system.

        If `in_chunks=True`, the grid is traversed in blocks (including 1-cell ghost zones), and the function
        is evaluated independently on each block. This enables out-of-core and lazy buffer evaluations.

        This method is commonly used to:

        - Construct analytic scalar/vector/tensor fields on structured grids
        - Convert closed-form physics models into discrete field data
        - Generate test fields for numerical method validation
        """
        # --- Handle fixed and free axes --- #
        output_axes = self.standardize_axes(output_axes)
        fixed_axes, fixed_values = self._compute_fixed_axes_and_values(
            free_axes=output_axes
        )

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        # We do have some additional leg work to determine the element shape in this case.
        result_shape = tuple(result_shape) if result_shape is not None else ()
        out = self._prepare_output_buffer(
            output_axes,
            output_element_shape=result_shape,
            out=out,
            include_ghosts=True,
            dtype=kwargs.pop("dtype", "f8"),
        )

        # --- Handle the eval function --- #
        # We need to handle the function because it might be
        # a function of the axes and not the full coordinate set,
        # but we want it to be a function of the full coordinate
        # set when passed through.
        if func_type == "axes":
            # Canonical list of coordinate names in this system
            all_axes = self.__cs__.__AXES__

            # Determine the indices of the output axes in the full system
            output_indices = [all_axes.index(ax) for ax in output_axes]

            # Define a new function that accepts *all_axes but internally calls func(*args_subset)
            # noinspection PyMissingOrEmptyDocstring
            def full_func(*_args, **_kwargs):
                # Subselect only the arguments that correspond to output_axes
                args_subset = [_args[i] for i in output_indices]
                return func(*args_subset, **_kwargs)

            eval_func = full_func
        else:
            eval_func = func

        # --- Perform the computation --- #
        if in_chunks:
            # Operation performed in chunks.
            self._ensure_supports_chunking()

            # Cycle through chunks.
            for chunk_slices in self.iter_chunk_slices(
                axes=output_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Compute coordinates. Cut down to the correct set of coordinates and
                # slices for the input field axes.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=output_axes, origin="global", __validate__=False
                )
                out[
                    (*chunk_slices, ...)
                ] = self.__cs__.compute_function_from_coordinates(
                    eval_func, coordinates, fixed_axes=fixed_values
                )
        else:
            coordinates = self.compute_domain_coords(
                axes=output_axes, origin="global", __validate__=False
            )
            out[...] = self.__cs__.compute_function_from_coordinates(
                eval_func, coordinates, fixed_axes=fixed_values
            )

        # Return the output buffer.
        return out

    # ======================================= #
    # General Dense Ops                       #
    # ======================================= #
    # recasting of functions from differential geometry's
    # general_ops module.
    def dense_element_wise_partial_derivatives(
        self: _SupDGMO,
        field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the element-wise partial derivatives of an array-valued field over the grid.

        This method computes the partial derivatives along each of the :attr:`~grids.base.GridBase.ndim` axes of the grid
        for each element of an array-valued input field. Thus,

        .. math::

            T_{ijk\ldots} \to T_{ijk\ldots;\mu} = \partial_\mu T_{ijk\ldots}.

        .. hint::

            Under the hood, this method wraps :func:`~differential_geometry.general_ops.dense_element_wise_partial_derivatives`.

        Parameters
        ----------
        field : array-like
            The array-valued field on which to compute the partial derivatives. This should be an array-like object
            with a compliant shape (see `Notes` below).
        field_axes : list of str
            The coordinate axes over which the `field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
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
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.general_ops.dense_element_wise_partial_derivatives`.

        Returns
        -------
        array-like
            The computed partial derivatives. he returned array has shape:

            .. code-block:: python

                (...grid_shape over `output_axes`, ...element_shape of `field`, ndim)

            The final axis contains the partial derivatives with respect to each coordinate axis
            in the grid’s coordinate system.

        Notes
        -----

        **Broadcasting and Array Shapes**:

        The input ``field`` must have a very precise shape to be valid for this operation. If the underlying grid
        (**including ghost zones**) has shape ``grid_shape = (G_1,...,G_ndim)``, then the spatial dimensions of the field must
        match exactly ``grid_shape[axes]``. Any remaining dimensions of the ``field`` are treated as elements of the field.

        Thus, if a ``(G1,G3,F1,F2,F3)`` field is passed with ``field_axes = ['x','z']`` (in cartesian coordinates), then
        the resulting output array will have shape ``(G1,G3,F1,F2,F3,3)`` and ``out[...,1] == 0`` because the field does not
        contain any variability over the ``y`` variable.

        If ``output_axes`` is specified, then that resulting grid will be broadcast to any additional grid axes necessary.

        When ``out`` is specified, it must match (exactly) the expected output buffer shape or an error will arise.

        **Chunking Semantics**:

        When ``in_chunks=True``, chunking is enabled for this operation. In that case, the ``out`` buffer is filled
        iteratively by performing computations on each chunk of the grid over the specified `output_axes`. When the
        computation is performed, each chunk is extracted with an additional 1-cell halo to ensure that :func:`numpy.gradient`
        attains its maximal accuracy on each chunk.

        .. note::

            Chunking is (generally) only useful when ``out`` and ``field`` are array-like lazy-loaded buffers like
            HDF5 datasets. In those cases, the maximum memory load is only that required to load each individual chunk.
            If the ``out`` buffer is not specified, it is allocated fully anyways, making chunking somewhat redundant.


        See Also
        --------
        dense_covariant_gradient: Covariant gradient of a tensor field.
        dense_contravariant_gradient: Contravariant gradient of a tensor field.
        ~differential_geometry.general_ops.dense_element_wise_partial_derivatives: Low-level callable version.

        Examples
        --------

        **Derivatives of a scalar field**:

        The easiest example is the derivative of a generic scalar field.

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import CartesianCoordinateSystem2D
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = CartesianCoordinateSystem2D()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]],center='cell')
            >>>
            >>> # Create the field
            >>> X,Y = grid.compute_domain_mesh(origin='global')
            >>> Z = np.sin((X**2+Y**2))
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives = grid.dense_element_wise_partial_derivatives(Z,['x','y'])
            >>>
            >>> fig,axes = plt.subplots(1,3,sharey=True,sharex=True,figsize=(7,3))
            >>> _ = axes[0].imshow(Z.T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1].imshow(derivatives[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[2].imshow(derivatives[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> plt.show()

        **Derivatives of an array field**:

        Similarly, this can be applied to array fields of any sort.

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import CartesianCoordinateSystem2D
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = CartesianCoordinateSystem2D()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]],center='cell')
            >>>
            >>> # Create the field
            >>> X,Y = grid.compute_domain_mesh(origin='global')
            >>> Z = np.stack([np.sin((X**2+Y**2)),np.sin(5*(X**2+Y**2))],axis=-1) # (504,504,2)
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives = grid.dense_element_wise_partial_derivatives(Z,['x','y'])
            >>>
            >>> fig,axes = plt.subplots(2,3,sharey=True,sharex=True,figsize=(7,6))
            >>> _ = axes[0,0].imshow(Z[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[0,1].imshow(derivatives[...,0,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[0,2].imshow(derivatives[...,0,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1,0].imshow(Z[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1,1].imshow(derivatives[...,1,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
            >>> _ = axes[1,2].imshow(derivatives[...,1,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
            >>> plt.show()

        **Expanding to output axes**:

        In some cases, you might have a field :math:`T_{ijk\ldots}(x,y)` and you may need
        :math:`\partial_\mu T_{ijk\ldots}(x,y,z)`. This can be achieved by declaring the `output_axes`
        argument.

        >>> from pymetric.coordinates import CartesianCoordinateSystem3D
        >>>
        >>> # Create the coordinate system
        >>> cs = CartesianCoordinateSystem3D()
        >>>
        >>> # Create the grid
        >>> grid = UniformGrid(cs,[[-1,-1,-1],[1,1,1]],[50,50,50],chunk_size=[5,5,5],ghost_zones=[2,2,2],center='cell')
        >>>
        >>> # Create the field
        >>> X,Y = grid.compute_domain_mesh(origin='global',axes=['x','y'])
        >>> Z = np.stack([np.sin((X**2+Y**2)),np.sin(5*(X**2+Y**2))],axis=-1) # (54,54,2)
        >>>
        >>> # Compute the partial derivatives.
        >>> derivatives = grid.dense_element_wise_partial_derivatives(Z,['x','y'],output_axes=['x','y','z'])
        >>> derivatives.shape
        (54, 54, 54, 2, 3)
        """
        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        field_axes, output_axes, field_axes_indices, _ = self._set_input_output_axes(
            field_axes, output_axes=output_axes
        )
        # Confirm that the field matches the expected shape.
        self.check_field_shape(field.shape, axes=field_axes)

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        # We do have some additional leg work to determine the element shape in this case.
        rank = field.ndim - len(field_axes)
        element_shape = field.shape[field.ndim - rank :]
        element_shape_out = tuple(element_shape) + (self.ndim,)
        out = self._prepare_output_buffer(
            output_axes,
            output_element_shape=element_shape_out,
            out=out,
            include_ghosts=True,
            dtype=field.dtype,
        )

        # --- Perform Gradient Operation --- #
        if in_chunks:
            # Operation performed in chunks.
            self._ensure_supports_chunking()

            for chunk_slices in self.iter_chunk_slices(
                axes=field_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Cut the slice out of the input tensor field.
                tensor_field_chunk = self.broadcast_array_to_axes(
                    field[(*chunk_slices, ...)],
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Construct the coordinates from the slices.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=field_axes, origin="global", __validate__=False
                )

                # Compute the covariant gradient.
                cov_grad = dense_element_wise_partial_derivatives(
                    tensor_field_chunk,
                    rank,
                    *coordinates,
                    output_indices=field_axes_indices,  # ensures we place grads into right slots.
                    edge_order=edge_order,
                    **kwargs,
                )

                # Assign to the output.
                out[
                    (
                        *chunk_slices,
                        ...,
                    )
                ] = cov_grad

        else:
            # Operation performed in single call. Compute all
            # coordinates over the entire domain and then compute.
            # Cast the input field to the output axes.
            reshaped_tensor_field = self.broadcast_array_to_axes(
                field, axes_in=field_axes, axes_out=output_axes
            )

            # Compute the coordinates.
            coordinates = self.compute_domain_coords(
                axes=field_axes, origin="global", __validate__=False
            )

            # Compute covariant gradient.
            dense_element_wise_partial_derivatives(
                reshaped_tensor_field,
                rank,
                *coordinates,
                output_indices=field_axes_indices,  # ensures we place grads into right slots.
                edge_order=edge_order,
                out=out,
                **kwargs,
            )

        # Return the output buffer.
        return out

    def dense_element_wise_laplacian(
        self: _SupDGMO,
        field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        Lterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        second_derivative_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the element-wise Laplacian of an array-valued field on the grid.

        For an array field :math:`T_{\ldots}^{\ldots}`, this method computes the Laplacian
        of each element individually. The scalar Laplacian of an element :math:`\phi` is:

        .. math::

            \nabla^2 \phi = \nabla \cdot \nabla \phi = \frac{1}{\rho} \partial_\mu \left( \rho g^{\mu\nu} \partial_\nu \phi \right)

        A numerically stable rearrangement is used in practice:

        .. math::

            \nabla^2 \phi = \frac{1}{\rho} \partial_\mu \left[ g^{\mu\nu} \rho \right] \partial_\nu \phi + g^{\mu\nu} \partial_\mu \partial_\nu \phi
                         = L^\nu \partial_\nu \phi + g^{\mu\nu} \partial_\mu \partial_\nu \phi

        This is the form used internally for stable, metric-aware computation.

        .. hint::

            Internally calls either :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_diag` (for diagonal metrics)
            or :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_full` (for full metrics), depending on the coordinate system.

        Parameters
        ----------
        field : array-like
            The array-valued field on which to operate. This must meet all the
            necessary shape criteria (see Notes).
        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `tensor_field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        Lterm_field : array-like, optional
            The volume log-derivative field :math:`L^\nu = \frac{1}{\rho} \partial_\mu [g^{\mu\nu} \rho]`.
            If not provided, it is computed automatically using the coordinate system. This argument can be filled to
            reduce numerical error and improve computational efficiency if it is known.

            If specified, `Lterm_field` must be shape compliant (see Notes).
        inverse_metric_field : array-like, optional
            A buffer containing the inverse metric field :math:`g^{\mu\nu}`. `inverse_metric_field`
            can be provided to improve computation speed (by avoiding computing it in stride);
            however, it is not required.

            The inverse metric can be derived from the coordinate system when this
            argument is not provided. See `Notes` below
            for details on the shape of `inverse_metric_field`.
        derivative_field : array-like, optional
            A buffer containing the first derivatives of the field. Can be provided to improve
            computation speed (by avoiding computing it in stride); however, it is not required.

            If specified, `derivative_field` must be shape compliant (see Notes).
        second_derivative_field : array-like, optional
            A buffer containing the second derivatives of the field. Can be provided to improve
            computation speed (by avoiding computing it in stride); however, it is not required.

            If specified, `second_derivative_field` must be shape compliant (see Notes).
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_full` or
            :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_diag`.

        Returns
        -------
        array-like
            Array of the same shape as `field`, optionally broadcast over additional axes.
            Contains the computed Laplacian :math:`\nabla^2 \phi` of each element in the input.

        Notes
        -----
        **Shape and Broadcasting Requirements**

        The spatial dimensions of `field` must match the grid shape exactly over the `field_axes`.
        For a scalar field on a grid with shape ``(G1, ..., Gm)``, and `field_axes = ['x', 'z']`,
        the field must have shape ``(Gₓ, G_z)``. For tensor fields, additional trailing dimensions
        (beyond the spatial ones) are interpreted as tensor indices and must either match `ndim` exactly
        or be nested in a form that makes the Laplacian contractable (i.e., act elementwise).

        The output shape will match the shape of `tensor_field` unless `output_axes` introduces
        additional broadcasting (e.g., singleton axes added by `broadcast_array_to_axes`).

        **Use of Auxiliary Fields**

        - `Lterm_field` is used to stabilize the Laplacian operator in curved coordinates.
        - `inverse_metric_field` allows skipping the on-the-fly metric computation.
        - `derivative_field` and `second_derivative_field` allow precomputing the necessary gradient terms.

        Each of these inputs must conform to specific shape constraints:

        - `Lterm_field`: (..., ndim)
        - `inverse_metric_field`: (..., ndim) or (..., ndim, ndim)
        - `derivative_field`: (..., ndim)
        - `second_derivative_field`: (..., ndim) or (..., ndim, ndim)

        **Chunked Execution**

        When `in_chunks=True`, the Laplacian is computed in small memory-efficient blocks with
        halo padding of 1 cell. This is especially useful when `tensor_field` and `out` are backed
        by HDF5 or other lazy-loading array backends. Chunking requires the grid to support
        `iter_chunk_slices(...)`.

        **Applicability**

        This method applies the scalar Laplacian element-wise to each component of the input field.
        It is appropriate for scalar fields, vector fields, or element-wise defined tensor fields
        in arbitrary curvilinear coordinates.

        See Also
        --------
        dense_element_wise_partial_derivatives: Generic form for general array-valued fields.
        dense_covariant_gradient: Covariant gradient of a tensor field.
        ~differential_geometry.dense_ops.dense_gradient_contravariant_full: Low-level callable version (full metric)
        ~differential_geometry.dense_ops.dense_gradient_contravariant_diag: Low-level callable version (diag metric)
        """
        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        (
            field_axes,
            output_axes,
            field_axes_indices,
            output_axes_indices,
        ) = self._set_input_output_axes(field_axes, output_axes=output_axes)
        differential_axes_indices = np.asarray(
            [_i for _i, _a in enumerate(output_axes) if _a in field_axes]
        )
        fixed_axes, fixed_values = self._compute_fixed_axes_and_values(
            free_axes=output_axes
        )

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        rank = field.ndim - len(field_axes)
        out = self._prepare_output_buffer(
            output_axes, out=out, include_ghosts=True, dtype=field.dtype
        )

        # --- Determine the correct operator --- #
        # We need to check the metric shape to determine.
        if len(self.__cs__.metric_tensor_symbol.shape) == 1:
            __op__ = dense_scalar_laplacian_diag
        else:
            __op__ = dense_scalar_laplacian_full

        # --- Perform the operation --- #
        if in_chunks:
            # Compute the divergence in chunks. Broadly speaking, this proceeds in
            # the following order of operations:
            # 1. Ensure that chunking is supported.
            # 2. Determine if we are given the D-term and (if so), mark that
            #    we don't need to try to compute on each round.
            self._ensure_supports_chunking()

            # Determine if we need to try to generate the
            # D-term field for each chunk or if we can just grab it.
            _try_F = Lterm_field is None
            _try_metric = inverse_metric_field is None
            _has_derivative = derivative_field is not None
            _has_second_derivative = second_derivative_field is not None

            # Iterate through each of the chunk slices in the
            # output space.
            for chunk_slices in self.iter_chunk_slices(
                axes=output_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Compute coordinates. Cut down to the correct set of coordinates and
                # slices for the input field axes.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=output_axes, origin="global", __validate__=False
                )
                differential_coordinates = [
                    coordinates[i] for i in differential_axes_indices
                ]
                differential_chunk_slices = [
                    chunk_slices[i] for i in differential_axes_indices
                ]

                # Broadcast the vector field onto the chunk.
                vector_field_chunk = self.broadcast_array_to_axes(
                    field[(*differential_chunk_slices, ...)],
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Attempt to build the D-term if it is needed.
                if _try_F:
                    Fterm_chunk = self.__cs__.compute_expression_from_coordinates(
                        "Lterm", coordinates, fixed_axes=fixed_values
                    )
                else:
                    Fterm_chunk = Lterm_field[(*chunk_slices, ...)]

                if _try_metric:
                    inverse_metric_field_chunk = (
                        self.__cs__.compute_expression_from_coordinates(
                            "inverse_metric_tensor",
                            coordinates,
                            fixed_axes=fixed_values,
                        )
                    )
                else:
                    inverse_metric_field_chunk = inverse_metric_field[
                        (*chunk_slices, ...)
                    ]

                # If we have the derivative field, we need to cut into it.
                if _has_derivative:
                    derivative_field_broadcast = self.broadcast_array_to_axes(
                        derivative_field[(*differential_chunk_slices, ...)],
                        axes_in=field_axes,
                        axes_out=output_axes,
                    )
                else:
                    derivative_field_broadcast = None

                if _has_second_derivative:
                    second_derivative_field_broadcast = self.broadcast_array_to_axes(
                        second_derivative_field[(*differential_chunk_slices, ...)],
                        axes_in=field_axes,
                        axes_out=output_axes,
                    )
                else:
                    second_derivative_field_broadcast = None

                # Compute the covariant gradient.
                out[(*chunk_slices, ...)] = __op__(
                    vector_field_chunk,
                    Fterm_chunk,
                    inverse_metric_field_chunk,
                    rank,
                    self.ndim,
                    *differential_coordinates,
                    derivative_field=derivative_field_broadcast,
                    second_derivative_field=second_derivative_field_broadcast,
                    field_axes=output_axes_indices,
                    derivative_axes=differential_axes_indices,
                    edge_order=edge_order,
                    **kwargs,
                )
        else:
            # Perform the operation in one pass. Broadly, the steps are
            # 1. Broadcast the field to the output axes for consistency.
            # 2. Compute the coordinates in the output axes space.
            # 3. Compute the Dterm_field if it is not provided.
            # 4. Broadcast the derivative field / 2nd derivative field if it is provided.

            # Broadcast to output axes. This will be (F1, ..., 1, ... FM) or something
            # of the sort.
            tensor_field_broadcast = self.broadcast_array_to_axes(
                field, axes_in=field_axes, axes_out=output_axes
            )

            # Compute the output coordinates so that we can
            # perform the differentiation operation.
            coordinates = self.compute_domain_coords(
                axes=output_axes, origin="global", __validate__=False
            )
            differential_coordinates = [
                coordinates[i] for i in differential_axes_indices
            ]

            # Broadcast required fields.
            if derivative_field is not None:
                derivative_field_broadcast = self.broadcast_array_to_axes(
                    derivative_field, axes_in=field_axes, axes_out=output_axes
                )
            else:
                derivative_field_broadcast = None

            if second_derivative_field is not None:
                second_derivative_field_broadcast = self.broadcast_array_to_axes(
                    second_derivative_field, axes_in=field_axes, axes_out=output_axes
                )
            else:
                second_derivative_field_broadcast = None

            # Create the D-term field over the free coordinates.
            Lterm_field = self.__cs__.requires_expression_from_coordinates(
                Lterm_field,
                "Lterm",
                coordinates,
                fixed_axes=fixed_values,
            )
            inverse_metric_field = self.__cs__.requires_expression_from_coordinates(
                inverse_metric_field,
                "inverse_metric_tensor",
                coordinates,
                fixed_axes=fixed_values,
            )

            __op__(
                tensor_field_broadcast,
                Lterm_field,
                inverse_metric_field,
                rank,
                self.ndim,
                *differential_coordinates,
                field_axes=output_axes_indices,
                derivative_axes=differential_axes_indices,
                edge_order=edge_order,
                out=out,
                first_derivative_field=derivative_field_broadcast,
                second_derivative_field=second_derivative_field_broadcast,
                **kwargs,
            )

        return out

    # ======================================= #
    # Dense Utils                             #
    # ======================================= #
    # These methods allow wrapping for methods
    # in `dense_utils`.
    def dense_contract_with_metric(
        self: _SupDGMO,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        index: int,
        /,
        mode: str = "lower",
        out: Optional[ArrayLike] = None,
        metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Contract a tensor index with the metric tensor or its inverse.

        This method raises or lowers a single index of a tensor field by contracting
        it with the metric tensor :math:`g_{\mu\nu}` or its inverse :math:`g^{\mu\nu}`.

        Depending on the mode:

        - ``mode='lower'`` applies :math:`T^\mu \mapsto T_\nu = g_{\mu\nu} T^\mu`
        - ``mode='raise'`` applies :math:`T_\mu \mapsto T^\nu = g^{\mu\nu} T_\mu`

        The appropriate form (full vs. diagonal) of the metric is automatically chosen based
        on the coordinate system's geometry.

        .. hint::

            Internally wraps :func:`~differential_geometry.dense_utils.dense_contract_with_metric`,
            using the diagonal or full version depending on the metric type.

        Parameters
        ----------
        tensor_field : array-like
            The tensor field whose index signature is to be adjusted.
            See `Notes` for more details on the shape requirements.
        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `tensor_field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        index : int
            Slot to raise or lower (``0 <= index < rank``).
        mode : {'raise', 'lower'}, default 'lower'
            Conversion direction. If ``'raise'``, then the inverse metric tensor is
            used in the contraction. Otherwise, the metric tensor is used.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        metric_field : array-like, optional
            Optional precomputed metric or inverse metric to use in the contraction:
            - For ``mode='raise'``, this must be the inverse metric.
            - For ``mode='lower'``, this must be the metric.

            May be diagonal (shape ``(..., ndim)``) or full (``(..., ndim, ndim)``).
            If not provided, it is computed from the coordinate system.
            See `Notes` for more details on the shape requirements.
        in_chunks : bool, default False
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        pbar : bool, default True
            Show a progress bar when ``in_chunks=True``.
        pbar_kwargs : dict, optional
            Extra keyword arguments forwarded to *tqdm*.
        **kwargs
            Passed straight to the low-level metric-contraction kernels
            (e.g. `where=` masks).

        Returns
        -------
        array-like
            The tensor with the selected slot converted.  Shape equals
            ``broadcast(grid_shape over output_axes) + element_shape``.

        Notes
        -----
        **Shape and Broadcasting Requirements**

        The shape of `tensor_field` must exactly match the grid shape over `field_axes`.
        Any trailing axes are treated as component dimensions of the tensor, and must match the
        expected rank.

        The `out` buffer (if supplied) must match the computed output shape.

        **Metric Input**

        If `metric_field` is not supplied, it is computed from the coordinate system.
        Supported metric shapes:

        - Diagonal metric: shape ``(..., ndim)``
        - Full metric: shape ``(..., ndim, ndim)``

        **Chunked Execution**

        When `in_chunks=True`, the grid is processed in small chunks (with ghost zones).
        This reduces memory overhead and improves performance with lazy buffers like HDF5.

        See Also
        --------
        dense_element_wise_partial_derivatives : Computes directional derivatives.
        dense_covariant_gradient : Computes the full covariant gradient of a tensor.
        ~differential_geometry.dense_utils.dense_contract_with_metric : Low-level backend for index contraction.
        """

        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        field_axes, output_axes, field_axes_indices, _ = self._set_input_output_axes(
            field_axes, output_axes=output_axes
        )
        fixed_axes, fixed_values = self._compute_fixed_axes_and_values(
            free_axes=output_axes
        )

        # --- Determine the correct operator --- #
        # We need to check the metric shape to determine.
        if len(self.__cs__.metric_tensor_symbol.shape) == 1:
            __op__ = _dense_contract_index_with_diagonal_metric
        else:
            __op__ = _dense_contract_index_with_metric

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        tensor_field_ndim, tensor_field_shape = tensor_field.ndim, tensor_field.shape
        rank = tensor_field_ndim - len(field_axes)
        tensor_field_element_shape = tensor_field_shape[tensor_field_ndim - rank :]
        output_field_element_shape = tuple(tensor_field_element_shape)
        out = self._prepare_output_buffer(
            output_axes,
            output_element_shape=output_field_element_shape,
            out=out,
            include_ghosts=True,
            dtype=tensor_field.dtype,
        )

        # --- Perform Gradient Operation --- #
        if in_chunks:
            # Operation performed in chunks.
            self._ensure_supports_chunking()

            # -- Create the chunk fetchers -- #
            # These allow us to seamlessly generate the correct
            # metric field over the chunks regardless of it being
            # pre-specified.
            _metric_fetcher_ = self._make_expression_chunk_fetcher(
                "inverse_metric_tensor" if mode == "raise" else "metric_tensor",
                fixed_values,
                value=metric_field,
            )

            # Cycle through chunks.
            for chunk_slices in self.iter_chunk_slices(
                axes=output_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Compute coordinates. Cut down to the correct set of coordinates and
                # slices for the input field axes.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=output_axes, origin="global", __validate__=False
                )

                tensor_field_chunk = self.broadcast_array_to_axes(
                    tensor_field,
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Attempt to build the metric tensor.
                metric_field_chunk = _metric_fetcher_(chunk_slices, coordinates)

                # Compute the covariant gradient.
                out[(*chunk_slices, ...)] = __op__(
                    tensor_field_chunk,
                    metric_field_chunk,
                    index,
                    rank,
                    **kwargs,
                )

        else:
            # Operation performed in single call. Compute all
            # coordinates over the entire domain and then compute.
            # Cast the input field to the output axes.
            reshaped_tensor_field = self.broadcast_array_to_axes(
                tensor_field, axes_in=field_axes, axes_out=output_axes
            )

            # Compute the coordinates over the output axes. The differential coordinates
            # are then also constructed.
            coordinates = self.compute_domain_coords(
                axes=output_axes, origin="global", __validate__=False
            )

            # Ensure that the inverse metric tensor has been computed
            # on the coordinates.
            metric_field = self.__cs__.requires_expression_from_coordinates(
                metric_field,
                "inverse_metric_tensor" if mode == "raise" else "metric_tensor",
                coordinates,
                fixed_axes=fixed_values,
            )

            # Compute covariant gradient.
            __op__(
                reshaped_tensor_field,
                metric_field,
                index,
                rank,
                out=out,
                **kwargs,
            )

        # Return the output buffer.
        return out

    def dense_adjust_tensor_signature(
        self: _SupDGMO,
        tensor_field,
        field_axes,
        indices,
        tensor_signature,
        /,
        out: Optional[ArrayLike] = None,
        metric_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Adjust the index signature of a tensor field by raising or lowering multiple indices.

        This method converts the variance (covariant vs. contravariant) of one or more indices
        of a tensor field to match a desired target signature. The transformation is applied in a
        single pass using the appropriate metric tensor, which may be full or diagonal depending on
        the coordinate system.

        It generalizes :meth:`dense_contract_with_metric` to allow simultaneous transformation
        of multiple indices and automatically selects raise/lower operations based on the
        desired `tensor_signature`.

        .. hint::

            Wraps :func:`~differential_geometry.dense_utils._dense_adjust_tensor_signature`
            or its diagonal-metric variant.

        Parameters
        ----------
        tensor_field : array-like
            The input tensor field whose indices will be transformed. The array shape must match
            the grid over `field_axes` followed by `rank` trailing axes representing tensor components.

        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `tensor_field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        indices : list of int
            Indices among the last `rank` axes to be modified.
        tensor_signature : list of int
            Current signature of the tensor's component axes. Must be of length `rank` and contain
            values `+1` for contravariant and `-1` for covariant indices.

        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        metric_field, inverse_metric_field : array-like, optional
            Pre-computed metric and inverse metric used for the
            conversions.  Provide one or both to avoid recomputation.
            Shape rules follow :meth:`dense_contract_with_metric`.
        in_chunks: bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        pbar : bool, default True
            Show a progress bar when ``in_chunks=True``.
        pbar_kwargs : dict, optional
            Extra keyword arguments forwarded to *tqdm*.
        **kwargs
            Passed straight to the low-level metric-contraction kernels
            (e.g. `where=` masks).

        Returns
        -------
        array-like
            Tensor with the specified slots in the requested variance.

        Notes
        -----

        **Signature Logic**

        For each index `k` in `indices`, this method compares the target signature with
        the current one from `tensor_signature`:

        - If ``tensor_signature[k] == -1`` and the index is currently contravariant, it is lowered.
        - If ``tensor_signature[k] == +1`` and the index is currently covariant, it is raised.
        - If already in the correct form, no operation is applied to that index.

        **Metric Field Requirements**

        Depending on the operation and coordinate system, the following metric shapes are supported:

        - Diagonal metric: shape ``(..., ndim)``
        - Full metric: shape ``(..., ndim, ndim)``

        If not provided, both metric and inverse metric are computed automatically
        from the coordinate system (globally or chunk-wise).

        **Chunked Execution**

        When `in_chunks=True`, the field is processed in chunks with halo padding.
        This is useful for HDF5-backed or large fields where full-domain memory
        use is undesirable.

        **Efficiency Tip**

        For repeated transformations over the same grid (e.g., adjusting multiple tensor fields),
        you can precompute `metric_field` and `inverse_metric_field` using
        :meth:`compute_function_on_grid` or similar and pass them in to avoid redundant evaluation.

        See Also
        --------
        dense_contract_with_metric : Lower- or raise a single tensor index.
        """
        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        field_axes, output_axes, field_axes_indices, _ = self._set_input_output_axes(
            field_axes, output_axes=output_axes
        )
        fixed_axes, fixed_values = self._compute_fixed_axes_and_values(
            free_axes=output_axes
        )

        # --- Determine the correct operator --- #
        # We need to check the metric shape to determine.
        if len(self.__cs__.metric_tensor_symbol.shape) == 1:
            __op__ = _dense_adjust_tensor_signature_diagonal_metric
        else:
            __op__ = _dense_adjust_tensor_signature

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        tensor_field_ndim, tensor_field_shape = tensor_field.ndim, tensor_field.shape
        rank = tensor_field_ndim - len(field_axes)
        tensor_field_element_shape = tensor_field_shape[tensor_field_ndim - rank :]
        output_field_element_shape = tuple(tensor_field_element_shape)
        out = self._prepare_output_buffer(
            output_axes,
            output_element_shape=output_field_element_shape,
            out=out,
            include_ghosts=True,
            dtype=tensor_field.dtype,
        )

        # --- Perform Gradient Operation --- #
        if in_chunks:
            # Operation performed in chunks.
            self._ensure_supports_chunking()

            # -- Create the chunk fetchers -- #
            # These allow us to seamlessly generate the correct
            # metric field over the chunks regardless of it being
            # pre-specified.
            _metric_fetcher_ = self._make_expression_chunk_fetcher(
                "metric_tensor",
                fixed_values,
                value=metric_field,
            )
            _inverse_metric_fetcher_ = self._make_expression_chunk_fetcher(
                "inverse_metric_tensor",
                fixed_values,
                value=inverse_metric_field,
            )

            # Cycle through chunks.
            for chunk_slices in self.iter_chunk_slices(
                axes=output_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Compute coordinates. Cut down to the correct set of coordinates and
                # slices for the input field axes.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=output_axes, origin="global", __validate__=False
                )

                tensor_field_chunk = self.broadcast_array_to_axes(
                    tensor_field,
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Attempt to build the metric tensor.
                metric_field_chunk = _metric_fetcher_(chunk_slices, coordinates)
                inv_metric_field_chunk = _inverse_metric_fetcher_(
                    chunk_slices, coordinates
                )

                # Compute the covariant gradient.
                out[(*chunk_slices, ...)] = __op__(
                    tensor_field_chunk,
                    indices,
                    tensor_signature,
                    metric_field=metric_field_chunk,
                    inverse_metric_field=inv_metric_field_chunk,
                    **kwargs,
                )

        else:
            # Operation performed in single call. Compute all
            # coordinates over the entire domain and then compute.
            # Cast the input field to the output axes.
            reshaped_tensor_field = self.broadcast_array_to_axes(
                tensor_field, axes_in=field_axes, axes_out=output_axes
            )

            # Compute the coordinates over the output axes. The differential coordinates
            # are then also constructed.
            coordinates = self.compute_domain_coords(
                axes=output_axes, origin="global", __validate__=False
            )

            # Ensure that the inverse metric tensor has been computed
            # on the coordinates.
            metric_field = self.__cs__.requires_expression_from_coordinates(
                metric_field,
                "metric_tensor",
                coordinates,
                fixed_axes=fixed_values,
            )
            inverse_metric_field = self.__cs__.requires_expression_from_coordinates(
                metric_field,
                "inverse_metric_tensor",
                coordinates,
                fixed_axes=fixed_values,
            )

            # Compute covariant gradient.
            __op__(
                reshaped_tensor_field,
                indices,
                tensor_signature,
                metric_field=metric_field,
                inverse_metric_field=inverse_metric_field,
                out=out,
                **kwargs,
            )

        # Return the output buffer.
        return out

    # ============================================================ #
    # Index–raising / –lowering convenience wrappers               #
    # ============================================================ #
    def dense_raise_index(
        self: _SupDGMO,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        index: int,
        /,
        out: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Raise a single covariant index of a tensor field.

        This method performs a metric contraction with the inverse metric tensor
        :math:`g^{\mu\nu}` to convert a covariant (lower) index to contravariant (upper)
        form. The `index` argument specifies which slot of the tensor should be raised.

        This is a specialized wrapper around :meth:`dense_contract_with_metric`
        with ``mode='raise'``.

        .. hint::
            Use this method to promote a component of a mixed or covariant tensor
            in curvilinear coordinates.

        Parameters
        ----------
        tensor_field : array-like
            The input tensor field. Its shape must be compatible with the grid dimensions
            over `field_axes`, followed by one or more component axes representing the tensor rank.

        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        index : int
            The index of the tensor field to raise. This should range from 0 to `rank` and is
            measured from the last spatial dimension of the tensor field.
        inverse_metric_field : array-like, optional
            A buffer containing the inverse metric field :math:`g^{\mu\nu}`. `inverse_metric_field`
            can be provided to improve computation speed (by avoiding computing it in stride);
            however, it is not required.

            The inverse metric can be derived from the coordinate system when this
            argument is not provided. See `Notes` below
            for details on the shape of `inverse_metric_field`.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        in_chunks: bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        pbar : bool, default True
            Show a progress bar when ``in_chunks=True``.
        pbar_kwargs : dict, optional
            Extra keyword arguments forwarded to *tqdm*.
        **kwargs
            Passed straight to the low-level metric-contraction kernels
            (e.g. `where=` masks).

        Returns
        -------
        array-like
            A tensor with the specified index raised. Shape equals
            ``broadcast(grid_shape over output_axes) + element_shape``.

        See Also
        --------
        dense_contract_with_metric : General routine for index contraction with a metric.
        dense_lower_index : Inverse operation that lowers an index using the metric.
        """
        return self.dense_contract_with_metric(
            tensor_field,
            field_axes,
            index,
            mode="raise",
            out=out,
            metric_field=inverse_metric_field,
            in_chunks=in_chunks,
            output_axes=output_axes,
            pbar=pbar,
            pbar_kwargs=pbar_kwargs,
            **kwargs,
        )

    def dense_lower_index(
        self: _SupDGMO,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        index: int,
        /,
        out: Optional[ArrayLike] = None,
        metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Lower a single contravariant index of a tensor field.

        This method performs a metric contraction with the metric tensor
        :math:`g_{\mu\nu}` to convert a contravariant (upper) index to covariant (lower)
        form. The `index` argument specifies which slot of the tensor should be lowered.

        This is a specialized wrapper around :meth:`dense_contract_with_metric`
        with ``mode='lower'``.

        .. hint::
            Use this method to convert a component of a vector or higher-rank tensor
            into covariant form in curved coordinate systems.

        Parameters
        ----------
        tensor_field : array-like
            The input tensor field. Its shape must be compatible with the grid dimensions
            over `field_axes`, followed by one or more component axes representing the tensor rank.

        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        index : int
            The index of the tensor field to raise. This should range from 0 to `rank` and is
            measured from the last spatial dimension of the tensor field.
        metric_field : array-like, optional
            A buffer containing the metric field :math:`g_{\mu\nu}`. `metric_field`
            can be provided to improve computation speed (by avoiding computing it in stride);
            however, it is not required.

            The metric can be derived from the coordinate system when this
            argument is not provided. See `Notes` below
            for details on the shape of `metric_field`.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        in_chunks: bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        pbar : bool, default True
            Show a progress bar when ``in_chunks=True``.
        pbar_kwargs : dict, optional
            Extra keyword arguments forwarded to *tqdm*.
        **kwargs
            Passed straight to the low-level metric-contraction kernels
            (e.g. `where=` masks).

        Returns
        -------
        array-like
            A tensor with the specified index lowered. Shape equals
            ``broadcast(grid_shape over output_axes) + element_shape``.

        See Also
        --------
        dense_contract_with_metric : General routine for index contraction with a metric.
        dense_raise_index : Inverse operation that raises an index using the inverse metric.
        """
        return self.dense_contract_with_metric(
            tensor_field,
            field_axes,
            index,
            mode="lower",
            out=out,
            metric_field=metric_field,
            in_chunks=in_chunks,
            output_axes=output_axes,
            pbar=pbar,
            pbar_kwargs=pbar_kwargs,
            **kwargs,
        )

    # ======================================= #
    # Gradient Methods                        #
    # ======================================= #
    # These methods are used for computing the gradient
    # on grids.
    def dense_covariant_gradient(
        self: _SupDGMO,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the covariant gradient of a dense-representation tensor field on a grid.

        For a tensor field :math:`T_{\ldots}^{\ldots}({\bf x})`, the covariant gradient is the
        rank :math:`rank(T)+1` tensor :math:`T_{\ldots\mu}^{\ldots}({\bf x})` such that

        .. math::

            T_{\ldots \mu}^{\ldots}({\bf x}) = \partial_\mu T_{\ldots}^{\ldots}({\bf x}).

        Parameters
        ----------
        tensor_field : array-like
            The tensor field on which to compute the partial derivatives. This should be an array-like object
            with a compliant shape (see `Notes` below).
        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `tensor_field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_compute_gradient_covariant`.

        Returns
        -------
        array-like
            The computed partial derivatives. The resulting array will have a field shape matching the grid's
            shape over the `output_axes` and an element shape matching that of `field` but with an additional `(ndim,)`
            sized dimension containing each of the partial derivatives for each index.

        Notes
        -----

        **Broadcasting and Array Shapes**:

        The input ``tensor_field`` must have a very precise shape to be valid for this operation. If the underlying grid
        (**including ghost zones**) has shape ``grid_shape = (G_1,...,G_ndim)``, then the spatial dimensions of the field must
        match exactly ``grid_shape[axes]``. Any remaining dimensions of the ``field`` are treated as densely populated tensor indices
        and therefore must **each** have :attr:`~grids.base.GridBase.ndim` elements.

        Thus, if a ``(G1,G3,ndim,ndim,...)`` field is passed with ``field_axes = ['x','z']`` (in cartesian coordinates), then
        the resulting output array will have shape ``(G1,G3,ndim,ndim,...,ndim)`` and ``out[...,1] == 0`` because the field does not
        contain any variability over the ``y`` variable.

        If ``output_axes`` is specified, then that resulting grid will be broadcast to any additional grid axes necessary.

        When ``out`` is specified, it must match (exactly) the expected output buffer shape or an error will arise.

        **Chunking Semantics**:

        When ``in_chunks=True``, chunking is enabled for this operation. In that case, the ``out`` buffer is filled
        iteratively by performing computations on each chunk of the grid over the specified `output_axes`. When the
        computation is performed, each chunk is extracted with an additional 1-cell halo to ensure that :func:`numpy.gradient`
        attains its maximal accuracy on each chunk.

        .. note::

            Chunking is (generally) only useful when ``out`` and ``tensor_field`` are array-like lazy-loaded buffers like
            HDF5 datasets. In those cases, the maximum memory load is only that required to load each individual chunk.
            If the ``out`` buffer is not specified, it is allocated fully anyways, making chunking somewhat redundant.


        See Also
        --------
        dense_element_wise_partial_derivatives: Generic form for general array-valued fields.
        dense_contravariant_gradient: Contravariant gradient of a tensor field.
        ~differential_geometry.dense_ops.dense_compute_gradient_covariant: Low-level callable version.

        Examples
        --------

        **Covariant Gradient of a Scalar Field**

        The easiest example is the derivative of a generic scalar field.

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import CartesianCoordinateSystem2D
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = CartesianCoordinateSystem2D()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]],center='cell')
            >>>
            >>> # Create the field
            >>> X,Y = grid.compute_domain_mesh(origin='global')
            >>> Z = np.sin((X**2+Y**2))
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives = grid.dense_covariant_gradient(Z,['x','y'])
            >>>
            >>> fig,axes = plt.subplots(1,3,sharey=True,sharex=True,figsize=(7,3))
            >>> _ = axes[0].imshow(Z.T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1].imshow(derivatives[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[2].imshow(derivatives[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> plt.show()

        **Derivatives of a vector field**:

        Similarly, this can be applied to vector field (or more general tensor field).

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import CartesianCoordinateSystem2D
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = CartesianCoordinateSystem2D()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]],center='cell')
            >>>
            >>> # Create the field
            >>> X,Y = grid.compute_domain_mesh(origin='global')
            >>> Z = np.stack([np.sin((X**2+Y**2)),np.sin(5*(X**2+Y**2))],axis=-1) # (504,504,2)
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives = grid.dense_covariant_gradient(Z,['x','y'])
            >>>
            >>> fig,axes = plt.subplots(2,3,sharey=True,sharex=True,figsize=(7,6))
            >>> _ = axes[0,0].imshow(Z[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[0,1].imshow(derivatives[...,0,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[0,2].imshow(derivatives[...,0,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1,0].imshow(Z[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1,1].imshow(derivatives[...,1,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
            >>> _ = axes[1,2].imshow(derivatives[...,1,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
            >>> plt.show()

        **Expanding to output axes**:

        In some cases, you might have a field :math:`T_{ijk\ldots}(x,y)` and you may need
        :math:`\partial_\mu T_{ijk\ldots}(x,y,z)`. This can be achieved by declaring the `output_axes`
        argument.

        >>> from pymetric.coordinates import CartesianCoordinateSystem3D
        >>>
        >>> # Create the coordinate system
        >>> cs = CartesianCoordinateSystem3D()
        >>>
        >>> # Create the grid
        >>> grid = UniformGrid(cs,[[-1,-1,-1],[1,1,1]],[50,50,50],chunk_size=[5,5,5],ghost_zones=[2,2,2],center='cell')
        >>>
        >>> # Create the field
        >>> X,Y = grid.compute_domain_mesh(origin='global',axes=['x','y'])
        >>> Z = np.stack([np.sin((X**2+Y**2)),np.sin(5*(X**2+Y**2)),np.zeros_like(X)],axis=-1) # (54,54,3)
        >>>
        >>> # Compute the partial derivatives.
        >>> derivatives = grid.dense_covariant_gradient(Z,['x','y'],output_axes=['x','y','z'])
        >>> derivatives.shape
        (54, 54, 54, 3, 3)

        **Inconsistent Input Field**:

        Note that in the previous example, we required

        .. code-block:: python

            Z = np.stack([np.sin((X**2+Y**2)),np.sin(5*(X**2+Y**2)),np.zeros_like(X)],axis=-1) # (54,54,3)

        to have an additional element. This is because :meth:`dense_covariant_gradient` requires dense representations. The
        same result can be achieved under less strict conventions with :meth:`dense_element_wise_partial_derivatives`.

        .. code-block:: python

            Z = np.stack([np.sin((X**2+Y**2)),np.sin(5*(X**2+Y**2))],axis=-1) # (54,54,3)
            derivatives = grid.dense_covariant_gradient(Z,['x','y'],output_axes=['x','y','z'])
            ValueError: Incompatible full field shape.
              Expected: (54, 54, 3)
              Found   : (54, 54, 2)

        """
        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        field_axes, output_axes, field_axes_indices, _ = self._set_input_output_axes(
            field_axes, output_axes=output_axes
        )

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        rank = tensor_field.ndim - len(field_axes)
        element_shape = tensor_field.shape[tensor_field.ndim - rank :]
        element_shape_out = tuple(element_shape) + (self.ndim,)

        # Check the input field first.
        self.check_field_shape(
            tensor_field.shape, axes=field_axes, field_shape=(self.ndim,) * rank
        )

        out = self._prepare_output_buffer(
            output_axes,
            output_element_shape=element_shape_out,
            out=out,
            include_ghosts=True,
            dtype=tensor_field.dtype,
        )

        # --- Perform Gradient Operation --- #
        if in_chunks:
            # Operation performed in chunks.
            self._ensure_supports_chunking()

            for chunk_slices in self.iter_chunk_slices(
                axes=field_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Cut the slice out of the input tensor field.
                tensor_field_chunk = self.broadcast_array_to_axes(
                    tensor_field[(*chunk_slices, ...)],
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Construct the coordinates from the slices.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=field_axes, origin="global", __validate__=False
                )

                # Compute the covariant gradient.
                cov_grad = dense_gradient_covariant(
                    tensor_field_chunk,
                    rank,
                    self.ndim,
                    *coordinates,
                    output_indices=field_axes_indices,  # ensures we place grads into right slots.
                    edge_order=edge_order,
                    **kwargs,
                )

                # Assign to the output.
                out[(*chunk_slices, ...)] = cov_grad

        else:
            # Operation performed in single call. Compute all
            # coordinates over the entire domain and then compute.
            # Cast the input field to the output axes.
            reshaped_tensor_field = self.broadcast_array_to_axes(
                tensor_field, axes_in=field_axes, axes_out=output_axes
            )

            # Compute the coordinates.
            coordinates = self.compute_domain_coords(
                axes=field_axes, origin="global", __validate__=False
            )

            # Compute covariant gradient.
            dense_gradient_covariant(
                reshaped_tensor_field,
                rank,
                self.ndim,
                *coordinates,
                output_indices=field_axes_indices,  # ensures we place grads into right slots.
                edge_order=edge_order,
                out=out,
                **kwargs,
            )

        # Return the output buffer.
        return out

    def dense_contravariant_gradient(
        self: _SupDGMO,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the contravariant gradient of a dense-representation tensor field on a grid.

        For a tensor field :math:`T_{\ldots}^{\ldots}({\bf x})`, the contravariant gradient is the
        rank :math:`rank(T)+1` tensor :math:`T_{\ldots\mu}^{\ldots}({\bf x})` such that

        .. math::

            T_{\ldots}^{\ldots\mu}({\bf x}) = g^{\mu\nu}\partial_\nu T_{\ldots}^{\ldots}({\bf x}).

        Parameters
        ----------
        tensor_field : array-like
            The tensor field on which to compute the partial derivatives. This should be an array-like object
            with a compliant shape (see `Notes` below).
        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `tensor_field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        inverse_metric_field : array-like, optional
            A buffer containing the inverse metric field :math:`g^{\mu\nu}`. `inverse_metric_field`
            can be provided to improve computation speed (by avoiding computing it in stride);
            however, it is not required.

            The inverse metric can be derived from the coordinate system when this
            argument is not provided. See `Notes` below
            for details on the shape of `inverse_metric_field`.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_compute_gradient_contravariant_full`.

        Returns
        -------
        array-like
            The computed partial derivatives. The resulting array will have a field shape matching the grid's
            shape over the `output_axes` and an element shape matching that of `field` but with an additional `(ndim,)`
            sized dimension containing each of the partial derivatives for each index.

        Notes
        -----

        **Broadcasting and Array Shapes**:

        The input ``tensor_field`` must have a very precise shape to be valid for this operation. If the underlying grid
        (**including ghost zones**) has shape ``grid_shape = (G_1,...,G_ndim)``, then the spatial dimensions of the field must
        match exactly ``grid_shape[axes]``. Any remaining dimensions of the ``field`` are treated as densely populated tensor indices
        and therefore must **each** have :attr:`~grids.base.GridBase.ndim` elements.

        Thus, if a ``(G1,G3,ndim,ndim,...)`` field is passed with ``field_axes = ['x','z']`` (in cartesian coordinates), then
        the resulting output array will have shape ``(G1,G3,ndim,ndim,...,ndim)`` and ``out[...,1] == 0`` because the field does not
        contain any variability over the ``y`` variable.

        If ``output_axes`` is specified, then that resulting grid will be broadcast to any additional grid axes necessary.

        When ``out`` is specified, it must match (exactly) the expected output buffer shape or an error will arise.

        **Chunking Semantics**:

        When ``in_chunks=True``, chunking is enabled for this operation. In that case, the ``out`` buffer is filled
        iteratively by performing computations on each chunk of the grid over the specified `output_axes`. When the
        computation is performed, each chunk is extracted with an additional 1-cell halo to ensure that :func:`numpy.gradient`
        attains its maximal accuracy on each chunk.

        .. note::

            Chunking is (generally) only useful when ``out`` and ``tensor_field`` are array-like lazy-loaded buffers like
            HDF5 datasets. In those cases, the maximum memory load is only that required to load each individual chunk.
            If the ``out`` buffer is not specified, it is allocated fully anyways, making chunking somewhat redundant.

        **Inverse Metric**:

        In most cases, the inverse metric is computed by the coordinate system behind the scenes; however, it may be
        provided directly in cases where doing so is convenient. If this is done, the provided field must have a
        spatial portion corresponding to the grid's shape (including ghost zones) over the **output axes**. Depending on
        the coordinate system, the provided metric may either be a rank-2 array (non-orthogonal coordinate systems) or
        a rank-1 array (orthogonal coordinate systems) in which each element corresponds to the diagonal element. The
        correct low-level callable is determined based on the coordinate system's type.

        See Also
        --------
        dense_element_wise_partial_derivatives: Generic form for general array-valued fields.
        dense_covariant_gradient: Covariant gradient of a tensor field.
        ~differential_geometry.dense_ops.dense_compute_gradient_contravariant_full: Low-level callable version (full metric)
        ~differential_geometry.dense_ops.dense_compute_gradient_contravariant_diag: Low-level callable version (diag metric)

        Examples
        --------

        **Contravariant Gradient of a Scalar Field**

        The easiest example is the derivative of a generic scalar field. In cartesian coordinates, this should
        behave exactly the same as the covariant gradient.

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import CartesianCoordinateSystem2D
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = CartesianCoordinateSystem2D()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]])
            >>>
            >>> # Create the field
            >>> X,Y = grid.compute_domain_mesh(origin='global')
            >>> Z = np.sin((X**2+Y**2))
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives = grid.dense_contravariant_gradient(Z,['x','y'])
            >>>
            >>> fig,axes = plt.subplots(1,3,sharey=True,sharex=True,figsize=(7,3))
            >>> _ = axes[0].imshow(Z.T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1].imshow(derivatives[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[2].imshow(derivatives[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> plt.show()

        In the more interesting case, we might consider the contravariant gradient in a non-cartesian coordinate system!
        Let

        .. math::

            \phi(r,\theta) = r^2 \cos(2\theta).

        The covariant gradient is

        .. math::

            \nabla_\mu \phi = \left[ 2r \cos(2\theta), \; -2r^2 \sin(2\theta) \right],

        while the contravariant gradient is

        .. math::

            \nabla^\mu \phi = g^{\mu\mu} \nabla_\mu \phi = \left[ 2r \cos(2\theta),\; -2\sin(2\theta) \right].

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import SphericalCoordinateSystem
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[0,1],[0,np.pi],[0,2*np.pi]],
            ...                   [500,50,50],
            ...                   chunk_size=[50,50,50],
            ...                   ghost_zones=[[2,2],[2,2],[2,2]])
            >>>
            >>> # Create the field
            >>> R, THETA = grid.compute_domain_mesh(origin='global',axes=['r','theta'])
            >>> Z = (R**2) * np.cos(2*THETA)
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives_cont = grid.dense_contravariant_gradient(Z,['r','theta'])
            >>> derivatives_co = grid.dense_covariant_gradient(Z,['r','theta'])
            >>>
            >>> fig,axes = plt.subplots(2,3,sharey=True,sharex=True,figsize=(7,6))
            >>> _ = axes[0,0].imshow(Z.T                 ,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[0,1].imshow(derivatives_co[...,0].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[0,2].imshow(derivatives_cont[...,0].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[1,0].imshow(Z.T                 ,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[1,1].imshow(derivatives_co[...,1].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[1,2].imshow(derivatives_cont[...,1].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> plt.show()
        """
        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        field_axes, output_axes, field_axes_indices, _ = self._set_input_output_axes(
            field_axes, output_axes=output_axes
        )
        differential_axes_indices = np.asarray(
            [_i for _i, _a in enumerate(output_axes) if _a in field_axes]
        )
        fixed_axes, fixed_values = self._compute_fixed_axes_and_values(
            free_axes=output_axes
        )

        # --- Determine the correct operator --- #
        # We need to check the metric shape to determine.
        if len(self.__cs__.metric_tensor_symbol.shape) == 1:
            __op__ = dense_gradient_contravariant_diag
        else:
            __op__ = dense_gradient_contravariant_full

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        tensor_field_ndim, tensor_field_shape = tensor_field.ndim, tensor_field.shape
        rank = tensor_field_ndim - len(field_axes)
        tensor_field_element_shape = tensor_field_shape[tensor_field_ndim - rank :]
        output_field_element_shape = tuple(tensor_field_element_shape) + (self.ndim,)
        out = self._prepare_output_buffer(
            output_axes,
            output_element_shape=output_field_element_shape,
            out=out,
            include_ghosts=True,
            dtype=tensor_field.dtype,
        )

        # --- Perform Gradient Operation --- #
        if in_chunks:
            # Operation performed in chunks.
            self._ensure_supports_chunking()

            # -- Create the chunk fetchers -- #
            # These allow us to seamlessly generate the correct
            # metric field over the chunks regardless of it being
            # pre-specified.
            _metric_fetcher_ = self._make_expression_chunk_fetcher(
                "inverse_metric_field",
                fixed_values,
                value=inverse_metric_field,
            )

            # Cycle through chunks.
            for chunk_slices in self.iter_chunk_slices(
                axes=output_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Compute coordinates. Cut down to the correct set of coordinates and
                # slices for the input field axes.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=output_axes, origin="global", __validate__=False
                )
                differential_coordinates = [
                    coordinates[i] for i in differential_axes_indices
                ]
                differential_chunk_slices = [
                    chunk_slices[i] for i in differential_axes_indices
                ]

                tensor_field_chunk = self.broadcast_array_to_axes(
                    tensor_field[(*differential_chunk_slices, ...)],
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Attempt to build the metric tensor.
                inverse_metric_field_chunk = _metric_fetcher_(chunk_slices, coordinates)

                # Compute the covariant gradient.
                out[(*chunk_slices, ...)] = __op__(
                    tensor_field_chunk,
                    inverse_metric_field_chunk,
                    rank,
                    self.ndim,
                    *differential_coordinates,
                    field_axes=differential_axes_indices,
                    output_indices=field_axes_indices,  # ensures we place grads into right slots.
                    edge_order=edge_order,
                    **kwargs,
                )

        else:
            # Operation performed in single call. Compute all
            # coordinates over the entire domain and then compute.
            # Cast the input field to the output axes.
            reshaped_tensor_field = self.broadcast_array_to_axes(
                tensor_field, axes_in=field_axes, axes_out=output_axes
            )

            # Compute the coordinates over the output axes. The differential coordinates
            # are then also constructed.
            coordinates = self.compute_domain_coords(
                axes=output_axes, origin="global", __validate__=False
            )
            differential_coordinates = [
                coordinates[i] for i in differential_axes_indices
            ]

            # Ensure that the inverse metric tensor has been computed
            # on the coordinates.
            inverse_metric_field = self.__cs__.requires_expression_from_coordinates(
                inverse_metric_field,
                "inverse_metric_tensor",
                coordinates,
                fixed_axes=fixed_values,
            )

            # Compute covariant gradient.
            __op__(
                reshaped_tensor_field,
                inverse_metric_field,
                rank,
                self.ndim,
                *differential_coordinates,
                field_axes=differential_axes_indices,
                output_indices=field_axes_indices,  # ensures we place grads into right slots.
                edge_order=edge_order,
                out=out,
                **kwargs,
            )

        # Return the output buffer.
        return out

    def dense_gradient(
        self: _SupDGMO,
        tensor_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        basis: Optional[Literal["contravariant", "covariant"]] = "covariant",
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the element-wise gradient of a tensor field over a grid.

        :meth:`dense_gradient` is a wrapper around the two basis-dependent gradient methods
        (:meth:`dense_contravariant_gradient` and :meth:`dense_covariant_gradient`) and uses the `basis` input
        to determine the correct method to direct the call sequence to.

        Parameters
        ----------
        tensor_field : array-like
            The tensor field on which to compute the partial derivatives. This should be an array-like object
            with a compliant shape (see `Notes` below).
        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `tensor_field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        inverse_metric_field : array-like, optional
            A buffer containing the inverse metric field :math:`g^{\mu\nu}`. `inverse_metric_field`
            can be provided to improve computation speed (by avoiding computing it in stride);
            however, it is not required.

            The inverse metric can be derived from the coordinate system when this
            argument is not provided. See `Notes` below
            for details on the shape of `inverse_metric_field`.
        basis: {'contravariant', 'covariant'}, optional
            The basis in which to compute the gradient tensor. If ``"covariant"``, then the gradient tensor will simply
            be the element-wise partial derivatives. If ``"contravariant"``, then the covariant solution will have its index
            raised using the `inverse_metric_field`.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_compute_gradient_covariant`,
            or :func:`~differential_geometry.dense_ops.dense_compute_gradient_contravariant_full`.

        Returns
        -------
        array-like
            The computed partial derivatives. The resulting array will have a field shape matching the grid's
            shape over the `output_axes` and an element shape matching that of `field` but with an additional `(ndim,)`
            sized dimension containing each of the partial derivatives for each index.

        Notes
        -----

        **Broadcasting and Array Shapes**:

        The input ``tensor_field`` must have a very precise shape to be valid for this operation. If the underlying grid
        (**including ghost zones**) has shape ``grid_shape = (G_1,...,G_ndim)``, then the spatial dimensions of the field must
        match exactly ``grid_shape[axes]``. Any remaining dimensions of the ``field`` are treated as densely populated tensor indices
        and therefore must **each** have :attr:`~grids.base.GridBase.ndim` elements.

        Thus, if a ``(G1,G3,ndim,ndim,...)`` field is passed with ``field_axes = ['x','z']`` (in cartesian coordinates), then
        the resulting output array will have shape ``(G1,G3,ndim,ndim,...,ndim)`` and ``out[...,1] == 0`` because the field does not
        contain any variability over the ``y`` variable.

        If ``output_axes`` is specified, then that resulting grid will be broadcast to any additional grid axes necessary.

        When ``out`` is specified, it must match (exactly) the expected output buffer shape or an error will arise.

        **Chunking Semantics**:

        When ``in_chunks=True``, chunking is enabled for this operation. In that case, the ``out`` buffer is filled
        iteratively by performing computations on each chunk of the grid over the specified `output_axes`. When the
        computation is performed, each chunk is extracted with an additional 1-cell halo to ensure that :func:`numpy.gradient`
        attains its maximal accuracy on each chunk.

        .. note::

            Chunking is (generally) only useful when ``out`` and ``tensor_field`` are array-like lazy-loaded buffers like
            HDF5 datasets. In those cases, the maximum memory load is only that required to load each individual chunk.
            If the ``out`` buffer is not specified, it is allocated fully anyways, making chunking somewhat redundant.

        **Inverse Metric**:

        In most cases, the inverse metric is computed by the coordinate system behind the scenes; however, it may be
        provided directly in cases where doing so is convenient. If this is done, the provided field must have a
        spatial portion corresponding to the grid's shape (including ghost zones) over the **output axes**. Depending on
        the coordinate system, the provided metric may either be a rank-2 array (non-orthogonal coordinate systems) or
        a rank-1 array (orthogonal coordinate systems) in which each element corresponds to the diagonal element. The
        correct low-level callable is determined based on the coordinate system's type.

        See Also
        --------
        dense_element_wise_partial_derivatives: Generic form for general array-valued fields.
        dense_covariant_gradient: Covariant gradient of a tensor field.
        dense_contravariant_gradient: Contravariant gradient of a tensor field.
        ~differential_geometry.dense_ops.dense_compute_gradient_contravariant_full: Low-level callable version (full metric)
        ~differential_geometry.dense_ops.dense_compute_gradient_contravariant_diag: Low-level callable version (diag metric)

        Examples
        --------

        **Contravariant Gradient of a Scalar Field**

        The easiest example is the derivative of a generic scalar field. In cartesian coordinates, this should
        behave exactly the same as the covariant gradient.

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import CartesianCoordinateSystem2D
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = CartesianCoordinateSystem2D()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]])
            >>>
            >>> # Create the field
            >>> X,Y = grid.compute_domain_mesh(origin='global')
            >>> Z = np.sin((X**2+Y**2))
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives = grid.dense_gradient(Z,['x','y'],basis='contravariant')
            >>>
            >>> fig,axes = plt.subplots(1,3,sharey=True,sharex=True,figsize=(7,3))
            >>> _ = axes[0].imshow(Z.T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[1].imshow(derivatives[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> _ = axes[2].imshow(derivatives[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
            >>> plt.show()

        In the more interesting case, we might consider the contravariant gradient in a non-cartesian coordinate system!
        Let

        .. math::

            \phi(r,\theta) = r^2 \cos(2\theta).

        The covariant gradient is

        .. math::

            \nabla_\mu \phi = \left[ 2r \cos(2\theta), \; -2r^2 \sin(2\theta) \right],

        while the contravariant gradient is

        .. math::

            \nabla^\mu \phi = g^{\mu\mu} \nabla_\mu \phi = \left[ 2r \cos(2\theta),\; -2\sin(2\theta) \right].

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import SphericalCoordinateSystem
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[0,1],[0,np.pi],[0,2*np.pi]],
            ...                   [500,50,50],
            ...                   chunk_size=[50,50,50],
            ...                   ghost_zones=[[2,2],[2,2],[2,2]])
            >>>
            >>> # Create the field
            >>> R, THETA = grid.compute_domain_mesh(origin='global',axes=['r','theta'])
            >>> Z = (R**2) * np.cos(2*THETA)
            >>>
            >>> # Compute the partial derivatives.
            >>> derivatives_cont = grid.dense_gradient(Z,['r','theta'],basis='contravariant')
            >>> derivatives_co = grid.dense_gradient(Z,['r','theta'],basis='covariant')
            >>>
            >>> fig,axes = plt.subplots(2,3,sharey=True,sharex=True,figsize=(7,6))
            >>> _ = axes[0,0].imshow(Z.T                 ,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[0,1].imshow(derivatives_co[...,0].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[0,2].imshow(derivatives_cont[...,0].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[1,0].imshow(Z.T                 ,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[1,1].imshow(derivatives_co[...,1].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> _ = axes[1,2].imshow(derivatives_cont[...,1].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
            >>> plt.show()
        """
        # Distinguish the basis and proceed to the low-level callable
        # depending on which basis is specified.
        if basis == "covariant":
            try:
                return self.dense_covariant_gradient(
                    tensor_field,
                    field_axes,
                    out=out,
                    in_chunks=in_chunks,
                    edge_order=edge_order,
                    output_axes=output_axes,
                    pbar=pbar,
                    pbar_kwargs=pbar_kwargs,
                    **kwargs,
                )
            except Exception as e:
                raise ValueError(f"Failed to compute covariant gradient: {e}") from e
        elif basis == "contravariant":
            try:
                return self.dense_contravariant_gradient(
                    tensor_field,
                    field_axes,
                    out=out,
                    inverse_metric_field=inverse_metric_field,
                    in_chunks=in_chunks,
                    edge_order=edge_order,
                    output_axes=output_axes,
                    pbar=pbar,
                    pbar_kwargs=pbar_kwargs,
                    **kwargs,
                )
            except Exception as e:
                raise ValueError(f"Failed to compute covariant gradient: {e}") from e

        else:
            raise ValueError(
                f"`basis` must be 'covariant' or 'contravariant', not '{basis}'."
            )

    # ======================================= #
    # Divergence Methods                      #
    # ======================================= #
    # These methods are used to compute the divergence of
    # a field.
    def dense_vector_divergence_contravariant(
        self: _SupDGMO,
        vector_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        Dterm_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the divergence of a contravariant vector field on a grid.

        For a contravariant vector field :math:`V^\mu`, the divergence is given by:

        .. math::

            \nabla \cdot V = \frac{1}{\rho} \partial_\mu(\rho V^\mu)
                           = \partial_\mu V^\mu + V^\mu \frac{\partial_\mu \rho}{\rho}

        This expanded form is used for improved numerical stability and implemented
        as a sum of a standard derivative and a geometry-aware "D-term".

        Parameters
        ----------
        vector_field : array-like
            The vector field on which to compute the divergence. This should be an array-like object
            with a compliant shape (see `Notes` below).
        field_axes : list of str
            The coordinate axes over which the `field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        Dterm_field : array-like, optional
            The D-term field for the specific coordinate system. This can be specified to improve computation speed; however,
            it can also be derived directly from the grid's coordinate system. If it is provided, it should be compliant with
            the shaping / broadcasting rules (see `Notes`).
        derivative_field : array-like, optional
            The first derivatives of the `vector_field`. `derivative_field` can be specified to improve computational speed or
            to improve accuracy if the derivatives are known analytically; however, they can also be computed numerically if
            not provided. If `derivative_field` is provided, it must comply with the shaping / broadcasting rules (see `Notes`).
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_vector_divergence_contravariant`.

        Returns
        -------
        array-like
            The computed partial derivatives. The resulting array will have a field shape matching the grid's
            shape over the `output_axes` and an element shape matching that of `field` but with an additional `(ndim,)`
            sized dimension containing each of the partial derivatives for each index.

        Notes
        -----

        **Broadcasting and Array Shapes**:

        The input ``vector_field`` must have a very precise shape to be valid for this operation. If the underlying grid
        (**including ghost zones**) has shape ``grid_shape = (G_1,...,G_ndim)``, then the spatial dimensions of the field must
        match exactly ``grid_shape[axes]``. ``vector_field`` should have 1 additional dimension representing the vector components
        which must have a size of :attr:`~grids.base.GridBase.ndim`. The resulting output array will be a scalar field over
        the relevant grid axes.

        Thus, if a ``(G1,G3,ndim)`` field is passed with ``field_axes = ['x','z']`` (in cartesian coordinates), then
        the resulting output array will have shape ``(G1,G3)``.

        If ``output_axes`` is specified, then that resulting grid will be broadcast to any additional grid axes necessary.

        If either ``derivative_field`` or ``Dterm_field`` are not provided, then they must each be provided over the ``output_axes``
        (if they are specified, otherwise ``field_axes``). The ``Dterm_field`` should have 1 additional dimension of size
        :attr:`~grid.base.GridBase.ndim` containing each of the elements of the covector field. The ``derivative_field`` must
        also be specified over the ``output_axes`` (if they are specified, otherwise ``field_axes``), but can have any number
        of elements in its 1 additional dimension (corresponding to the set of non-zero derivatives).

        When ``out`` is specified, it must match (exactly) the expected output buffer shape or an error will arise.

        **Chunking Semantics**:

        When ``in_chunks=True``, chunking is enabled for this operation. In that case, the ``out`` buffer is filled
        iteratively by performing computations on each chunk of the grid over the specified `output_axes`. When the
        computation is performed, each chunk is extracted with an additional 1-cell halo to ensure that :func:`numpy.gradient`
        attains its maximal accuracy on each chunk.

        .. note::

            Chunking is (generally) only useful when ``out`` and ``vector_field`` are array-like lazy-loaded buffers like
            HDF5 datasets. In those cases, the maximum memory load is only that required to load each individual chunk.
            If the ``out`` buffer is not specified, it is allocated fully anyways, making chunking somewhat redundant.

        Examples
        --------

        In Spherical Coordinates, the divergence is

        .. math::

            \nabla \cdot V = \frac{1}{r^2}\partial_r\left(r^2V^r\right) + \frac{1}{\sin \theta} \partial_\theta \left(\sin \theta V^\theta\right)
            + \partial_\phi V^\phi.

        As such, if :math:`V^\mu = \left(r cos \theta,\; \sin\theta,\; 0\right)`, then

        .. math::

            \nabla \cdot V = \frac{\cos \theta}{r^2}\partial_r r^3 + \frac{1}{\sin \theta}\partial_\theta \sin^2\theta = 5 \cos \theta.

        To see this numerically, we can do the following:

        .. plot::
            :include-source:

            >>> from pymetric.grids.core import UniformGrid
            >>> from pymetric.coordinates import SphericalCoordinateSystem
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Create the coordinate system
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>> # Create the grid
            >>> grid = UniformGrid(cs,[[0,0,0],[1,np.pi,2*np.pi]],
            ...                   [500,50,50],
            ...                   chunk_size=[50,50,50],
            ...                   ghost_zones=[2,2,2],center='cell')
            >>>
            >>> # Create the field
            >>> R, THETA = grid.compute_domain_mesh(origin='global',axes=['r','theta'])
            >>> Z = np.stack([R * np.cos(THETA), np.sin(THETA), np.zeros_like(R)],axis=-1)
            >>> Z.shape
            (504, 54, 3)
            >>>
            >>> # Compute the divergence.
            >>> DivZ = grid.dense_vector_divergence_contravariant(Z,['r','theta'],in_chunks=True)
            >>>
            >>> fig,axes = plt.subplots(1,1)
            >>> _ = axes.imshow(DivZ.T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
            >>> plt.show()
        """
        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        (
            field_axes,
            output_axes,
            field_axes_indices,
            output_axes_indices,
        ) = self._set_input_output_axes(field_axes, output_axes=output_axes)
        differential_axes_indices = np.asarray(
            [_i for _i, _a in enumerate(output_axes) if _a in field_axes]
        )
        fixed_axes, fixed_values = self._compute_fixed_axes_and_values(
            free_axes=output_axes
        )

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        out = self._prepare_output_buffer(
            output_axes, out=out, include_ghosts=True, dtype=vector_field.dtype
        )

        # --- Perform the operation --- #
        if in_chunks:
            # Compute the divergence in chunks. Broadly speaking, this proceeds in
            # the following order of operations:
            # 1. Ensure that chunking is supported.
            # 2. Determine if we are given the D-term and (if so), mark that
            #    we don't need to try to compute on each round.
            self._ensure_supports_chunking()

            # Determine if we need to try to generate the
            # D-term field for each chunk or if we can just grab it.
            _Dterm_generator_ = self._make_expression_chunk_fetcher(
                "Dterm", fixed_values, value=Dterm_field
            )

            _has_derivative = derivative_field is not None

            # Iterate through each of the chunk slices in the
            # output space.
            for chunk_slices in self.iter_chunk_slices(
                axes=output_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Compute coordinates. Cut down to the correct set of coordinates and
                # slices for the input field axes.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=output_axes, origin="global", __validate__=False
                )
                differential_coordinates = [
                    coordinates[i] for i in differential_axes_indices
                ]
                differential_chunk_slices = [
                    chunk_slices[i] for i in differential_axes_indices
                ]

                # Broadcast the vector field onto the chunk.
                vector_field_chunk = self.broadcast_array_to_axes(
                    vector_field[(*differential_chunk_slices, ...)],
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Attempt to build the D-term if it is needed.
                Dterm_chunk = _Dterm_generator_(chunk_slices, coordinates)

                # If we have the derivative field, we need to cut into it.
                if _has_derivative:
                    derivative_field_broadcast = self.broadcast_array_to_axes(
                        derivative_field[(*differential_chunk_slices, ...)],
                        axes_in=field_axes,
                        axes_out=output_axes,
                    )
                else:
                    derivative_field_broadcast = None

                # Compute the covariant gradient.
                out[(*chunk_slices, ...)] = dense_vector_divergence_contravariant(
                    vector_field_chunk,
                    Dterm_chunk,
                    *differential_coordinates,
                    derivative_field=derivative_field_broadcast,
                    field_axes=output_axes_indices,
                    derivative_axes=differential_axes_indices,
                    edge_order=edge_order,
                    **kwargs,
                )
        else:
            # Perform the operation in one pass. Broadly, the steps are
            # 1. Broadcast the field to the output axes for consistency.
            # 2. Compute the coordinates in the output axes space.
            # 3. Compute the Dterm_field if it is not provided.
            # 4. Broadcast the derivative field if it is provided.

            # Broadcast to output axes. This will be (F1, ..., 1, ... FM) or something
            # of the sort.
            vector_field_broadcast = self.broadcast_array_to_axes(
                vector_field, axes_in=field_axes, axes_out=output_axes
            )
            if derivative_field is not None:
                derivative_field_broadcast = self.broadcast_array_to_axes(
                    derivative_field, axes_in=field_axes, axes_out=output_axes
                )
            else:
                derivative_field_broadcast = None

            # Compute the output coordinates so that we can
            # perform the differentiation operation.
            coordinates = self.compute_domain_coords(
                axes=output_axes, origin="global", __validate__=False
            )
            differential_coordinates = [
                coordinates[i] for i in differential_axes_indices
            ]

            # Create the D-term field over the free coordinates.
            Dterm_field = self.__cs__.requires_expression_from_coordinates(
                Dterm_field,
                "Dterm",
                coordinates,
                fixed_axes=fixed_values,
            )

            dense_vector_divergence_contravariant(
                vector_field_broadcast,
                Dterm_field,
                *differential_coordinates,
                derivative_field=derivative_field_broadcast,
                field_axes=output_axes_indices,
                derivative_axes=differential_axes_indices,
                edge_order=edge_order,
                out=out,
                **kwargs,
            )

        return out

    def dense_vector_divergence_covariant(
        self: _SupDGMO,
        vector_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        Dterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the divergence of a covariant vector field on a grid.

        For a covariant vector field :math:`V_\mu`, the divergence is:

        .. math::

            \nabla \cdot V = \frac{1}{\rho} \partial_\mu(\rho g^{\mu\nu} V_\nu)

        Expanded for numerical stability:

        .. math::

            \nabla \cdot V = g^{\mu\nu} V_\nu \frac{\partial_\mu \rho}{\rho}
                           + \partial_\mu(g^{\mu\nu} V_\nu)

        This form is used internally, with precomputed or on-the-fly inverse metrics.

        Parameters
        ----------
        vector_field : array-like
            The vector field on which to compute the divergence. This should be an array-like object
            with a compliant shape (see `Notes` below).
        field_axes : list of str
            The coordinate axes over which the `field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        Dterm_field : array-like, optional
            The D-term field for the specific coordinate system. This can be specified to improve computation speed; however,
            it can also be derived directly from the grid's coordinate system. If it is provided, it should be compliant with
            the shaping / broadcasting rules (see `Notes`).
        inverse_metric_field : array-like, optional
            A buffer containing the inverse metric field :math:`g^{\mu\nu}`. `inverse_metric_field`
            can be provided to improve computation speed (by avoiding computing it in stride);
            however, it is not required.

            The inverse metric can be derived from the coordinate system when this
            argument is not provided. See `Notes` below
            for details on the shape of `inverse_metric_field`.
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_vector_divergence_contravariant`.

        Returns
        -------
        array‑like
            Scalar divergence on the grid.  The shape equals the broadcasted
            grid shape over ``output_axes`` (ghost zones included when the grid
            carries them).

        Notes
        -----
        **Broadcasting & shapes**

        If the full grid (including ghosts) has shape
        ``grid_shape = (G1, …, G_ndim)``, then

        * ``vector_field[..., k]`` must match ``grid_shape[field_axes[k]]``.
        * ``out`` must match ``grid_shape[output_axes]``.
        * ``Dterm_field`` and ``inverse_metric_field`` (when supplied) must be
          broadcast‑compatible with the same grid shape.

        **Chunking semantics**

        When ``in_chunks=True`` the routine iterates over the grid’s stored
        chunks, fetching or generating the necessary D‑terms and inverse
        metric on each sub‑domain.  A one‑cell halo is automatically included
        to maintain gradient accuracy, and progress is reported through
        *tqdm*.

        **Inverse Metric**:

        In most cases, the inverse metric is computed by the coordinate system behind the scenes; however, it may be
        provided directly in cases where doing so is convenient. If this is done, the provided field must have a
        spatial portion corresponding to the grid's shape (including ghost zones) over the **output axes**. Depending on
        the coordinate system, the provided metric may either be a rank-2 array (non-orthogonal coordinate systems) or
        a rank-1 array (orthogonal coordinate systems) in which each element corresponds to the diagonal element. The
        correct low-level callable is determined based on the coordinate system's type.

        Examples
        --------
        Divergence of a covariant field in *2‑D cylindrical coordinates*
        \((r, z)\):

        >>> from pymetric.grids.core import UniformGrid
        >>> from pymetric.coordinates import CylindricalCoordinateSystem
        >>> import numpy as np
        >>>
        >>> # Coordinate system & grid
        >>> cs = CylindricalCoordinateSystem()       # (r, φ, z) – φ suppressed
        >>> grid = UniformGrid(cs,
        ...     [[0,0,0],[1,2*np.pi,1]],
        ...     [400, 10, 200],ghost_zones=1,center='cell')
        >>>
        >>> # Covariant vector: V_r = r, V_z = z
        >>> R, Z = grid.compute_domain_mesh(origin='global', axes=['rho', 'z'])
        >>> Vcov = np.stack([R,np.zeros_like(R), Z], axis=-1)
        >>>
        >>> # Divergence (automatic metric & D‑term)
        >>> div = grid.dense_vector_divergence_covariant(
        ...     Vcov, ['rho', 'z'])
        >>> div.shape
        (402, 202)
        >>> np.isclose(div.mean(),3.0)   # → analytical result for this field is constant 3
        True
        """
        # --- Preparing axes --- #
        # To prepare the axes, we need to ensure that they are standardized and
        # then check for subsets. We also extract the indices so that they
        # can be used for various low-level callables.
        (
            field_axes,
            output_axes,
            field_axes_indices,
            output_axes_indices,
        ) = self._set_input_output_axes(field_axes, output_axes=output_axes)
        differential_axes_indices = np.asarray(
            [_i for _i, _a in enumerate(output_axes) if _a in field_axes]
        )
        fixed_axes, fixed_values = self._compute_fixed_axes_and_values(
            free_axes=output_axes
        )

        # --- Allocate `out` --- #
        # Having now determined the correct output axes, we can
        # simply generate the output. This logic is encapsulated in `_prepare_output_buffer`.
        out = self._prepare_output_buffer(
            output_axes, out=out, include_ghosts=True, dtype=vector_field.dtype
        )

        # --- Determine the correct operator --- #
        # We need to check the metric shape to determine.
        if len(self.__cs__.metric_tensor_symbol.shape) == 1:
            __op__ = dense_vector_divergence_covariant_diag
        else:
            __op__ = dense_vector_divergence_covariant_full

        # --- Perform the operation --- #
        if in_chunks:
            # Compute the divergence in chunks. Broadly speaking, this proceeds in
            # the following order of operations:
            # 1. Ensure that chunking is supported.
            # 2. Determine if we are given the D-term and (if so), mark that
            #    we don't need to try to compute on each round.
            self._ensure_supports_chunking()

            # Determine if we need to try to generate the
            # D-term field for each chunk or if we can just grab it.
            _try_D = Dterm_field is None
            _try_metric = inverse_metric_field is None

            # Iterate through each of the chunk slices in the
            # output space.
            for chunk_slices in self.iter_chunk_slices(
                axes=output_axes,
                include_ghosts=True,
                halo_offsets=1,
                oob_behavior="clip",
                pbar=pbar,
                pbar_kwargs=pbar_kwargs,
            ):
                # Compute coordinates. Cut down to the correct set of coordinates and
                # slices for the input field axes.
                coordinates = self.compute_coords_from_slices(
                    chunk_slices, axes=output_axes, origin="global", __validate__=False
                )
                differential_coordinates = [
                    coordinates[i] for i in differential_axes_indices
                ]
                differential_chunk_slices = [
                    chunk_slices[i] for i in differential_axes_indices
                ]

                # Broadcast the vector field onto the chunk.
                vector_field_chunk = self.broadcast_array_to_axes(
                    vector_field[(*differential_chunk_slices, ...)],
                    axes_in=field_axes,
                    axes_out=output_axes,
                )

                # Attempt to build the D-term if it is needed.
                if _try_D:
                    Dterm_chunk = self.__cs__.compute_expression_from_coordinates(
                        "Dterm", coordinates, fixed_axes=fixed_values
                    )
                else:
                    Dterm_chunk = Dterm_field[(*chunk_slices, ...)]

                # Attempt to build the metric tensor.
                if _try_metric:
                    inverse_metric_field_chunk = (
                        self.__cs__.compute_expression_from_coordinates(
                            "inverse_metric_tensor",
                            coordinates,
                            fixed_axes=fixed_values,
                        )
                    )
                else:
                    inverse_metric_field_chunk = inverse_metric_field[
                        (*chunk_slices, ...)
                    ]

                # Compute the covariant gradient.
                out[(*chunk_slices, ...)] = __op__(
                    vector_field_chunk,
                    Dterm_chunk,
                    inverse_metric_field_chunk,
                    *differential_coordinates,
                    field_axes=output_axes_indices,
                    derivative_axes=differential_axes_indices,
                    edge_order=edge_order,
                    **kwargs,
                )
        else:
            # Perform the operation in one pass. Broadly, the steps are
            # 1. Broadcast the field to the output axes for consistency.
            # 2. Compute the coordinates in the output axes space.
            # 3. Compute the Dterm_field if it is not provided.
            # 4. Broadcast the derivative field if it is provided.

            # Broadcast to output axes. This will be (F1, ..., 1, ... FM) or something
            # of the sort.
            vector_field_broadcast = self.broadcast_array_to_axes(
                vector_field, axes_in=field_axes, axes_out=output_axes
            )

            # Compute the output coordinates so that we can
            # perform the differentiation operation.
            coordinates = self.compute_domain_coords(
                axes=output_axes, origin="global", __validate__=False
            )
            differential_coordinates = [
                coordinates[i] for i in differential_axes_indices
            ]

            # Create the D-term field over the free coordinates.
            Dterm_field = self.__cs__.requires_expression_from_coordinates(
                Dterm_field,
                "Dterm",
                coordinates,
                fixed_axes=fixed_values,
            )
            inverse_metric_field = self.__cs__.requires_expression_from_coordinates(
                inverse_metric_field,
                "inverse_metric_tensor",
                coordinates,
                fixed_axes=fixed_values,
            )
            __op__(
                vector_field_broadcast,
                Dterm_field,
                inverse_metric_field,
                *differential_coordinates,
                field_axes=output_axes_indices,
                derivative_axes=differential_axes_indices,
                edge_order=edge_order,
                out=out,
                **kwargs,
            )

        return out

    def dense_vector_divergence(
        self: _SupDGMO,
        vector_field: ArrayLike,
        field_axes: Sequence[str],
        /,
        basis: Literal["covariant", "contravariant"] = "contravariant",
        out: Optional[ArrayLike] = None,
        Dterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the divergence of a vector field, with automatic basis dispatching.

        This is a convenience wrapper around:
        - :meth:`dense_vector_divergence_contravariant` if `basis='contravariant'`
        - :meth:`dense_vector_divergence_covariant` if `basis='covariant'`

        Parameters
        ----------
        vector_field : array-like
            The vector field on which to compute the divergence. This should be an array-like object
            with a compliant shape (see `Notes` below).
        field_axes : list of str
            The coordinate axes over which the `field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        basis : {'contravariant', 'covariant'}, optional
            The basis in which the input vector field is represented:

            - If ``'contravariant'`` (default), the input is assumed to be a contravariant vector
              :math:`V^\mu`, and the divergence is computed directly using:

              .. math:: \nabla \cdot V = \partial_\mu V^\mu + D_\mu V^\mu

              The `inverse_metric_field` is **not** required or used. The `derivative_field` can be
              supplied in this case.

            - If ``'covariant'``, the input is assumed to be a covector field :math:`V_\mu`.
              In this case, the divergence is computed via:

              .. math:: \nabla \cdot V = \frac{1}{\rho} \partial_\mu(\rho g^{\mu\nu} V_\nu)

              which requires contraction with the **inverse metric** :math:`g^{\mu\nu}`.
              If not provided, this is derived automatically from the coordinate system. The
              `derivative_field` is ignored for this basis.
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        Dterm_field : array-like, optional
            The D-term field :math:`D_\mu = \frac{\partial_\mu \rho}{\rho}` for the coordinate system,
            where :math:`\rho = \sqrt{|g|}` is the metric volume element.

            If not provided, it is computed automatically from the coordinate system. Supplying it
            can improve performance or accuracy.

            The array must be broadcast-compatible with the grid shape over `output_axes`, and have a final
            dimension of size equal to `ndim`, one per coordinate direction.

        inverse_metric_field : array-like, optional
            The inverse metric tensor :math:`g^{\mu\nu}` used to raise covariant components.

            This field is only required if ``basis='covariant'`` and will be ignored in contravariant mode.

            If omitted, it is computed from the coordinate system. The expected shape depends on the metric type:

            - For diagonal (orthogonal) metrics: shape `(..., ndim)`
            - For full (non-orthogonal) metrics: shape `(..., ndim, ndim)`

            In both cases, the leading shape must match the grid domain over `output_axes`.

        derivative_field : array-like, optional
            The first derivatives of the input vector field :math:`\partial_\mu V^\nu`.

            This is optional—if not provided, derivatives are computed numerically. If known analytically
            or precomputed, supplying this can improve accuracy and reduce compute time.

            The array must be broadcast-compatible with the grid over `output_axes`, and its final dimension
            should match the number of coordinate axes over which derivatives are taken (typically `ndim`).
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_vector_divergence_contravariant`.

        Returns
        -------
        array‑like
            Scalar divergence on the grid.  The shape equals the broadcasted
            grid shape over ``output_axes`` (ghost zones included when the grid
            carries them).

        Notes
        -----

        - The divergence formula used depends on the input `basis`. The `contravariant` case requires only the vector field
          and optionally its derivatives. The `covariant` case requires the inverse metric for contraction.

        - If `output_axes` is provided, all fields (`vector_field`, `Dterm_field`, etc.) must be broadcastable over that domain.

        - If `in_chunks=True`, each chunk is extended by a 1-cell halo for stencil accuracy and processed independently.
          Chunking is useful when working with HDF5-backed or lazily loaded buffers.

        See Also
        --------
        dense_vector_divergence_contravariant : Compute divergence for contravariant vectors.
        dense_vector_divergence_covariant : Compute divergence for covariant vectors.
        dense_divergence : Basis-dispatching divergence for arbitrary-rank tensors.
        """
        # Distinguish the basis and proceed to the low-level callable
        # depending on which basis is specified.
        if basis == "covariant":
            try:
                return self.dense_vector_divergence_covariant(
                    vector_field,
                    field_axes,
                    out=out,
                    Dterm_field=Dterm_field,
                    inverse_metric_field=inverse_metric_field,
                    in_chunks=in_chunks,
                    edge_order=edge_order,
                    output_axes=output_axes,
                    pbar=pbar,
                    pbar_kwargs=pbar_kwargs,
                    **kwargs,
                )
            except Exception as e:
                raise ValueError(f"Failed to compute covariant gradient: {e}") from e
        elif basis == "contravariant":
            try:
                return self.dense_vector_divergence_contravariant(
                    vector_field,
                    field_axes,
                    out=out,
                    Dterm_field=Dterm_field,
                    derivative_field=derivative_field,
                    in_chunks=in_chunks,
                    edge_order=edge_order,
                    output_axes=output_axes,
                    pbar=pbar,
                    pbar_kwargs=pbar_kwargs,
                    **kwargs,
                )
            except Exception as e:
                raise ValueError(f"Failed to compute covariant gradient: {e}") from e

        else:
            raise ValueError(
                f"`basis` must be 'covariant' or 'contravariant', not '{basis}'."
            )

    def dense_scalar_laplacian(
        self: _SupDGMO,
        field: ArrayLike,
        field_axes: Sequence[str],
        /,
        out: Optional[ArrayLike] = None,
        Lterm_field: Optional[ArrayLike] = None,
        inverse_metric_field: Optional[ArrayLike] = None,
        derivative_field: Optional[ArrayLike] = None,
        second_derivative_field: Optional[ArrayLike] = None,
        *,
        in_chunks: bool = False,
        edge_order: Literal[1, 2] = 2,
        output_axes: Optional[Sequence[str]] = None,
        pbar: bool = True,
        pbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Compute the element-wise Laplacian of a tensor field.

        For a generic array field :math:`T_{\ldots}^{\ldots}`, the Laplacian of a given element (denoted :math:`\phi`) is

        .. math::

            \nabla^2 \phi = \nabla \cdot \nabla \phi = \frac{1}{\rho}\partial_\mu \left(\rho g^{\mu\nu} \partial_\nu \phi\right).

        A more numerically stable expression of this result is

        .. math::

            \nabla^2\phi = \frac{1}{\rho}\partial_\mu \left[g^{\mu\nu} \rho\right] \partial_\nu \phi + g^{\mu\nu} \partial_\mu\partial_\nu \phi
            = L^\nu \partial_\nu \phi + g^{\mu\nu} \partial_\mu\partial_\nu \phi.

        This is the formula used here.

        .. hint::

            This method wraps the low-level :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_full` or
            (if the metric tensor is diagonal), it wraps :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_diag`.

        Parameters
        ----------
        field : array-like
            The tensor field on which to operate. This must meet all the
            necessary shape criteria (see Notes).
        field_axes : list of str
            The coordinate axes over which the `tensor_field` spans. This should be a sequence of strings referring to
            the various coordinates of the underlying :attr:`~grids.base.GridBase.coordinate_system` of the grid.
            For each element in `field_axes`, `tensor_field`'s `i`-th index should match the shape
            of the grid for that coordinate. (See `Notes` for more details).
        out : array-like, optional
            An optional buffer in which to store the result.
            This can be used to reduce memory usage when performing
            computations. The shape of `out` must be compliant
            with broadcasting rules (see `Notes`). `out` may be a buffer or any
            other array-like object.
        Lterm_field : array-like, optional
            The volume log-derivative field :math:`L^\nu = \frac{1}{\rho} \partial_\mu [g^{\mu\nu} \rho]`.
            If not provided, it is computed automatically using the coordinate system. This argument can be filled to
            reduce numerical error and improve computational efficiency if it is known.

            If specified, `Lterm_field` must be shape compliant (see Notes).
        inverse_metric_field : array-like, optional
            A buffer containing the inverse metric field :math:`g^{\mu\nu}`. `inverse_metric_field`
            can be provided to improve computation speed (by avoiding computing it in stride);
            however, it is not required.

            The inverse metric can be derived from the coordinate system when this
            argument is not provided. See `Notes` below
            for details on the shape of `inverse_metric_field`.
        derivative_field : array-like, optional
            A buffer containing the first derivatives of the field. Can be provided to improve
            computation speed (by avoiding computing it in stride); however, it is not required.

            If specified, `derivative_field` must be shape compliant (see Notes).
        second_derivative_field : array-like, optional
            A buffer containing the second derivatives of the field. Can be provided to improve
            computation speed (by avoiding computing it in stride); however, it is not required.

            If specified, `second_derivative_field` must be shape compliant (see Notes).
        output_axes : list of str, optional
            The axes of the coordinate system over which the result should
            span. By default, `output_axes` is the same as `field_axes` and
            the output field matches the span of the input field.

            This argument may be specified to expand the number of axes onto which
            the output field is computed. `output_axes` must be a superset of the
            `field_axes`.
        in_chunks : bool, optional
            Whether to perform the computation in chunks. This can help reduce memory usage during the operation but
            will increase runtime due to increased computational load. If input buffers are all fully-loaded into memory,
            chunked performance will only marginally improve; however, if buffers are lazy loaded, then chunked operations
            will significantly improve efficiency. Defaults to ``False``.
        edge_order : {1, 2}, optional
            Order of the finite difference scheme to use when computing derivatives. See :func:`numpy.gradient` for more
            details on this argument. Defaults to ``2``.
        pbar : bool, optional
            Whether to display a progress bar when `in_chunks=True`.
        pbar_kwargs : dict, optional
            Additional keyword arguments passed to the progress bar utility. These can be any valid arguments
            to :func:`tqdm.tqdm`.
        **kwargs
            Additional keyword arguments passed to :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_full` or
            :func:`~differential_geometry.dense_ops.dense_scalar_laplacian_diag`.

        Returns
        -------
        array-like
            The computed partial derivatives. The resulting array will have a field shape matching the grid's
            shape over the `output_axes` and an element shape matching that of `field` but with an additional `(ndim,)`
            sized dimension containing each of the partial derivatives for each index.

        Notes
        -----
        **Shape and Broadcasting Requirements**

        The spatial dimensions of `field` must match the grid shape exactly over the `field_axes`.
        For a scalar field on a grid with shape ``(G1, ..., Gm)``, and `field_axes = ['x', 'z']`,
        the field must have shape ``(Gₓ, G_z)``. For tensor fields, additional trailing dimensions
        (beyond the spatial ones) are interpreted as tensor indices and must either match `ndim` exactly
        or be nested in a form that makes the Laplacian contractable (i.e., act elementwise).

        The output shape will match the shape of `tensor_field` unless `output_axes` introduces
        additional broadcasting (e.g., singleton axes added by `broadcast_array_to_axes`).

        **Lterm and Inverse Metric**

        The Laplacian operator requires knowledge of both the inverse metric and the volume derivative term (Lterm).
        These are automatically computed from the coordinate system unless explicitly provided.
        If supplied manually:

        - `Lterm_field` must have shape ``(..., ndim)``
        - `inverse_metric_field` must be either ``(..., ndim)`` (diagonal) or ``(..., ndim, ndim)`` (full)

        Additionally, the derivative fields may be supplied. In that case,

        - `derivative_field` must have shape ``(..., ndim)``
        - `second_derivative_field` must have shape ``(..., ndim)`` if the metric is diagonal and
          ``(..., ndim, ndim)`` if it is full.

        **Chunked Execution**

        When `in_chunks=True`, the Laplacian is computed in small memory-efficient blocks with
        halo padding of 1 cell. This is especially useful when `tensor_field` and `out` are backed
        by HDF5 or other lazy-loading array backends. Chunking requires the grid to support
        `iter_chunk_slices(...)`.

        **When to Use This**

        This method is suitable for computing the Laplace–Beltrami operator in arbitrary curvilinear
        coordinate systems. It generalizes to higher-rank tensors when `tensor_field` contains dense
        component axes. For fields with symbolic or sparse component structure, see symbolic APIs.

        See Also
        --------
        dense_element_wise_partial_derivatives: Generic form for general array-valued fields.
        dense_covariant_gradient: Covariant gradient of a tensor field.
        ~differential_geometry.dense_ops.dense_gradient_contravariant_full: Low-level callable version (full metric)
        ~differential_geometry.dense_ops.dense_gradient_contravariant_diag: Low-level callable version (diag metric)
        """
        return self.dense_element_wise_laplacian(
            field,
            field_axes,
            out=out,
            Lterm_field=Lterm_field,
            inverse_metric_field=inverse_metric_field,
            derivative_field=derivative_field,
            second_derivative_field=second_derivative_field,
            in_chunks=in_chunks,
            edge_order=edge_order,
            output_axes=output_axes,
            pbar=pbar,
            pbar_kwargs=pbar_kwargs,
            **kwargs,
        )
