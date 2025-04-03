"""
Support for tensor-field structures in Pisces-Geometry.

This module defines the :py:class:`TensorField` class, a general representation of tensor-valued
fields defined on a geometric grid, along with several specialized subclasses for
common low-rank tensor types: scalars, vectors, and covectors.

At its core, :py:class:`TensorField` allows the representation of a field whose values are
multi-index tensors, with explicit control over variance (contravariant vs covariant),
axis labeling, and index interpretation. These fields are typically defined on a grid
in some coordinate system and are backed by a buffer storing the actual numerical values.
"""
from typing import Any, Dict, Sequence, Tuple, Type, Union

import numpy as np

from pisces_geometry.fields.base import _BaseField
from pisces_geometry.fields.buffers import _Buffer


class TensorField(_BaseField):
    """
    Representation of a general `tensor field <https://en.wikipedia.org/wiki/Tensor_field>`_ on a
    geometric grid.

    The :py:class:`TensorField` class is a subclass of :py:class:`fields.base.GenericField`, designed
    to represent tensor-valued fields in arbitrary coordinate systems. Each tensor field is defined
    over a grid and stores values that are multi-index tensors (e.g., scalars, vectors, matrices),
    with full support for symbolic labeling, rank/variance tracking, and coordinate-aware operations.

    The type of tensor stored at each grid point is determined by the **tensor class** :math:`(p, q)`,
    which encodes the number of contravariant (upper) and covariant (lower) indices, respectively.
    This structure enables consistent handling of tensor transformation rules and differential geometry
    operations such as contraction, pullbacks, and projections.

    The trailing dimensions of the data buffer represent the tensor indices. For example:

    - A scalar field (rank-(0,0)) has shape ``grid.shape``.
    - A vector field (rank-(1,0)) has shape ``grid.shape + (n,)``, where ``n`` is the grid dimensionality.
    - A rank-(1,1) tensor field has shape ``grid.shape + (n, n)``, and so on.

    Fields may carry symbolic labels for each component (e.g., ``['r', 'theta']``), enabling
    introspection and selective operations aligned with coordinate directions.

    Notes
    -----
    Users should not typically instantiate :py:class:`TensorField` directly unless working with
    higher-rank or mixed-variance tensors. For common cases, the constructor automatically dispatches
    to a more specific subclass:

    - ``(0, 0)`` → :py:class:`ScalarField`
    - ``(1, 0)`` → :py:class:`VectorField`
    - ``(0, 1)`` → :py:class:`CovectorField`

    This design provides a unified interface for scalar, vector, and tensor field types while preserving
    the mathematical structure necessary for tensor calculus in curvilinear and non-Euclidean spaces.
    """

    # ================================= #
    # Initialization                    #
    # ================================= #
    # These methods are all delegated to during __init__ and can
    # be modified as needed when subclassing. Read the docstrings
    # for each in order to get a clear explanation of required tasks
    # for each method.
    def _construct_and_validate_components(self, *_, components=None, **__):
        """
        Construct or validate the tensor components list. This is provided as a list of
        lists of axes strings. Each element corresponds to a specific index of the
        tensor field. We check that the number of components matches the rank of the
        tensor and that each has the correct number of axes in it.

        Parameters
        ----------
        components : list of list of str or None
            Component labels to validate or construct.

        Returns
        -------
        list of list of str
            Component names per tensor index.
        """
        # Extract the shape of the tensor element. This is what
        # we check the components against.
        element_shape = self.__buffer__.shape[self.axes_ndim :]

        if components is None:
            # Default behavior: use the first `n` axes repeatedly.
            # For example, for a 2x2 tensor and a grid with axes ['x', 'y', 'z'],
            # components = [['x', 'y'], ['x', 'y']] for shape (2, 2)
            result = []
            for dim_size in element_shape:
                if dim_size > self.__grid__.ndim:
                    raise ValueError(
                        f"Cannot have a tensor field with more than {self.__grid__.ndim} components from only "
                        f"{len(self.__grid__.axes)} grid axes."
                    )
                result.append(list(self.__grid__.axes[:dim_size]))
            return result

        # User provided components: validate shape match
        if len(components) != len(element_shape):
            raise ValueError(
                f"`components` has {len(components)} index dimensions, but "
                f"buffer has element shape {element_shape} (rank {len(element_shape)})."
            )

        for i, (comp, expected_size) in enumerate(zip(components, element_shape)):
            if not hasattr(comp, "__len__") or isinstance(comp, str):
                raise ValueError(
                    f"`components[{i}]` must be a sequence, got {type(comp).__name__}"
                )
            if len(comp) != expected_size:
                raise ValueError(
                    f"`components[{i}]` has length {len(comp)}, expected {expected_size} "
                    f"based on buffer shape."
                )

        return [list(c) for c in components]  # Ensure list of lists

    def __configure_and_validate_tensor__(
        self, tensor_class, *args, components=None, tensor_signature=None, **kwargs
    ):
        """
        Validate the tensor's components, rank, and signature.
        """
        # Validate and set components
        self.__components__ = self._construct_and_validate_components(
            *args, components=components, **kwargs
        )

        # Compute and validate tensor rank
        rank = sum(tensor_class)
        element_shape = self.__buffer__.shape[self.axes_ndim :]

        if len(element_shape) != rank:
            raise ValueError(
                f"tensor_class={tensor_class} implies rank {rank}, "
                f"but buffer has {len(element_shape)} extra dimensions (shape {element_shape})."
            )

        if any(e > self.__grid__.ndim for e in element_shape):
            raise ValueError(
                "Tensor component dimensions cannot exceed grid dimensionality."
            )

        self.__tensor_class__ = tensor_class

        # Handle variance (signature of +1 for contravariant, -1 for covariant)
        if tensor_signature is not None:
            if len(tensor_signature) != rank:
                raise ValueError(
                    f"tensor_signature has length {len(tensor_signature)}, "
                    f"expected {rank} from tensor_class={tensor_class}."
                )
            if any(v not in (1, -1) for v in tensor_signature):
                raise ValueError("tensor_signature must only contain +1 or -1.")
            self.__variance__ = list(tensor_signature)
        else:
            self.__variance__ = [1] * tensor_class[0] + [-1] * tensor_class[1]

    def __new__(
        cls, buffer: Any, grid: Any, tensor_class: Tuple[int, int], *args, **kwargs
    ):
        # If the user is calling `TensorField(...)` but the rank matches known simpler classes,
        # automatically dispatch to those classes.
        # If `cls` is already e.g. VectorField, or the user gave a custom rank, do nothing special.
        if cls is TensorField:
            if tensor_class == (0, 0):
                return ScalarField(buffer, grid, *args, **kwargs)
            elif tensor_class == (1, 0):
                return VectorField(buffer, grid, *args, **kwargs)
            elif tensor_class == (0, 1):
                return CovectorField(buffer, grid, *args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        buffer: Any,
        grid: Any,
        tensor_class: Tuple[int, int],
        *args,
        axes: Sequence[str] = None,
        components: Sequence[Sequence[str]] = None,
        tensor_signature: Sequence[int] = None,
        __buffer_type__: Type[_Buffer] = None,
        __buffer_kwargs__: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize a general tensor-valued field on a geometric grid.

        This constructor sets up the base field structure and performs validation for the tensor
        rank, component labeling, and variance (covariant/contravariant nature of each index). It
        is typically not called directly for common tensor ranks like scalars, vectors, or covectors.
        Instead, use :py:class:`ScalarField`, :py:class:`VectorField`, or :py:class:`CovectorField` directly, or let :py:class:`TensorField`
        dispatch to the appropriate subclass automatically based on `tensor_class`.

        Parameters
        ----------
        buffer : array-like or _Buffer
            The underlying data representing the field values. This must be either:

            - A NumPy array with shape equal to ``grid.shape + tensor_shape``, where ``tensor_shape``
              is determined by the rank implied from ``tensor_class``.
            - An existing instance of a buffer class derived from :py:class:`~fields.buffers._Buffer`.

            If a raw array-like is provided, it will be automatically coerced into a buffer using the buffer system
            from :py:mod:`~fields.buffers`. The buffer type is inferred from context unless explicitly
            specified via ``__buffer_type__``. Additional configuration parameters for the buffer coercion process can
            be supplied via ``__buffer_kwargs__``.

        grid : :py:class:`~grids.base.GenericGrid`
            The grid object on which this field is defined. This provides the geometric and coordinate context for the field.
            The grid must support coordinate axes and shape introspection, and should be compatible with the Pisces coordinate
            system framework (e.g., :py:class:`~grids.base.GenericGrid`, :py:class:`~grids.base.UniformGrid`, etc.).
        tensor_class : tuple of int
            A 2-tuple ``(p, q)`` representing the tensor's type:

            - ``p``: number of contravariant (upper) indices,
            - ``q``: number of covariant (lower) indices.

            This determines the rank and interpretation of the tensor field. For example, ``(1, 0)`` indicates a vector field,
            ``(0, 1)`` a covector (dual vector) field, and ``(0, 0)`` a scalar field.
        *args :
            Additional positional arguments forwarded to internal methods or subclasses. Typically unused in the base class
            but supported for extensibility.
        axes : list of str, optional
            A list of symbolic axis names used to describe the grid axes (e.g., ``['r', 'theta', 'phi']``). If not provided,
            this defaults to the axes defined on the input grid. The axes provided must always match those present in the
            underlying coordinate system and be in the same order; however, axes may be omitted if they are not relevant
            to the dependence of the grid.
        components : list of list of str, optional
            A list of symbolic component labels for each tensor index. Each entry in the outer list corresponds to one tensor
            index and should contain axis names for that index. If not provided, defaults to using the first ``n`` grid axes
            for each dimension of the tensor.

            For example, a rank-2 tensor over a 3D grid might use:

            .. code-block:: python

                components = [['x', 'y', 'z'], ['x', 'y', 'z']]

        tensor_signature : list of int, optional
            A list of integers (+1 or -1) indicating the variance of each index in the tensor:

            - +1: contravariant (upper index),
            - -1: covariant (lower index).

            If not provided, this is inferred from ``tensor_class`` as ``[+1]*p + [-1]*q``.
        __buffer_type__ : type, optional
            The buffer wrapper class to use when coercing a raw array into a field buffer. Must be a subclass of
            :py:class:`~pisces_geometry.fields.buffers._Buffer`. If not set, the default buffer class for the field type
            will be used (e.g., NumPy-backed buffer by default).
        __buffer_kwargs__ : dict, optional
            Additional keyword arguments passed to the buffer's coercion logic. Useful for customizing memory layout,
            compression options, or I/O behavior depending on the chosen buffer backend (e.g., unyt, HDF5).
        **kwargs :
            Additional keyword arguments passed to internal field machinery or reserved for future use.

        Raises
        ------
        ValueError
            If the buffer shape does not match the grid + tensor shape, or if the component labels or
            variance are inconsistent with the tensor rank.
        """
        # Initialize at the superclass level to get the grid and
        # the buffer configured.
        super().__init__(
            buffer,
            grid,
            *args,
            axes=axes,
            __buffer_type__=__buffer_type__,
            __buffer_kwargs__=__buffer_kwargs__,
            **kwargs,
        )

        # Now we need to configure the tensor and do the remaining checks to ensure
        # that the tensor is self consistently defined.
        self.__configure_and_validate_tensor__(
            tensor_class,
            *args,
            components=components,
            tensor_signature=tensor_signature,
            **kwargs,
        )

    # ================================= #
    # Dunder Methods                    #
    # ================================= #
    def __str__(self):
        return f"<{self.__tensor_class__[0]}-{self.__tensor_class__[1]} Tensor Field | {self.grid.shape}>"

    # ================================= #
    # Properties                        #
    # ================================= #
    @property
    def components(self) -> Sequence[Sequence[str]]:
        """
        The symbolic labels for each component of this field's tensor structure.

        Each tensor index (e.g., vector or covector index) is associated with a list of axis names,
        indicating which coordinate directions it applies to. The number of outer lists matches the
        rank of the tensor, and each inner list has length equal to the size of that index.

        This enables introspection of field components and coordinate-aware slicing or contraction.

        Examples
        --------
        - For a rank-(1,0) vector field in polar coordinates:

          .. code-block:: python

              components = [['r', 'theta']]

        - For a rank-(1,1) mixed tensor:

          .. code-block:: python

              components = [['r', 'theta'], ['r', 'theta']]

        Returns
        -------
        list of list of str
            A list of component labels, each itself a list of axis names per tensor index.
        """
        return self.__components__

    @property
    def components_mask(self) -> np.ndarray:
        """
        Boolean masks identifying which coordinate axes are involved in each tensor component.

        This method maps the symbolic component labels (from `components`) into a boolean array,
        indicating presence of each coordinate axis across tensor indices. The output has shape
        ``(grid.ndim, rank)``, with one column per tensor index.

        It can be used for selecting components in a coordinate-aware fashion (e.g., projecting
        along a subset of axes).

        Returns
        -------
        numpy.ndarray of bool
            A boolean mask array with shape ``(grid.ndim, rank)``, where each column indicates
            which coordinate axes are involved in that index position of the tensor.

        Raises
        ------
        ValueError
            If called on a scalar field (rank 0), which has no tensor indices.
        """
        if self.rank == 0:
            raise ValueError("This is a scalar field.")
        return np.stack(
            [
                self.grid.coordinate_system.build_axes_mask(_comp)
                for _comp in self.components
            ],
            axis=-1,
        )

    @property
    def variance(self) -> np.ndarray:
        """
        Tensor index variance for each dimension of the tensor.

        Variance describes whether each index of the tensor is contravariant (+1) or covariant (-1).
        This array has length equal to the tensor rank and is used for applying proper transformation
        rules under coordinate changes.

        Returns
        -------
        numpy.ndarray of int
            An array of length equal to the tensor rank. Entries are +1 (contravariant) or -1 (covariant).
        """
        return np.asarray(self.__variance__, dtype=int)

    @property
    def contravariant_mask(self) -> np.ndarray:
        """
        Boolean mask indicating contravariant indices (upper indices) in the tensor.

        Returns
        -------
        numpy.ndarray of bool
            A boolean array of length equal to the tensor rank. `True` indicates a contravariant index.

        Raises
        ------
        ValueError
            If called on a scalar field (rank 0), which has no indices.
        """
        if self.rank == 0:
            raise ValueError("This is a scalar field.")
        return np.equal(self.variance, 1)

    @property
    def covariant_mask(self) -> np.ndarray:
        """
        Boolean mask indicating covariant indices (lower indices) in the tensor.

        Returns
        -------
        numpy.ndarray of bool
            A boolean array of length equal to the tensor rank. `True` indicates a covariant index.

        Raises
        ------
        ValueError
            If called on a scalar field (rank 0), which has no indices.
        """
        if self.rank == 0:
            raise ValueError("This is a scalar field.")
        return np.equal(self.variance, -1)

    def raise_index(self):
        pass

    def lower_index(self):
        pass

    def adjust_signature(self):
        pass

    def gradient(self):
        pass

    def tensor_product(self):
        pass


class VectorField(TensorField):
    r"""
    Representation of a contravariant vector field (:math:`(1,0)` tensor field) on a geometric grid.

    The :py:class:`VectorField` class is a specialized subclass of :py:class:`TensorField`, representing
    fields that associate a contravariant vector with each point in space. This corresponds to a tensor
    with one upper index, and is a common structure in physics and differential geometry (e.g., velocity fields,
    electric fields, displacement vectors).

    Each vector is stored in the trailing dimension of the buffer, which must match the number of coordinate axes
    in the grid. The data buffer must therefore have shape ``grid.shape + (n,)``, where ``n = grid.ndim``.

    The tensor class is automatically fixed to ``(1, 0)``, and the variance signature is set to ``[+1]``.

    Examples
    --------
    A 3D vector field in spherical coordinates might use:

    .. code-block:: python

        buffer.shape == (Nr, Ntheta, Nphi, 3)
        components = [['r', 'theta', 'phi']]

    Notes
    -----
    - Equivalent to a rank-1 contravariant tensor.
    - Automatically dispatched by :py:class:`TensorField` when ``tensor_class=(1,0)``.
    - Supports component-wise masking and coordinate-aware tensor operations.
    """

    # ================================= #
    # Initialization                    #
    # ================================= #
    # These methods are all delegated to during __init__ and can
    # be modified as needed when subclassing. Read the docstrings
    # for each in order to get a clear explanation of required tasks
    # for each method.
    def __new__(cls, buffer, grid, *args, **kwargs):
        return super().__new__(cls, buffer, grid, (1, 0), *args, **kwargs)

    def __init__(
        self,
        buffer: Any,
        grid: Any,
        *args,
        axes: Sequence[str] = None,
        components: Union[str, Sequence[str], Sequence[Sequence[str]]] = None,
        __buffer_type__: Type[_Buffer] = None,
        __buffer_kwargs__: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize a vector field (rank-(1,0)) on a geometric grid.

        A vector field associates a contravariant vector with each point on the grid.
        This class automatically sets the tensor type to ``(1, 0)`` and variance to ``[+1]``.

        Parameters
        ----------
        buffer : array-like, numpy.ndarray, or _Buffer
            The underlying data representing the vector field values. Must be one of:

            - A NumPy array with shape equal to ``grid.shape + (ndim,)``, where ``ndim`` is the number
              of dimensions in the grid (i.e., the number of coordinate axes).
            - An instance of a buffer class derived from :py:class:`~pisces_geometry.fields.buffers._Buffer`.

            If a raw array is provided, it will be automatically wrapped using the internal buffer system.
            You may override the default buffer backend using ``__buffer_type__``, and customize coercion
            with ``__buffer_kwargs__``.

        grid : :py:class:`~pisces_geometry.grids.base.GenericGrid`
            The grid over which the vector field is defined. The grid provides the geometric structure
            (including shape and axes) and must be compatible with the Pisces coordinate system framework,
            such as :py:class:`~pisces_geometry.grids.GenericGrid` or :py:class:`~pisces_geometry.grids.UniformGrid`.

        *args :
            Additional positional arguments passed to internal methods (typically ignored in this subclass).

        axes : list of str, optional
            Names of the grid axes (e.g., ``['r', 'theta', 'phi']``). These define the spatial ordering of the grid
            dimensions. If not specified, they are inferred from the grid. Axes must match the coordinate system
            defined in the grid and be ordered accordingly.

        components : list of list of str, optional
            Symbolic labels for the vector components. Since this is a rank-1 tensor, this must be a single list
            containing axis names for the single vector index (e.g., ``[['r', 'theta', 'phi']]``). If not provided,
            defaults to the first ``ndim`` axes in the grid.

        __buffer_type__ : type, optional
            Optional buffer backend to wrap the data (e.g., a custom subclass of
            :py:class:`~pisces_geometry.fields.buffers._Buffer`). If omitted, a default (e.g., NumPy-backed) buffer is used.

        __buffer_kwargs__ : dict, optional
            Keyword arguments passed to the buffer's coercion method. Useful for controlling memory layout, unit handling,
            or lazy-loading (e.g., HDF5 or unyt buffers).

        **kwargs :
            Additional keyword arguments forwarded to internal field logic or future extensions.

        Raises
        ------
        ValueError
            If the shape of the buffer does not match ``grid.shape + (ndim,)``, or if the provided components or
            tensor signature are inconsistent with a rank-(1,0) vector field.
        """
        # Setup the tensor class and correct the components.
        tensor_class = (1, 0)
        if components is not None:
            if isinstance(components, str):
                components = [[components]]
            elif (isinstance(components, Sequence)) and isinstance(components[0], str):
                components = [components]
            else:
                pass

        # Initialize at the superclass level to get the grid and
        # the buffer configured.
        super().__init__(
            buffer,
            grid,
            tensor_class,
            *args,
            axes=axes,
            __buffer_type__=__buffer_kwargs__,
            __buffer_kwargs__=__buffer_kwargs__,
            components=components,
            tensor_signature=[1],
            **kwargs,
        )

    # ================================= #
    # Properties                        #
    # ================================= #
    @property
    def components(self) -> Sequence[str]:
        """
        The symbolic labels for each component of this field's tensor structure.

        Each tensor index (e.g., vector or covector index) is associated with a list of axis names,
        indicating which coordinate directions it applies to. The number of outer lists matches the
        rank of the tensor, and each inner list has length equal to the size of that index.

        This enables introspection of field components and coordinate-aware slicing or contraction.

        Examples
        --------
        - For a rank-(1,0) vector field in polar coordinates:

          .. code-block:: python

              components = [['r', 'theta']]

        - For a rank-(1,1) mixed tensor:

          .. code-block:: python

              components = [['r', 'theta'], ['r', 'theta']]

        Returns
        -------
        list of list of str
            A list of component labels, each itself a list of axis names per tensor index.
        """
        return self.__components__[0]

    @property
    def components_mask(self) -> np.ndarray:
        return self.grid.coordinate_system.build_axes_mask(self.components)

    @property
    def variance(self) -> np.ndarray:
        """
        Tensor index variance for each dimension of the tensor.

        Variance describes whether each index of the tensor is contravariant (+1) or covariant (-1).
        This array has length equal to the tensor rank and is used for applying proper transformation
        rules under coordinate changes.

        Returns
        -------
        numpy.ndarray of int
            An array of length equal to the tensor rank. Entries are +1 (contravariant) or -1 (covariant).
        """
        return np.asarray(self.__variance__, dtype=int)

    @property
    def contravariant_mask(self) -> np.ndarray:
        """
        Boolean mask indicating contravariant indices (upper indices) in the tensor.

        Returns
        -------
        numpy.ndarray of bool
            A boolean array of length equal to the tensor rank. `True` indicates a contravariant index.

        Raises
        ------
        ValueError
            If called on a scalar field (rank 0), which has no indices.
        """
        if self.rank == 0:
            raise ValueError("This is a scalar field.")
        return np.equal(self.variance, 1)

    @property
    def covariant_mask(self) -> np.ndarray:
        """
        Boolean mask indicating covariant indices (lower indices) in the tensor.

        Returns
        -------
        numpy.ndarray of bool
            A boolean array of length equal to the tensor rank. `True` indicates a covariant index.

        Raises
        ------
        ValueError
            If called on a scalar field (rank 0), which has no indices.
        """
        if self.rank == 0:
            raise ValueError("This is a scalar field.")
        return np.equal(self.variance, -1)


class CovectorField(TensorField):
    r"""
    Representation of a covariant vector field (:math:`(0,1)` tensor field) on a geometric grid.

    The :py:class:`CovectorField` class is a specialized subclass of :py:class:`TensorField`, representing
    fields that associate a covariant vector (a 1-form or dual vector) with each point in space. This corresponds
    to a tensor with one lower index and arises in contexts such as gradients, differential forms, and metric-induced
    mappings of vectors.

    Each covector is stored in the trailing dimension of the buffer, which must match the number of coordinate axes
    in the grid. The data buffer must therefore have shape ``grid.shape + (n,)``, where ``n = grid.ndim``.

    The tensor class is automatically fixed to ``(0, 1)``, and the variance signature is set to ``[-1]``.

    Examples
    --------
    A covector field representing a gradient might use:

    .. code-block:: python

        buffer.shape == (Nr, Ntheta, Nphi, 3)
        components = [['r', 'theta', 'phi']]

    Notes
    -----
    - Equivalent to a rank-1 covariant tensor (1-form).
    - Automatically dispatched by :py:class:`TensorField` when ``tensor_class=(0,1)``.
    - Fully compatible with metric duality and inner products with vector fields.
    """

    def __new__(cls, buffer, grid, *args, **kwargs):
        return object.__new__(CovectorField)

    def __init__(
        self,
        buffer: Any,
        grid: Any,
        *args,
        axes: Sequence[str] = None,
        components: Union[str, Sequence[str], Sequence[Sequence[str]]] = None,
        __buffer_type__: Type[_Buffer] = None,
        __buffer_kwargs__: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize a co-vector field (rank-(1,0)) on a geometric grid.

        A co-vector field associates a covariant vector with each point on the grid.
        This class automatically sets the tensor type to ``(0,1)`` and variance to ``[-1]``.

        Parameters
        ----------
        buffer : array-like, numpy.ndarray, or _Buffer
            The underlying data representing the co-vector field values. Must be one of:

            - A NumPy array with shape equal to ``grid.shape + (ndim,)``, where ``ndim`` is the number
              of dimensions in the grid (i.e., the number of coordinate axes).
            - An instance of a buffer class derived from :py:class:`~pisces_geometry.fields.buffers._Buffer`.

            If a raw array is provided, it will be automatically wrapped using the internal buffer system.
            You may override the default buffer backend using ``__buffer_type__``, and customize coercion
            with ``__buffer_kwargs__``.

        grid : :py:class:`~pisces_geometry.grids.base.GenericGrid`
            The grid over which the co-vector field is defined. The grid provides the geometric structure
            (including shape and axes) and must be compatible with the Pisces coordinate system framework,
            such as :py:class:`~pisces_geometry.grids.GenericGrid` or :py:class:`~pisces_geometry.grids.UniformGrid`.

        *args :
            Additional positional arguments passed to internal methods (typically ignored in this subclass).

        axes : list of str, optional
            Names of the grid axes (e.g., ``['r', 'theta', 'phi']``). These define the spatial ordering of the grid
            dimensions. If not specified, they are inferred from the grid. Axes must match the coordinate system
            defined in the grid and be ordered accordingly.

        components : list of list of str, optional
            Symbolic labels for the co-vector components. Since this is a rank-1 tensor, this must be a single list
            containing axis names for the single co-vector index (e.g., ``[['r', 'theta', 'phi']]``). If not provided,
            defaults to the first ``ndim`` axes in the grid.

        __buffer_type__ : type, optional
            Optional buffer backend to wrap the data (e.g., a custom subclass of
            :py:class:`~pisces_geometry.fields.buffers._Buffer`). If omitted, a default (e.g., NumPy-backed) buffer is used.

        __buffer_kwargs__ : dict, optional
            Keyword arguments passed to the buffer's coercion method. Useful for controlling memory layout, unit handling,
            or lazy-loading (e.g., HDF5 or unyt buffers).

        **kwargs :
            Additional keyword arguments forwarded to internal field logic or future extensions.

        Raises
        ------
        ValueError
            If the shape of the buffer does not match ``grid.shape + (ndim,)``, or if the provided components or
            tensor signature are inconsistent with a rank-(1,0) co-vector field.
        """
        # Setup the tensor class and correct the components.
        tensor_class = (0, 1)
        if components is not None:
            if isinstance(components, str):
                components = [[components]]
            elif (isinstance(components, Sequence)) and isinstance(components[0], str):
                components = [components]
            else:
                pass

        # Initialize at the superclass level to get the grid and
        # the buffer configured.
        super().__init__(
            buffer,
            grid,
            tensor_class,
            *args,
            axes=axes,
            __buffer_type__=__buffer_kwargs__,
            __buffer_kwargs__=__buffer_kwargs__,
            components=components,
            tensor_signature=[1],
            **kwargs,
        )


class ScalarField(TensorField):
    r"""
    Representation of a scalar field (:math:`(0,0)` tensor field) on a geometric grid.

    The :py:class:`ScalarField` class is a specialized subclass of :py:class:`TensorField`, representing
    fields that associate a single scalar value with each point in space. Scalars have no indices and are
    invariant under coordinate transformations, making them the simplest type of tensor field.

    The data buffer must have shape exactly equal to ``grid.shape``, with no trailing tensor dimensions.

    The tensor class is automatically fixed to ``(0, 0)``, and the field carries no variance or component labels.

    Examples
    --------
    A temperature field on a spherical grid might look like:

    .. code-block:: python

        buffer.shape == (Nr, Ntheta, Nphi)

    Notes
    -----
    - Equivalent to a rank-0 tensor (pure scalar).
    - Automatically dispatched by :py:class:`TensorField` when ``tensor_class=(0,0)``.
    - Supports arithmetic operations, broadcasting, and interpolation like any numeric field.
    """

    def __new__(cls, buffer, grid, *args, **kwargs):
        return object.__new__(ScalarField)

    def __init__(
        self,
        buffer: Any,
        grid: Any,
        *args,
        axes: Sequence[str] = None,
        __buffer_type__: Type[_Buffer] = None,
        __buffer_kwargs__: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize a scalar field (A :math:`(0,0)` tensor field) on a geometric grid.

        A scalar field assigns a single scalar value to each point on the grid.
        This class automatically sets the tensor type to ``(0, 0)`` and has no variance.

        Parameters
        ----------
        buffer : array-like, numpy.ndarray, or _Buffer
            The underlying data representing the scalar field values. Must be one of:

            - A NumPy array with shape exactly equal to ``grid.shape``.
            - An instance of a buffer class derived from :py:class:`~pisces_geometry.fields.buffers._Buffer`.

            If a raw array is provided, it will be automatically wrapped using the internal buffer system.
            You may override the default buffer backend using ``__buffer_type__``, and customize coercion
            with ``__buffer_kwargs__``.

        grid : :py:class:`~pisces_geometry.grids.base.GenericGrid`
            The grid over which the scalar field is defined. The grid provides the geometric structure
            (including shape and axes) and must be compatible with the Pisces coordinate system framework,
            such as :py:class:`~pisces_geometry.grids.GenericGrid` or :py:class:`~pisces_geometry.grids.UniformGrid`.

        *args :
            Additional positional arguments passed to internal methods (typically ignored in this subclass).

        axes : list of str, optional
            Names of the grid axes (e.g., ``['r', 'theta', 'phi']``). These define the spatial ordering of the grid
            dimensions. If not specified, they are inferred from the grid. Axes must match the coordinate system
            defined in the grid and be ordered accordingly.

        __buffer_type__ : type, optional
            Optional buffer backend to wrap the data (e.g., a custom subclass of
            :py:class:`~pisces_geometry.fields.buffers._Buffer`). If omitted, a default (e.g., NumPy-backed) buffer is used.

        __buffer_kwargs__ : dict, optional
            Keyword arguments passed to the buffer's coercion method. Useful for controlling memory layout, unit handling,
            or lazy-loading (e.g., HDF5 or unyt buffers).

        **kwargs :
            Additional keyword arguments forwarded to internal field logic or reserved for future use.

        Raises
        ------
        ValueError
            If the shape of the buffer does not exactly match ``grid.shape`` or if the buffer cannot be coerced
            to a valid scalar buffer.
        """
        # Define the tensor class and then proceed.
        tensor_class = (0, 0)
        super().__init__(
            buffer,
            grid,
            tensor_class,
            *args,
            axes=axes,
            components=None,  # scalar: no components
            tensor_signature=None,
            __buffer_type__=__buffer_type__,
            __buffer_kwargs__=__buffer_kwargs__,
            **kwargs,
        )

    @property
    def components(self) -> Sequence[Sequence[str]]:
        """
        The symbolic labels for each component of this field's tensor-like structure.

        This property must be implemented by subclasses that define the tensor component labeling
        (e.g., VectorField, CovectorField). Each index of the tensor should have a corresponding
        list of symbolic axis names.

        Returns
        -------
        list of list of str
            A list of component labels for each tensor index.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError("components must be implemented by subclasses.")

    @property
    def components_mask(self) -> np.ndarray:
        """
        Boolean mask indicating which grid axes are involved in each tensor component.

        This property must be implemented by subclasses that define component-axis associations.

        Returns
        -------
        numpy.ndarray
            A mask array of shape (grid.ndim, rank) or similar.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError("components_mask must be implemented by subclasses.")

    @property
    def variance(self) -> np.ndarray:
        """
        Tensor index variance for each dimension of the tensor.

        Variance describes whether each index is contravariant (+1) or covariant (-1).
        Must be implemented by subclasses to enable proper transformation behavior.

        Returns
        -------
        numpy.ndarray of int
            A 1D array of +1 or -1 values indicating variance per index.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError("variance must be implemented by subclasses.")

    @property
    def contravariant_mask(self) -> np.ndarray:
        """
        Boolean mask indicating which indices of the tensor are contravariant (upper indices).

        Returns
        -------
        numpy.ndarray of bool
            A boolean array where True marks a contravariant index.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError(
            "contravariant_mask must be implemented by subclasses."
        )

    @property
    def covariant_mask(self) -> np.ndarray:
        """
        Boolean mask indicating which indices of the tensor are covariant (lower indices).

        Returns
        -------
        numpy.ndarray of bool
            A boolean array where True marks a covariant index.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError("covariant_mask must be implemented by subclasses.")


if __name__ == "__main__":
    from pisces_geometry.coordinates import SphericalCoordinateSystem
    from pisces_geometry.grids import GenericGrid

    u = SphericalCoordinateSystem()
    g = GenericGrid(u, [[0, 1], [0, 1], [0, 1]])
    _buffer = np.zeros((2, 2, 2, 3))
    l = TensorField(_buffer, g, (1, 0))
    print(l, type(l))
    l.set_boundary_condition("theta", 1, "both")
    l.list_boundary_conditions()
