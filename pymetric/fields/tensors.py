"""
Tensor Fields.
"""
from typing import Literal, Optional, Sequence, Tuple

from pymetric.differential_geometry.dependence import DenseTensorDependence
from pymetric.grids.base import GridBase

from .base import DenseField
from .components import FieldComponent
from .mixins.base import DTensorFieldCoreMixin
from .mixins.dense_mathops import DenseTensorFieldDMOMixin
from .utils import signature_to_tensor_class, validate_rank_signature


class DenseTensorField(DTensorFieldCoreMixin, DenseTensorFieldDMOMixin, DenseField):
    """
    A dense tensor field with a defined rank and tensor signature over a structured grid.

    :class:`DenseTensorField` represents a tensor-valued field defined continuously
    across a structured coordinate grid. Each field stores its data in a single
    :class:`~fields.components.FieldComponent`, with tensor structure encoded in
    the trailing dimensions of the buffer.

    Tensor indices are explicitly tracked using a signature, which defines whether each
    index is covariant (−1) or contravariant (+1). This signature is used to apply
    correct transformations under coordinate operations, including gradient, divergence,
    index raising/lowering, and contraction.

    This class extends :class:`DenseField` with tensor semantics, automatic index
    handling, and support for symbolic dependence metadata.

    See Also
    --------
    :class:`~fields.base.DenseField`
    :class:`~fields.components.FieldComponent`
    :func:`~differential_geometry.dense_utils.validate_rank_signature`
    """

    def __init__(
        self,
        grid: GridBase,
        component: FieldComponent,
        signature: Optional[Sequence[Literal[1, -1]]] = None,
    ):
        """
        Initialize a dense tensor field over a structured coordinate grid.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
            This determines the spatial axes, metric, and chunking behavior.

        component : ~fields.components.FieldComponent
            The single component storing the field data. The component must
            have trailing element dimensions that match the expected tensor shape:
            one dimension per tensor index, each of size equal to `grid.ndim`.

            For example, a rank-2 tensor field in 3D requires `component.element_shape = (3, 3)`.

        signature : list of {+1,-1}, optional
            Tensor signature indicating the variance of each index:

            - `+1` → contravariant index (superscript)
            - `-1` → covariant index (subscript)

            If not specified, a default signature of all +1 (fully contravariant) is assumed.

        Raises
        ------
        ValueError
            If the component's element shape is not consistent with the tensor rank
            or the grid dimensionality, or if the signature is invalid.
        """
        # Initialize the parent object.
        super().__init__(grid, component)

        # Validate the relevant tensor behavior of the
        # class.
        expected_shape = component.element_ndim * (grid.ndim,)
        if component.element_shape != expected_shape:
            raise ValueError(
                f"Component has element shape {component.element_shape}, expected {expected_shape} "
                f"for a rank-{component.element_ndim} tensor in {grid.ndim}D."
            )

        # Manage the signature.
        self.__signature__: Tuple[int, ...] = validate_rank_signature(
            component.element_ndim, signature=signature
        )

    # ------------------------------------ #
    # Properties                           #
    # ------------------------------------ #
    @property
    def signature(self) -> Tuple[int, ...]:
        """
        The tensor signature specifying variance of each index.

        Returns
        -------
        tuple of int
            A tuple of `1` (contravariant) and `-1` (covariant) values.
        """
        return self.__signature__

    @property
    def is_scalar(self) -> bool:
        """
        Whether this tensor is a scalar (rank-0).

        Returns
        -------
        bool
        """
        return self.rank == 0

    @property
    def is_vector(self) -> bool:
        """
        Whether this tensor is a vector (rank-1, contravariant).

        Returns
        -------
        bool
        """
        return self.signature == (1,)

    @property
    def is_covector(self) -> bool:
        """
        Whether this tensor is a covector (rank-1, covariant).

        Returns
        -------
        bool
        """
        return self.signature == (-1,)

    @property
    def tensor_class(self) -> Tuple[int, int]:
        """
        Return the (p, q) signature of the tensor.

        This represents the number of contravariant (p) and covariant (q) indices.

        Returns
        -------
        tuple of int
            A tuple (p, q) where:
            - p = number of contravariant indices (1)
            - q = number of covariant indices (-1)
        """
        return signature_to_tensor_class(self.__signature__)

    @property
    def dependence(self) -> DenseTensorDependence:
        """
        The symbolic coordinate dependence object for this tensor field.

        This property returns a :class:`~differential_geometry.dependence.DenseTensorDependence`
        instance that encodes the symbolic dependence of each component of the tensor field
        on the coordinate axes. It is used in symbolic differential geometry operations
        (e.g., gradient, divergence, Laplacian) to track and propagate analytical dependence
        information through transformations.

        The object is lazily constructed on first access, using:

        - the coordinate system associated with the grid
        - the tensor rank of the field
        - the set of grid axes on which the field is defined

        Returns
        -------
        DenseTensorDependence
            Symbolic dependence tracker for this tensor field.

        Notes
        -----

        - This object is automatically populated with the correct tensor rank.
        - The `dependent_axes` argument is inferred from the field’s spatial axes.
        - This is used internally for symbolic propagation in operations like `gradient()` or `raise_index()`.

        See Also
        --------
        :class:`~differential_geometry.symbolic.DenseTensorDependence`
        :meth:`DenseTensorField.gradient`
        :meth:`DenseTensorField.raise_index`
        """
        if self.__dependence__ is None:
            self.__dependence__ = DenseTensorDependence(
                self.__grid__.__cs__, self.rank, dependent_axes=self.axes
            )

        return self.__dependence__
