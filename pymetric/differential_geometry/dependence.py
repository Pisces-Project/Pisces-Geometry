"""
Symbolic operational dependence manager classes.

This module contains `SymPy <https://docs.sympy.org/latest/index.html>`_ based handlers
for keeping track of dependence in differential operations.
"""
import operator
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import sympy as sp

from pymetric.differential_geometry.symbolic import (
    adjust_tensor_signature,
    compute_divergence,
    compute_gradient,
    compute_laplacian,
    compute_tensor_gradient,
    compute_tensor_laplacian,
    lower_index,
    raise_index,
)

# =================== Typing Utilities ================== #
# These are utilities for handling typing in this module. They should
# not need modification in most cases and add little to the
# understanding of the code.
if TYPE_CHECKING:
    from pymetric.coordinates.base import _CoordinateSystemBase

# Define a generic field type so that we can shorten type hints.
_GenericFieldType = Union[sp.Function, sp.MutableDenseNDimArray, sp.Basic]


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _TensorDependenceProto(Protocol):
    # data attributes the mixin touches
    coordinate_system: "_CoordinateSystemBase"
    rank: int
    symbolic_proxy: _GenericFieldType

    def to_symbolic_proxy(self) -> _GenericFieldType:
        ...

    @classmethod
    def from_symbolic_proxy(
        cls,
        coordinate_system: "_CoordinateSystemBase",
        symbolic_proxy: _GenericFieldType,
    ) -> "DependenceObject":
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _OperatorsDependenceProto(Protocol):
    # data attributes the mixin touches
    coordinate_system: "_CoordinateSystemBase"
    shape: Tuple[int, ...]
    is_scalar: bool
    symbolic_proxy: _GenericFieldType

    def to_symbolic_proxy(self) -> _GenericFieldType:
        ...

    @classmethod
    def from_symbolic_proxy(
        cls,
        coordinate_system: "_CoordinateSystemBase",
        symbolic_proxy: _GenericFieldType,
    ) -> "DependenceObject":
        ...


_DepT = TypeVar("_DepT", bound=_TensorDependenceProto)
_DepA = TypeVar("_DepA", bound=_OperatorsDependenceProto)


# ============= Tensor Support Mixins =================== #
# These mixin classes allow us to support operations on different
# dependence structures. The OperatorsMixin supports operations
# on generic arrays / fields and TensorDependenceMixin allows support
# on tensor fields.
class OperatorsMixin(Generic[_DepA]):
    """
    Mixin class providing basic arithmetic operations and differential operators
    for symbolic dependence objects in a coordinate system.

    Supports elementwise binary operations (e.g., addition, subtraction, multiplication,
    division) with other compatible dependence objects or scalars. Also provides
    methods to compute symbolic gradients and Laplacians on each component.

    Intended to be used with classes that define a symbolic proxy, coordinate system,
    and a `from_symbolic_proxy` constructor method.
    """

    # @@ Basic Operation Support @@ #
    # These methods provide mixin support for basic operations
    # like addition, subtraction, etc.
    def __binary_operation__(
        self: _DepA,
        other: Union[int, float, complex, np.ndarray, sp.Basic, _DepA],
        op: Callable[[Any, Any], Any],
        op_name: str,
    ):
        """
        Perform an elementwise binary operation between this object and another operand.

        At it's core, this method performs the underlying operation on this object's
        symbolic proxy and the other objects symbolic proxy. If the other object doesn't have
        a symbolic proxy, we attempt to apply it directly.

        Furthermore, the __is_binary_op_compatible__ method allows for other
        objects with symbolic proxies to be checked for compatibility.
        """
        if hasattr(other, "symbolic_proxy"):
            # This object has a symbolic proxy so we need to
            # check for compatibility and then pass the proxy forward.
            if not self.__is_binary_op_compatible__(other, op_name):
                raise ValueError(
                    f"Cannot perform {op_name}: incompatible objects: {self}, {other})"
                )

            # Now simply apply the operation between the two elements
            try:
                result_exp = op(self.symbolic_proxy, other.__symbolic_proxy__)
            except Exception as e:
                raise ValueError(
                    f"Failed to perform {op_name} between"
                    f" {type(self).__name__} and {type(other).__name__}: {e}"
                ) from e
        else:
            # We don't need to validate. We can simply be direct and try to
            # perform the operation.
            try:
                result_exp = op(self.symbolic_proxy, other)
            except Exception as e:
                raise ValueError(
                    f"Failed to perform {op_name} between"
                    f" {type(self).__name__} and {type(other).__name__}: {e}"
                ) from e

        # Now attempt to convert the result expression into a new
        # object using the symbolic proxy.
        try:
            return self.from_symbolic_proxy(self.coordinate_system, result_exp)
        except Exception as e:
            raise ValueError(
                f"Failed to perform {op_name} between"
                f" {type(self).__name__} and {type(other).__name__} due"
                f" to failure in proxy reconstruction."
            ) from e

    def __is_binary_op_compatible__(self: _DepA, other: Any, op_name: str) -> bool:
        """
        Check whether a binary operation with `other` is valid.

        Compatibility is defined by matching coordinate systems and shape.

        Parameters
        ----------
        other : Any
            The second operand in the binary operation.
        op_name : str
            The name of the operation being attempted (e.g., '__add__').

        Returns
        -------
        bool
            True if the operation is allowed; False otherwise.
        """
        # Ensure the other object has necessary attributes
        if not hasattr(other, "coordinate_system") or not hasattr(other, "shape"):
            return False

        return (
            self.coordinate_system == other.coordinate_system
            and self.shape == other.shape
        )

    def __add__(self, other):
        return self.__binary_operation__(other, operator.add, "__add__")

    def __sub__(self, other):
        return self.__binary_operation__(other, operator.sub, "__sub__")

    def __mul__(self, other):
        return self.__binary_operation__(other, operator.mul, "__mul__")

    def __truediv__(self, other):
        return self.__binary_operation__(other, operator.truediv, "__truediv__")

    # @@ Element Wise Differential Operations @@ #
    # These are standard operations that are available to all array structured
    # dependence classes for computing gradients and
    # @@ Element-Wise Differential Operations @@ #
    # These operations compute symbolic derivatives for each component
    # of a scalar or tensor field over a coordinate system.

    def element_wise_gradient(
        self: _DepA,
        *,
        basis: Literal["covariant", "contravariant"] = "covariant",
        as_field: bool = False,
    ):
        """
        Compute the component-wise gradient of the field.

        Parameters
        ----------
        basis : {"covariant", "contravariant"}, optional
            Whether to compute the gradient in the covariant or contravariant basis.
        as_field : bool, optional
            If True, return the raw symbolic expression.
            If False (default), wrap the result in a dependence object.

        Returns
        -------
        Expr or _DepA
            Symbolic gradient expression or a new dependence object.
        """
        inverse_metric = (
            self.coordinate_system.get_expression("inverse_metric_tensor")
            if basis == "contravariant"
            else None
        )

        if hasattr(self.symbolic_proxy, "shape"):
            # We have a shape, so this needs the tensorial
            # treatment.
            grad = compute_tensor_gradient(
                self.symbolic_proxy,
                self.coordinate_system.axes_symbols,
                basis=basis,
                inverse_metric=inverse_metric,
            )
        else:
            grad = compute_gradient(
                self.symbolic_proxy,
                self.coordinate_system.axes_symbols,
                basis=basis,
                inverse_metric=inverse_metric,
            )
        return (
            grad if as_field else self.from_symbolic_proxy(self.coordinate_system, grad)
        )

    def element_wise_laplacian(self: _DepA, *, as_field: bool = False):
        """
        Compute the component-wise Laplacian of the field.

        For scalar fields, computes the scalar Laplacian.
        For tensor fields, applies the Laplace–Beltrami operator to each component.

        Parameters
        ----------
        as_field : bool, optional
            If True, return the raw symbolic expression.
            If False (default), wrap the result in a dependence object.

        Returns
        -------
        Expr or _DepA
            Symbolic Laplacian expression or a new dependence object.
        """
        inverse_metric = self.coordinate_system.get_expression("inverse_metric_tensor")
        metric_density = self.coordinate_system.get_expression("metric_density")

        if len(self.shape) == 0:
            lap = compute_laplacian(
                self.symbolic_proxy,
                self.coordinate_system.axes_symbols,
                inverse_metric=inverse_metric,
                metric_density=metric_density,
            )
        else:
            lap = compute_tensor_laplacian(
                self.symbolic_proxy,
                self.coordinate_system.axes_symbols,
                inverse_metric=inverse_metric,
                metric_density=metric_density,
            )

        return (
            lap if as_field else self.from_symbolic_proxy(self.coordinate_system, lap)
        )


class TensorDependenceMixin(OperatorsMixin, Generic[_DepT]):
    """
    Mixin class providing tensor-specific operations for symbolic dependence objects.

    This mixin extends `OperatorsMixin` by adding operations that rely on tensor structure,
    such as raising/lowering indices, adjusting variance signatures, and computing
    divergence in addition to the general-purpose gradient and Laplacian.

    It assumes the object exposes:
        - `coordinate_system`: a coordinate system with symbolic metric data,
        - `rank`: an integer rank for the tensor (0 = scalar, 1 = vector, etc.),
        - `symbolic_proxy`: a SymPy scalar or array-like expression,
        - `from_symbolic_proxy`: a method to reconstruct a new instance from a proxy.

    Notes
    -----
    Unlike `OperatorsMixin`, which applies differential operators to each component
    independently, this mixin enables true tensorial operations that respect index
    variance (covariant/contravariant behavior).
    """

    def raise_index(self: _DepT, axis: int, /, *, as_field: bool = False):
        """
        Raise a tensor index along the specified axis using the inverse metric tensor.

        Parameters
        ----------
        axis : int
            The index axis to raise.
        as_field : bool, optional
            If True, return a symbolic expression. If False, return a new dependence object.

        Returns
        -------
        _GenericFieldType or _DepT
            Symbolic tensor with raised index or a new dependence object.

        Raises
        ------
        ValueError
            If the tensor is scalar (rank 0).
        """
        if self.rank == 0:
            raise ValueError("Cannot raise an index on a scalar field.")
        proxy = raise_index(
            self.symbolic_proxy,
            self.coordinate_system.get_expression("inverse_metric_tensor"),
            axis=axis,
        )
        return (
            proxy
            if as_field
            else self.from_symbolic_proxy(self.coordinate_system, proxy)
        )

    def lower_index(self: _DepT, axis: int, /, *, as_field: bool = False):
        """
        Lower a tensor index along the specified axis using the metric tensor.

        Parameters
        ----------
        axis : int
            The index axis to lower.
        as_field : bool, optional
            If True, return a symbolic expression. If False, return a new dependence object.

        Returns
        -------
        _GenericFieldType or _DepT
            Symbolic tensor with lowered index or a new dependence object.

        Raises
        ------
        ValueError
            If the tensor is scalar (rank 0).
        """
        if self.rank == 0:
            raise ValueError("Cannot lower an index on a scalar field.")
        proxy = lower_index(
            self.symbolic_proxy,
            self.coordinate_system.get_expression("metric_tensor"),
            axis=axis,
        )
        return (
            proxy
            if as_field
            else self.from_symbolic_proxy(self.coordinate_system, proxy)
        )

    def adjust_tensor_signature(
        self: _DepT,
        variance_in: Sequence[int],
        variance_out: Sequence[int],
        /,
        *,
        as_field: bool = False,
    ):
        """
        Adjust the tensor variance (covariant/contravariant) of each index.

        Parameters
        ----------
        variance_in : Sequence[int]
            The current variance of each index (0 = covariant, 1 = contravariant).
        variance_out : Sequence[int]
            The target variance signature to convert to.
        as_field : bool, optional
            If True, return the symbolic expression. If False, return a new dependence object.

        Returns
        -------
        _GenericFieldType or _DepT
            Symbolic tensor with adjusted signature or a new dependence object.

        Raises
        ------
        ValueError
            If the variance vector lengths do not match the tensor rank.
        """
        if len(variance_in) != self.rank or len(variance_out) != self.rank:
            raise ValueError("Variance vectors must match tensor rank.")
        metric = self.coordinate_system.get_expression("metric_tensor")
        inv_metric = self.coordinate_system.get_expression("inverse_metric_tensor")
        transformed = adjust_tensor_signature(
            self.symbolic_proxy,
            variance_in,
            variance_out,
            metric=metric,
            inverse_metric=inv_metric,
        )
        return (
            transformed
            if as_field
            else self.from_symbolic_proxy(self.coordinate_system, transformed)
        )

    def gradient(
        self: _DepT,
        *,
        basis: Literal["covariant", "contravariant"] = "covariant",
        as_field: bool = False,
    ):
        """
        Compute the elementwise gradient of the tensor field.

        This delegates to `element_wise_gradient()` from `OperatorsMixin`, treating each
        tensor component as an independent scalar function.

        Parameters
        ----------
        basis : {"covariant", "contravariant"}, optional
            Whether to return the result in the covariant or contravariant basis.
        as_field : bool, optional
            Whether to return the raw symbolic result or wrap it as a dependence object.

        Returns
        -------
        _GenericFieldType or _DepT
            The gradient expression or new dependence object.
        """
        return self.element_wise_gradient(basis=basis, as_field=as_field)

    def divergence(
        self: _DepT,
        *,
        basis: Literal["covariant", "contravariant"] = "contravariant",
        as_field: bool = False,
    ):
        """
        Compute the divergence of a rank-1 tensor field.

        This is a true tensorial divergence that contracts the covariant derivative
        with the appropriate metric volume form and raises indices if needed.

        Parameters
        ----------
        basis : {"covariant", "contravariant"}, optional
            Basis in which to compute the divergence.
        as_field : bool, optional
            Whether to return the raw symbolic result or wrap it as a dependence object.

        Returns
        -------
        _GenericFieldType or _DepT
            The divergence expression or new dependence object.

        Raises
        ------
        ValueError
            If the tensor rank is not 1 (i.e., not a vector field).
        """
        if self.rank != 1:
            raise ValueError("Divergence only defined for rank‑1 tensors.")
        inv_metric = self.coordinate_system.get_expression("inverse_metric_tensor")
        metric_density = self.coordinate_system.get_expression("metric_density")
        div = compute_divergence(
            self.symbolic_proxy,
            self.coordinate_system.axes_symbols,
            basis=basis,
            inverse_metric=inv_metric,
            metric_density=metric_density,
        )
        return (
            div if as_field else self.from_symbolic_proxy(self.coordinate_system, div)
        )

    def laplacian(self: _DepT, *, as_field: bool = False):
        """
        Compute the Laplacian of the tensor field (component-wise).

        This calls `element_wise_laplacian()` from `OperatorsMixin`, applying the
        Laplace–Beltrami operator to each component independently.

        Parameters
        ----------
        as_field : bool, optional
            Whether to return the raw symbolic result or wrap it as a dependence object.

        Returns
        -------
        _GenericFieldType or _DepT
            The Laplacian expression or new dependence object.
        """
        return self.element_wise_laplacian(as_field=as_field)


# ============= Dependence Tensor Generators ============ #
# This section of the code provides a couple of methods for converting
# common tensor conventions (dense, sparse, etc.) to symbolic models
# that reflect the correct tensor dependence.
class DependenceObject(ABC):
    """
    Base class for representing symbolic dependence of tensors on coordinates.

    This class serves as the foundation for modeling whether a scalar or tensor field
    depends on specific coordinate axes in a given coordinate system. It provides
    interfaces to construct a symbolic proxy of the tensor and to reconstruct a dependence
    object from a symbolic expression.

    Subclasses must implement:
        - :meth:`to_symbolic_proxy`: Create a dummy symbolic tensor/field to represent dependence.
        - :meth:`from_symbolic_proxy`: Reconstruct a dependence object from an existing symbolic proxy.

    Parameters
    ----------
    coordinate_system : _CoordinateSystemBase
        The coordinate system in which the dependence is defined.
    """

    # @@ Initialization @@ #
    def __init__(self, coordinate_system: "_CoordinateSystemBase") -> None:
        """
        Initialize the base dependence object with a coordinate system.

        This constructor sets up the foundational context for tensor dependence by
        storing the coordinate system and deferring construction of the symbolic proxy
        until it is explicitly requested via `symbolic_proxy`.

        Subclasses should extend this constructor to store additional structural
        properties (e.g., shape, dependent axes, tensor rank), but **must always**
        call `super().__init__(coordinate_system)` to ensure proper initialization.

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system that defines the geometric context for symbolic dependence.
        """
        self.__cs__: "_CoordinateSystemBase" = coordinate_system
        self.__symbolic_proxy__: Optional[_GenericFieldType] = None

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the object."""
        return f"<{self.__class__.__name__} | {self.coordinate_system}>"

    def __str__(self) -> str:
        return self.__repr__()

    # @@ Universal Properties @@ #
    # These are universal properties of all
    # DependenceObjects, but may be supplemented in
    # subclasses.
    @property
    def coordinate_system(self) -> "_CoordinateSystemBase":
        """
        The coordinate system associated with this dependence object.

        Returns
        -------
        _CoordinateSystemBase
            The coordinate system in which the tensor is defined.
        """
        return self.__cs__

    @property
    def coordinates_ndim(self) -> int:
        """
        Number of coordinate dimensions in the associated coordinate system.

        Returns
        -------
        int
            Dimensionality of the coordinate system.
        """
        return self.__cs__.ndim

    @property
    def symbolic_proxy(self) -> _GenericFieldType:
        """
        A symbolic proxy field or tensor that represents coordinate dependence.

        This property lazily constructs a symbolic representation of the field or tensor
        that depends on specific coordinates, useful for symbolic differential operations
        or analysis of tensor structure.

        Returns
        -------
        _GenericFieldType
            A SymPy function or tensor representing the tensor's coordinate dependence.
        """
        if self.__symbolic_proxy__ is None:
            self.__symbolic_proxy__ = self.to_symbolic_proxy()
        return self.__symbolic_proxy__

    # @@ Abstract Methods @@ #
    # These two methods are critical for all subclasses and
    # are used to convert symbolic proxies to / from the base class.
    @abstractmethod
    def to_symbolic_proxy(self) -> _GenericFieldType:
        """
        Construct a symbolic proxy representing this dependence.

        This method should return a dummy symbolic field or tensor using SymPy,
        where the coordinate axes on which the tensor depends are expressed
        via symbolic function arguments.

        Returns
        -------
        _GenericFieldType
            A symbolic field or tensor constructed using SymPy.
        """
        ...

    @classmethod
    @abstractmethod
    def from_symbolic_proxy(
        cls,
        coordinate_system: "_CoordinateSystemBase",
        symbolic_proxy: _GenericFieldType,
    ) -> "DependenceObject":
        """
        Construct a dependence object by analyzing a symbolic proxy.

        This method should parse a symbolic expression and infer the shape,
        rank, and axes of dependence in order to reconstruct a subclass instance.

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system in which the symbolic expression is defined.
        symbolic_proxy : _GenericFieldType
            A symbolic representation (scalar or tensor) of the field.

        Returns
        -------
        DependenceObject
            An instance of the subclass representing the symbolic dependence.
        """
        ...


class DenseDependenceObject(DependenceObject, OperatorsMixin):
    """
    Represents a dense symbolic dependence on coordinate axes within a coordinate system.

    This class models a scalar or tensor field where every component shares the same
    symbolic dependence on one or more coordinate axes. In a dense dependence object,
    the entire field is treated uniformly: all components are assumed to depend on the
    same subset of coordinate axes (e.g., ["r", "theta"]). This contrasts with sparse
    or component-wise symbolic representations where each component could depend on
    different variables.

    DenseDependenceObject supports:
    - Uniform shape and rank definitions for the symbolic field.
    - Construction of full symbolic proxies (scalars or tensors).
    - Element-wise symbolic operations and differential operators.
    - Dependence introspection and shape-level comparison.

    It builds on the abstract `DependenceObject` base class by adding tensor shape and
    axis metadata, and it mixes in `OperatorsMixin` to enable symbolic arithmetic and
    differential operations.

    Notes
    -----
    The symbolic proxy is lazily constructed using SymPy. For tensor fields, a
    `MutableDenseNDimArray` of symbolic functions is used, with each component named
    based on its index.

    Examples
    --------
    >>> from pymetric.coordinates import SphericalCoordinateSystem
    >>> from pymetric.utilities.logging import pg_log
    >>> pg_log.disabled = True
    >>>
    >>> u = SphericalCoordinateSystem() # doctest: +ELLIPSIS
    >>>
    >>> # Construct the dependence
    >>> # object.
    >>> obj = DenseDependenceObject(u, (3,), dependent_axes=["r", "theta"])
    >>> obj.shape
    (3,)
    >>> obj.depends_on("r")
    True
    >>> obj.symbolic_proxy  # Returns a symbolic vector field
    [T_r(r, theta), T_theta(r, theta), T_phi(r, theta)]

    """

    # @@ Initialization @@ #
    # The __init__ here builds on that of DependenceObject
    # to clarify structure for dense objects.
    def __init__(
        self,
        coordinate_system: "_CoordinateSystemBase",
        shape: Sequence[int],
        /,
        *,
        dependent_axes: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """
        Initialize a dense dependence object with shape and dependent axes.

        This constructor defines the tensor shape and determines which coordinate
        axes the symbolic expression should depend on. If `dependent_axes` is not
        provided, it defaults to full dependence on all coordinate axes.

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system that defines the geometric context.
        shape : Sequence[int] or Tuple[int, ...]
            The tensor shape of the symbolic object (e.g., (3,) for a vector).
        dependent_axes : str or Sequence[str], optional
            The coordinate axes on which the tensor depends. Can be a single axis name
            (e.g., "r") or a list of axis names. If None, the tensor is assumed to
            depend on all axes in the coordinate system.
        """
        super().__init__(coordinate_system)

        # Store shape and scalar flag
        self.__shape__: Tuple[int, ...] = tuple(shape)
        self.__is_scalar__: bool = len(self.__shape__) == 0

        # Normalize the dependent axes
        if dependent_axes is None:
            # Default: depend on all axes
            dependent_axes = self.__cs__.__AXES__[:]
        elif isinstance(dependent_axes, str):
            # Single axis string → list
            dependent_axes = [dependent_axes]

        # Canonicalize and store ordered dependent axes and their index positions
        self.__dependent_axes__: List[str] = self.__cs__.order_axes_canonical(
            list(dependent_axes)
        )
        self.__dependent_axes_idx__: List[int] = [
            self.__cs__.convert_axes_to_indices(ax) for ax in self.__dependent_axes__
        ]

    # @@ Dunder Methods @@ #
    def __repr__(self) -> str:
        return (
            f"DenseDependenceObject(cs={self.coordinate_system}, shape={self.shape}, "
            f"axes={self.dependent_axes})"
        )

    def __str__(self) -> str:
        summary = "scalar" if self.is_scalar else f"tensor of shape {self.shape}"
        return f"<DenseDependenceObject over {self.coordinate_system} | {summary} | depends on {self.dependent_axes}>"

    def __eq__(self, other: "DenseDependenceObject") -> bool:
        # For two DenseDependenceObjects to be equivalent, they must
        # be of the same type, share the same coordinate system, have
        # the same shape, and have the same dependence.
        return (
            type(self) is type(other)
            and self.coordinate_system == other.coordinate_system
            and self.shape == other.shape
            and self.dependent_axes == other.dependent_axes
        )

    # @@ Universal Properties @@ #
    # These are common structural attributes of all DependenceObjects.
    # Subclasses may override or extend them, but the base logic
    # supports consistent behavior for symbolic tensor fields.

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The tensor shape of this symbolic field.

        Returns
        -------
        Tuple[int, ...]
            A tuple describing the shape (e.g., (), (3,), (3, 3)).
        """
        return self.__shape__

    @property
    def rank(self) -> int:
        """
        The tensor rank (number of indices) of this field.

        Returns
        -------
        int
            The number of tensor indices, equivalent to `len(shape)`.
        """
        return 0 if self.__is_scalar__ else len(self.__shape__)

    @property
    def is_scalar(self) -> bool:
        """
        Whether this object represents a scalar field (i.e., rank 0).

        Returns
        -------
        bool
            True if this is a scalar field; False otherwise.
        """
        return self.__is_scalar__

    @property
    def dependent_axes(self) -> List[str]:
        """
        The list of coordinate axis names this field depends on.

        This is returned as a copy to prevent accidental mutation.

        Returns
        -------
        List[str]
            Ordered list of axis names (e.g., ['r', 'theta']).
        """
        return self.__dependent_axes__[:]

    @property
    def axes_symbols(self) -> List[sp.Symbol]:
        """
        The symbolic representations of the dependent coordinate axes.

        These are SymPy symbols corresponding to each axis name
        in `dependent_axes`.

        Returns
        -------
        List[sympy.Symbol]
            List of SymPy axis symbols.
        """
        return [self.__cs__.axes_symbols[i] for i in self.__dependent_axes_idx__]

    # @@ Abstract Methods @@ #
    # These two methods are critical for all subclasses and
    # are used to convert symbolic proxies to / from the base class.
    def to_symbolic_proxy(self) -> _GenericFieldType:
        """
        Construct a symbolic proxy representing this tensor or scalar field.

        This method builds a symbolic field or tensor using SymPy function objects.
        For scalar fields, it returns a single symbolic function depending on the
        relevant coordinate symbols. For tensors, it returns a dense symbolic array
        of the appropriate shape, with each component named based on its index.

        Returns
        -------
        sympy.Function or sympy.MutableDenseNDimArray
            A symbolic scalar or tensor function with coordinate dependence.
        """
        if self.is_scalar:
            # Scalar field: return a single symbolic function depending on axes
            return sp.Function("T")(*self.axes_symbols)

        # Tensor field: construct a dense symbolic array with named components
        proxy = sp.MutableDenseNDimArray.zeros(*self.__shape__)

        for idx in np.ndindex(*self.__shape__):
            # Build a label like T_r or T_01 depending on coordinate naming
            label = "".join(str(idx))
            proxy[idx] = sp.Function(f"T_{label}")(*self.axes_symbols)

        return proxy

    @classmethod
    def from_symbolic_proxy(
        cls,
        coordinate_system: "_CoordinateSystemBase",
        symbolic_proxy: _GenericFieldType,
    ) -> "DenseDependenceObject":
        """
        Reconstruct a DenseDependenceObject from a symbolic proxy expression.

        This method is the inverse of `to_symbolic_proxy`. It inspects a symbolic expression—
        either a scalar symbolic function or a dense symbolic tensor array—to determine:

        - its shape (based on `.shape` if present),
        - its dependent coordinate axes (based on free symbols).

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system context in which the symbolic proxy is defined.
        symbolic_proxy : _GenericFieldType
            A symbolic scalar or tensor expression (e.g., from SymPy).

        Returns
        -------
        DenseDependenceObject
            A new instance initialized from the inferred shape and dependencies.
        """
        # Extract all free symbols from the expression
        free_syms = symbolic_proxy.free_symbols
        dependent_axes = [str(sym) for sym in free_syms]

        # Determine shape: assume scalar if `.shape` is not available
        shape = () if not hasattr(symbolic_proxy, "shape") else symbolic_proxy.shape

        return cls(coordinate_system, shape, dependent_axes=dependent_axes)

    # @@ Basic Methods @@ #
    # These are methods specific to these dense structural
    # methods.
    def depends_on(self, axis: str) -> bool:
        """
        Check whether this object symbolically depends on a given coordinate axis.

        This method inspects the symbolic structure and returns whether the field
        (scalar or tensor) has symbolic dependence on the specified axis.

        Parameters
        ----------
        axis : str
            The name of the coordinate axis (e.g., "r", "theta", "z").

        Returns
        -------
        bool
            True if this object depends on the given axis; False otherwise.
        """
        return axis in self.dependent_axes


class DenseTensorDependence(DenseDependenceObject, TensorDependenceMixin):
    """
    Dense symbolic tensor dependence with attached tensor-aware operators.

    This subclass uses the tensor rank to define the shape and supports symbolic
    tensor operations (e.g., raising/lowering indices, divergence), in addition to
    element-wise operations.

    Parameters
    ----------
    coordinate_system : _CoordinateSystemBase
        The coordinate system in which the tensor is defined.
    rank : int
        The rank (number of tensor indices) for the tensor. Shape will be (ndim,)*rank.
    dependent_axes : str or Sequence[str], optional
        The coordinate axes on which the tensor depends.
    """

    def __init__(
        self,
        coordinate_system: "_CoordinateSystemBase",
        rank: int,
        /,
        *,
        dependent_axes: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """
        Initialize a dense tensor dependence object with a given rank and coordinate dependence.

        This constructor extends `DenseDependenceObject` by inferring the shape of the tensor
        based on its rank and the dimensionality of the coordinate system. For example, a
        rank-2 tensor in a 3D coordinate system will have shape (3, 3).

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
            The coordinate system that defines the geometric context.
        rank : int
            The rank (number of tensor indices) of the tensor field.
            A rank of 0 indicates a scalar field.
        dependent_axes : str or Sequence[str], optional
            The coordinate axes the tensor depends on. Can be a single axis name or a list.
            If None, the tensor is assumed to depend on all coordinate axes.
        """
        # Infer shape from rank and coordinate dimensionality
        shape = () if rank == 0 else (coordinate_system.ndim,) * rank

        # Delegate to the base DenseDependenceObject constructor
        super().__init__(coordinate_system, shape, dependent_axes=dependent_axes)

        # Store rank explicitly for clarity and subclass use
        self.__rank__: int = rank

    # @@ Dunder Methods @@ #
    def __repr__(self) -> str:
        return (
            f"DenseTensorDependence(cs={self.coordinate_system}, rank={self.rank}, "
            f"axes={self.dependent_axes})"
        )

    def __str__(self) -> str:
        summary = "scalar" if self.rank == 0 else f"tensor of rank {self.rank}"
        return f"<DenseTensorDependence over {self.coordinate_system} | {summary} | depends on {self.dependent_axes}>"

    # @@ Universal Properties @@ #
    # These are universal properties of all
    # DependenceObjects, but may be supplemented in
    # subclasses.
    @property
    def rank(self) -> int:
        """
        The tensor rank (number of indices) of this field.

        Returns
        -------
        int
            The rank explicitly stored during construction.
        """
        return self.__rank__

    # @@ Abstract Methods @@ #
    # These two methods are critical for all subclasses and
    # are used to convert symbolic proxies to / from the base class.
    def to_symbolic_proxy(self) -> _GenericFieldType:
        """
        Construct a symbolic proxy representing this tensor or scalar field.

        This method builds a symbolic field or tensor using SymPy function objects.
        For scalar fields, it returns a single symbolic function depending on the
        relevant coordinate symbols. For tensors, it returns a dense symbolic array
        of the appropriate shape, with each component named based on its index.

        Returns
        -------
        sympy.Function or sympy.MutableDenseNDimArray
            A symbolic scalar or tensor function with coordinate dependence.
        """
        if self.is_scalar:
            # Scalar field: return a single symbolic function depending on axes
            return sp.Function("T")(*self.axes_symbols)

        # Tensor field: construct a dense symbolic array with named components
        coord_labels = np.asarray(self.__cs__.__AXES__)
        proxy = sp.MutableDenseNDimArray.zeros(*self.__shape__)

        for idx in np.ndindex(*self.__shape__):
            # Build a label like T_r or T_01 depending on coordinate naming
            label = "".join(coord_labels[list(idx)])
            proxy[idx] = sp.Function(f"T_{label}")(*self.axes_symbols)

        return proxy

    @classmethod
    def from_symbolic_proxy(
        cls,
        coordinate_system: "_CoordinateSystemBase",
        symbolic_proxy: _GenericFieldType,
    ) -> "DenseTensorDependence":
        """
        Rebuild a DenseTensorDependence from a symbolic proxy.

        Parameters
        ----------
        coordinate_system : _CoordinateSystemBase
        symbolic_proxy : sympy expression

        Returns
        -------
        DenseTensorDependence
        """
        free_syms = symbolic_proxy.free_symbols
        dependent_axes = [str(sym) for sym in free_syms]
        rank = 0 if not hasattr(symbolic_proxy, "shape") else len(symbolic_proxy.shape)
        return cls(coordinate_system, rank, dependent_axes=dependent_axes)
