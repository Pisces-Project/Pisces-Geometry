"""
Mixin typing protocols to support various mixin classes for fields.
"""
# mypy: ignore-errors
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import numpy as np
from fields.utils._typing import ComponentDictionary
from numpy.typing import ArrayLike

# ============= Typing Management ================== #
# Load in type checking only imports so that mypy and other
# static checkers can pass.
if TYPE_CHECKING:
    import unyt

    from pymetric.coordinates.base import _CoordinateSystemBase
    from pymetric.fields.buffers import BufferRegistry
    from pymetric.fields.buffers.base import BufferBase
    from pymetric.fields.components import FieldComponent
    from pymetric.fields.utils._typing import ComponentIndex, SignatureInput
    from pymetric.grids.base import GridBase
    from pymetric.grids.utils._typing import AxesInput


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsFieldCore(Protocol):
    __grid__: "GridBase"
    __components__: ComponentDictionary
    __components_view__: MappingProxyType
    grid: "GridBase"
    coordinate_system: "_CoordinateSystemBase"
    components: MappingProxyType


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsSFieldCore(_SupportsFieldCore):
    num_components: int
    component_list: List["ComponentIndex"]

    def __getitem__(self, index: "ComponentIndex") -> FieldComponent:
        ...

    def __setitem__(self, index: "ComponentIndex", value: FieldComponent) -> None:
        ...

    def __delitem__(self, index: "ComponentIndex"):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator["ComponentIndex"]:
        ...

    def __contains__(self, item: "ComponentIndex") -> bool:
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsDFieldCore(_SupportsFieldCore):
    component: "FieldComponent"
    __component__: "FieldComponent"
    axes: List[str]
    rank: int
    naxes: int
    buffer: "BufferBase"
    shape: Tuple[int, ...]
    spatial_shape: Tuple[int, ...]
    element_shape: Tuple[int, ...]
    units: "unyt.Unit"
    size: int
    ndim: int
    spatial_ndim: int
    element_ndim: int
    dtype: np.dtype

    def __getitem__(self, idx: Any) -> Any:
        ...

    def __setitem__(self, idx: Any, value: Any) -> Any:
        ...

    @classmethod
    def _wrap_comp_from_op(cls, operation: str, *args, **kwargs):
        ...

    @classmethod
    def from_func(cls, *args, **kwargs) -> "_SupportsDFieldCore":
        ...

    @classmethod
    def from_array(cls, *args, **kwargs) -> "_SupportsDFieldCore":
        ...

    @classmethod
    def zeros(cls, *args, **kwargs) -> "_SupportsDFieldCore":
        ...

    @classmethod
    def ones(cls, *args, **kwargs) -> "_SupportsDFieldCore":
        ...

    @classmethod
    def full(cls, *args, **kwargs) -> "_SupportsDFieldCore":
        ...

    @classmethod
    def zeros_like(cls, other, *args, **kwargs) -> "_SupportsDFieldCore":
        ...

    @classmethod
    def ones_like(cls, other, *args, **kwargs) -> "_SupportsDFieldCore":
        ...

    @classmethod
    def full_like(cls, other, *args, **kwargs) -> "_SupportsDFieldCore":
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsDFieldDMathOps(_SupportsDFieldCore):
    # --- Utility Methods --- #
    def determine_op_dependence(self, opname: str, *args, **kwargs) -> List[str]:
        ...

    def _process_output_to_field_or_array(
        self,
        output: Any,
        *args,
        as_array: bool = False,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        buffer_args: Optional[Sequence[Any]] = None,
        buffer_kwargs: Optional[Dict] = None,
        output_axes: Optional[Sequence[Any]] = None,
        **kwargs,
    ) -> Union["_SupportsDFieldDMathOps", np.ndarray]:
        ...

    # ======================================= #
    # General Dense Ops                       #
    # ======================================= #
    # recasting of functions from differential geometry's
    # general_ops module.
    def element_wise_partial_derivatives(
        self,
        out: Optional[ArrayLike] = None,
        output_axes: Optional[Sequence[str]] = None,
        as_array: bool = False,
        **kwargs,
    ) -> "_SupportsDFieldDMathOps":
        ...

    def element_wise_laplacian(
        self,
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
    ) -> "_SupportsDFieldDMathOps":
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsDTFieldCore(_SupportsDFieldCore):
    signature: Tuple[Literal[-1, 1], ...]
    is_scalar: bool
    is_vector: bool
    is_covector: bool
    tensor_class: Tuple[int, int]

    @classmethod
    def from_func(
        cls, *args, signature: Optional["SignatureInput"] = None, **kwargs
    ) -> "_SupportsDTFieldCore":
        ...

    @classmethod
    def from_array(
        cls, *args, signature: Optional["SignatureInput"] = None, **kwargs
    ) -> "_SupportsDTFieldCore":
        ...

    @classmethod
    def zeros(
        cls, *args, signature: Optional["SignatureInput"] = None, **kwargs
    ) -> "_SupportsDTFieldCore":
        ...

    @classmethod
    def ones(
        cls, *args, signature: Optional["SignatureInput"] = None, **kwargs
    ) -> "_SupportsDTFieldCore":
        ...

    @classmethod
    def full(
        cls, *args, signature: Optional["SignatureInput"] = None, **kwargs
    ) -> "_SupportsDTFieldCore":
        ...


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsDTFieldDMathOps(_SupportsDTFieldCore):
    pass


# noinspection PyMissingOrEmptyDocstring
@runtime_checkable
class _SupportsFieldComponentCore(Protocol):
    __grid__: "GridBase"
    __axes__: List[str]
    __buffer__: "BufferBase"
    grid: "GridBase"
    axes: List[str]
    naxes: int
    buffer: "BufferBase"
    shape: Tuple[int, ...]
    spatial_shape: Tuple[int, ...]
    element_shape: Tuple[int, ...]
    units: "unyt.Unit"
    size: int
    ndim: int
    spatial_ndim: int
    element_ndim: int
    dtype: np.dtype

    def as_array(self) -> np.ndarray:
        ...

    def as_buffer_core(self) -> Any:
        ...

    def as_buffer_repr(self) -> Any:
        ...

    def as_unyt_array(self) -> "unyt.unyt_array":
        ...

    def as_array_in_axes(self, axes: "AxesInput", **kwargs) -> np.ndarray:
        ...

    def as_buffer_core_in_axes(self, axes: "AxesInput", **kwargs) -> Any:
        ...

    def as_buffer_repr_in_axes(self, axes: "AxesInput", **kwargs) -> Any:
        ...

    def as_unyt_array_in_axes(self, axes: "AxesInput", **kwargs) -> "unyt.unyt_array":
        ...

    @classmethod
    def zeros(
        cls,
        grid: "GridBase",
        axes,
        *args,
        element_shape=None,
        buffer_class=None,
        buffer_registry=None,
        **kwargs,
    ) -> "_SupportsFieldComponentCore":
        ...

    @classmethod
    def ones(
        cls,
        grid: "GridBase",
        axes,
        *args,
        element_shape=None,
        buffer_class=None,
        buffer_registry=None,
        **kwargs,
    ) -> "_SupportsFieldComponentCore":
        ...

    @classmethod
    def full(
        cls,
        grid: "GridBase",
        axes,
        *args,
        fill_value: float = 0.0,
        element_shape=None,
        buffer_class=None,
        buffer_registry=None,
        **kwargs,
    ) -> "_SupportsFieldComponentCore":
        ...

    @classmethod
    def zeros_like(
        cls, other: "_SupportsFieldComponentCore", *args, **kwargs
    ) -> "_SupportsFieldComponentCore":
        ...

    @classmethod
    def ones_like(
        cls, other: "_SupportsFieldComponentCore", *args, **kwargs
    ) -> "_SupportsFieldComponentCore":
        ...

    @classmethod
    def full_like(
        cls, other: "_SupportsFieldComponentCore", *args, **kwargs
    ) -> "_SupportsFieldComponentCore":
        ...

    # ------------------------------ #
    # Unit Handling                  #
    # ------------------------------ #
    # These method supplement those above to help with
    # unit handling.

    # === Inplace unit manipulation === #
    def convert_to_units(
        self,
        units: Union[str, unyt.Unit],
        equivalence: Optional[str] = None,
        **kwargs,
    ):
        ...

    def convert_to_base(
        self,
        unit_system: Optional[str] = None,
        equivalence: Optional[str] = None,
        **kwargs,
    ):
        ...

    # === Casting Unit Manipulation === #

    def in_units(
        self,
        units: Union[str, unyt.Unit],
        *args,
        as_array: bool = False,
        equivalence: Optional[str] = None,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        equiv_kw: Optional[dict] = None,
        **kwargs,
    ):
        ...

    def to(
        self,
        units: Union[str, unyt.Unit],
        *args,
        equivalence: Optional[str] = None,
        buffer_class: Optional[Type["BufferBase"]] = None,
        buffer_registry: Optional["BufferRegistry"] = None,
        as_array: bool = False,
        **kwargs,
    ):
        ...

    def to_value(
        self,
        units: Union[str, unyt.Unit],
        equivalence: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        ...
