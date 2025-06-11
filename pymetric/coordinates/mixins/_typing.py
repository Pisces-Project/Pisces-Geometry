"""
Typing infrastructure for the mixin classes. These are used for
mypy compatibility in mixin classes.
"""
from pathlib import Path

# mypy: ignore-errors
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import sympy as sp

if TYPE_CHECKING:
    import unyt

    from pymetric.coordinates.base import _CoordinateSystemBase

_ExpressionType = Union[
    sp.Symbol,
    sp.Expr,
    sp.Matrix,
    sp.MutableDenseMatrix,
    sp.MutableDenseNDimArray,
    sp.ImmutableDenseMatrix,
    sp.ImmutableDenseNDimArray,
]


# noinspection PyMissingOrEmptyDocstring
class _SupportsCoordinateSystemBase(Protocol):
    """Typing protocol for Pisces Geometry coordinate system classes."""

    __is_abstract__: bool
    __setup_point__: Literal["init", "import"]
    __is_setup__: bool
    __DEFAULT_REGISTRY__: Dict[str, Any]

    # @@ CLASS ATTRIBUTES @@ #
    # The CoordinateSystem class attributes provide some of the core attributes
    # for all coordinate systems and should be adjusted in all subclasses to initialize
    # the correct axes, dimensionality, etc.
    __AXES__: List[str]
    __PARAMETERS__: Dict[str, Any]
    __AXES_DIMENSIONS__: List[sp.Basic]
    __AXES_LATEX__: Dict[str, str]

    # @@ CLASS BUILDING PROCEDURES @@ #
    # During either import or init, the class needs to build its symbolic attributes in order to
    # be usable. The class attributes and relevant class methods are defined in this section
    # of the class object.
    __axes_symbols__: List[sp.Symbol]
    __parameter_symbols__: Dict[str, sp.Symbol]
    __class_expressions__: Dict[str, Any]
    __NDIM__: int

    # --- Core properties --- #
    ndim: int
    axes: List[str]
    parameters: Dict[str, Any]
    metric_tensor_symbol: _ExpressionType
    inverse_metric_tensor_symbol: _ExpressionType
    metric_tensor: Callable
    inverse_metric_tensor: Callable
    axes_symbols: List[sp.Symbol]
    axes_dimensions: List[sp.Basic]
    parameter_symbols: Dict[str, sp.Symbol]
    __expressions__: Dict[str, _ExpressionType] = dict()
    __numerical_expressions__: Dict[str, Callable]

    @classmethod
    def __setup_symbols__(cls):
        ...

    @classmethod
    def __construct_class_expressions__(cls):
        ...

    @classmethod
    def __construct_explicit_class_expressions__(cls):
        ...

    @classmethod
    def __setup_class__(cls):
        ...

    # =============================== #
    # INITIALIZATION                  #
    # =============================== #
    # Many method play into the initialization procedure. To ensure extensibility,
    # these are broken down into sub-methods which can be altered when subclassing the
    # base class.
    def _setup_parameters(self, **kwargs):
        ...

    def _setup_explicit_expressions(self):
        ...

    def __getitem__(self, index: int) -> str:
        ...

    # @@ COORDINATE METHODS @@ #
    # These methods dictate the behavior of the coordinate system including how
    # coordinate conversions behave and how the coordinate system handles differential
    # operations.
    @staticmethod
    def __construct_metric_tensor_symbol__(*args, **kwargs) -> sp.Matrix:
        ...

    @classmethod
    def __compute_Dterm__(cls, *args, **kwargs):
        ...

    @classmethod
    def __compute_Lterm__(cls, *args, **kwargs):
        ...

    # @@ EXPRESSION METHODS @@ #
    # These methods allow the user to interact with derived, symbolic, and numeric expressions.
    def substitute_expression(self, expression: _ExpressionType) -> _ExpressionType:
        ...

    def lambdify_expression(self, expression: Union[str, sp.Basic]) -> Callable:
        ...

    @classmethod
    def get_class_expression(cls, expression_name: str) -> _ExpressionType:
        ...

    @classmethod
    def list_class_expressions(cls) -> List[str]:
        ...

    @classmethod
    def has_class_expression(cls, expression_name: str) -> bool:
        ...

    def get_expression(self, expression_name: str) -> _ExpressionType:
        ...

    def set_expression(
        self, expression_name: str, expression: _ExpressionType, overwrite: bool = False
    ):
        ...

    def list_expressions(self) -> List[str]:
        ...

    def has_expression(self, expression_name: str) -> bool:
        ...

    def get_numeric_expression(self, expression_name: str) -> Callable:
        ...

    # @@ CONVERSION @@ #
    # Perform conversions to / from cartesian coordinates.
    def _convert_native_to_cartesian(self, *args):
        ...

    def _convert_cartesian_to_native(self, *args):
        ...


# noinspection PyMissingOrEmptyDocstring
class _SupportsCoordinateSystemCore(_SupportsCoordinateSystemBase):
    # -------------------------- #
    # Basic Utility Functions    #
    # -------------------------- #
    def pprint(self) -> None:
        ...

    # ------------------------------- #
    # Coordinate Conversion Utilities #
    # ------------------------------- #
    # These methods provide access to the API for
    # coordinate conversion.
    def _check_same_dimension(self, other: "_SupportsCoordinateSystemCore") -> None:
        ...

    def to_cartesian(self, *coords: Any) -> Tuple[np.ndarray, ...]:
        ...

    def from_cartesian(self, *coords: Any) -> Tuple[np.ndarray, ...]:
        ...

    def convert_to(
        self, target_system: "_CoordinateSystemBase", *native_coords: Any
    ) -> Tuple[np.ndarray, ...]:
        ...

    def get_conversion_transform(self, other: "_CoordinateSystemBase") -> Any:
        ...

    def to_hdf5(
        self,
        filename: Union[str, Path],
        group_name: Optional[str] = None,
        overwrite: bool = False,
    ):
        ...

    @classmethod
    def from_hdf5(
        cls,
        filename: Union[str, Path],
        group_name: Optional[str] = None,
        registry: Optional[Dict] = None,
    ):
        ...

    def to_json(self, filepath: Union[str, Path], overwrite: bool = False):
        ...

    @classmethod
    def from_json(cls, filepath: Union[str, Path], registry: Optional[Dict] = None):
        ...

    def to_yaml(self, filepath: Union[str, Path], overwrite: bool = False):
        ...

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path], registry: Optional[Dict] = None):
        ...


# noinspection PyMissingOrEmptyDocstring
class _SupportsCoordinateSystemAxes(_SupportsCoordinateSystemCore):
    # -------------------------------- #
    # Basic Axes Utilities             #
    # -------------------------------- #
    # These basic utilities are really just simple wrappers
    # around logic that could be easily implemented independently.
    def convert_indices_to_axes(
        self, axes_indices: Union[int, Sequence[int]]
    ) -> Union[str, List[str]]:
        ...

    def convert_axes_to_indices(
        self, axes: Union[str, Sequence[str]]
    ) -> Union[int, List[int]]:
        ...

    def build_axes_mask(self, axes: Sequence[str]) -> np.ndarray:
        ...

    def get_axes_from_mask(self, mask: np.ndarray) -> List[str]:
        ...

    def get_mask_from_axes(self, axes: Union[str, Sequence[str]]) -> np.ndarray:
        ...

    def get_mask_from_indices(self, indices: Union[int, Sequence[int]]) -> np.ndarray:
        ...

    def get_indices_from_mask(self, mask: np.ndarray) -> Union[int, List[int]]:
        ...

    # -------------------------------- #
    # Permutations and Order           #
    # -------------------------------- #
    # These methods help with permuting objects and
    # ordering objects according to axes.
    def axes_complement(self, axes: Sequence[str]) -> List[str]:
        ...

    def is_axis(self, axis: Union[str, Sequence[str]]) -> Union[bool, List[bool]]:
        ...

    @staticmethod
    def is_axes_subset(axes_a: Sequence[str], axes_b: Sequence[str]) -> bool:
        ...

    @staticmethod
    def is_axes_superset(axes_a: Sequence[str], axes_b: Sequence[str]) -> bool:
        ...

    def get_free_fixed(
        self,
        axes: Optional[Sequence[str]] = None,
        *,
        fixed_axes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        ...

    @staticmethod
    def get_axes_permutation(
        src_axes: Sequence[str], dst_axes: Sequence[str]
    ) -> List[int]:
        ...

    def get_canonical_axes_permutation(self, axes: Sequence[str]) -> List[int]:
        ...

    @staticmethod
    def get_axes_order(src_axes: Sequence[str], dst_axes: Sequence[str]) -> List[int]:
        ...

    @staticmethod
    def order_axes(src_axes: Sequence[str], dst_axes: Sequence[str]) -> List[str]:
        ...

    @staticmethod
    def in_axes_order(
        iterable: Sequence[Any], src_axes: Sequence[str], dst_axes: Sequence[str]
    ) -> List[Any]:
        ...

    @staticmethod
    def get_canonical_axes_order(src_axes: Sequence[str]) -> List[int]:
        ...

    def order_axes_canonical(self, src_axes: Sequence[str]) -> List[str]:
        ...

    def in_canonical_order(
        self, iterable: Sequence[Any], src_axes: Sequence[str]
    ) -> List[Any]:
        ...

    def resolve_axes(
        self,
        axes: Optional[Sequence[str]] = None,
        *,
        require_subset: bool = True,
        require_order: bool = False,
    ) -> List[str]:
        ...

    def insert_fixed_axes(
        self,
        iterable: Sequence[Any],
        src_axes: Sequence[str],
        fixed_axes: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        ...

    # -------------------------------- #
    # Latex                            #
    # -------------------------------- #
    # This connects axes to latex.
    def get_axes_latex(self, axes: Union[str, Sequence[str]]) -> Union[str, List[str]]:
        ...

    # -------------------------------- #
    # Units                            #
    # -------------------------------- #
    def get_axes_units(self, unit_system: "unyt.UnitSystem") -> List["unyt.Unit"]:
        ...


# noinspection PyMissingOrEmptyDocstring
class _SupportsCoordinateSystemCoordinates(_SupportsCoordinateSystemAxes):
    def coordinate_meshgrid(
        self,
        *coordinate_arrays,
        axes: Optional[Sequence[str]] = None,
        copy: bool = True,
        sparse: bool = False,
    ):
        ...

    def create_coordinate_list(
        self,
        coordinate_arrays: Sequence[np.ndarray],
        /,
        axes: Optional[Sequence[str]] = None,
        *,
        fixed_axes: Optional[Dict[str, float]] = None,
    ) -> List[Union[np.ndarray, float]]:
        ...
