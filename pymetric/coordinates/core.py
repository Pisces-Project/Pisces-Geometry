"""
Core coordinate system classes for user / developer subclassing.

This module (as opposed to :py:mod:`~coordinates.base`) provides stub classes for defining
custom coordinate systems that fall into a few standard types:

1. **Curvilinear Coordinate Systems**: should be descended from :py:class:`CurvilinearCoordinateSystem`.
2. **Orthogonal Coordinate Systems**: should be descended from :py:class:`OrthogonalCoordinateSystem`.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal

import sympy as sp

from pymetric.coordinates.base import _CoordinateSystemBase


class CurvilinearCoordinateSystem(_CoordinateSystemBase, ABC):
    """
    Base class for general curvilinear coordinate systems.

    This class provides the foundational interface for coordinate systems in which the axes may be
    curved or non-uniform but are not necessarily orthogonal. Unlike `OrthogonalCoordinateSystem`,
    the metric tensor for a `CurvilinearCoordinateSystem` may contain off-diagonal components,
    allowing it to support a broader range of geometries including skewed, stretched, or sheared
    coordinate systems.

    Subclasses must define the coordinate axes and parameters using the ``__AXES__`` and ``__PARAMETERS__``
    class attributes, and must implement the ``__construct_metric_tensor_symbol__`` method to return
    the full symbolic metric tensor (including both diagonal and off-diagonal elements).

    Symbolic processing is triggered either lazily upon instantiation or eagerly at import time,
    depending on the ``__setup_point__`` attribute. Most systems should default to lazy initialization
    (``"init"``) for improved performance and reduced import time.

    Features
    --------
    - Symbolic representation and evaluation of the full (non-diagonal) metric tensor
    - Placeholder for extension to general differential geometry operations (gradient, divergence, etc.)
    - Support for user-defined coordinate systems with custom metric behavior
    - Encodes system axes, parameters, and symbolic processing hooks for downstream utilities

    Notes
    -----
    This class provides minimal functionality on its own, as it is intended to be subclassed
    by users creating general curvilinear systems. Users must implement custom logic for
    differential geometry operations unless handled externally.

    See Also
    --------
    OrthogonalCoordinateSystem : Specialized base class for systems with diagonal metrics.
    _CoordinateSystemBase : Internal base class that defines the shared core logic.
    """

    pass


class OrthogonalCoordinateSystem(_CoordinateSystemBase, ABC):
    r"""
    Base class for orthogonal curvilinear coordinate systems.

    This class provides the foundational symbolic and numerical machinery for defining and working
    with orthogonal coordinate systems—those in which the metric tensor is diagonal and basis vectors
    are mutually perpendicular. It supports automatic symbolic generation of differential geometry
    operators such as the gradient, divergence, and Laplacian, as well as index manipulation utilities
    for raising and lowering tensor components.

    Subclasses should define the coordinate axes and parameter set via the ``__AXES__`` and ``__PARAMETERS__``
    class attributes, and must implement the ``__construct_metric_tensor_symbol__`` method to specify
    the diagonal components of the metric tensor symbolically.

    Coordinate systems may be initialized lazily (at instance creation) or eagerly (at import time) depending
    on the ``__setup_point__`` class attribute. Most systems should use lazy setup for performance.

    Features
    --------
    - Symbolic and numeric computation of metric, inverse metric, and metric density
    - Raising and lowering of tensor indices using orthogonal metric contractions
    - Computation of gradient, divergence, and Laplacian in both covariant and contravariant bases
    - Automatic handling of coordinate slices via `fixed_axes`
    - Support for symbolic dependence analysis for field operations


    Notes
    -----
    This class assumes that the metric tensor is diagonal, as is true for all orthogonal systems.
    Mixed metric components (i.e., :math:`g_{ij}` for :math:`i \\ne j`) are assumed to be zero.
    """

    # @@ CLASS FLAGS @@ #
    # CoordinateSystem flags are used to indicate to the metaclass whether
    # certain procedures should be executed on the class.
    __is_abstract__: bool = (
        True  # Marks this class as abstract - no symbolic processing (unusable)
    )
    __setup_point__: Literal[
        "init", "import"
    ] = "init"  # Determines when symbolic processing should occur.
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
    __axes_symbols__: List[
        sp.Symbol
    ] = None  # The symbolic representations of each axis.
    __parameter_symbols__: Dict[
        str, sp.Symbol
    ] = None  # The symbolic representation of each of the parameters.
    __class_expressions__: Dict[
        str, Any
    ] = {}  # The expressions that are generated for this class.
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
        # Bugfix: 05/27/25 -- enforce __class_expressions__ to prevent corruption.
        cls.__class_expressions__ = {
            "metric_tensor": cls.__construct_metric_tensor_symbol__(
                *cls.__axes_symbols__, **cls.__parameter_symbols__
            )
        }

        cls.__class_expressions__["inverse_metric_tensor"] = sp.Array(
            [1 / _element for _element in cls.__class_expressions__["metric_tensor"]]
        )
        cls.__class_expressions__["metric_density"] = sp.sqrt(
            sp.prod(cls.__class_expressions__["metric_tensor"])
        )

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
        Construct the metric tensor for the coordinate system.

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
