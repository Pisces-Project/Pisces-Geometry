"""
Base classes and metaclass infrastructure for defining coordinate systems inPyMetric.

This module provides the foundational machinery for all coordinate system definitions. It includes:

- ``_CoordinateSystemBase``: the abstract base class for coordinate systems, supporting symbolic and numerical operations,
- ``_CoordinateMeta``: a metaclass that handles automatic symbolic construction and validation of coordinate classes,
- :py:func:`class_expression`: a decorator to mark symbolic methods that are evaluated on demand.

Coordinate systems built on this foundation can define custom metric tensors, symbolic expressions, and conversions
to/from Cartesian coordinates. These systems support tensor calculus operations such as gradients, divergences, and
Laplacians, all respecting the underlying geometry.
"""
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import sympy as sp

from pymetric.differential_geometry import compute_Dterm, compute_Lterm
from pymetric.utilities.logging import pg_log
from pymetric.utilities.symbolic import lambdify_expression

from ._exceptions import CoordinateClassException
from .mixins import (
    CoordinateOperationsMixin,
    CoordinateSystemAxesMixin,
    CoordinateSystemCoreMixin,
    CoordinateSystemIOMixin,
    CoordinateSystemMathMixin,
)

# ============================ #
# Typing Utilities             #
# ============================ #
_ExpressionType = Union[
    sp.Symbol,
    sp.Expr,
    sp.Matrix,
    sp.MutableDenseMatrix,
    sp.MutableDenseNDimArray,
    sp.ImmutableDenseMatrix,
    sp.ImmutableDenseNDimArray,
]

# ============================ #
# Coordinate Base              #
# ============================ #
DEFAULT_COORDINATE_REGISTRY: Dict[str, Any] = {}
""" dict of str, Any: The default registry containing all initialized coordinate system classes.
"""


# noinspection PyTypeChecker
def class_expression(name: Optional[str] = None) -> classmethod:
    """
    Mark a class method as a symbolic expression ("class expression") in a coordinate system.

    Class expressions are symbolic methods that define expressions such as metric tensors,
    Jacobians, or differential geometry terms. When decorated with this function, the method
    is automatically registered during class construction and evaluated on demand via
    :meth:`~coordinates.core.CurvilinearCoordinateSystem.get_class_expression`.

    The decorated method must be a ``@classmethod`` with the following signature:

    .. code-block:: python

        def some_expr(cls, *axes_symbols, **parameter_symbols): ...

    Parameters
    ----------
    name : str, optional
        A custom name to assign to the expression. If omitted, the method name is used.

    Returns
    -------
    classmethod
        The decorated class method with metadata attached for registration and deferred evaluation.

    Notes
    -----
    - The decorator only works on methods already marked as ``@classmethod``.
    - Registered expressions are stored on the class and evaluated once when first accessed.
    - Use :meth:`get_class_expression(name)` to access the expression symbolically.

    Example
    -------
    .. code-block:: python

        class MySystem(CoordinateSystemBase):
            __AXES__ = ["r", "theta"]
            __PARAMETERS__ = {}

            @class_expression()
            @classmethod
            def metric_tensor(cls, r, theta):
                return sp.Matrix([[1, 0], [0, r**2]])

        expr = MySystem.get_class_expression("metric_tensor")
        print(expr)  # => Matrix([[1, 0], [0, r**2]])
    """

    def decorator(func):
        """Add the wrapper around the class expression."""
        if not isinstance(func, classmethod):
            raise TypeError(
                "The @class_expression decorator must be applied to a @classmethod."
            )

        original_func = func.__func__  # Extract underlying function from classmethod

        @wraps(original_func)
        def wrapper(*args, **kwargs):
            """Wrap original function."""
            return original_func(*args, **kwargs)

        # Rewrap as a classmethod
        wrapped_method = classmethod(wrapper)

        # Attach metadata
        wrapped_method.class_expression = True
        wrapped_method.expression_name = name or original_func.__name__

        return wrapped_method

    return decorator


class _CoordinateMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Generate the class object using the basic object
        # procedure. We then make modifications to this.
        cls_object = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Fetch the class flags from the class object. Based on these values, we then
        # make decisions about how to process the class during setup.
        _cls_is_abstract = getattr(cls_object, "__is_abstract__", False)
        _cls_setup_point = getattr(cls_object, "__setup_point__", "init")

        if _cls_is_abstract:
            # We do not process this class at all.
            return cls_object

        # Now validate the class - This is performed even if the initialization is
        # actually performed at init time because it is a very quick function call.
        # noinspection PyTypeChecker
        mcs.validate_coordinate_system_class(cls_object)

        # Add the class to the registry.
        DEFAULT_COORDINATE_REGISTRY[cls_object.__name__] = cls_object

        # Check if the class is supposed to be set up immediately or if we
        # delay.
        if _cls_setup_point == "import":
            # noinspection PyUnresolvedReferences
            cls_object.__setup_class__()

        return cls_object

    @staticmethod
    def validate_coordinate_system_class(cls):
        """
        Validate a new coordinate system class.

        This includes determining the number of dimensions and ensuring
        that bounds and coordinates are all accurate.
        """
        # Check the new class for the required attributes that all classes should have.
        __required_elements__ = [
            "__AXES__",
            "__PARAMETERS__",
            "__axes_symbols__",
            "__parameter_symbols__",
            "__NDIM__",
        ]
        for _re_ in __required_elements__:
            if not hasattr(cls, _re_):
                raise CoordinateClassException(
                    f"Coordinate system {cls.__name__} does not define or inherit an expected "
                    f"class attribute: `{_re_}`."
                )

        # Ensure that we have specified axes and that they have the correct length.
        # The AXES_BOUNDS need to be validated to ensure that they have the correct
        # structure and only specify valid conventions for boundaries.
        if cls.__AXES__ is None:
            raise CoordinateClassException(
                f"Coordinate system {cls.__name__} does not define a set of axes"
                "using the `__AXES__` attribute."
            )

        # Determine the number of dimensions from __AXES__ and ensure that __AXES_BOUNDS__ is
        # the same length as axes.
        cls.__NDIM__ = len(cls.__AXES__)


class _CoordinateSystemBase(
    CoordinateSystemCoreMixin,
    CoordinateSystemIOMixin,
    CoordinateSystemAxesMixin,
    CoordinateOperationsMixin,
    CoordinateSystemMathMixin,
    ABC,
    metaclass=_CoordinateMeta,
):
    """
    Base class for allPyMetric coordinate system classes.

    :py:class:`CoordinateSystemBase` provides the backbone for the symbolic / numerical structure
    of coordinate systems and also acts as a template for developers to use when developing custom coordinate system classes.

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

    # =============================== #
    # CLASS FLAGS / CONFIG            #
    # =============================== #
    # CoordinateSystem flags are used to indicate to the metaclass whether
    # certain procedures should be executed on the class.
    __is_abstract__: bool = (
        True  # Marks this class as abstract - no symbolic processing (unusable)
    )
    __setup_point__: Literal[
        "init", "import"
    ] = "init"  # Determines when symbolic processing should occur.
    __is_setup__: bool = False  # Used to check if the class has already been set up.
    __DEFAULT_REGISTRY__: Dict = DEFAULT_COORDINATE_REGISTRY

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
    __AXES_LATEX__: Dict[str, str] = None
    """LaTeX representations of the coordinate axes in this coordinate system.

    This class flag is entirely optional when implementing new coordinate systems. If
    it is not set, then the axes names are used as the latex representations.
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
    def __setup_symbols__(cls):
        """
        Create symbolic representations of coordinate axes and parameters.

        Populates:
        - `__axes_symbols__`: sympy Symbols for each coordinate axis.
        - `__parameter_symbols__`: sympy Symbols for each parameter.

        This method is called automatically during symbolic setup.
        """
        # For each of the parameters and for each of the axes, generate a
        # symbol and store in the correct class variable.
        cls.__axes_symbols__ = [sp.Symbol(_ax) for _ax in cls.__AXES__]
        cls.__parameter_symbols__ = {_pn: sp.Symbol(_pn) for _pn in cls.__PARAMETERS__}
        pg_log.debug(
            f"Configured symbols for {cls.__name__}: {cls.__axes_symbols__} and {cls.__parameter_symbols__}."
        )

    @classmethod
    def __construct_class_expressions__(cls):
        """
        Register all class-level symbolic expressions defined with @class_expression.

        Scans the method resolution order (MRO) of the class and identifies class methods
        tagged as symbolic expressions.

        Adds them to the `__class_expressions__` dictionary. The expressions are evaluated
        on demand when requested via `get_class_expression()`.

        Notes
        -----
        This only registers the expression. Evaluation is deferred until the first access.
        """
        # begin the iteration through the class __mro__ to find objects
        # in the entire inheritance structure.
        seen = set()
        for base in reversed(cls.__mro__):  # reversed to ensure subclass -> baseclass
            # Check if we need to search this element of the __mro__. We only exit if we find
            # `object` because it's not going to have any worthwhile symbolics.
            if base is object:
                continue

            # Check this element of the __mro__ for any relevant elements that
            # we might want to attach to this class.
            for attr_name, method in base.__dict__.items():
                # Check if we have any interest in processing these methods. If the method is already
                # seen, then we skip it. Additionally, if the class expression is missing the correct
                # attributes, we skip it.
                if (base, attr_name) in seen:
                    continue
                if (not isinstance(method, classmethod)) and not (
                    callable(method) and getattr(method, "class_expression", False)
                ):
                    seen.add((base, attr_name))
                    continue
                elif (isinstance(method, classmethod)) and not (
                    callable(method.__func__)
                    and getattr(method, "class_expression", False)
                ):
                    seen.add((base, attr_name))
                    continue
                seen.add((base, attr_name))

                # At this point, any remaining methods are relevant class expressions which should
                # be registered. Everything is loaded on demand, so we just add the method to the
                # expression dictionary and then (when loading) check it to see if it's loaded or not.
                pg_log.debug(f"Registering {method} to {cls.__name__}.")
                expression_name = getattr(method, "expression_name", attr_name)
                cls.__class_expressions__[expression_name] = method

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
        # Bugfix: 05/27/25 -- enforce __class_expressions__ to prevent corruption.
        cls.__class_expressions__ = {
            "metric_tensor": cls.__construct_metric_tensor_symbol__(
                *cls.__axes_symbols__, **cls.__parameter_symbols__
            )
        }
        # Derive the metric, inverse metric, and the metric density. We call to the
        # __construct_metric_tensor_symbol__ and then take the inverse and the determinant of
        # the matrices.
        cls.__class_expressions__["inverse_metric_tensor"] = cls.__class_expressions__[
            "metric_tensor"
        ].inv()
        cls.__class_expressions__["metric_density"] = sp.sqrt(
            cls.__class_expressions__["metric_tensor"].det()
        )

        # Any additional core expressions can be added here. The ones above can also be modified as
        # needed.

    @classmethod
    def __setup_class__(cls):
        """
        Orchestrates the symbolic setup for a coordinate system class.

        This is the main entry point used during class construction. It performs the following steps:

        1. Initializes coordinate and parameter symbols.
        2. Builds explicit class symbols (things like the metric and metric density)
        3. Registers class expressions.
        4. Sets up internal flags to avoid re-processing.

        Raises
        ------
        CoordinateClassException
            If any part of the symbolic setup fails (e.g., axes, metric, or expressions).
        """
        # Validate the necessity of this procedure. If __is_abstract__, then this should never be reached and
        # if __is_set_up__, then we don't actually need to run it.
        pg_log.debug(f"Setting up coordinate system class: {cls.__name__}.")
        if cls.__is_abstract__:
            raise TypeError(
                f"CoordinateSystem class {cls.__name__} is abstract and cannot be instantiated or constructed."
            )

        if cls.__is_setup__:
            return

        # Set up checks have passed. Now we need to proceed to constructing the axes symbols and
        # the parameter symbols and then constructing the symbolic attributes.
        try:
            cls.__setup_symbols__()
        except Exception as e:
            raise CoordinateClassException(
                f"Failed to setup the coordinate symbols for coordinate system class {cls.__name__} due to"
                f" an error: {e}."
            ) from e

        # Construct the explicitly declared class expressions. These are class expressions which are
        # still registered in `__class_expressions__` but are constructed explicitly as part of class
        # setup. Additional entries can be declared in the `cls.__setup_class_symbolic_attributes__` method.
        # Generally, this is used for things like the metric and inverse metric.
        try:
            cls.__construct_explicit_class_expressions__()
        except Exception as e:
            raise CoordinateClassException(
                f"Failed to setup the metric tensor for coordinate system class {cls.__name__} due to"
                f" an error: {e}."
            ) from e

        # Identify the class expressions and register them in __class_expressions__.
        try:
            cls.__construct_class_expressions__()
        except Exception as e:
            raise CoordinateClassException(
                f"Failed to setup derived class expressions for coordinate system class {cls.__name__} due to"
                f" an error: {e}."
            ) from e

    # =============================== #
    # INITIALIZATION                  #
    # =============================== #
    # Many method play into the initialization procedure. To ensure extensibility,
    # these are broken down into sub-methods which can be altered when subclassing the
    # base class.
    def _setup_parameters(self, **kwargs):
        # Start by creating a carbon-copy of the default parameters.
        _parameters = self.__class__.__PARAMETERS__.copy()

        # For each of the provided kwargs, we need to check that the kwarg is
        # in the _parameters dictionary and then set the value.
        for _parameter_name, _parameter_value in kwargs.items():
            if _parameter_name not in _parameters:
                raise ValueError(
                    f"Parameter `{_parameter_name}` is not a recognized parameter of the {self.__class__.__name__} coordinate system."
                )

            # The parameter name is valid, we just need to set the value.
            _parameters[_parameter_name] = _parameter_value

        return _parameters

    def _setup_explicit_expressions(self):
        """Set up any special symbolic expressions or numerical instances."""
        # Setup the metric, inverse_metric, and the metric density at the instance level.
        self.__expressions__["metric_tensor"] = self.substitute_expression(
            self.__class_expressions__["metric_tensor"]
        )
        self.__expressions__["inverse_metric_tensor"] = self.substitute_expression(
            self.__class_expressions__["inverse_metric_tensor"]
        )
        self.__expressions__["metric_density"] = self.substitute_expression(
            self.__class_expressions__["metric_density"]
        )

        # Setup the numerical metric and other parameters.
        self.__numerical_expressions__["metric_tensor"] = self.lambdify_expression(
            self.__expressions__["metric_tensor"]
        )
        self.__numerical_expressions__[
            "inverse_metric_tensor"
        ] = self.lambdify_expression(self.__expressions__["inverse_metric_tensor"])
        self.__numerical_expressions__["metric_density"] = self.lambdify_expression(
            self.__expressions__["metric_density"]
        )

    def __init__(self, **kwargs):
        """
        Initialize a coordinate system instance with specific parameter values.

        This constructor sets up the symbolic and numerical infrastructure for the coordinate system
        by performing the following steps:

        1. If the class has not already been set up, trigger symbolic construction of metric tensors,
           symbols, and expressions.
        2. Validate and store user-provided parameter values, overriding class defaults.
        3. Substitute parameter values into symbolic expressions to produce instance-specific forms.
        4. Lambdify key expressions (metric tensor, inverse metric, metric density) for numerical evaluation.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specifying values for coordinate system parameters. Each key should match
            a parameter name defined in ``__PARAMETERS__``. Any unspecified parameters will use the class-defined
            default values.

        Raises
        ------
        ValueError
            If a provided parameter name is not defined in the coordinate system.

        """
        # -- Class Initialization -- #
        # For coordinate systems with setup flags for 'init', it is necessary to process
        # symbolics at this point if the class is not initialized.
        self.__class__.__setup_class__()

        # -- Parameter Creation -- #
        # The coordinate system takes a set of kwargs (potentially empty) which specify
        # the parameters of the coordinate system. Each should be adapted into a self.__parameters__ dictionary.
        self.__parameters__ = self._setup_parameters(**kwargs)

        # -- Base Symbol Manipulations -- #
        # Once the class is set up, we need to simplify the metric and other class
        # level symbols to construct the instance level symbols.
        # noinspection PyTypeChecker
        self.__expressions__: Dict[str, _ExpressionType] = dict()
        self.__numerical_expressions__ = dict()
        self._setup_explicit_expressions()

    # =============================== #
    # CORE FUNCTIONALITY              #
    # =============================== #
    # @@ DUNDER METHODS @@ #
    # These should not be altered.
    def __repr__(self):
        return f"<{self.__class__.__name__} - Parameters={self.__parameters__}> "

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    def __len__(self) -> int:
        """
        Return the number of axes in the coordinate system.

        Example
        -------
        >>> cs = MyCoordinateSystem()
        >>> print(len(cs))
        3
        """
        return self.ndim

    def __hash__(self):
        r"""
        Compute a hash value for the CoordinateSystem instance.

        The hash is based on the class name and keyword arguments (``__parameters__``).
        This ensures that two instances with the same class and initialization parameters produce the same hash.

        Returns
        -------
        int
            The hash value of the instance.
        """
        return hash(
            (self.__class__.__name__, tuple(sorted(self.__parameters__.items())))
        )

    def __getitem__(self, index: int) -> str:
        """
        Return the axis name at the specified index.

        Parameters
        ----------
        index : int
            The index of the axis to retrieve.

        Returns
        -------
        str
            The name of the axis at the given index.

        Raises
        ------
        IndexError
            If the index is out of range.

        Example
        -------
        >>> cs = MyCoordinateSystem()
        >>> axis_name = cs[0]
        >>> print(axis_name)
        'r'
        """
        return self.axes[index]

    def __contains__(self, axis_name: str) -> bool:
        """
        Check whether a given axis name is part of the coordinate system.

        Parameters
        ----------
        axis_name : str
            The axis name to check.

        Returns
        -------
        bool
            True if the axis name is present; False otherwise.

        Example
        -------
        >>> cs = MyCoordinateSystem()
        >>> 'r' in cs
        True
        """
        return axis_name in self.axes

    def __eq__(self, other: object) -> bool:
        """
        Check equality between two coordinate system instances.

        Returns True if:
          1. They are instances of the same class.
          2. They have the same axes (in the same order).
          3. They have the same parameter keys and values.

        Parameters
        ----------
        other : object
            The object to compare with this coordinate system.

        Returns
        -------
        bool
            True if they are considered equal, False otherwise.

        Example
        -------
        >>> cs1 = MyCoordinateSystem(a=3)
        >>> cs2 = MyCoordinateSystem(a=3)
        >>> print(cs1 == cs2)
        True
        """
        if not isinstance(other, self.__class__):
            return False
        # Compare axes
        if self.axes != other.axes:
            return False
        # Compare parameters
        if self.parameters != other.parameters:
            return False
        return True

    def __copy__(self):
        """
        Create a shallow copy of the coordinate system.

        Returns
        -------
        _CoordinateSystemBase
            A new instance of the same class, initialized with the same parameters.

        Example
        -------
        >>> import copy
        >>> cs1 = MyCoordinateSystem(a=3)
        >>> cs2 = copy.copy(cs1)
        >>> print(cs1 == cs2)
        True
        """
        # Shallow copy: re-init with same parameters
        cls = self.__class__
        new_obj = cls(**self.parameters)
        return new_obj

    # @@ PROPERTIES @@ #
    # These should not be altered in subclasses.
    @property
    def ndim(self) -> int:
        """The number of dimensions spanned by this coordinate system."""
        return self.__NDIM__

    @property
    def axes(self) -> List[str]:
        """The axes names present in this coordinate system."""
        return self.__AXES__[:]

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        The parameters of this coordinate system. Note that modifications made to the returned dictionary
        are not reflected in the class itself. To change a parameter value, the class must be re-instantiated.
        """
        return self.__parameters__.copy()

    @property
    def metric_tensor_symbol(self) -> _ExpressionType:
        """The symbolic metric tensor for this coordinate system instance."""
        return self.__class_expressions__["metric_tensor"]

    @property
    def inverse_metric_tensor_symbol(self) -> _ExpressionType:
        """
        The symbolic inverse metric tensor for this coordinate system instance.
        """
        return self.__class_expressions__["inverse_metric_tensor"]

    @property
    def metric_tensor(self) -> Callable:
        """
        Returns the callable function for the metric tensor of the coordinate system.

        The metric tensor :math:`g_{ij}` defines the inner product structure of the coordinate system.
        It is used for measuring distances, computing derivatives, and raising/lowering indices.
        This function returns the precomputed metric tensor as a callable function, which can be
        evaluated at specific coordinates.

        Returns
        -------
        Callable
            A function that computes the metric tensor :math:`g_{ij}` when evaluated at specific coordinates.
            The returned function takes numerical coordinate values as inputs and outputs a NumPy array
            of shape ``(ndim, ndim)``.

        Example
        -------
        .. code-block:: python

            cs = MyCoordinateSystem()
            g_ij = cs.metric_tensor(x=1, y=2, z=3)  # Evaluates the metric at (1,2,3)
            print(g_ij.shape)  # Output: (ndim, ndim)

        """
        return self.__numerical_expressions__["metric_tensor"]

    @property
    def inverse_metric_tensor(self) -> Callable:
        """
        Returns the callable function for the inverse metric tensor of the coordinate system.

        The inverse metric tensor :math:`g^{ij}` is the inverse of :math:`g_{ij}` and is used to raise indices,
        compute dual bases, and perform coordinate transformations. This function returns a callable
        representation of :math:`g^{ij}`, allowing evaluation at specific coordinate points.

        Returns
        -------
        Callable
            A function that computes the inverse metric tensor :math:`g^{ij}` when evaluated at specific coordinates.
            The returned function takes numerical coordinate values as inputs and outputs a NumPy array
            of shape ``(ndim, ndim)``.

        Example
        -------
        .. code-block:: python

            cs = MyCoordinateSystem()
            g_inv = cs.inverse_metric_tensor(x=1, y=2, z=3)  # Evaluates g^{ij} at (1,2,3)
            print(g_inv.shape)  # Output: (ndim, ndim)

        """
        return self.__numerical_expressions__["inverse_metric_tensor"]

    @property
    def axes_symbols(self) -> List[sp.Symbol]:
        """
        The symbols representing each of the coordinate axes in this coordinate system.
        """
        return self.__class__.__axes_symbols__[:]

    @property
    def parameter_symbols(self):
        """
        Get the symbolic representations of the coordinate system parameters.

        Returns
        -------
        dict of str, ~sympy.core.symbol.Symbol
            A dictionary mapping parameter names to their corresponding SymPy symbols.
            These symbols are used in all symbolic expressions defined by the coordinate system.

        Notes
        -----
        - The returned dictionary is a copy, so modifying it will not affect the internal state.
        - These symbols are created during class setup and correspond to keys in `self.parameters`.
        """
        return self.__parameter_symbols__.copy()

    # @@ COORDINATE METHODS @@ #
    # These methods dictate the behavior of the coordinate system including how
    # coordinate conversions behave and how the coordinate system handles differential
    # operations.
    @staticmethod
    @abstractmethod
    def __construct_metric_tensor_symbol__(*args, **kwargs) -> sp.Matrix:
        r"""
        Construct the metric tensor for the coordinate system.

        The metric tensor defines the way distances and angles are measured in the given coordinate system.
        It is used extensively in differential geometry and tensor calculus, particularly in transformations
        between coordinate systems.

        This method must be implemented by subclasses to specify how the metric tensor is computed.
        The returned matrix should contain symbolic expressions that define the metric components.

        Parameters
        ----------
        *args : tuple of sympy.Symbol
            The symbolic representations of each coordinate axis.
        **kwargs : dict of sympy.Symbol
            The symbolic representations of the coordinate system parameters.

        Returns
        -------
        sp.Matrix
            A symbolic ``NDIM x NDIM`` matrix representing the metric tensor.

        Notes
        -----
        - This method is abstract and must be overridden in derived classes.
        - The metric tensor is used to compute distances, gradients, and other differential operations.
        - In orthogonal coordinate systems, the metric tensor is diagonal.

        Example
        -------
        In a cylindrical coordinate system (r, θ, z), the metric tensor is:

        .. math::
            g_{ij} =
            \\begin{bmatrix}
            1 & 0 & 0 \\\\
            0 & r^2 & 0 \\\\
            0 & 0 & 1
            \\end{bmatrix}

        For a custom coordinate system, this function should return an equivalent symbolic representation.
        """
        pass

    @class_expression(name="Dterm")
    @classmethod
    def __compute_Dterm__(cls, *_, **__):
        r"""
        Compute the D-term :math:`(1/\rho)\partial_\mu \rho` for use in
        computing the divergence numerically.
        """
        _metric_density = cls.__class_expressions__["metric_density"]
        _axes = cls.__axes_symbols__

        return compute_Dterm(_metric_density, _axes)

    @class_expression(name="Lterm")
    @classmethod
    def __compute_Lterm__(cls, *_, **__):
        r"""
        Compute the D-term :math:`(1/\rho)\partial_\mu \rho` for use in
        computing the divergence numerically.
        """
        _metric_density = cls.__class_expressions__["metric_density"]
        _inverse_metric_tensor = cls.__class_expressions__["inverse_metric_tensor"]
        _axes = cls.__axes_symbols__

        return compute_Lterm(_inverse_metric_tensor, _metric_density, _axes)

    # @@ EXPRESSION METHODS @@ #
    # These methods allow the user to interact with derived, symbolic, and numeric expressions.
    def substitute_expression(self, expression: _ExpressionType) -> _ExpressionType:
        """
        Replace symbolic parameters with numerical values in an expression.

        This method takes a symbolic expression that may include parameter symbols and
        substitutes them with the numerical values assigned at instantiation.

        Parameters
        ----------
        expression : str or sympy expression
            The symbolic expression to substitute parameter values into.

        Returns
        -------
        sympy expression
            The expression with parameters replaced by their numeric values.

        Notes
        -----
        - Only parameters defined in ``self.__parameters__`` are substituted.
        - If an expression does not contain any parameters, it remains unchanged.
        - This method is useful for obtaining instance-specific symbolic representations.

        Example
        -------

        .. code-block:: python

            from sympy import Symbol
            expr = Symbol('a') * Symbol('x')
            coords = MyCoordinateSystem(a=3)
            print(coords.substitute_expression(expr))
            3*x

        """
        # Substitute in each of the parameter values.
        _params = {k: v for k, v in self.__parameters__.items()}
        return sp.simplify(sp.sympify(expression).subs(_params))

    def lambdify_expression(self, expression: Union[str, sp.Basic]) -> Callable:
        """
        Convert a symbolic expression into a callable function.

        Parameters
        ----------
        expression : :py:class:`str` or sp.Basic
            The symbolic expression to lambdify.

        Returns
        -------
        Callable
            A callable numerical function.
        """
        return lambdify_expression(
            expression, self.__axes_symbols__, self.__parameters__
        )

    @classmethod
    def get_class_expression(cls, expression_name: str) -> _ExpressionType:
        """
        Retrieve a derived expression for this coordinate system by name. The returned expression will include
        symbolic representations for all the axes as well as the parameters.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve.

        Returns
        -------
        The requested symbolic expression.

        Raises
        ------
        KeyError
            If the requested expression name does not exist.
        """
        # Validate that the expression actually exists. If it does not, return an error indicating
        # that the expression isn't known.
        # Check for the metric first
        if expression_name in cls.__class_expressions__:
            _class_expr = cls.__class_expressions__[expression_name]
        else:
            raise ValueError(
                f"Coordinate system {cls.__name__} doesn't have an expression '{expression_name}'."
            )

        # If the expression hasn't been loaded at the class level yet, we need to execute that
        # code to ensure that it does get loaded.
        if isinstance(_class_expr, classmethod):
            try:
                pg_log.debug(
                    f"Retrieving symbolic expression `{expression_name}` for class {cls.__name__}."
                )
                # Extract the class method and evaluate it to get the symbolic expression.
                _class_expr_function = _class_expr.__func__  # The underlying callable.
                _class_expr = _class_expr_function(
                    cls, *cls.__axes_symbols__, **cls.__parameter_symbols__
                )

                # Now simplify the expression.
                _class_expr = sp.simplify(_class_expr)

                # Now register in the expression dictionary.
                cls.__class_expressions__[expression_name] = _class_expr
            except Exception as e:
                raise CoordinateClassException(
                    f"Failed to evaluate class expression {expression_name} (linked to {_class_expr.__func__}) due to"
                    f" an error: {e}. "
                ) from e

        # Now that the expression is certainly loaded, we can simply return the class-level expression.
        return _class_expr

    @classmethod
    def list_class_expressions(cls) -> List[str]:
        """
        List the available coordinate system expressions.

        Returns
        -------
        list of str
            The list of available class-level expressions.
        """
        return list(cls.__class_expressions__.keys())

    @classmethod
    def has_class_expression(cls, expression_name: str) -> bool:
        """
        Check if the coordinate system has a specific expression registered to it.

        Parameters
        ----------
        expression_name: str
            The name of the symbolic expression to check.

        Returns
        -------
        bool
            ``True`` if the symbolic expression is registered at the class level.
        """
        return expression_name in cls.__class_expressions__

    def get_expression(self, expression_name: str) -> _ExpressionType:
        """
        Retrieve an instance-specific symbolic expression.

        Unlike :py:meth:`get_class_expression`, this method returns an expression where
        parameter values have been substituted. The returned expression retains symbolic
        representations of coordinate axes but replaces any parameter symbols with their
        numerical values assigned at instantiation.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve.

        Returns
        -------
        The symbolic expression with parameters substituted.

        Raises
        ------
        ValueError
            If the expression is not found at either the instance or class level.

        Notes
        -----
        - This method allows retrieving instance-specific symbolic expressions where numerical
          parameter values have been applied.
        - If an expression has not been previously computed for the instance, it is derived
          from the class-level expression and stored in ``self.__expressions__``.

        Example
        -------

        .. code-block:: python

            class CylindricalCoordinateSystem(CoordinateSystemBase):
                __AXES__ = ['r', 'theta', 'z']
                __PARAMETERS__ = {'scale': 2}

                @staticmethod
                def __construct_metric_tensor_symbol__(*args, **kwargs):
                    return sp.Matrix([[1, 0, 0], [0, args[0]**2, 0], [0, 0, 1]])

            coords = CylindricalCoordinateSystem(scale=3)
            expr = coords.get_expression('metric_tensor')
            print(expr)
            Matrix([
                [1, 0, 0],
                [0, r**2, 0],
                [0, 0, 1]
            ])

        """
        # Look for the expression in the instance directory first.
        if expression_name in self.__expressions__:
            return self.__expressions__[expression_name]

        # We couldn't find it in the instance directory, now we try to fetch it
        # and perform a substitution.
        if expression_name in self.__class__.__class_expressions__:
            _substituted_expression = self.substitute_expression(
                self.get_class_expression(expression_name)
            )
            self.__expressions__[expression_name] = _substituted_expression
            return _substituted_expression

        raise ValueError(
            f"Coordinate system {self.__class__.__name__} doesn't have an expression '{expression_name}'."
        )

    def set_expression(
        self, expression_name: str, expression: _ExpressionType, overwrite: bool = False
    ):
        """
        Set a symbolic expression at the instance level.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to register.
        expression : sympy expression
            The symbolic expression to register.
        overwrite : bool, optional
            If True, overwrite an existing expression with the same name. Defaults to False.

        Raises
        ------
        ValueError
            If the expression name already exists and `overwrite` is False.
        """
        if (expression_name in self.__expressions__) and (not overwrite):
            raise ValueError(
                f"Expression '{expression_name}' already exists. Use `overwrite=True` to replace it."
            )
        self.__expressions__[expression_name] = expression

    def list_expressions(self) -> List[str]:
        """
        List the available instance-level expressions.

        Returns
        -------
        list of str
            The list of available class-level expressions.
        """
        return list(
            set(self.__class_expressions__.keys()) | set(self.__expressions__.keys())
        )

    def has_expression(self, expression_name: str) -> bool:
        """
        Check if a symbolic expression is registered at the instance level.

        Parameters
        ----------
        expression_name: str
            The name of the symbolic expression to check.

        Returns
        -------
        bool
            ``True`` if the symbolic expression is registered.
        """
        return expression_name in self.list_expressions()

    def get_numeric_expression(self, expression_name: str) -> Callable:
        """
        Retrieve a numerically evaluable version of a coordinate system expression given the expression name.

        This method will search through the numerical expressions already generated in the instance and return the
        numerical version if it finds it. It will also search through all the symbolic expressions and try to perform
        a conversion to numerical.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve or convert.

        Returns
        -------
        Callable
            A numeric (callable) version of the symbolic expression.

        Raises
        ------
        KeyError
            If the symbolic expression is not found.
        """
        if expression_name not in self.__numerical_expressions__:
            symbolic_expression = self.get_expression(expression_name)
            self.__numerical_expressions__[expression_name] = self.lambdify_expression(
                symbolic_expression
            )
        return self.__numerical_expressions__[expression_name]

    # @@ CONVERSION @@ #
    # Perform conversions to / from cartesian coordinates.
    @abstractmethod
    def _convert_native_to_cartesian(self, *args):
        pass

    @abstractmethod
    def _convert_cartesian_to_native(self, *args):
        pass
