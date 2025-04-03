"""
Base classes and metaclass infrastructure for defining coordinate systems in Pisces-Geometry.

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
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import sympy
import sympy as sp
from numpy._typing import ArrayLike

from pisces_geometry.coordinates._exceptions import CoordinateClassException
from pisces_geometry.differential_geometry import (
    compute_Dterm,
    compute_Lterm,
    contract_index_with_metric,
    gdiv_cl_contravariant,
    gdiv_cl_covariant,
    get_divergence_dependence,
    get_gradient_dependence,
    get_laplacian_dependence,
    get_lowering_dependence,
    get_raising_dependence,
    ggrad_cl_contravariant,
    ggrad_cl_covariant,
    glap_cl,
    lower_index,
    raise_index,
)
from pisces_geometry.utilities.arrays import get_grid_spacing
from pisces_geometry.utilities.logging import pg_log
from pisces_geometry.utilities.symbolic import lambdify_expression

_ExpressionType = Union[
    sp.Symbol, sp.Expr, sp.Matrix, sp.MutableDenseMatrix, sp.MutableDenseNDimArray
]

# Construct the buffer.
DEFAULT_COORDINATE_REGISTRY: Dict[str, Any] = {}
""" dict of str, Any: The default registry containing all initialized coordinate system classes.
"""


# noinspection PyTypeChecker
def class_expression(name: str = None) -> classmethod:
    """
    A decorator to be used when building custom coordinate system classes which marks a
    class method of the coordinate system as a "class expression".

    A class expression should be a class method with signature ``_func(cls, *args, **kwargs)`` which,
    when provided with the coordinate system's axes symbols (``args``) and the parameter symbols (``kwargs``),
    returns a ``sympy`` expression incorporating those symbols.

    Once registered, users can access the class expression using the :py:meth:`CoordinateSystem.get_class_expression` method.

    Parameters
    ----------
        name: (str, optional)
            Custom name for the expression. Defaults to the function name.

    Notes
    -----
    All class expressions are loaded on demand meaning that the expression is only evaluated once a user attempts
    to access it for the first time.

    Under the hood, this just adds markers to the function so that the meta-class will identify the expressions.

    Returns
    -------
    classmethod:
        A wrapped class method with additional attributes.
    """

    # noinspection PyTypeChecker
    def decorator(func):
        """
        The decorator that adds the wrapper around the class expression.
        """
        if not isinstance(func, classmethod):
            raise TypeError(
                "The @class_expression decorator must be applied to a @classmethod."
            )

        original_func = func.__func__  # Extract underlying function from classmethod

        @wraps(original_func)
        def wrapper(*args, **kwargs):
            """Wrapper that preserves class method behavior."""
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
        Validate a new coordinate system class. This includes determining the number of
        dimensions and ensuring that bounds and coordinates are all accurate.
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


class _CoordinateSystemBase(ABC, metaclass=_CoordinateMeta):
    """
    Base class for all Pisces-Geometry coordinate system classes. :py:class:`CoordinateSystemBase` provides the backbone
    for the symbolic / numerical structure of coordinate systems and also acts as a template for developers to use
    when developing custom coordinate system classes.

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
        # Derive the metric, inverse metric, and the metric density. We call to the
        # __construct_metric_tensor_symbol__ and then take the inverse and the determinant of
        # the matrices.
        cls.__class_expressions__[
            "metric_tensor"
        ] = cls.__construct_metric_tensor_symbol__(
            *cls.__axes_symbols__, **cls.__parameter_symbols__
        )
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

    # @@ INITIALIZATION PROCEDURES @@ #
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
        """
        Set up any special symbolic expressions or numerical instances.
        """
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
        self.__expressions__ = dict()
        self.__numerical_expressions__ = dict()
        self._setup_explicit_expressions()

    # @@ DUNDER METHODS @@ #
    # These should not be altered.
    def __repr__(self):
        return f"<{self.__class__.__name__} - Parameters={self.__parameters__}> "

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    def __len__(self) -> int:
        """
        Returns the number of axes in the coordinate system.

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
        Returns the axis name at the specified index.

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
        Checks whether a given axis name is part of the coordinate system.

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
        Checks equality between two coordinate system instances.

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
        Creates a shallow copy of the coordinate system.

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

    # @@ Properties @@ #
    # These should not be altered in subclasses.
    @property
    def ndim(self) -> int:
        """
        The number of dimensions spanned by this coordinate system.
        """
        return self.__NDIM__

    @property
    def axes(self) -> List[str]:
        """
        The axes (strings) present in this coordinate system.
        """
        return self.__AXES__[:]

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        The parameters of this coordinate system. Note that modifications made to the returned dictionary
        are not reflected in the class itself. To change a parameter value, the class must be re-instantiated.
        """
        return self.__parameters__.copy()

    @property
    def metric_tensor_symbol(self) -> Any:
        """
        The symbolic metric tensor for this coordinate system instance.
        """
        return self.__class_expressions__["metric_tensor"]

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
        Constructs the metric tensor for the coordinate system.

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

    # @@ EXPRESSION METHODS @@ #
    # These methods allow the user to interact with derived, symbolic, and numeric expressions.
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
        sp.Basic
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
        Retrieves an instance-specific symbolic expression.

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
        :py:class:`sympy.core.Basic`
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
        self, expression_name: str, expression: sp.Basic, overwrite: bool = False
    ):
        """
        Set a symbolic expression at the instance level.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to register.
        expression : sp.Basic
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

    # @@ Conversion Functions @@ #
    # Perform conversions to / from cartesian coordinates.
    @abstractmethod
    def _convert_native_to_cartesian(self, *args):
        pass

    @abstractmethod
    def _convert_cartesian_to_native(self, *args):
        pass

    # @@ Utility Functions @@ #
    # These utility methods provide interaction for the symbolic / numerical
    # expressions and some other features of the class which are useful.
    def substitute_expression(self, expression: Union[str, sp.Basic]) -> sp.Basic:
        """
        Replaces symbolic parameters with numerical values in an expression.

        This method takes a symbolic expression that may include parameter symbols and
        substitutes them with the numerical values assigned at instantiation.

        Parameters
        ----------
        expression : str or sp.Basic
            The symbolic expression to substitute parameter values into.

        Returns
        -------
        sp.Basic
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

    def pprint(self) -> None:
        """
        Prints a detailed description of the coordinate system, including its axes, parameters, and expressions.

        Example
        -------
        .. code-block:: python

            cs = MyCoordinateSystem(a=3, b=4)
            cs.describe()
            Coordinate System: MyCoordinateSystem
            Axes: ['x', 'y', 'z']
            Parameters: {'a': 3, 'b': 4}
            Available Expressions: ['jacobian', 'metric_tensor']

        """
        print(f"Coordinate System: {self.__class__.__name__}")
        print(f"Axes: {self.axes}")
        print(f"Parameters: {self.parameters}")
        print(f"Available Expressions: {self.list_expressions()}")

    def axes_index_to_string(self, axes_index: int) -> str:
        """
        Converts an axis index to its corresponding axis name.

        Parameters
        ----------
        axes_index : int
            The index of the axis to retrieve.

        Returns
        -------
        str
            The name of the corresponding axis.

        Raises
        ------
        IndexError
            If the provided index is out of range.

        """
        return self.__AXES__[axes_index]

    def axes_string_to_index(self, axes_string: str) -> int:
        """
        Converts an axis name to its corresponding index.

        Parameters
        ----------
        axes_string : str
            The name of the axis to retrieve its index.

        Returns
        -------
        int
            The index of the corresponding axis.

        Raises
        ------
        ValueError
            If the provided axis name is not found in the coordinate system.
        """
        return self.__AXES__.index(axes_string)

    def build_axes_mask(self, axes: Sequence[str]) -> np.ndarray:
        r"""
        Construct a boolean mask array indicating which axes are in ``axes``.

        Parameters
        ----------
        axes: list of str or int

        Returns
        -------
        numpy.ndarray
        A boolean mask array indicating which axes are in ``axes``.
        """
        # Set up the indices for the axes.
        _mask = np.zeros(len(self.__AXES__), dtype=bool)
        _axes = np.asarray([self.axes_string_to_index(ax) for ax in axes], dtype=int)

        # Fill the mask values.
        _mask[_axes] = True
        return _mask

    @classmethod
    def _standardize_coordinates(cls, *args) -> np.ndarray:
        # Ensure that all the arguments are the same shape and that
        # there are NDIM of them.
        if len(args) != cls.__NDIM__:
            raise ValueError(f"Expected {cls.__NDIM__} arguments, got {len(args)}.")

        args = [np.asarray(_a) for _a in args]

        # Check that the dimensionality of the coordinates is homogeneous so that
        # they can be made into a standardized array.
        _shapes = set(_a.shape for _a in args)
        if len(_shapes) != 1:
            raise ValueError(
                f"Coordinates provided have incompatible shapes: {[_a.shape for _a in args]}."
            )

        # Now join them all together and return
        return np.stack(args, axis=-1)

    # @@ Mathematical Validation Operations @@ #
    # These methods are used for various aspects of the mathematical backbone
    # when dealing with things like incomplete tensor inputs and other common but
    # annoying issues that need to be managed.
    @classmethod
    def get_free_fixed_axes(
        cls, grid_dimensions: int, fixed_axes: Dict[str, float] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Determine the names of free and fixed axes based on the number of spatial dimensions
        and any axes that are held constant (fixed).

        This utility is useful when working with reduced-dimensional coordinate grids (e.g.,
        a 2D slice of a 3D system), where some axes are held at constant values.

        Parameters
        ----------
        grid_dimensions : int
            The number of axes present in the input grid (i.e., the number of dimensions that vary).
        fixed_axes : dict of str, float, optional
            A dictionary specifying fixed axes and their values. The keys are axis names and
            the values are constants to be inserted into those dimensions.

        Returns
        -------
        (free_axes, fixed_axes_list) : tuple of lists
            A tuple containing:

                - free_axes: list of axis names that correspond to the free (grid) axes.
                - fixed_axes_list: list of axis names that are held constant (from the fixed_axes dictionary).

        Raises
        ------
        ValueError
            If the number of free + fixed axes does not match the number of dimensions of the coordinate system.
        """
        fixed_axes = fixed_axes or {}

        if len(fixed_axes) + grid_dimensions != cls.__NDIM__:
            raise ValueError(
                f"Mismatch in free/fixed axes: got {grid_dimensions} grid dimensions and {len(fixed_axes)} fixed axes, "
                f"but coordinate system has NDIM = {cls.__NDIM__}."
            )

        free_axes = [_ax for _ax in cls.__AXES__ if _ax not in fixed_axes]
        return free_axes, list(fixed_axes.keys())

    @classmethod
    def fill_coordinates(
        cls, coordinates: Sequence[Any], fixed_axes: Dict[str, float] = None
    ) -> List[Any]:
        """
        Fill in any missing coordinates in a list of coordinates using a set of fixed axes values.

        This method adds new entries to ``coordinates`` so that any ``fixed_axes`` are placed
        in the coordinate list in the correct order.

        Parameters
        ----------
        coordinates: list
            The list of coordinates to fill. This should be a list with length smaller than the coordinate system's dimension.
            These coordinates are not changed at all during the procedure, the ordering of the list simply changes to insert
            any missing coordinates.
        fixed_axes: dict of str, float, optional
            The fixed axes (axes not already in ``coordinates``) to fill in. Each entry in the dictionary should be a coordinate
            axis (key, ``str``) and a value (any type). The result will then insert these values in the correct positions.

        Returns
        -------
        list
            The reordered and inserted list of coordinate items.

        Raises
        ------
        ValueError
            A value error is raised in one of two scenarios:

            1. If any of the axes in ``fixed_axes`` are not valid axes of the coordinate system.
            2. If the number of elements in ``coordinates`` plus the number of elements in ``fixed_axes`` is not
               equal to the number of dimensions in the coordinate system.

        """
        # Validate that none of the fixed axes are invalid and that the
        # length of the coordinates + length of the fixed axes is the full set of
        # axes.
        fixed_axes = fixed_axes if fixed_axes is not None else {}
        if any(_fa not in cls.__AXES__ for _fa in fixed_axes):
            raise ValueError(
                f"Some axes in `fixed_axes` are not valid axes: {[_fa for _fa in fixed_axes if _fa not in cls.__AXES__]}."
            )
        if len(coordinates) + len(fixed_axes) != cls.__NDIM__:
            raise ValueError(
                f"Could not fill in coordinates with `fixed axes`={fixed_axes}. The sum of free and fixed axes is not"
                f" equal to the number of coordinate dimensions."
            )

        # Identify the free axes based on the provided fixed axes.
        free_axes, _ = cls.get_free_fixed_axes(len(coordinates), fixed_axes)
        return [
            coordinates[free_axes.index(ax)] if ax in free_axes else fixed_axes[ax]
            for ax in cls.__AXES__
        ]

    @classmethod
    def fill_coordinate_grid(
        cls, coordinate_grid: np.ndarray, fixed_axes: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Fill in a full coordinate grid of shape ``(..., NDIM)`` by inserting fixed axis values
        into a partial grid of shape ``(..., k)``, where ``k < NDIM``.

        This is useful when the input coordinate grid represents only a subset of the full
        coordinate system's dimensions and the remainder must be filled in from fixed_axes.

        Parameters
        ----------
        coordinate_grid : numpy.ndarray
            A NumPy array of shape ``(..., k)`` representing the grid in the free coordinate directions.
            Each trailing dimension corresponds to one of the free axes.
        fixed_axes : dict of str, float, optional
            Dictionary mapping axis names to fixed values. These are inserted in their correct positions.

        Returns
        -------
        numpy.ndarray
            A coordinate grid of shape ``(..., NDIM)`` with all axes present in the correct order.

        Raises
        ------
        ValueError
            If the number of free + fixed axes does not match the number of coordinate system dimensions.
        """
        # Validate the number of axes to ensure that we have all the information
        # necessary to actually perform this insertion.
        fixed_axes = fixed_axes or {}
        k = coordinate_grid.shape[-1]
        free_axes, _ = cls.get_free_fixed_axes(k, fixed_axes)
        full_axes = cls.__AXES__.copy()

        # Initialize result array
        output_shape = coordinate_grid.shape[:-1] + (cls.__NDIM__,)
        full_grid = np.empty(output_shape, dtype=coordinate_grid.dtype)

        # Assign free axis values from the coordinate grid
        for i, ax in enumerate(free_axes):
            full_idx = full_axes.index(ax)
            full_grid[..., full_idx] = coordinate_grid[..., i]

        # Fill fixed axis values as constant arrays
        for ax, value in fixed_axes.items():
            full_idx = full_axes.index(ax)
            full_grid[..., full_idx] = value

        return full_grid

    @classmethod
    def get_axis_connector(
        cls, grid_axes: List[str], components: List[str]
    ) -> List[int]:
        """
        Map component names to their indices in the grid_axes list.

        Parameters
        ----------
        grid_axes : list of str
            Axes present in the coordinate grid (i.e., free axes).
        components : list of str
            Axes for which indices are required.

        Returns
        -------
        list of int
            Indices of components within the grid_axes list.

        Raises
        ------
        ValueError
            If any component is not found in grid_axes.
        """
        try:
            return [grid_axes.index(c) if c in grid_axes else None for c in components]
        except ValueError as e:
            raise ValueError(
                f"Component {e.args[0].split()[-1]} not found in grid axes {grid_axes}."
            ) from e

    def compute_expression_on_coordinates(
        self,
        expression: str,
        coordinates: Optional[List[Any]] = None,
        /,
        fixed_axes: Dict[str, float] = None,
        value: Optional[np.ndarray] = None,
    ):
        """
        Evaluate a symbolic expression on the coordinate grid.

        Parameters
        ----------
        expression : str
            The name of the symbolic or numerical expression to evaluate.
        coordinates : list of Any, optional
            List of values corresponding to the free axes. If not provided,
            `value` must be given directly.
        fixed_axes : dict, optional
            Fixed axis values used to construct the full coordinate input.
        value : numpy.ndarray, optional
            Optional precomputed result to return directly. If provided, skips evaluation.

        Returns
        -------
        numpy.ndarray
            Evaluated expression at the provided coordinates.

        Raises
        ------
        ValueError
            If neither `coordinates` nor `value` are provided.
        """
        if value is not None:
            return value

        if coordinates is None:
            raise ValueError(
                "Must provide either `value` or `coordinates` for evaluation."
            )

        _coordinates = self.fill_coordinates(coordinates, fixed_axes=fixed_axes)
        _expression = self.get_numeric_expression(expression)
        return _expression(*_coordinates)

    def compute_metric_on_coordinates(
        self,
        coordinates: Optional[Sequence[Any]] = None,
        /,
        inverse: bool = False,
        fixed_axes: Dict[str, float] = None,
        value: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate the (inverse) metric tensor at a set of coordinates, or return an explicitly provided value.

        Parameters
        ----------
        coordinates : list or array, optional
            Coordinates for evaluating the metric. If not provided, `value` must be passed.
        inverse : bool, optional
            If True, returns the inverse metric tensor instead of the metric tensor.
        fixed_axes : dict, optional
            Fixed coordinate values to use for filling out missing axes.
        value : numpy.ndarray, optional
            Optional precomputed metric (or inverse metric) to return directly.

        Returns
        -------
        numpy.ndarray
            The evaluated metric tensor.

        Raises
        ------
        ValueError
            If neither `coordinates` nor `value` are provided.
        """
        if value is not None:
            return value

        if coordinates is None:
            raise ValueError(
                "Must provide either `value` or `coordinates` for metric evaluation."
            )

        _coordinates = self.fill_coordinates(coordinates, fixed_axes=fixed_axes)
        return (
            self.inverse_metric_tensor(*_coordinates)
            if inverse
            else self.metric_tensor(*_coordinates)
        )

    def compute_expression_on_grid(
        self,
        expression: str,
        coordinate_grid: Optional[np.ndarray] = None,
        /,
        fixed_axes: Dict[str, float] = None,
        value: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate a symbolic expression over a full coordinate grid.

        Parameters
        ----------
        expression : str
            The name of the symbolic expression to evaluate.
        coordinate_grid : numpy.ndarray, optional
            A grid of shape ``(..., k)`` representing the free coordinate directions. If not provided,
            `value` must be given directly.
        fixed_axes : dict, optional
            A mapping of fixed axis names to their constant values.
        value : numpy.ndarray, optional
            If provided, returned directly without computation.

        Returns
        -------
        numpy.ndarray
            The evaluated expression over the full grid of shape (..., NDIM).

        Raises
        ------
        ValueError
            If neither `coordinate_grid` nor `value` are provided.
        """
        if coordinate_grid is not None:
            _coordinates = np.moveaxis(coordinate_grid, -1, 0)
        else:
            _coordinates = None
        return self.compute_expression_on_coordinates(
            expression, _coordinates, fixed_axes=fixed_axes, value=value
        )

    def compute_metric_on_grid(
        self,
        coordinate_grid: Optional[np.ndarray] = None,
        /,
        inverse: bool = False,
        fixed_axes: Dict[str, float] = None,
        value: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate the metric or inverse metric tensor over a full coordinate grid.

        Parameters
        ----------
        coordinate_grid : numpy.ndarray, optional
            A grid of shape (..., k) representing the free coordinate directions. If not provided,
            `value` must be given directly.
        inverse : bool, optional
            If True, compute inverse metric instead of metric.
        fixed_axes : dict, optional
            A mapping of fixed axis names to their constant values.
        value : numpy.ndarray, optional
            If provided, returned directly without computation.

        Returns
        -------
        numpy.ndarray
            The evaluated metric tensor over the full grid, shape (..., NDIM, NDIM).

        Raises
        ------
        ValueError
            If neither `coordinate_grid` nor `value` are provided.
        """
        if coordinate_grid is not None:
            _coordinates = np.moveaxis(coordinate_grid, -1, 0)
        else:
            _coordinates = None
        return self.compute_metric_on_coordinates(
            _coordinates, inverse=inverse, fixed_axes=fixed_axes, value=value
        )

    # @@ Mathematical Operations @@ #
    # These should only be changed in fundamental subclasses where
    # new mathematical approaches become available.
    def raise_index(
        self,
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        inverse_metric: np.ndarray = None,
        coordinate_grid: np.ndarray = None,
        fixed_axes: Dict[str, float] = None,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Raise a single index of a tensor field using the inverse metric of this coordinate system.

        Parameters
        ----------
        tensor_field : numpy.ndarray
            The input tensor field. The tensor field should be an array of shape ``(..., ndim, ndim, ...)``, where
            each of the later axes is a tensor component axis for a particular rank.
        index : int
            Index (0-based, relative to tensor rank) to raise.
        rank : int
            Number of tensor dimensions (excluding grid).
        inverse_metric : numpy.ndarray, optional
            Inverse metric tensor :math:`g^{ij}` of shape ``(..., ndim, ndim)``. If not provided,
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

            T^{a} = g^{ab} T_b

        The index position to be raised is specified by `index`.

        Examples
        --------
        In spherical coordinates, the metric tensor is

        .. math::

            g_{\mu\nu} = \begin{bmatrix}1&0&0\\0&r^2&0\\0&0&r^2\sin(\theta)\end{bmatrix}.

        As such, the vector :math:`{\bf v}` with covariant elements

        .. math::

            {\bf v} = r^2 {\bf e}^\theta

        can be converted to its contravariant form via

        .. math::

            v^\theta = g^{\theta \theta} v_{\theta} = 1.

        Let's see this happen computationally:

        .. plot::
            :include-source:

            >>> # Start by importing the spherical coordinate system.
            >>> from pisces_geometry.coordinates.coordinate_systems import SphericalCoordinateSystem
            >>> from numpy import array
            >>>
            >>> # Construct the spherical coordinate system
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>> # Create a grid of x,y values
            >>> x,y,z = np.linspace(-5,5,100), np.linspace(-5,5,100), np.zeros(100)
            >>> X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
            >>>
            >>> # Convert the grid to spherical coordinates.
            >>> R,THETA,PHI = cs._convert_cartesian_to_native(X,Y,Z)
            >>>
            >>> C = np.moveaxis(np.stack([R,THETA,PHI]), 0, -1)
            >>> # Define the tensor field.
            >>> tensor_field = np.zeros(C.shape )
            >>> tensor_field[...,1] = R**2
            >>>
            >>> # Raise the tensor field's index.
            >>> raised_tensor_field = cs.raise_index(tensor_field,index=0,rank=1,coordinate_grid=C)
            >>>
            >>> # Plot the two fields
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots(2,1,figsize=(6,8))
            >>> I = ax[0].imshow(tensor_field[...,0,1],extent=[-5,5,-5,5])
            >>> _ = plt.colorbar(I, ax=ax[0])
            >>> I = ax[1].imshow(raised_tensor_field[...,0,1],extent=[-5,5,-5,5])
            >>> _ = plt.colorbar(I,ax=ax[1])
            >>> plt.show()
        """
        inverse_metric = self.compute_metric_on_grid(
            coordinate_grid, inverse=True, fixed_axes=fixed_axes, value=inverse_metric
        )
        return raise_index(tensor_field, index, rank, inverse_metric, **kwargs)

    def lower_index(
        self,
        tensor_field: np.ndarray,
        index: int,
        rank: int,
        metric: np.ndarray = None,
        coordinate_grid: np.ndarray = None,
        fixed_axes: Dict[str, float] = None,
        **kwargs,
    ) -> np.ndarray:
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

            T^{a} = g^{ab} T_b

        The index position to be raised is specified by `index`.

        Examples
        --------
        In spherical coordinates, the metric tensor is

        .. math::

            g_{\mu\nu} = \begin{bmatrix}1&0&0\\0&r^2&0\\0&0&r^2\sin(\theta)\end{bmatrix}.

        As such, the vector :math:`{\bf v}` with covariant elements

        .. math::

            {\bf v} = r^2 {\bf e}^\theta

        can be converted to its contravariant form via

        .. math::

            v^\theta = g^{\theta \theta} v_{\theta} = 1.

        Let's see this happen computationally:

        .. plot::
            :include-source:

            >>> # Start by importing the spherical coordinate system.
            >>> from pisces_geometry.coordinates.coordinate_systems import SphericalCoordinateSystem
            >>> from numpy import array
            >>>
            >>> # Construct the spherical coordinate system
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>> # Create a grid of x,y values
            >>> x,y,z = np.linspace(-5,5,100), np.linspace(-5,5,100), np.zeros(100)
            >>> X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
            >>>
            >>> # Convert the grid to spherical coordinates.
            >>> R,THETA,PHI = cs._convert_cartesian_to_native(X,Y,Z)
            >>>
            >>> C = np.moveaxis(np.stack([R,THETA,PHI]), 0, -1)
            >>> # Define the tensor field.
            >>> tensor_field = np.zeros(C.shape )
            >>> tensor_field[...,1] = 1
            >>>
            >>> # Raise the tensor field's index.
            >>> raised_tensor_field = cs.lower_index(tensor_field,index=0,rank=1,coordinate_grid=C)
            >>>
            >>> # Plot the two fields
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots(2,1,figsize=(6,8))
            >>> I = ax[0].imshow(tensor_field[...,0,1],extent=[-5,5,-5,5])
            >>> _ = plt.colorbar(I, ax=ax[0])
            >>> I = ax[1].imshow(raised_tensor_field[...,0,1],extent=[-5,5,-5,5])
            >>> _ = plt.colorbar(I,ax=ax[1])
            >>> plt.show()
        """
        metric = self.compute_metric_on_grid(
            coordinate_grid, inverse=False, fixed_axes=fixed_axes, value=metric
        )
        return lower_index(tensor_field, index, rank, metric, **kwargs)

    def adjust_tensor_signature(
        self,
        tensor_field: np.ndarray,
        indices: List[int],
        modes: List[Literal["upper", "lower"]],
        rank: int,
        metric: np.ndarray = None,
        inverse_metric: np.ndarray = None,
        coordinate_grid: np.ndarray = None,
        component_masks: Optional[List[np.ndarray]] = None,
        inplace: bool = False,
        fixed_axes: Dict[str, float] = None,
    ) -> np.ndarray:
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
            The full metric tensor g_{ij}. Required for lowering unless `coordinate_grid`
            is provided.
        inverse_metric : numpy.ndarray, optional
            The inverse metric tensor g^{ij}. Required for raising unless `coordinate_grid`
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

        working_tensor = np.copy(tensor_field) if not inplace else tensor_field
        for i, (idx, mode) in enumerate(zip(indices, modes)):
            mask = component_masks[i] if component_masks else slice(None)

            if mode == "lower":
                metric = self.compute_metric_on_grid(
                    coordinate_grid, inverse=False, fixed_axes=fixed_axes, value=metric
                )
                current_metric = (
                    metric[..., mask, :][..., :, mask]
                    if isinstance(mask, np.ndarray)
                    else metric
                )
            elif mode == "upper":
                inverse_metric = self.compute_metric_on_grid(
                    coordinate_grid,
                    inverse=True,
                    fixed_axes=fixed_axes,
                    value=inverse_metric,
                )
                current_metric = (
                    inverse_metric[..., mask, :][..., :, mask]
                    if isinstance(mask, np.ndarray)
                    else inverse_metric
                )
            else:
                raise ValueError(f"Invalid mode '{mode}' for index {idx}")

            working_tensor = contract_index_with_metric(
                working_tensor, current_metric, idx, rank
            )

        return working_tensor

    @class_expression(name="Dterm")
    @classmethod
    def __compute_Dterm__(cls, *args, **kwargs):
        r"""
        Computes the D-term :math:`(1/\rho)\partial_\mu \rho` for use in
        computing the divergence numerically.
        """
        _metric_density = cls.__class_expressions__["metric_density"]
        _axes = cls.__axes_symbols__

        return compute_Dterm(_metric_density, _axes)

    @class_expression(name="Lterm")
    @classmethod
    def __compute_Lterm__(cls, *args, **kwargs):
        r"""
        Computes the D-term :math:`(1/\rho)\partial_\mu \rho` for use in
        computing the divergence numerically.
        """
        _metric_density = cls.__class_expressions__["metric_density"]
        _inverse_metric_tensor = cls.__class_expressions__["inverse_metric_tensor"]
        _axes = cls.__axes_symbols__

        return compute_Lterm(_inverse_metric_tensor, _metric_density, _axes)

    def compute_gradient(
        self,
        scalar_field: np.ndarray,
        /,
        spacing: Sequence[int] = None,
        coordinate_grid: np.ndarray = None,
        derivative_field: np.ndarray = None,
        inverse_metric: np.ndarray = None,
        *,
        basis: Literal["covariant", "contravariant"] = "covariant",
        fixed_axes: Dict[str, float] = None,
        is_uniform: bool = False,
        **kwargs,
    ):
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
        inverse_metric : numpy.ndarray, optional
            The inverse metric tensor :math:`g^{ij}` which is used when evaluating the contravariant components. If it is
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

        Examples
        --------
        As a basic example:

        .. plot::
            :include-source:

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from pisces_geometry.coordinates.coordinate_systems import SphericalCoordinateSystem
            >>>
            >>> # Initialize coordinate system
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>>
            >>> # Create the spherical grid
            >>> r,theta,phi = np.linspace(0.1,2,100),np.linspace(0.1,np.pi,100),np.linspace(0.1,2*np.pi,100)
            >>> R, THETA, PHI = np.meshgrid(r,theta,phi,indexing='ij')
            >>> coords = np.moveaxis(np.stack([R, THETA, PHI]), 0, -1)
            >>>
            >>> # Define two scalar fields: r and r * theta
            >>> field_r = R
            >>> field_r_theta = R * np.sin(THETA)
            >>>
            >>> # Compute covariant gradients
            >>> grad_r_cov = cs.compute_gradient(field_r, coordinate_grid=coords, basis='covariant')
            >>> grad_rtheta_cov = cs.compute_gradient(field_r_theta, coordinate_grid=coords, basis='covariant')
            >>>
            >>> # Compute contravariant gradients
            >>> grad_r_contra = cs.compute_gradient(field_r, coordinate_grid=coords, basis='contravariant')
            >>> grad_rtheta_contra = cs.compute_gradient(field_r_theta, coordinate_grid=coords, basis='contravariant')
            >>> # Plotting
            >>> fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            >>> titles = ["∇r (covariant)", "∇r (contravariant)",
            ...           "∇(rθ) (covariant)", "∇(rθ) (contravariant)"]
            >>> gradients = [grad_r_cov[..., 0], grad_r_contra[..., 0],
            ...              grad_rtheta_cov[..., 1], grad_rtheta_contra[..., 1]]
            >>>
            >>> for ax, grad, title in zip(axes.flat, gradients, titles):
            ...     im = ax.imshow(grad[..., 0], extent=[0.1,2,0,np.pi], origin='lower',vmin=-2,vmax=2)
            ...     _ = ax.set_title(title)
            ...     _ = fig.colorbar(im, ax=ax)
            >>> plt.tight_layout()
            >>> plt.show()

        """
        # Determine the number of axes in the grid.
        grid_ndim = scalar_field.ndim

        # Determine the grid spacing if it is necessary (derivative field not provided). Either the
        # spacing has been provided or we need to get it from the coordinate grid.
        if (
            (derivative_field is None)
            and (spacing is None)
            and (coordinate_grid is None)
        ):
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

            spacing = get_grid_spacing(coordinate_grid, is_uniform=is_uniform)

        # The spacing or the derivative field is now available, we have everything we need
        # to at least compute the covariant case. For the contravariant case, we'll still need to
        # establish a metric.
        if basis == "covariant":
            return ggrad_cl_covariant(
                scalar_field,
                spacing=spacing,
                derivative_field=derivative_field,
                **kwargs,
            )
        if basis == "contravariant":
            # The contravariant approach will require an inverse metric to be established
            # on the coordinate grid. If we don't have it, then we need to build it from scratch
            # using the coordinate system internals.
            inverse_metric = self.compute_metric_on_grid(
                coordinate_grid,
                inverse=True,
                fixed_axes=fixed_axes,
                value=inverse_metric,
            )

            # Now the inverse metric is assuredly available and we can pass to the lower
            # level callable.
            return ggrad_cl_contravariant(
                scalar_field,
                spacing,
                inverse_metric,
                derivative_field=derivative_field,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown basis {basis}.")

    def compute_divergence(
        self,
        vector_field: np.ndarray,
        /,
        dterm_field: np.ndarray = None,
        coordinate_grid: np.ndarray = None,
        spacing: np.ndarray = None,
        inverse_metric: np.ndarray = None,
        derivative_field: np.ndarray = None,
        *,
        basis: str = "contravariant",
        fixed_axes: Dict[str, float] = None,
        components: List[str] = None,
        is_uniform: bool = False,
        **kwargs,
    ):
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
        inverse_metric : numpy.ndarray, optional
            The inverse metric tensor (:math:`g^{\mu\nu}`), used only for covariant basis computations. If ``coordinate_grid`` is not
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

        Examples
        --------
        Consider the divergence of the vector field :math:`\mathbf{v} = r^k \sin \theta {\bf e}_{r}`. In spherical coordinates,

        .. math::

            \nabla \cdot {\bf v} = \frac{1}{r^2\sin\theta} \partial_r \left(r^{k+2} \sin^2 \theta \right) = (k+2)r^{k-1}\sin(\theta).

        We can do this computationally with relative ease:

        .. plot::
            :include-source:

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from pisces_geometry.coordinates.coordinate_systems import SphericalCoordinateSystem
            >>>
            >>> # Initialize coordinate system
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>>
            >>> # Create the spherical grid
            >>> r,theta,phi = np.linspace(0.1,2,100),np.linspace(0.1,np.pi,100),np.linspace(0.1,2*np.pi,100)
            >>> R, THETA = np.meshgrid(r,theta,indexing='ij')
            >>> coords = np.moveaxis(np.stack([R, THETA]), 0, -1)
            >>>
            >>> # Define two scalar fields: r and r * theta
            >>> field_1,field_2 = np.zeros(R.shape + (3,)),np.zeros(R.shape + (3,))
            >>> field_1[...,0] = R * np.sin(THETA)
            >>> field_2[...,0] = (R**2) * np.sin(THETA)
            >>>
            >>> # Compute the divergences.
            >>> div_1 = cs.compute_divergence(field_1,coordinate_grid=coords,fixed_axes={'phi':0})
            >>> div_2 = cs.compute_divergence(field_2,coordinate_grid=coords,fixed_axes={'phi':0})
            >>>
            >>> # Create the plot.
            >>> fig,ax = plt.subplots(2,1,figsize=[6,4],sharex=True,sharey=True)
            >>> I1 = ax[0].imshow(div_1.T,origin='lower',extent=[0.1,2.0,0,np.pi],aspect='auto')
            >>> I2 = ax[1].imshow(div_2.T,origin='lower',extent=[0.1,2.0,0,np.pi],aspect='auto')
            >>> _ = plt.colorbar(I1,ax=ax[0])
            >>> _ = plt.colorbar(I2,ax=ax[1])
            >>> _ = ax[1].set_xlabel(r'$r$')
            >>> _ = ax[0].set_ylabel(r'$\theta$')
            >>> _ = ax[1].set_ylabel(r'$\theta$')
            >>> plt.show()

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
        if (
            (derivative_field is None)
            and (spacing is None)
            and (coordinate_grid is None)
        ):
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
            spacing = get_grid_spacing(coordinate_grid, is_uniform=is_uniform)

        # Compute the D-term fields if they are not already available. Validate the shape of the dterms.
        dterm_field = self.compute_expression_on_grid(
            "Dterm", coordinate_grid, fixed_axes=fixed_axes, value=dterm_field
        )
        if dterm_field.shape == (comp_ndim,) + vector_field.shape[:-1]:
            dterm_field = np.moveaxis(dterm_field, 0, -1)

        # The spacing or the derivative field is now available, we have everything we need
        # to at least compute the covariant case. For the contravariant case, we'll still need to
        # establish a metric.
        if basis == "contravariant":
            # We can plug into the low level `gdiv_cl_contravariant` which doesn't need us
            # to provide a metric tensor because this is the "natural" basis.
            return gdiv_cl_contravariant(
                vector_field,
                dterm_field,
                spacing=spacing,
                derivative_field=derivative_field,
                axis_connector=axis_connector,
            )
        elif basis == "covariant":
            # We do need to use the metric tensor in `gdiv_cl_covariant`, which may require
            # computing the metric tensor. We'll follow the same procedure as in compute_gradient to
            # establish the correct inverse metric tensor.
            inverse_metric = self.compute_metric_on_grid(
                coordinate_grid,
                inverse=True,
                fixed_axes=fixed_axes,
                value=inverse_metric,
            )

            # Hand off the computation to the covariant solver.
            return gdiv_cl_covariant(
                vector_field,
                dterm_field,
                inverse_metric,
                spacing=spacing,
                derivative_field=None,
                axis_connector=axis_connector,
            )

        else:
            raise ValueError(f"Unknown basis {basis}.")

    def compute_laplacian(
        self,
        scalar_field: np.ndarray,
        /,
        lterm_field: np.ndarray = None,
        coordinate_grid: np.ndarray = None,
        spacing: Sequence[float] = None,
        inverse_metric: np.ndarray = None,
        derivative_field: np.ndarray = None,
        second_derivative_field: np.ndarray = None,
        fixed_axes: Dict[str, float] = None,
        is_uniform: bool = False,
        **kwargs,
    ) -> np.ndarray:
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

        inverse_metric : numpy.ndarray, optional
            Optional precomputed values of the inverse metric tensor :math:`g^{\mu\nu}` on the grid. This should have
            shape ``(..., ndim, ndim)`` or ``(..., k, k)`` depending on whether a full or sliced grid is used.

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

        Examples
        --------

        .. plot::
            :include-source:

            >>> # Import modules
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from pisces_geometry.coordinates.coordinate_systems import SphericalCoordinateSystem
            >>> cs = SphericalCoordinateSystem()
            >>>
            >>> # Construct the coordinates and the coordinate grid.
            >>> # For this example, we just use R and THETA (phi held constant)
            >>> r,theta = np.linspace(0.1,2,100),np.linspace(np.pi/4,np.pi*(3/4),100)
            >>> R,THETA = np.meshgrid(r,theta,indexing='ij')
            >>> C = np.stack([R,THETA],axis=-1)
            >>>
            >>> # Define the scalar field sin(theta) on the grid.
            >>> F = np.sin(THETA)
            >>>
            >>>
            >>> # Compute the laplacian and the "true" (theoretical) laplacian.
            >>> L = cs.compute_laplacian(F,coordinate_grid=C,fixed_axes={'phi':0})
            >>> Ltrue = np.cos(2*THETA)/(np.sin(THETA) * R**2)
            >>>
            >>> # Plotting data
            >>> fig,axes = plt.subplots(2,1,figsize=(6,8),sharex=True)
            >>> extent = [0.1,2,np.pi/4,np.pi*(3/4)]
            >>>
            >>> # Plot the computed case:
            >>> Q = axes[0].imshow(L.T,vmin=-20,vmax=1,origin='lower',extent=extent)
            >>> _ = plt.colorbar(Q)
            >>> P = axes[1].imshow(Ltrue.T,vmin=-20,vmax=1,origin='lower',extent=extent)
            >>> _ = plt.colorbar(P)
            >>> _ = axes[1].set_xlabel('r')
            >>> _ = axes[0].set_ylabel('theta')
            >>> _ = axes[1].set_ylabel('theta')
            >>>
            >>> plt.show()

        """
        # Identify the grid dimension and construct the free / fixed axes so that we can
        # use them for shape compliance and other construction procedures.
        grid_ndim = scalar_field.ndim
        free_axes, _ = self.get_free_fixed_axes(grid_ndim, fixed_axes=fixed_axes)
        free_mask = self.build_axes_mask(free_axes)

        # Determine the grid spacing if it is necessary (derivative field not provided). Either the
        # spacing has been provided or we need to get it from the coordinate grid.
        if (
            ((derivative_field is None) or (second_derivative_field is None))
            and (spacing is None)
            and (coordinate_grid is None)
        ):
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
            spacing = get_grid_spacing(coordinate_grid, is_uniform=is_uniform)

        # Compute the inverse metric and check its shape. We are either given this as an argument or
        # the coordinate system will try to compute it from the coordinate grid. The output may be in
        # some set of odd array shapes that need to be corrected.
        inverse_metric = self.compute_metric_on_grid(
            coordinate_grid, inverse=True, fixed_axes=fixed_axes, value=inverse_metric
        )
        # Correct the possibly incorrect inverse_metric shapes.
        if inverse_metric.shape == scalar_field.shape + (
            self.ndim,
            self.ndim,
        ):
            # The inverse metric was returned for the full coordinate system but we
            # only need the relevant grid axes (in BOTH indices).
            inverse_metric = inverse_metric[..., :, free_mask][..., free_mask, :]

        # Compute the L-term field and check its shape. We are either given this as an argument or
        # the coordinate system will try to compute it from the coordinate grid. The output may be in
        # some set of odd array shapes that need to be corrected.
        lterm_field = self.compute_expression_on_grid(
            "Lterm", coordinate_grid, fixed_axes=fixed_axes, value=lterm_field
        )
        # Correct the shape issues if they arise.
        if lterm_field.shape == (self.ndim,) + scalar_field.shape:
            lterm_field = np.moveaxis(lterm_field, 0, -1)[..., free_mask]
        elif lterm_field.shape == (grid_ndim,) + scalar_field.shape:
            lterm_field = np.moveaxis(lterm_field, 0, -1)

        # Compute the Laplacian using the core differential geometry operator
        return glap_cl(
            scalar_field,
            lterm_field,
            inverse_metric,
            spacing=spacing,
            derivative_field=derivative_field,
            second_derivative_field=second_derivative_field,
            **kwargs,
        )

    def get_raise_dependence(
        self,
        tensor_field_dependence: ArrayLike,
        axis: int = 0,
    ) -> np.ndarray:
        """
        Determine the symbolic dependence of the tensor field components after raising an index.

        Parameters
        ----------
        tensor_field_dependence : array-like
            Symbolic structure of the tensor field. Each leaf may be a sympy expression, symbol,
            or a sequence of symbols indicating dependencies.
        axis : int, optional
            Axis along which the index should be raised. Defaults to 0.

        Returns
        -------
        numpy.ndarray
            An array of the same shape as the input, where each element is either 0
            (if the component is identically zero) or a set of free symbols.
        """
        return get_raising_dependence(
            tensor_field_dependence=tensor_field_dependence,
            inverse_metric=self.get_expression("inverse_metric_tensor"),
            axis=axis,
        )

    def get_lower_dependence(
        self,
        tensor_field_dependence: ArrayLike,
        axis: int = 0,
    ) -> np.ndarray:
        """
        Determine the symbolic dependence of the tensor field components after lowering an index.

        Parameters
        ----------
        tensor_field_dependence : array-like
            Symbolic structure of the tensor field. Each leaf may be a sympy expression, symbol,
            or a sequence of symbols indicating dependencies.
        axis : int, optional
            Axis along which the index should be lowered. Defaults to 0.

        Returns
        -------
        numpy.ndarray
            An array of the same shape as the input, where each element is either 0
            (if the component is identically zero) or a set of free symbols.
        """
        return get_lowering_dependence(
            tensor_field_dependence=tensor_field_dependence,
            metric=self.get_expression("metric_tensor"),
            axis=axis,
        )

    def get_gradient_dependence(
        self,
        scalar_field_dependence: Sequence[sp.Symbol],
        basis: Literal["covariant", "contravariant"] = "covariant",
    ) -> np.ndarray:
        """
        Determine which coordinate axes each component of the gradient depends on.

        Parameters
        ----------
        scalar_field_dependence : list of sympy.Symbol
            Symbols on which the scalar field depends.
        basis : {'covariant', 'contravariant'}, optional
            Which basis to compute the gradient in. Default is 'covariant'.

        Returns
        -------
        numpy.ndarray
            Array of sets (or 0) indicating the dependence of each gradient component.

        Examples
        --------
        Let's examine the gradient dependence of a spherical coordinate system.

        >>> from pisces_geometry.coordinates import OblateHomoeoidalCoordinateSystem
        >>> u = OblateHomoeoidalCoordinateSystem(ecc=0.5)
        >>> print(u.get_gradient_dependence(['xi']))
        [{xi} 0 0]
        >>> print(u.get_gradient_dependence(['xi'],basis='contravariant'))
        [{xi, theta} {xi, theta} 0]
        """
        try:
            return get_gradient_dependence(
                scalar_field_dependence,
                self.axes_symbols,
                basis=basis,
                inverse_metric=self.get_expression("inverse_metric_tensor")
                if basis == "contravariant"
                else None,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to compute gradient dependence due to an error: {e}"
            )

    def get_divergence_dependence(
        self,
        vector_field_dependence: Sequence[Sequence[sp.Symbol]],
        basis: Literal["covariant", "contravariant"] = "contravariant",
    ) -> Union[set, int]:
        """
        Determine which coordinate axes the divergence of a vector field depends on.

        Parameters
        ----------
        vector_field_dependence : list of list of sympy.Symbol
            Variables each vector component depends on.
        basis : {'covariant', 'contravariant'}, optional
            The basis in which the vector field is expressed. Default is 'contravariant'.

        Returns
        -------
        set or 0
            Set of dependent symbols, or 0 if divergence is identically zero.

        Examples
        --------
        Let's examine the gradient dependence of a spherical coordinate system.

        >>> from pisces_geometry.coordinates import OblateHomoeoidalCoordinateSystem
        >>> u = OblateHomoeoidalCoordinateSystem(ecc=0.5)
        >>> print(u.get_divergence_dependence([['xi'],[0],[0]]))
        {theta, xi}
        >>> print(u.get_divergence_dependence([['xi'],[0],[0]],basis='contravariant'))
        {theta, xi}

        """
        # noinspection PyTypeChecker
        dterms = list(self.get_expression("Dterm"))
        return get_divergence_dependence(
            vector_field_dependence,
            dterms,
            self.axes_symbols,
            inverse_metric=self.get_expression("inverse_metric_tensor")
            if basis == "covariant"
            else None,
            basis=basis,
        )

    def get_laplacian_dependence(
        self, scalar_field_dependence: Sequence[sp.Symbol]
    ) -> Union[set, int]:
        """
        Determine which coordinate axes the Laplacian of a scalar field depends on.

        Parameters
        ----------
        scalar_field_dependence : list of sympy.Symbol
            Variables on which the scalar field depends.

        Returns
        -------
        set or 0
            Set of dependent symbols, or 0 if Laplacian is identically zero.
        """
        return get_laplacian_dependence(
            scalar_field_dependence,
            self.axes_symbols,
            self.get_expression("inverse_metric_tensor"),
            l_term=sp.Array(self.get_expression("Lterm")),
            metric_density=self.get_class_expression("metric_density"),
        )

    # @@ IO Operations @@ #
    def to_hdf5(
        self,
        filename: Union[str, Path],
        group_name: str = None,
        overwrite: bool = False,
    ):
        r"""
        Save this coordinate system to HDF5.

        Parameters
        ----------
        filename : str
            The path to the output HDF5 file.
        group_name : str, optional
            The name of the group in which to store the grid data. If None, data is stored at the root level.
        overwrite : bool, default=False
            Whether to overwrite existing data. If False, raises an error when attempting to overwrite.
        """
        import json

        import h5py

        # Ensure that the filename is a Path object and then check for existence and overwrite violations.
        # These are only relevant at this stage if a particular group has not yet been specified.
        filename = Path(filename)
        if filename.exists():
            # Check if there are overwrite issues and then delete it if it is
            # relevant to do so.
            if (not overwrite) and (group_name is None):
                # We can't overwrite and there is data. Raise an error.
                raise IOError(
                    f"File '{filename}' already exists and overwrite=False. "
                    "To store data in a specific group, provide `group_name`."
                )
            elif overwrite and group_name is None:
                # We are writing to the core dir and overwrite is true.
                # delete the entire file and rebuild it.
                filename.unlink()
                with h5py.File(filename, "w"):
                    pass
        else:
            # The file didn't already exist, we simply create it and then
            # let it close again so that we can reopen it in the next phase.
            with h5py.File(filename, "w"):
                pass

        # Now that the file has been opened at least once and looks clean, we can
        # proceed with the actual write process. This will involve first checking
        # if there are overwrite violations when ``group_name`` is actually specified. Then
        # we can proceed with actually writing the data.
        with h5py.File(filename, "r+") as f:
            # Start checking for overwrite violations and the group information.
            if group_name is None:
                group = f
            else:
                # If the group exists, handle overwrite flag
                if group_name in f:
                    if overwrite:
                        del f[group_name]
                        group = f.create_group(group_name)
                    else:
                        raise IOError(
                            f"Group '{group_name}' already exists in '{filename}' and overwrite=False."
                        )
                else:
                    group = f.create_group(group_name)

            # Now start writing the core data to the disk. The coordinate system
            # MUST have the class name and then any optional parameters.
            group.attrs["class_name"] = str(self.__class__.__name__)

            # Save each kwarg individually as an attribute
            for key, value in self.parameters.items():
                if key in self.__PARAMETERS__:
                    if isinstance(value, (int, float, str)):
                        group.attrs[key] = value
                    else:
                        group.attrs[key] = json.dumps(value)  # serialize complex data

    # noinspection PyCallingNonCallable
    @classmethod
    def from_hdf5(cls, filename: Union[str, Path], group_name: str = None):
        r"""
        Save this coordinate system to HDF5.

        Parameters
        ----------
        filename : str
            The path to the output HDF5 file.
        group_name : str, optional
            The name of the group in which to store the grid data. If None, data is stored at the root level.
        """
        import json

        import h5py

        # Ensure that we have a connection to the file and that we can
        # actually open it in hdf5.
        filename = Path(filename)
        if not filename.exists():
            raise IOError(f"File '{filename}' does not exist.")

        # Now open the hdf5 file and look for the group name.
        with h5py.File(filename, "r") as f:
            # Identify the data storage group.
            if group_name is None:
                group = f
            else:
                if group_name in f:
                    group = f[group_name]
                else:
                    raise IOError(
                        f"Group '{group_name}' does not exist in '{filename}'."
                    )

            # Now load the class name from the group.
            __class_name__ = group.attrs["class_name"]

            # Load kwargs, deserializing complex data as needed
            kwargs = {}
            for key, value in group.attrs.items():
                if key != "class_name":
                    try:
                        kwargs[key] = json.loads(
                            value
                        )  # try to parse complex JSON data
                    except (TypeError, json.JSONDecodeError):
                        kwargs[key] = value  # simple data types remain as is

        try:
            _cls = DEFAULT_COORDINATE_REGISTRY[__class_name__]
        except KeyError:
            raise IOError(
                f"Failed to find the coordinate system class {__class_name__}. Ensure you have imported any"
                " relevant coordinate system modules."
            )
        return _cls(**kwargs)
