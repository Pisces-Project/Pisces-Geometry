"""
Base classes for geometric field representations.

:mod:`~fields.base` provides the core classes defining the behavior of all geometric fields in Pymetric. This
includes the base class :class:`_FieldBase` and its two descendants :class:`DenseField` and :class:`SparseField`.

For must use-cases, only the dense and sparse fields are used.

.. warning::

    Support for sparse fields is not yet fully implemented.
"""
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

from pymetric.differential_geometry.dependence import DenseDependenceObject
from pymetric.grids.base import GridBase

from .components import FieldComponent
from .mixins._generic import NumpyArithmeticMixin
from .mixins.base import DFieldCoreMixin, FieldCoreMixin, SFieldCoreMixin
from .mixins.dense_mathops import DenseFieldDMOMixin

# ==================================== #
# Static Typing Support                #
# ==================================== #
# Load in type checking only imports so that mypy and other
# static checkers can pass.
if TYPE_CHECKING:
    from pymetric.coordinates.base import _CoordinateSystemBase
    from pymetric.fields.buffers.base import BufferBase
    from pymetric.fields.utils._typing import ComponentDictionary, ComponentIndex


# ==================================== #
# Classes                              #
# ==================================== #
class _FieldBase(FieldCoreMixin):
    """
    Abstract base class for geometric data fields on structured grids.

    :class:`FieldBase` serves as the foundational interface for all field representations in the
    Pymetric library, enabling storage and manipulation of scalar, vector, or tensor-valued
    data defined over structured grids in curvilinear coordinate systems.

    This class is not intended to be used directly. Instead, concrete subclasses such as
    :class:`DenseField` and :class:`SparseField` implement specific data layouts and behavior. The base
    class provides core infrastructure for:

    - Attaching a field to a geometric `GridBase` object.
    - Managing a dictionary of `FieldComponent` objects.
    - Supporting both dense (single-component) and sparse (multi-component) layouts.
    - Validating component compatibility with the grid and coordinate system.
    - Providing consistent introspection through `grid`, `components`, and `coordinate_system`.

    Fields store values over structured domains, and each component can include additional
    trailing element-wise dimensions to represent vector or tensor values. All components
    must share the same underlying grid and be accessed via a canonicalized dictionary of
    component indices (typically integers or tuples of integers).

    Notes
    -----
    Subclasses must implement field-specific logic such as numerical operations, indexing,
    and transformations. This class assumes that all components are already validated and normalized.

    The base class (``_FieldBase``) lacks most relevant structures because they are implemented differently
    in each of the relevant subclasses. As such, the class lacks basic arithmetic support and a variety of
    other useful methods.
    """

    # --- Class Level Flags --- #
    # These are logical flags for determining
    # finer details of class behavior and triage.
    __array_priority__ = 3.0
    """
    The priority of the component class in numpy operations.
    """
    __array_function_dispatch__: Optional[Dict[Callable, Callable]] = {}
    """
    `__array_function_dispatch__` is a dictionary which can optionally map
    NumPy callables to internal implementations to allow overriding of default behavior.

    By default, when a NumPy function (non ufunc) is called on a Component, the buffer
    is stripped and the operation occurs on the underlying representation. If a callable
    is specified here, then `__array_function__()` will catch the redirect and
    triage accordingly.
    """

    # ------------------------------------ #
    # Initialization Processes             #
    # ------------------------------------ #
    def __init__(self, grid: GridBase, components: "ComponentDictionary"):
        """
        Initialize a new field with a structured grid and one or more field components.

        This constructor validates and registers all components, ensuring they are compatible
        with the specified grid. It supports both single-component and multi-component (sparse)
        layouts by storing components in a canonicalized dictionary.

        Parameters
        ----------
        grid : GridBase
            The structured grid over which the field is defined. All components must share this grid.
        components : dict[ComponentIndex, FieldComponent]
            A mapping from integer or tuple indices to `FieldComponent` instances.
            Each component must:
              - Be an instance of `FieldComponent`
              - Use the same grid as the field
              - Have a valid index (int or tuple of ints)

        Raises
        ------
        TypeError
            If `grid` is not an instance of `GridBase`.
        TypeError
            If any component is not a `FieldComponent`.
        ValueError
            If any component uses a different grid or has an invalid index.
        """
        # Begin by setting the grid after checking the type.
        if not isinstance(grid, GridBase):
            raise TypeError(f"`grid` argument must be a GridBase, not {type(grid)}.")
        self.__grid__: GridBase = grid

        # Now start handling components. Each component needs
        # to be validated to ensure that the elements are components,
        # that they have matching grids, and that the indices are
        # universally tuples.
        self.validate_components(components)
        self.__components__: ComponentDictionary = components
        self.__components_view__: MappingProxyType = MappingProxyType(
            self.__components__
        )

    # ------------------------------------ #
    # Core Methods                         #
    # ------------------------------------ #
    # These are dunder methods for basic interaction
    # with the FieldBase class. In cases where behaviors
    # are divergent between the sparse and dense subclasses,
    # the definitions are delegated to that level.
    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the field,
        including the grid and its component dictionary.

        Returns
        -------
        str
            Full constructor-style representation of the object.
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"  grid={self.grid!r},\n"
            f"  components={dict(self.components)!r}\n)"
        )

    def __str__(self) -> str:
        """
        Return a concise string representation summarizing grid and component count.

        Returns
        -------
        str
            User-friendly summary string.
        """
        return f"<{self.__class__.__name__} grid={self.grid} | Ncomponents={len(self.components)}>"

    # ------------------------------------ #
    # Properties                           #
    # ------------------------------------ #
    # These are the core properties of the class. They
    # can be altered in subclasses, but doing so should
    # be done cautiously.
    @property
    def grid(self) -> GridBase:
        """
        The structured grid associated with this field.

        All field components share this grid, which defines the coordinate
        system, resolution, and spatial layout of the domain over which the
        field is defined.

        Returns
        -------
        GridBase
            The grid object that defines the spatial structure of the field.
        """
        return self.__grid__

    @property
    def coordinate_system(self) -> "_CoordinateSystemBase":
        """
        The coordinate system (e.g. a subclass of :py:class:`~coordinates.core.OrthogonalCoordinateSystem`) which
        underlies this grid.

        The coordinate system determines which axes are available in the grid (:py:attr:`axes`) and also determines
        how various differential procedures are performed in this grid structure.
        """
        return self.__grid__.__cs__

    @property
    def components(self) -> MappingProxyType:
        """
        Read-only dictionary of field components.

        Each entry maps a component index (typically an integer or tuple of integers)
        to a `FieldComponent` that stores data for that part of the field. This structure
        supports both dense (single component) and sparse (multi-component) representations.

        Returns
        -------
        MappingProxyType
            An immutable view of the field’s component dictionary.
        """
        return self.__components_view__


class DenseField(_FieldBase, DFieldCoreMixin, DenseFieldDMOMixin, NumpyArithmeticMixin):
    """
    Concrete field class for dense, single-component data on structured grids.

    :class:`DenseField` represents the most common type of geometric field in Pymetric:
    a uniformly defined scalar, vector, or tensor field backed by a single data buffer and
    aligned with a structured :class:`~grids.base.GridBase`. This class is ideal for
    continuous fields that span the entire domain and share a consistent coordinate and metric context.

    Internally, the field is implemented using a single :class:`~fields.components.FieldComponent`,
    which encodes both the array data and its spatial alignment. The field supports geometric
    operations, symbolic dependence tracking, coordinate-aware transformations, and NumPy-style
    arithmetic and ufuncs.

    Typical use cases include:

    - Scalar fields (e.g., temperature, pressure)
    - Vector fields (e.g., velocity, electric field)
    - Tensor fields (e.g., stress, strain)

    See Also
    --------
    ~fields.components.FieldComponent :
        Underlying data wrapper used by this class.
    ~fields.tensors.DenseTensorField :
        Tensor-valued field supporting variance-aware operations.
    ~grids.base.GridBase :
        Structured grid class supporting coordinate systems and chunking.
    """

    # ------------------------------------ #
    # Initialization Processes             #
    # ------------------------------------ #
    def __init__(self, grid: GridBase, component: FieldComponent):
        """
        Initialize a dense, single-component field on a structured grid.

        Parameters
        ----------
        grid : ~grids.base.GridBase
            The structured grid over which the field is defined.
            This provides coordinate access, chunking logic, and metric information
            needed for geometric operations.

        component : ~fields.components.FieldComponent
            The sole field component representing the field data.
            This encapsulates the field buffer, axis metadata, and tensor rank,
            and is used for all internal computations.

        Notes
        -----
        This class is intended for dense fields—those with a single buffer
        representing either scalar, vector, or tensor values over the entire domain.
        Multi-component or symbolic fields should use the corresponding multi-component field classes.
        """
        # Allocate the single component in the new __component__
        # attribute.
        self.__component__: FieldComponent = component

        # Now create a singleton dictionary and pass
        # off to the super().
        components = {0: component}
        super().__init__(grid, components)

        # Construct the dependence object
        self.__dependence__: DenseDependenceObject = None

    # ------------------------------------ #
    # Core Methods                         #
    # ------------------------------------ #
    # These are dunder methods for basic interaction
    # with the FieldBase class.
    def __getitem__(self, idx: Any) -> Any:
        return self.__component__.__getitem__(idx)

    def __setitem__(self, idx: Any, value: Any) -> Any:
        return self.__component__.__setitem__(idx, value)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Numpy ufunc standardization. For this, we forward the operations to
        the underlying components and then re-wrap the resulting component that
        is returned if we can.
        """
        # Standardize the inputs so that they are all relegated to the component
        # level for ufuncs.
        inputs = [
            inp.__component__ if isinstance(inp, self.__class__) else inp
            for inp in inputs
        ]

        # Catch any specifications for `out`. The approach here is to have
        # out pass down to the relevant component and then simply return the
        # already-wrapped field.
        # Handle any instances where `out` is specified. We require that
        # the two field components have the same axes. All other checks
        # are delegated to the operation on the underlying buffer.
        out = kwargs.get("out", None)
        if out is not None:
            # Normalize to a tuple for uniform processing
            is_tuple = isinstance(out, tuple)
            out_tuple = out if is_tuple else (out,)

            # Unwrap buffers
            unwrapped_out = tuple(
                o.__component__ if isinstance(o, self.__class__) else o
                for o in out_tuple
            )
            kwargs["out"] = unwrapped_out if is_tuple else unwrapped_out[0]

            # Apply the ufunc
            result = getattr(ufunc, method)(*inputs, **kwargs)

            # check if result is None or if we got something back.
            # If something came back we actually want to pass out.
            if result is not None:
                return out
            else:
                return result
        else:
            # Now calculate the results of the operation.
            result = getattr(ufunc, method)(*inputs, **kwargs)

            # check if the result is actually a component. If it is,
            # we need to re-wrap the result.
            if isinstance(result, FieldComponent):
                return self.__class__(result.grid, result)
            else:
                return result

    def __array_function__(self, func, types, args, kwargs):
        """
        Override NumPy high-level functions for BufferBase.

        The heuristic for this behavior is to simply delegate operations to
        the buffer representation unless there is a specific override in place.
        """
        # Check for custom forwarding implementations via
        # the __array_functions_dispatch__.
        if all(issubclass(t, self.__class__) for t in types):
            # Fetch the dispatch and check for the override of
            # this function.
            redirect_func = getattr(self, "__array_function_dispatch__", {}).get(
                func, None
            )
            if redirect_func is not None:
                # We have a redirection, we now delegate to that.
                return redirect_func(*args, **kwargs)

        # No valid dispatch found. We now strip the args down and
        # pass through without and further alterations.
        unwrapped_args = tuple(
            a.__component__ if isinstance(a, self.__class__) else a for a in args
        )
        unwrapped_kwargs = {
            _k: _v.__component__ if isinstance(_v, self.__class__) else _v
            for _k, _v in kwargs.items()
        }
        return func(*unwrapped_args, **unwrapped_kwargs)

    # ------------------------------------ #
    # Properties                           #
    # ------------------------------------ #
    # These are the core properties of the class. They
    # can be altered in subclasses, but doing so should
    # be done cautiously.
    @property
    def rank(self) -> int:
        """
        The rank [number of indices] of the tensor.

        Returns
        -------
        int
            The number of element-wise dimensions of the component.
        """
        return self.__component__.element_ndim

    @property
    def axes(self) -> List[str]:
        """
        The spatial axes along which this field component is defined.

        These axes correspond to the leading dimensions of the field buffer.

        Returns
        -------
        list of str
            Canonical list of spatial axis labels.
        """
        return self.__component__.axes

    @property
    def naxes(self) -> int:
        """
        Number of spatial axes the field is defined over.

        Returns
        -------
        int
            The number of named spatial axes.
        """
        return self.__component__.naxes

    @property
    def buffer(self) -> "BufferBase":
        """
        The internal buffer storing this field’s data.

        This buffer provides backend-specific logic (NumPy, unyt, HDF5, etc.)
        for data access, arithmetic, and I/O.

        Returns
        -------
        ~fields.buffers.base.BufferBase
            The underlying storage buffer.
        """
        return self.__component__.buffer

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the full data array, including spatial and element dimensions.

        Returns
        -------
        tuple of int
            Full shape of the field buffer.
        """
        return self.__component__.shape

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """
        The shape of the field over the spatial axes (grid-aligned dimensions).

        Returns
        -------
        tuple of int
            Shape of the field over the named spatial axes only.
        """
        return self.__component__.spatial_shape

    @property
    def element_shape(self) -> Tuple[int, ...]:
        """
        The shape of the element-wise structure (e.g., vector or tensor components).

        Returns
        -------
        tuple of int
            Shape of the trailing element-wise dimensions.
        """
        return self.__component__.element_shape

    @property
    def is_scalar(self) -> bool:
        """Return True if the field has no element-wise structure."""
        return self.__component__.is_scalar

    @property
    def size(self) -> int:
        """
        Total number of elements in the buffer.

        Returns
        -------
        int
            Product of all dimensions in the field shape.
        """
        return self.__component__.size

    @property
    def element_size(self) -> int:
        """Total number of element-wise components."""
        return self.__component__.element_size

    @property
    def spatial_size(self) -> int:
        """Total number of spatial elements (grid cells)."""
        return self.__component__.spatial_size

    @property
    def ndim(self) -> int:
        """
        Total number of dimensions in the field buffer.

        This includes both spatial (grid-aligned) and trailing element dimensions.

        Returns
        -------
        int
            Total rank of the field array.
        """
        return self.__component__.ndim

    @property
    def spatial_ndim(self) -> int:
        """
        Number of spatial dimensions (i.e., number of named axes).

        Returns
        -------
        int
            Number of dimensions aligned with the grid.
        """
        return self.__component__.spatial_ndim

    @property
    def element_ndim(self) -> int:
        """
        Number of trailing element-wise dimensions (e.g., vector or tensor structure).

        Returns
        -------
        int
            Number of dimensions not aligned with spatial grid axes.
        """
        return self.__component__.element_ndim

    @property
    def dtype(self) -> Any:
        """
        The data type of the elements stored in the buffer.

        Returns
        -------
        dtype
            The NumPy dtype or equivalent backend type.
        """
        return self.__component__.dtype

    @property
    def c(self):
        """
        Shorthand to obtain the underlying buffer's core
        object.

        Returns
        -------
        ArrayLike
            The backend-native data structure stored in this buffer.
        """
        return self.__component__.c

    @property
    def dependence(self) -> DenseDependenceObject:
        """
        The symbolic coordinate dependence object for this tensor field.

        This property returns a :class:`~differential_geometry.dependence.DenseDependenceObject`
        instance that encodes the symbolic dependence of each component of the field
        on the coordinate axes. It is used in symbolic differential geometry operations
        (e.g., gradient, divergence, Laplacian) to track and propagate analytical dependence
        information through transformations.

        The object is lazily constructed on first access, using:

        - the coordinate system associated with the grid
        - the tensor rank of the field
        - the set of grid axes on which the field is defined

        Returns
        -------
        DenseDependenceObject
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
            self.__dependence__ = DenseDependenceObject(
                self.__grid__.__cs__, self.element_shape, dependent_axes=self.axes
            )

        return self.__dependence__


class SparseField(_FieldBase, SFieldCoreMixin):
    """
    Field class for sparse, multi-component representations on structured grids.

    `SparseField` represents a field composed of multiple independent `FieldComponent` objects,
    each identified by a unique index (typically an integer or a tuple of integers). This class
    is designed to support fields with:

    - Explicitly indexed tensor components (e.g., Tᵢⱼ)
    - Direction-dependent or variably-defined field data
    - Sparse physical quantities (e.g., only select components are populated)

    Unlike `DenseField`, which wraps a single component and assumes uniform structure,
    `SparseField` allows for heterogeneous or partially defined field data across multiple
    indexed subfields.
    """

    # ------------------------------------ #
    # Initialization Processes             #
    # ------------------------------------ #
    # MultiComponentFieldBase inherits the same initialization
    # procedures as the FieldBase does.
    def __init__(self, *args, **kwargs):
        """
        .. important::

            Not yet implemented.
        """
        raise NotImplementedError("Support for sparse fields is not yet available.")

    # ------------------------------------ #
    # Core Methods                         #
    # ------------------------------------ #
    # These are dunder methods for basic interaction
    # with the FieldBase class.
    def __getitem__(self, index: "ComponentIndex") -> FieldComponent:
        """
        Retrieve a specific field component by index.

        Parameters
        ----------
        index : ComponentIndex
            The index (typically an integer or tuple) identifying the component.

        Returns
        -------
        FieldComponent
            The component corresponding to the given index.

        Raises
        ------
        KeyError
            If the index does not exist in the component dictionary.
        """
        return self.__components__[index]

    def __setitem__(self, index: "ComponentIndex", value: FieldComponent) -> None:
        """
        Set or replace a component at a specific index.

        Validation is performed to ensure the component is valid and grid-consistent.

        Parameters
        ----------
        index : ComponentIndex
            Index of the component to replace or insert.
        value : FieldComponent
            The new component.

        Raises
        ------
        TypeError or ValueError
            If the component is invalid or incompatible with the field grid.
        """
        self.validate_component(index, value)
        self.__components__[index] = value

    def __delitem__(self, index: "ComponentIndex"):
        del self.__components__[index]

    def __len__(self) -> int:
        return len(self.__components__)

    def __iter__(self) -> Iterator["ComponentIndex"]:
        return iter(self.__components__)

    def __contains__(self, item: "ComponentIndex") -> bool:
        return item in self.__components__

    # ------------------------------------ #
    # Properties                           #
    # ------------------------------------ #
    # These are the core properties of the class. They
    # can be altered in subclasses, but doing so should
    # be done cautiously.
    @property
    def num_components(self) -> int:
        """
        Total number of components in the sparse field.

        Each component corresponds to a distinct entry in the internal component
        dictionary, typically indexed by an integer or tuple of integers. This
        property provides a quick way to determine the field's structural richness
        or tensorial complexity.

        Returns
        -------
        int
            The number of active components stored in the field.
        """
        return len(self.__components__)

    @property
    def component_list(self) -> List["ComponentIndex"]:
        """
        List of all component indices present in the field.

        These indices identify each subfield (or `FieldComponent`) in the sparse
        field structure. Component indices are typically integers (e.g., 0, 1) or
        tuples of integers (e.g., (0,), (1, 1)) used to represent tensor positions
        or other indexing schemes.

        Returns
        -------
        list of ComponentIndex
            A list of keys corresponding to the active components of the field.
        """
        return list(self.__components__.keys())
