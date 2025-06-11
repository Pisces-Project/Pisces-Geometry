"""
Core mixin classes inherited by all coordinate system subclasses.

This module provides the coordinate system support for IO operations as well as
interactions with coordinate axes, managing and manipulating coordinate order, and
other supplemental methods.
"""
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import sympy as sp

from pymetric.utilities.arrays import normalize_index

# ================================== #
# TYPING SUPPORT                     #
# ================================== #
if TYPE_CHECKING:
    import unyt

    from pymetric.coordinates.base import _CoordinateSystemBase
    from pymetric.coordinates.mixins._typing import (
        _SupportsCoordinateSystemBase,
        _SupportsCoordinateSystemCore,
    )

_ExpressionType = Union[
    sp.Symbol,
    sp.Expr,
    sp.Matrix,
    sp.MutableDenseMatrix,
    sp.MutableDenseNDimArray,
    sp.ImmutableDenseMatrix,
    sp.ImmutableDenseNDimArray,
]
_SupCoordSystemBase = TypeVar(
    "_SupCoordSystemBase", bound="_SupportsCoordinateSystemBase"
)
_SupCoordSystemCore = TypeVar(
    "_SupCoordSystemCore", bound="_SupportsCoordinateSystemCore"
)


# ================================== #
# Mixin Classes                      #
# ================================== #
# These classes form the core mixins of the base coordinate system class.
class CoordinateSystemCoreMixin(Generic[_SupCoordSystemBase]):
    """
    Core methods for coordinate systems wrapped in a Mixin for
    readability.
    """

    # -------------------------- #
    # Basic Utility Functions    #
    # -------------------------- #
    def pprint(self: _SupCoordSystemBase) -> None:
        """
        Print a detailed description of the coordinate system, including its axes, parameters, and expressions.

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

    # ------------------------------- #
    # Coordinate Conversion Utilities #
    # ------------------------------- #
    # These methods provide access to the API for
    # coordinate conversion.
    def _check_same_dimension(
        self: _SupCoordSystemBase, other: _SupCoordSystemBase
    ) -> None:
        if self.ndim != other.ndim:
            raise ValueError(
                "Coordinate systems must have the same number of dimensions."
            )

    def to_cartesian(self: _SupCoordSystemBase, *coords: Any) -> Tuple[np.ndarray, ...]:
        """
        Convert native coordinates to Cartesian coordinates.

        Parameters
        ----------
        *coords : float or array-like
            Coordinates in this system's native basis.

        Returns
        -------
        tuple of np.ndarray
            Cartesian coordinates (x, y, z) or lower-dimensional equivalent.
        """
        return self._convert_native_to_cartesian(*coords)

    def from_cartesian(
        self: _SupCoordSystemBase, *coords: Any
    ) -> Tuple[np.ndarray, ...]:
        """
        Convert Cartesian coordinates to native coordinates in this system.

        Parameters
        ----------
        *coords : float or array-like
            Cartesian coordinates (x, y, z) or similar.

        Returns
        -------
        tuple of np.ndarray
            Native coordinates for this coordinate system.
        """
        return self._convert_cartesian_to_native(*coords)

    def convert_to(
        self: _SupCoordSystemBase,
        target_system: "_CoordinateSystemBase",
        *native_coords: Any,
    ) -> Tuple[np.ndarray, ...]:
        """
        Convert coordinates from this system to another coordinate system via Cartesian intermediate.

        Parameters
        ----------
        target_system : _CoordinateSystemBase
            The target coordinate system to convert to.
        *native_coords : float or array-like
            Coordinates in this system's native basis.

        Returns
        -------
        tuple of np.ndarray
            Coordinates expressed in the target coordinate system.

        Example
        -------
        .. code-block:: python

            from pymetric.coordinates import SphericalCoordinateSystem, CylindricalCoordinateSystem

            sph = SphericalCoordinateSystem()
            cyl = CylindricalCoordinateSystem()
            rho, phi, z = sph.convert_to(cyl, 1.0, np.pi/2, 0.0)
        """
        self._check_same_dimension(target_system)
        cartesian_coords = self.to_cartesian(*native_coords)
        return target_system.from_cartesian(*cartesian_coords)

    def get_conversion_transform(
        self: _SupCoordSystemBase, other: "_CoordinateSystemBase"
    ) -> Callable:
        """
        Construct a coordinate transformation function that maps native coordinates
        from this coordinate system to the target coordinate system.

        The returned function can be used to convert any valid input (scalars or arrays)
        in the native coordinate system of `self` into the native coordinate system of `other`.

        Parameters
        ----------
        other : _CoordinateSystemBase
            The target coordinate system to transform into.

        Returns
        -------
        Callable
            A function that takes native coordinates of `self` and returns native coordinates of `other`.

        Example
        -------
        .. code-block:: python

            sph = SphericalCoordinateSystem()
            cyl = CylindricalCoordinateSystem()

            transform = sph.get_conversion_transform(cyl)
            rho, phi, z = transform(1.0, np.pi / 2, 0.0)

        Notes
        -----
        This conversion is performed via Cartesian coordinates:
        native (self) -> Cartesian -> native (other).
        """
        # Validate that the coordinate systems are of the
        # same overall dimension.
        self._check_same_dimension(other)

        # Construct the function to pass over.
        # noinspection PyMissingOrEmptyDocstring
        def transform(*native_coords):
            cartesian = self.to_cartesian(*native_coords)
            return other.from_cartesian(*cartesian)

        return transform


class CoordinateSystemIOMixin(Generic[_SupCoordSystemCore]):
    """
    Mixin class for :py:class:`coordinates.core.CurvilinearCoordinateSystem` that provides
    serialization support for saving and loading coordinate systems to and from HDF5.

    This mixin implements convenient methods for persisting coordinate system instances,
    including all user-specified parameters, to HDF5 files. It supports both flat and
    group-based storage within the file, and includes registry-aware deserialization
    to recover the correct class type.

    Key Capabilities
    ----------------

    - Save a coordinate system instance to disk with :meth:`to_hdf5`.
    - Restore a coordinate system instance from disk with :meth:`from_hdf5`.
    - Automatically serialize parameters, including support for JSON-encoded complex values.
    - Supports hierarchical group-based storage in HDF5 files.
    - Uses a registry to resolve class names to actual coordinate system types on load.
    """

    def to_hdf5(
        self: _SupCoordSystemCore,
        filename: Union[str, Path],
        group_name: Optional[str] = None,
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
                raise OSError(
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
                        raise OSError(
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

    @classmethod
    def from_hdf5(
        cls: _SupCoordSystemCore,
        filename: Union[str, Path],
        group_name: Optional[str] = None,
        registry: Optional[Dict] = None,
    ):
        r"""
        Save this coordinate system to HDF5.

        Parameters
        ----------
        filename : str
            The path to the output HDF5 file.
        group_name : str, optional
            The name of the group in which to store the grid data. If None, data is stored at the root level.
        registry : dict, optional
            Dictionary mapping class names to coordinate system classes. If None, uses the class's default registry.

        """
        import json

        import h5py

        # Fill in the registry assignment.
        if registry is None:
            registry = cls.__DEFAULT_REGISTRY__

        # Ensure that we have a connection to the file and that we can
        # actually open it in hdf5.
        filename = Path(filename)
        if not filename.exists():
            raise OSError(f"File '{filename}' does not exist.")

        # Now open the hdf5 file and look for the group name.
        with h5py.File(filename, "r") as f:
            # Identify the data storage group.
            if group_name is None:
                group = f
            else:
                if group_name in f:
                    group = f[group_name]
                else:
                    raise OSError(
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
            _cls = registry[__class_name__]
        except KeyError:
            raise OSError(
                f"Failed to find the coordinate system class {__class_name__}. Ensure you have imported any"
                " relevant coordinate system modules."
            )
        return _cls(**kwargs)

    def to_json(
        self: _SupCoordSystemCore, filepath: Union[str, Path], overwrite: bool = False
    ):
        """
        Save the coordinate system to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output JSON file.
        overwrite : bool, optional
            If True, overwrite the file if it already exists. Default is False.

        Notes
        -----
        When called, this method will serialize the coordinate system's class name and
        its parameters so that they can be dumped to a JSON file. When reloaded, the coordinate
        system will be reinitialized from that data.
        """
        import json

        # Standardize the filepath and check if it exists or not. We'll
        # need to check the overwrite parameter if it does exist.
        filepath = Path(filepath)

        if filepath.exists() and overwrite:
            filepath.unlink()
        elif filepath.exists() and (not overwrite):
            raise ValueError(
                "File `{filepath}` already exists. To overwrite, set `overwrite=True`."
            )
        else:
            pass

        # Serialize the class name and data so that they can be written
        # to JSON. In general, this shouldn't be an issue because the parameters aren't
        # exotic types.
        data = {
            "class_name": self.__class__.__name__,
            "parameters": {
                k: v for k, v in self.parameters.items() if k in self.__PARAMETERS__
            },
        }

        # Now write the data to the file.
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(
        cls: _SupCoordSystemCore,
        filepath: Union[str, Path],
        registry: Optional[Dict] = None,
    ):
        """
        Load a coordinate system from a JSON file. The
        coordinate system name is resolved by reference to the
        `registry`, which can be inferred from the default registry or
        be specified by the user.

        Parameters
        ----------
        filepath : str or Path
            Path to the input JSON file.
        registry : dict, optional
            Registry mapping class names to coordinate system classes.
        """
        import json

        # Coerce the filepath and try to open the file
        # with JSON protocol.
        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)

        # Parse the registry and read the
        # data from the JSON file.
        if registry is None:
            registry = cls.__DEFAULT_REGISTRY__

        class_name = data["class_name"]
        parameters = data["parameters"]

        try:
            target_cls = registry[class_name]
        except KeyError:
            raise OSError(
                f"Unknown coordinate system class '{class_name}' in registry."
            )

        return target_cls(**parameters)

    def to_yaml(
        self: _SupCoordSystemCore, filepath: Union[str, Path], overwrite: bool = False
    ):
        """
        Save the coordinate system to a YAML file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output YAML file.
        overwrite : bool, optional
            If True, overwrite the file if it already exists. Default is False.
        """
        import yaml

        filepath = Path(filepath)

        if filepath.exists() and overwrite:
            filepath.unlink()
        elif filepath.exists() and (not overwrite):
            raise ValueError(
                f"File `{filepath}` already exists. To overwrite, set `overwrite=True`."
            )

        data = {
            "class_name": self.__class__.__name__,
            "parameters": {
                k: v for k, v in self.parameters.items() if k in self.__PARAMETERS__
            },
        }

        with open(filepath, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def from_yaml(
        cls: _SupCoordSystemCore,
        filepath: Union[str, Path],
        registry: Optional[Dict] = None,
    ):
        """
        Load a coordinate system from a YAML file.

        Parameters
        ----------
        filepath : str or Path
            Path to the input YAML file.
        registry : dict, optional
            Registry mapping class names to coordinate system classes.
        """
        import yaml

        filepath = Path(filepath)
        with open(filepath) as f:
            data = yaml.safe_load(f)

        if registry is None:
            registry = cls.__DEFAULT_REGISTRY__

        class_name = data["class_name"]
        parameters = data["parameters"]

        try:
            target_cls = registry[class_name]
        except KeyError:
            raise OSError(
                f"Unknown coordinate system class '{class_name}' in registry."
            )

        return target_cls(**parameters)


class CoordinateSystemAxesMixin(Generic[_SupCoordSystemCore]):
    """
    Mixin class for :py:class:`coordinates.core.CurvilinearCoordinateSystem` which provides
    support for axis manipulation and logic in coordinate systems.

    This class defines a comprehensive suite of methods for manipulating and validating
    axis names, indices, masks, permutations, and orderings in the context of a coordinate system.
    It is designed to be mixed into coordinate system base classes in the Pisces Geometry library.

    Key Capabilities
    ----------------

    - Convert between axis names and numeric indices.
    - Build and interpret boolean axis masks.
    - Validate and normalize axis inputs (with optional order enforcement).
    - Compute permutations and reorderings for axes and associated data.
    - Insert or complete axis-aligned iterables using fixed axes.
    - Provide LaTeX representations of axes for display purposes.
    """

    # -------------------------------- #
    # Basic Axes Utilities             #
    # -------------------------------- #
    # These basic utilities are really just simple wrappers
    # around logic that could be easily implemented independently.
    def convert_indices_to_axes(
        self: _SupCoordSystemCore, axes_indices: Union[int, Sequence[int]]
    ) -> Union[str, List[str]]:
        """
        Convert axis index or indices to their corresponding axis name(s).

        This method maps a single axis index or a sequence of axis indices to the
        canonical axis name(s) as defined in the coordinate system's ``__AXES__`` attribute.

        Parameters
        ----------
        axes_indices : int or Sequence[int]
            An axis index or list/tuple of axis indices.
            Negative indices are supported and interpreted as in standard Python indexing.

        Returns
        -------
        str or list of str
            The axis name(s) corresponding to the provided index or indices.
            Returns a string for a single index and a list of strings for a sequence.

        Raises
        ------
        IndexError
            If any index is out of bounds for the dimensionality of the coordinate system.

        Notes
        -----
        This method is useful for converting internal numeric axis representations
        (e.g., from grid shape or tensor slots) into symbolic or user-facing axis names
        (like "r", "theta", "z", etc.).

        Examples
        --------
        This method allows for various index to axes conversions. Notably, if
        the input is scalar, the output will also be scalar and if the input is
        an iterable, then so too will the output. For example, if ``0`` is put in,
        the result will look like:

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> u = SphericalCoordinateSystem()

        >>> u.convert_indices_to_axes(0)
        'r'

        Likewise, providing ``[0,2]`` yields

        >>> u.convert_indices_to_axes([0,2])
        ['r', 'phi']

        Scalar axes can also be provided inside of iterables to ensure consistent
        typing:

        >>> u.convert_indices_to_axes([0])
        ['r']

        """
        # Enforce the typing on the axes indices so
        # that they are a uniform iterable type.
        if not hasattr(axes_indices, "__len__"):
            axes_indices = [axes_indices]
            _as_scalar = True
        else:
            axes_indices = list(axes_indices)
            _as_scalar = False
        # Normalize the indices.
        axes_indices = [normalize_index(index, self.ndim) for index in axes_indices]

        # Now perform the indexing procedure.
        if _as_scalar:
            return self.__AXES__[axes_indices[0]]
        else:
            return [self.__AXES__[axes_index] for axes_index in axes_indices]

    def convert_axes_to_indices(
        self: _SupCoordSystemCore, axes: Union[str, Sequence[str]]
    ) -> Union[int, List[int]]:
        """
        Convert axis name(s) to their corresponding index or indices.

        This method maps a single axis name or a sequence of axis names to their
        numeric index as defined by the order of the coordinate system’s ``__AXES__``
        attribute.

        Parameters
        ----------
        axes : str or Sequence[str]
            A single axis name or a list/tuple of axis names to convert.

        Returns
        -------
        int or list of int
            The index/indices corresponding to the given axis name(s). Returns an
            integer for a single axis and a list of integers for multiple axes.

        Raises
        ------
        ValueError
            If any axis name is not found in the coordinate system.

        Notes
        -----
        This method provides the inverse of :meth:`convert_indices_to_axes`, allowing
        user-facing axis names (like "r", "theta", "phi") to be mapped to their internal
        numeric indices (e.g. 0, 1, 2). This is commonly used when aligning field data,
        slicing tensors, or resolving axis permutations for broadcasting and contraction.

        Examples
        --------
        Axis names may be passed individually or as sequences. Scalar inputs yield scalar
        outputs, and sequences yield lists:

        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> u = SphericalCoordinateSystem()

        >>> u.convert_axes_to_indices("r")
        0

        >>> u.convert_axes_to_indices(["r", "phi"])
        [0, 2]

        >>> u.convert_axes_to_indices(["theta"])
        [1]
        """
        # Enforce the typing on the axes indices so
        # that they are a uniform iterable type.
        if isinstance(axes, str):
            axes = [axes]
            _as_scalar = True
        elif hasattr(axes, "__len__"):
            axes = list(axes)
            _as_scalar = False
        else:
            raise TypeError(f"Invalid type {type(axes)} for `axes`.")

        # Check that all axes are in the __AXES__.
        if any(ax not in self.__AXES__ for ax in axes):
            raise ValueError(
                f"Invalid axes: {[ax for ax in axes if ax not in self.__AXES__]}"
            )

        # Now perform the indexing procedure.
        if _as_scalar:
            return self.__AXES__.index(axes[0])
        else:
            return [self.__AXES__.index(ax) for ax in axes]

    def build_axes_mask(self: _SupCoordSystemCore, axes: Sequence[str]) -> np.ndarray:
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
        _axes = np.asarray([self.convert_axes_to_indices(ax) for ax in axes], dtype=int)

        # Fill the mask values.
        _mask[_axes] = True
        return _mask

    def get_axes_from_mask(self: _SupCoordSystemCore, mask: np.ndarray) -> List[str]:
        """
        Convert a boolean axis mask into a list of axis names.

        This method reverses the effect of :meth:`build_axes_mask` by returning
        the axis names corresponding to ``True`` values in the provided mask.

        Parameters
        ----------
        mask : np.ndarray
            A boolean array of length ``ndim`` where each ``True`` value indicates
            that the corresponding axis is selected. Must match the length of
            ``self.__AXES__``.

        Returns
        -------
        list of str
            The names of the axes that are selected in the mask.

        Raises
        ------
        ValueError
            If the mask does not have the same length as the number of coordinate axes.

        Notes
        -----
        This is the inverse of :meth:`build_axes_mask`:

        >>> mask = cs.build_axes_mask(["r", "phi"])
        >>> cs.get_axes_from_mask(mask)
        ['r', 'phi']

        Examples
        --------
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> cs = SphericalCoordinateSystem()
        >>> mask = np.array([True, False, True])
        >>> cs.get_axes_from_mask(mask)
        ['r', 'phi']
        """
        if mask.shape[0] != len(self.__AXES__):
            raise ValueError(
                f"Mask length {mask.shape[0]} does not match number of axes ({len(self.__AXES__)})."
            )

        return list(np.asarray(self.__AXES__, dtype=str)[mask])

    def get_mask_from_axes(
        self: _SupCoordSystemCore, axes: Union[str, Sequence[str]]
    ) -> np.ndarray:
        """
        Return a boolean mask of shape ``(ndim,)`` with *True* on the
        positions corresponding to ``axes``.

        Parameters
        ----------
        axes : str or Sequence[str]
            An axis name or iterable of axis names.

        Returns
        -------
        numpy.ndarray
            Boolean mask selecting those axes.

        Examples
        --------
        >>> cs.get_mask_from_axes("phi")
        array([False, False,  True])
        >>> cs.get_mask_from_axes(["r", "theta"])
        array([ True,  True, False])
        """
        # Normalise to list of canonical names then reuse existing helper
        axes_list = [axes] if isinstance(axes, str) else list(axes)
        return self.build_axes_mask(axes_list)

    def get_mask_from_indices(
        self: _SupCoordSystemCore, indices: Union[int, Sequence[int]]
    ) -> np.ndarray:
        """
        Boolean mask that is *True* at the supplied numeric indices.

        Negative indices are handled exactly like standard Python indexing.

        Parameters
        ----------
        indices : int or Sequence[int]

        Returns
        -------
        numpy.ndarray
            Mask of length ``ndim``.

        Raises
        ------
        IndexError
            If any index is out of range.
        """
        if isinstance(indices, int):
            idx_list = [indices]
        else:
            idx_list = list(indices)

        # Normalise negatives / validate range
        idx_list = [normalize_index(i, self.ndim) for i in idx_list]

        mask = np.zeros(self.ndim, dtype=bool)
        mask[idx_list] = True
        return mask

    def get_indices_from_mask(
        self: _SupCoordSystemCore, mask: np.ndarray
    ) -> Union[int, List[int]]:
        """
        Convert a boolean mask of length ``ndim`` back to numeric indices.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean selector for axes.

        Returns
        -------
        int or list[int]
            * An ``int`` if exactly one element is *True*.
            * A ``list`` of ints if multiple elements are *True*.

        Raises
        ------
        ValueError
            If the mask length does not equal ``ndim``.
        """
        if mask.shape[0] != self.ndim:
            raise ValueError(f"Mask length {mask.shape[0]} != ndim ({self.ndim}).")

        idx = np.nonzero(mask)[0]
        return int(idx[0]) if idx.size == 1 else idx.tolist()

    # -------------------------------- #
    # Permutations and Order           #
    # -------------------------------- #
    # These methods help with permuting objects and
    # ordering objects according to axes.
    def axes_complement(self: _SupCoordSystemCore, axes: Sequence[str]) -> List[str]:
        """
        Return all axes in the coordinate system that are not present in `axes`.

        Parameters
        ----------
        axes : list of str
            Subset of axes to exclude.

        Returns
        -------
        list of str
            Canonically ordered axes not included in `axes`.

        Examples
        --------
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> cs = SphericalCoordinateSystem()
        >>> cs.axes
        ['r', 'theta', 'phi']
        >>> cs.axes_complement(["theta"])
        ['r', 'phi']
        """
        return [ax for ax in self.axes if ax not in axes]

    def is_axis(
        self: _SupCoordSystemCore, axis: Union[str, Sequence[str]]
    ) -> Union[bool, List[bool]]:
        """
        Check whether the given axis name(s) exist in this coordinate system.

        Parameters
        ----------
        axis : str or list of str
            One or more axis names to validate.

        Returns
        -------
        bool or list of bool
            True/False for single input; list of bools for multiple inputs.

        Examples
        --------
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> cs = SphericalCoordinateSystem()
        >>> cs.is_axis("theta")
        True

        >>> cs.is_axis(["r", "x"])
        [True, False]
        """
        if isinstance(axis, str):
            return axis in self.axes
        return [ax in self.axes for ax in axis]

    @staticmethod
    def is_axes_subset(axes_a: Sequence[str], axes_b: Sequence[str]) -> bool:
        """
        Check if `axes_a` is a subset of `axes_b`.

        Parameters
        ----------
        axes_a : Sequence[str]
            The axes to check as a potential subset.
        axes_b : Sequence[str]
            The reference axes that should include all of `axes_a`.

        Returns
        -------
        bool
            True if every axis in `axes_a` is in `axes_b`, else False.

        Examples
        --------
        >>> cs.is_subset(["r", "theta"], ["r", "theta", "phi"])
        True
        >>> cs.is_subset(["phi", "z"], ["r", "theta", "phi"])
        False
        """
        return set(axes_a).issubset(set(axes_b))

    @staticmethod
    def is_axes_superset(axes_a: Sequence[str], axes_b: Sequence[str]) -> bool:
        """
        Check if `axes_a` is a superset of `axes_b`.

        Parameters
        ----------
        axes_a : Sequence[str]
            The axes to check as a potential superset.
        axes_b : Sequence[str]
            The reference axes that should be contained within `axes_a`.

        Returns
        -------
        bool
            True if every axis in `axes_b` is in `axes_a`, else False.

        Examples
        --------
        >>> cs.is_superset(["r", "theta", "phi"], ["theta"])
        True
        >>> cs.is_superset(["theta"], ["r", "phi"])
        False
        """
        return set(axes_a).issuperset(set(axes_b))

    def get_free_fixed(
        self: _SupCoordSystemCore,
        axes: Optional[Sequence[str]] = None,
        *,
        fixed_axes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Split a list of coordinate axes into fixed and free components.

        This utility verifies that all fixed axes are:
        - present in the coordinate system
        - included in the axes list being considered

        It then returns a list of free axes (i.e., axes not fixed) and the fixed axis dictionary.

        Parameters
        ----------
        axes : list of str, optional
            The axes to consider. If not provided, uses all coordinate system axes.
        fixed_axes : dict of {str: Any}, optional
            A mapping of fixed axis names to values.

        Returns
        -------
        (list of str, dict of str → Any)
            A tuple of (free_axes, fixed_axes) where:
            - `free_axes` is a list of axes in `axes` that are not fixed.
            - `fixed_axes` is the same dictionary (possibly empty), but validated.

        Raises
        ------
        ValueError
            If any fixed axis is not in the coordinate system.
            If any fixed axis is not in the provided axes list.

        Examples
        --------
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> cs = SphericalCoordinateSystem()
        >>> cs.get_free_fixed(axes=["r", "theta", "phi"], fixed_axes={"theta": 0.0})
        (['r', 'phi'], {'theta': 0.0})
        """
        # Default to all axes in the coordinate system
        axes = list(self.resolve_axes(axes))
        fixed_axes = fixed_axes or {}

        # Validate that all fixed axes are in the coordinate system
        unknown_fixed = [ax for ax in fixed_axes if ax not in self.axes]
        if unknown_fixed:
            raise ValueError(
                f"Fixed axes not in coordinate system: {unknown_fixed}. Valid axes: {self.axes}"
            )

        # Validate that all fixed axes are in the provided axes list
        not_in_axes = [ax for ax in fixed_axes if ax not in axes]
        if not_in_axes:
            raise ValueError(
                f"Fixed axes {not_in_axes} are not included in the target axes: {axes}"
            )

        # Compute the list of free axes
        free_axes = [ax for ax in axes if ax not in fixed_axes]

        return free_axes, fixed_axes

    @staticmethod
    def get_axes_permutation(
        src_axes: Sequence[str], dst_axes: Sequence[str]
    ) -> List[int]:
        """
        Compute the permutation needed to reorder `src_axes` into `dst_axes`.

        Parameters
        ----------
        src_axes : list of str
            The current ordering of axes.
        dst_axes : list of str
            The desired target ordering.

        Returns
        -------
        list of int
            Indices describing how to reorder `src_axes` to match `dst_axes`.

        Raises
        ------
        ValueError
            If the two lists are not permutations of each other.

        Examples
        --------
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> cs = SphericalCoordinateSystem()
        >>> cs.get_axes_permutation(["theta", "r"], ["r", "theta"])
        [1, 0]

        If an element is not in one or the other sets, then an error
        occurs.

        >>> cs.get_axes_permutation(["theta", "r", 'phi'], ["r", "theta"]) # doctest: +ELLIPSIS +SKIP
        ValueError: `src_axes` and `dst_axes` must be permutations of each other.
        """
        if set(src_axes) != set(dst_axes):
            raise ValueError(
                "`src_axes` and `dst_axes` must be permutations of each other."
            )

        return [src_axes.index(ax) for ax in dst_axes]

    def get_canonical_axes_permutation(
        self: _SupCoordSystemCore, axes: Sequence[str]
    ) -> List[int]:
        """
        Compute the permutation needed to reorder `axes` into the canonical order defined by the coordinate system.

        Parameters
        ----------
        axes : list of str
            A list of axis names to permute.

        Returns
        -------
        list of int
            Indices describing how to reorder `axes` to match the canonical order (`self.axes`).

        """
        return self.get_axes_permutation(axes, self.axes)

    @staticmethod
    def get_axes_order(src_axes: Sequence[str], dst_axes: Sequence[str]) -> List[int]:
        """
        Compute the reordering indices that will reorder `src_axes` into the order of `dst_axes`.

        This function returns a list of indices that can be used to rearrange `src_axes` so that its
        elements appear in the same order as in `dst_axes`, skipping any elements of `dst_axes` that are
        not present in `src_axes`.

        Parameters
        ----------
        src_axes : list of str
            The current ordering of a subset of axes (e.g., axes labeling a tensor).
        dst_axes : list of str
            The desired target ordering (typically canonical order).

        Returns
        -------
        list of int
            A permutation `P` such that `[src_axes[i] for i in P]` gives the axes in `dst_axes` order.

        Raises
        ------
        ValueError
            If any element of `src_axes` is not found in `dst_axes`.

        Examples
        --------
        >>> get_axes_order(["phi", "r"], ["r", "theta", "phi"])
        [1, 0]  # "r" comes before "phi" in dst_axes

        >>> get_axes_order(["x", "y"], ["y", "z", "x"])
        [1, 0]  # reorder to ["y", "x"]
        """
        src_set = set(src_axes)
        if not src_set.issubset(set(dst_axes)):
            missing = src_set - set(dst_axes)
            raise ValueError(
                f"Some source axes are not present in destination: {missing}"
            )

        return [src_axes.index(ax) for ax in dst_axes if ax in src_axes]

    @staticmethod
    def order_axes(src_axes: Sequence[str], dst_axes: Sequence[str]) -> List[str]:
        """
        Reorder `src_axes` into the order defined by `dst_axes`.

        Parameters
        ----------
        src_axes : list of str
            A subset of axis names to reorder.
        dst_axes : list of str
            The desired axis ordering to match (typically canonical axes).

        Returns
        -------
        list of str
            Reordered version of `src_axes` to match the order in `dst_axes`.

        Raises
        ------
        ValueError
            If any element in `src_axes` is not present in `dst_axes`.
        """
        missing = [ax for ax in src_axes if ax not in dst_axes]
        if missing:
            raise ValueError(
                f"Unknown axis name(s): {missing}. Must be present in destination order: {dst_axes}"
            )

        return [ax for ax in dst_axes if ax in src_axes]

    @staticmethod
    def in_axes_order(
        iterable: Sequence[Any], src_axes: Sequence[str], dst_axes: Sequence[str]
    ) -> List[Any]:
        """
        Reorder a sequence of values from `src_axes` order to `dst_axes` order.

        Parameters
        ----------
        iterable : list
            Items corresponding to axes in `src_axes` order.
        src_axes : list of str
            Axis names corresponding to the order of `iterable`.
        dst_axes : list of str
            Desired axis ordering to match.

        Returns
        -------
        list:
            Reordered iterable in the `dst_axes` order.

        Raises
        ------
        ValueError
            If the lengths don't match or any axes are unknown.
        """
        if len(iterable) != len(src_axes):
            raise ValueError(
                f"Length mismatch: {len(iterable)} items vs {len(src_axes)} axes."
            )

        missing = [ax for ax in src_axes if ax not in dst_axes]
        if missing:
            raise ValueError(
                f"Unknown axis name(s): {missing}. Must be present in destination order: {dst_axes}"
            )

        ordered_axes = [ax for ax in dst_axes if ax in src_axes]
        mapping = dict(zip(src_axes, iterable))
        return [mapping[ax] for ax in ordered_axes]

    @staticmethod
    def get_canonical_axes_order(src_axes: Sequence[str]) -> List[int]:
        """
        Compute the permutation indices to reorder `src_axes` into sorted alphabetical order.

        This function is useful for contexts where canonical order is alphabetical,
        or where symbolic systems (without a defined canonical axis list) use string sorting as a fallback.

        Parameters
        ----------
        src_axes : list of str
            A list of axis names.

        Returns
        -------
        list of int
            A permutation `P` such that `[src_axes[i] for i in P]` gives `sorted(src_axes)`.

        Examples
        --------
        >>> get_canonical_axes_order(["theta", "r", "phi"])
        [2, 1, 0]
        """
        return sorted(range(len(src_axes)), key=lambda i: src_axes[i])

    def order_axes_canonical(
        self: _SupCoordSystemCore, src_axes: Sequence[str]
    ) -> List[str]:
        """
        Reorder a list of axis names into the canonical order of this coordinate system.

        Parameters
        ----------
        src_axes : list of str
            A subset of axis names to reorder.

        Returns
        -------
        list of str
            Reordered list of axis names, matching the order in `self.axes`.

        Raises
        ------
        ValueError
            If any element in `src_axes` is not present in `self.axes`.
        """
        return self.order_axes(src_axes, self.axes)

    def in_canonical_order(
        self: _SupCoordSystemCore, iterable: Sequence[Any], src_axes: Sequence[str]
    ) -> List[Any]:
        """
        Reorder a sequence of values from `src_axes` order to canonical axis order.

        Parameters
        ----------
        iterable : list
            Items corresponding to axes in `src_axes` order.
        src_axes : list of str
            Axis names corresponding to the order of `iterable`.

        Returns
        -------
        list
            Reordered iterable in the canonical axis order (`self.axes`).

        Raises
        ------
        ValueError
            If the lengths don't match or any axes are unknown.
        """
        return self.in_axes_order(iterable, src_axes, self.axes)

    def resolve_axes(
        self: _SupCoordSystemCore,
        axes: Optional[Sequence[str]] = None,
        *,
        require_subset: bool = True,
        require_order: bool = False,
    ) -> List[str]:
        """
        Normalize and validate a user-supplied list of axis names.

        This utility resolves the canonical ordering and performs consistency checks
        such as subset membership, uniqueness, and order compliance.

        Parameters
        ----------
        axes : list of str or None
            The axis names to validate. If None, returns the full list of canonical axes (`self.axes`).
        require_subset : bool, default=True
            If True, all entries in `axes` must be present in `self.axes`.
        require_order : bool, default=False
            If True, `axes` must appear in the same order as they do in `self.axes`.

        Returns
        -------
        list of str
            A concrete list of axis names, validated and normalized.

        Raises
        ------
        ValueError
            If duplicate, unknown, or misordered axes are found.

        Examples
        --------
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> cs = SphericalCoordinateSystem()
        >>> cs.resolve_axes(["phi", "r"])
        ['phi', 'r']

        >>> cs.resolve_axes(["phi", "r"], require_order=True)  # doctest: +SKIP
        ValueError: Axes must appear in canonical order r → theta → phi; received ['phi', 'r']
        """
        # Default: use full canonical axes
        if axes is None:
            return list(self.axes)

        # Normalize to mutable list
        axes = list(axes)

        # Check for duplicates
        if len(set(axes)) != len(axes):
            dup = [ax for ax in axes if axes.count(ax) > 1]
            raise ValueError(f"Duplicate axis/axes in input: {sorted(set(dup))}")

        # Check for unknown axes
        if require_subset:
            unknown = [ax for ax in axes if ax not in self.axes]
            if unknown:
                raise ValueError(
                    f"Unknown axis/axes {unknown!r} – valid axes are {self.axes}"
                )

        # Check order matches canonical
        if require_order:
            canonical_index = [self.axes.index(ax) for ax in axes]
            if canonical_index != sorted(canonical_index):
                raise ValueError(
                    "Axes must appear in canonical order "
                    f"{' → '.join(self.axes)}; received {axes}"
                )

        return axes

    def insert_fixed_axes(
        self: _SupCoordSystemCore,
        iterable: Sequence[Any],
        src_axes: Sequence[str],
        fixed_axes: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Insert fixed axis values into an iterable of values according to canonical axis order.

        This is used to construct a complete value list (e.g., coordinate components) from:
        - a partial set of values aligned with `src_axes`, and
        - a dictionary of fixed scalar values for other axes (`fixed_axes`).

        The result is a new list with one value per coordinate system axis, aligned to `self.axes`.

        Parameters
        ----------
        iterable : list
            Values corresponding to `src_axes`.
        src_axes : list of str
            Axis names corresponding to the entries in `iterable`.
        fixed_axes : dict of {str: Any}, optional
            A dictionary of fixed axis values to insert into the output.

        Returns
        -------
        list
            Values reordered and filled to match `self.axes`.

        Raises
        ------
        ValueError
            If `src_axes` and `fixed_axes` overlap.
            If any axis in `src_axes` or `fixed_axes` is not part of the coordinate system.

        Examples
        --------
        >>> from pymetric.coordinates import SphericalCoordinateSystem
        >>> cs = SphericalCoordinateSystem()
        >>> cs.axes
        ['r', 'theta', 'phi']
        >>> cs.insert_fixed_axes(["R","PHI"], ['r', 'phi'], fixed_axes={'theta': "THETA"})
        ['R', 'THETA', 'PHI']

        This will also reorder entries that are not in canonical order:

        >>> cs.insert_fixed_axes(["PHI","R"], ['phi', 'r'], fixed_axes={'theta': "THETA"})
        ['R', 'THETA', 'PHI']

        """
        fixed_axes = fixed_axes or {}

        # Check for illegal overlaps
        overlap = set(src_axes) & set(fixed_axes)
        if overlap:
            raise ValueError(
                f"`src_axes` and `fixed_axes` must not overlap: {sorted(overlap)}"
            )

        # Check for unknown axes
        unknown_src = [ax for ax in src_axes if ax not in self.axes]
        unknown_fixed = [ax for ax in fixed_axes if ax not in self.axes]
        if unknown_src or unknown_fixed:
            raise ValueError(
                f"Unknown axes: {unknown_src + unknown_fixed}. Must be a subset of: {self.axes}"
            )

        # Build the mapping
        mapping = dict(zip(src_axes, iterable))
        mapping.update(fixed_axes)

        # Fill values in canonical order
        return [mapping[ax] for ax in self.axes if ax in mapping]

    # -------------------------------- #
    # Latex                            #
    # -------------------------------- #
    # This connects axes to latex.
    def get_axes_latex(
        self: _SupCoordSystemCore, axes: Union[str, Sequence[str]]
    ) -> Union[str, List[str]]:
        """
        Return the LaTeX representation(s) of one or more axis names.

        Parameters
        ----------
        axes : str or Sequence[str]
            A single axis name or a list/tuple of axis names.

        Returns
        -------
        str or list of str
            The LaTeX representation(s) of the provided axis/axes. Returns a single string
            if a scalar input is given, and a list of strings if a sequence is provided.

        Notes
        -----
        - If ``__AXES_LATEX__`` is not defined for the coordinate system, this falls back
          to wrapping each axis in ``$...$``.
        - Axis names must be valid entries in ``__AXES__``.
        """
        if isinstance(axes, str):
            axes_list = [axes]
            is_scalar = True
        else:
            axes_list = list(axes)
            is_scalar = False

        if self.__AXES_LATEX__ is None:
            latex_list = [f"${ax}$" for ax in axes_list]
        else:
            try:
                latex_list = [self.__AXES_LATEX__[ax] for ax in axes_list]
            except KeyError as e:
                raise ValueError(
                    f"Axis {e.args[0]} does not have a defined LaTeX representation."
                ) from e

        return latex_list[0] if is_scalar else latex_list

    # -------------------------------- #
    # Units                            #
    # -------------------------------- #
    def get_axes_units(
        self: _SupCoordSystemCore, unit_system: "unyt.UnitSystem"
    ) -> List["unyt.Unit"]:
        """
        Resolve the physical units for each axis in a given unit system.

        Parameters
        ----------
        unit_system : unyt.UnitSystem
            The unit system used to resolve the symbolic axis dimensions.

        Returns
        -------
        list of unyt.Unit
            The resolved unit for each axis, in canonical order.
        """
        return [unit_system[dim] for dim in self.axes_dimensions]
