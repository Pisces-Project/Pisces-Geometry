.. _components:

========================
Fields: Field Components
========================

The :class:`~fields.components.FieldComponent` class is the
primary data container in the PyMetric fields. It serves as a structured
wrapper for discretized data over a coordinate-aware grid.
It combines geometric information with backend-specific storage,
making it easy to define, manipulate, and broadcast physical fields
in a scientifically meaningful way.

Field components are designed to integrate tightly with grid geometry,
storage backends, and broadcasting logic, enabling clean and flexible construction
of scalar, vector, and tensor fields.

.. note::

    In general, :class:`~fields.components.FieldComponent`'s are intended to be wrapped in
    a true field class (i.e. :class:`~fields.base.DenseField`) when being used for most
    purposes.

.. contents::
   :local:
   :depth: 2

Overview
--------

A :class:`~fields.components.FieldComponent` ties together three core ideas:

- A structured **grid** (:attr:`~fields.components.FieldComponent.grid`) defines the spatial geometry — the physical layout of the domain.
  It is typically constructed first and contains coordinate mappings, shape information, and axis labels.
- A **buffer** (:attr:`~fields.components.FieldComponent.buffer`) holds the numerical data associated with the field. This can be a raw NumPy array,
  a unit-tagged array, or a disk-backed HDF5 dataset.
- A set of **spatial axes** (:attr:`~fields.components.FieldComponent.axes`) specifies which leading dimensions of the buffer are aligned with
  the spatial structure of the grid. These spatial axes ensure compatibility with broadcasting and differential
  operations.

This design separates the grid structure from the field’s numerical structure, allowing flexible
representation of scalar fields, vector fields, and high-rank tensors over the same domain.


A :class:`~fields.components.FieldComponent` exposes a variety of different attributes (see the API doc for a full list); however,
the most frequently useful are those pertaining to the shape, size, and dimensions of the :class:`~fields.components.FieldComponent` in
reference to its behavior as an array. There are actually three different sizes, dimensions, etc. that you can access:

.. tab-set::
    .. tab-item:: Spatial (Grid)

        Spatial attributes describe the layout of the field over the **coordinate-aligned grid axes**.
        These are the dimensions of the domain over which the field is
        defined — one per axis in :attr:`~fields.components.FieldComponent.axes`.

        These attributes are especially useful for looping over the grid or determining resolution.

        **Attributes:**

        - :attr:`~fields.components.FieldComponent.spatial_shape` — The shape of the buffer over the grid axes (e.g., ``(nx, ny, nz)``).
        - :attr:`~fields.components.FieldComponent.spatial_ndim` — Number of grid-aligned axes.
        - :attr:`~fields.components.FieldComponent.spatial_size` — Total number of spatial points, i.e. ``np.prod(A.spatial_shape)``.

    .. tab-item:: Element (Field)

        Element attributes describe the internal structure of the field *at each grid point*. This is what
        distinguishes a scalar from a vector or a tensor — not the domain, but what is stored per cell.

        These attributes are important for understanding the "rank" of the field and working with
        vector or tensor operations.

        **Attributes:**

        - :attr:`~fields.components.FieldComponent.element_shape` — The shape of the trailing (non-spatial) dimensions.
        - :attr:`~fields.components.FieldComponent.element_ndim` — The number of element dimensions (0 for scalar fields).
        - :attr:`~fields.components.FieldComponent.element_size` — Total number of values per spatial location.

        A field is considered **scalar** if :attr:`~fields.components.FieldComponent.element_ndim` is zero
        and :attr:`~fields.components.FieldComponent.is_scalar` returns `True`.

    .. tab-item:: Full

        The full shape of the component includes both spatial and element-wise dimensions.
        These correspond directly to the shape of the underlying array (buffer), making them
        useful for reshaping, indexing, or NumPy operations.

        **Attributes:**

        - :attr:`~fields.components.FieldComponent.shape` — Total shape of the buffer: ``.spatial_shape + .element_shape``.
        - :attr:`~fields.components.FieldComponent.ndim` — Total number of dimensions.
        - :attr:`~fields.components.FieldComponent.size` — Total number of elements (equal to ``np.prod(A.shape)``).
        - :attr:`~fields.components.FieldComponent.dtype` — Data type of the buffer elements.
        - :attr:`~fields.components.FieldComponent.units` — Physical units associated with the buffer, if present.

        These attributes are compatible with typical NumPy-like semantics,
        including slicing and broadcasting.


Creating Components
---------------------

There are a number of ways to generate components depending on the desired behavior
and the backend storage format. The most generic approach is to simply use
the :class:`~fields.components.FieldComponent` constructor:

.. code-block:: python

    from pymetric.fields.components import FieldComponent

    component = FieldComponent(grid, np.random.randn(32, 32), axes=["x", "y"])

This approach will check that the provided array matches the expected shape of
the grid in its leading dimensions and then convert the provided array to a valid
:py:class:`~fields.buffers.base.BufferBase` subclass using **buffer resolution**.

.. hint::

    If the :py:class:`~fields.buffers.base.BufferBase` is new to you, you should
    read :ref:`buffers`. These are the backend of all field components because they
    are themselves backend agnostic allowing users to work with lazy-loaded arrays
    or other (more complex / specialized) array backends.

An equivalent approach is the use the :meth:`~fields.components.FieldComponent.from_array` constructor, which likewise
allows the user to provide a generic array like object and let PyMetric deal with wrapping
it in a suitable buffer class. :meth:`~fields.components.FieldComponent.from_array` does allow finer control over the
buffer resolution and also allows for additional properties to be set.

For example,

.. code-block:: python

    FieldComponent.from_array(array_like=my_data,
                              grid=grid,
                              axes=["r", "theta"],
                              buffer_class="unyt",
                              units="keV")

This method validates that the buffer shape is consistent with the grid and axes,
and allows explicit backend selection.

Convenience Constructors
^^^^^^^^^^^^^^^^^^^^^^^^

Like `NumPy <https://numpy.org/doc/stable/index.html>`_, PyMetric supports generating
a number of simple components with direct functions:

- :meth:`~fields.components.FieldComponent.zeros`
- :meth:`~fields.components.FieldComponent.ones`
- :meth:`~fields.components.FieldComponent.full`
- :meth:`~fields.components.FieldComponent.empty`

These methods each require an existing grid to be provided, but will perform
the logic regarding the spatial shape for you:

.. code-block:: python

    from pymetric import FieldComponent, GenericGrid

    # Create a grid representing your physical
    # system.
    grid = ...

    # Create a vector field component.
    comp = FieldComponent.zeros(grid, ['x','y'], element_shape=(3,))

In many cases, this is the nicest way to build a component. It saves you from having
to go to all the effort of figuring out what the grid expects your array to look like.

Advanced Creation:
^^^^^^^^^^^^^^^^^^
In addition to the core construction methods presented above,
a few additional methods are available to construct field
components from more esoteric origins. The most significant of these is the
:meth:`~fields.components.FieldComponent.from_function` which allows
users to create fields by specifying directly a function :math:`f(x^1,x^2,\ldots,x^n)`.
The following example illustrates the basic usage:

.. plot::
    :include-source:

    import numpy as np
    from pymetric import FieldComponent, CartesianCoordinateSystem2D, GenericGrid
    import matplotlib.pyplot as plt

    # Create the coordinate system and the grid.
    cs = CartesianCoordinateSystem2D()
    x, y = (np.linspace(0,1,100),
            np.linspace(0,1,100))
    g = GenericGrid(cs, [x, y])

    # Define a function of the coords.
    func = lambda _x,_y: np.sin(10*np.sqrt(_x**2+_y**2))

    # Create the dense field from the function.
    f = FieldComponent.from_function(func, g, ['x','y'])

    fig,axes = plt.subplots(1,1)
    Q = axes.imshow(f[...].T,extent=(0,1,0,1))
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    plt.colorbar(Q,ax=axes)
    plt.show()

Operations on Field Components
------------------------------

Like fields (see :ref:`fields`) and buffers (see :ref:`buffers`),
:class:`~fields.components.FieldComponent` are designed to be easy to work with in
a manner which reflects the intuitive NumPy-like style that most python users
are familiar with. In this section, we'll go over some of the more important
elements of PyMetric's behavior when using :class:`~fields.components.FieldComponent`.

Broadcasting and Axes
^^^^^^^^^^^^^^^^^^^^^

One of the most powerful features of PyMetric’s :class:`~fields.components.FieldComponent` class is its support
for axis-aware broadcasting. Components may be defined over a subset of a grid’s axes,
but many operations (e.g., arithmetic, tensor contractions, derivatives)
require consistent alignment of all participating arrays.
PyMetric handles this using semantic broadcasting utilities
that operate at the level of coordinate axes.

These utilities allow for **non-destructive reshaping**, **dimensional promotion**, and **alignment-aware arithmetic**,
all while respecting grid geometry and physical units.

Key concepts:

- **Axis-aware** broadcasting is performed by matching named coordinate axes, not just array positions.
- Components can be **expanded** (e.g., from scalar to tensor) or **reduced** (e.g., slicing at fixed index) with axis-preserving semantics.
- All operations use **semantic broadcasting**, ensuring safe and interpretable manipulation of tensor components.

.. hint::

   Unlike :func:`numpy.broadcast_to`, PyMetric's broadcasting retains geometric meaning — you broadcast *over axes*, not just dimensions.

.. rubric:: Broadcasting Methods

PyMetric provides several methods to reshape, align, or slice field components in a geometry- and unit-aware way:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description

   * - :meth:`~fields.components.FieldComponent.broadcast_to_array_in_axes`
     - Return a NumPy array broadcasted to a new set of axes.

   * - :meth:`~fields.components.FieldComponent.broadcast_to_unyt_array_in_axes`
     - Return a unit-aware array broadcasted to a new set of axes.

   * - :meth:`~fields.components.FieldComponent.broadcast_to_buffer_core_in_axes`
     - Return the core backend array (e.g., NumPy, HDF5) broadcasted to new axes.

   * - :meth:`~fields.components.FieldComponent.broadcast_to_buffer_repr_in_axes`
     - Return the NumPy-compatible view of the buffer broadcasted to new axes.

   * - :meth:`~fields.components.FieldComponent.broadcast_buffer_to_axes`
     - Return a new buffer object aligned to a different set of axes.

   * - :meth:`~fields.components.FieldComponent.expand_axes`
     - Materialize all specified axes as physical dimensions by tiling and copying. Useful when buffer views are insufficient.

   * - :meth:`~fields.components.FieldComponent.reduce_axes`
     - Reduce the dimensionality of the field by slicing specific axes at fixed indices.

   * - :meth:`~fields.components.FieldComponent.reshape_element`
     - Reshape the element-wise portion (i.e., tensor structure) of the field to a new shape.

   * - :meth:`~fields.components.FieldComponent.reshape_element_like`
     - Reshape the element dimensions to match another field component’s element shape.

For example, a field component of axes ``['x','y']`` can be cast to one over ``['x','y','z']`` using the
following syntax:

.. code-block:: python

    from pymetric import FieldComponent

    # Start with a scalar field defined over (x, y)
    f = FieldComponent.zeros(grid, axes=['x', 'y'])

    # Expand it to align with (x, y, z) using singleton expansion
    g = f.broadcast_to_array_in_axes(['x', 'y', 'z'])  # returns np.ndarray

    # Materialize the result as a true tensor over (x, y, z)
    g_full = f.expand_axes(['x', 'y', 'z'])  # returns new FieldComponent

The first broadcast operation returns a NumPy array view over the additional axis (``z``),
while the second call to :meth:`~fields.components.FieldComponent.expand_axes`
returns a fully realized :class:`~fields.components.FieldComponent` where data
has been tiled to physically exist on all specified axes.

It is also possible to reduce the number of axes present in a component. However,
this comes at the expense of data generality — reducing a field requires slicing
into its buffer, assuming that its value at a particular coordinate is representative
for all purposes.

.. code-block:: python

    # Fix 'z' index at z = 5 to get a 2D slice
    f_xy = f.reduce_axes(['z'], [5])

    assert f_xy.axes == ['x', 'y']

Here, the field ``f`` originally lives on axes ``['x', 'y', 'z']``, and we reduce it
to ``['x', 'y']`` by selecting a single index along ``z``. This effectively slices
a 2D surface from a 3D volume. The resulting field remains aligned with the grid geometry
but is now one dimension lower in structure.

.. warning::

    Unlike broadcasting, reduction is not always lossless. Use it with care when slicing fields
    that may contain spatial variation along reduced axes.

Extracting Data
^^^^^^^^^^^^^^^^^^^^^^^^^

Field components provide multiple methods for accessing the raw data stored in the buffer.
These methods return views or copies of the internal data in forms that are compatible
with NumPy and related tools:

- :meth:`~fields.components.FieldComponent.as_array()` — Returns the data as a plain `numpy.ndarray` without units.
- :meth:`~fields.components.FieldComponent.as_unyt_array()` — Returns a `unyt_array` with attached physical units.
- :meth:`~fields.components.FieldComponent.as_buffer_core()` — Returns the core backend buffer (e.g., raw NumPy or HDF5 dataset).
- :meth:`~fields.components.FieldComponent.as_buffer_repr()` — Returns the preferred array-like representation for external use, respecting units or formatting.

These methods allow seamless integration with common scientific Python tools such as Matplotlib, SciPy, or custom numerical routines.

.. note::

    If you're working with units, always prefer :meth:`~fields.components.FieldComponent.as_unyt_array`
    to ensure correct dimensional analysis and physical semantics.

Slicing into Components
~~~~~~~~~~~~~~~~~~~~~~~~

Field components support NumPy-like indexing and slicing behavior. Slicing into a component
returns the underlying buffer content with grid-aligned semantics preserved. Axes follow
canonical ordering, so slicing always proceeds in spatial axis order, followed by element axes.

.. code-block:: python

    # Create a 3D scalar field component
    f = FieldComponent.zeros(grid, axes=['x', 'y', 'z'])

    # Extract a 2D xy-slice at z=5
    slice_xy = f[:, :, 5]

    # Extract the raw array directly
    data = f.as_array()
    value = data[10, 20, 5]

    # Slicing with units preserved
    value_with_units = f.as_unyt_array()[10, 20, 5]

This behavior allows for precise control over data access, while still leveraging the underlying
coordinate-aware infrastructure. If you need to reduce a field's dimensionality with updated metadata,
use :meth:`~fields.components.FieldComponent.reduce_axes` instead.

.. warning::

    Direct slicing returns NumPy or :class:`~unyt.array.unyt_array` arrays, **not** new :class:`~fields.components.FieldComponent` instances.
    Use :meth:`~fields.components.FieldComponent.reduce_axes` if you want to retain coordinate-aware semantics after dimensional reduction.

Unit Management
----------------

Field components support both in-place and non-destructive unit conversion via a
clean wrapper over their underlying buffers. These conversions are powered by `unyt`
and ensure physical correctness in unit-aware computations.

There are two primary ways to manipulate units:

- **In-place conversion** using :meth:`~fields.components.FieldComponent.convert_to_units`
  or :meth:`~fields.components.FieldComponent.convert_to_base`, which updates the buffer directly.
- **Casting to new units** using :meth:`~fields.components.FieldComponent.in_units`,
  :meth:`~fields.components.FieldComponent.to`, or :meth:`~fields.components.FieldComponent.to_value`,
  which produce a new component or raw array in the target units.

.. code-block:: python

    # In-place unit conversion
    field.convert_to_units("erg / cm**3")

    # Return a copy in new units
    field_new = field.to("keV")

    # Return a NumPy array with units stripped
    values = field.to_value("J")

These methods mirror the buffer-level API but operate at the field component level.
For detailed control or custom unit logic, refer to the :ref:`buffers` documentation.

.. note::

    Fields without attached units (i.e., purely numeric buffers) will accept
    unit assignment during in-place conversion without modifying data.

Subclassing FieldComponent
---------------------------

FieldComponent is designed to be subclassed for domain-specific needs. You may add:

- Custom postprocessing behavior
- Additional validation logic
- Domain-specific coordinate or field metadata
- Redefinitions of NumPy dispatch behavior via `__array_function_dispatch__`

See also
--------

- :mod:`~pymetric.fields.buffers` — Backends used for storage
- :mod:`~pymetric.grids` — Grid and coordinate system classes
- :class:`~fields.components.FieldComponent` — Main class reference
- :mod:`~pymetric.fields.mixins.components` — Core behavior mixin
- :mod:`~pymetric.utilities.arrays` — Tools for broadcasting and ufunc alignment
