.. _buffers:

====================
Fields: Data Buffers
====================

PyMetric is a scientific computing library focused on performing differential geometry operations on structured grids.
To support flexible and scalable numerical workflows, PyMetric employs a unified buffer abstraction that decouples
mathematical logic from the details of in-memory or on-disk data storage.

This buffer system enables seamless integration with multiple backend array storage solutions, including:

- `NumPy <https://numpy.org/doc/stable/index.html>`__ for standard in-memory arrays,
- HDF5 datasets via `h5py <https://docs.h5py.org/>`__ for efficient on-disk access and large-scale data persistence.

Rather than directly operating on raw array types, PyMetric introduces a lightweight buffer API layer,
into which different array libraries can be plugged. This API abstracts away the specific behavior of each backend
while maintaining NumPy compatibility and enabling advanced features such as lazy loading, unit tracking, and
backend-specific transforms.

All numerical field data in PyMetric—such as scalars, vectors, and tensors defined over coordinate grids—
is backed by a concrete implementation of a :class:`~fields.buffers.base.BufferBase` subclass, such as
:class:`~fields.buffers.core.ArrayBuffer`, or :class:`~fields.buffers.core.HDF5Buffer`. These buffers provide
a uniform interface for:

- Element-wise and indexed access,
- Broadcasting and assignment,
- Participation in NumPy operations and universal functions (ufuncs),
- Serialization, transformation, and type-aware metadata inspection.

High-level interfaces such as :class:`~fields.base.DenseField` automatically wrap user-provided arrays into
conforming buffer types when necessary, ensuring a consistent and extensible interface throughout the PyMetric library.

This document provides a comprehensive overview of the buffer architecture, including its design philosophy,
core protocol, indexing semantics, transformation logic, NumPy integration, and subclassing guidelines.
It is intended for developers implementing custom buffers, as well as advanced users who need fine-grained control
over memory layout, or file I/O.

.. contents::
   :local:
   :depth: 2

Overview
--------

Buffers are the low-level storage abstraction in the :mod:`~fields` module. They are responsible for:

- Managing actual data (values and memory layout)
- Interfacing with array backends (e.g., NumPy, HDF5)
- Supporting broadcasting, NumPy semantics, and I/O
- Providing a clean interface for the :class:`~fields.components.FieldComponent` system

Each buffer class inherits from :class:`~fields.buffers.base.BufferBase`, which provides a uniform API across backends.

Creating Buffers of Different Types
-----------------------------------

There are currently **two core buffer types**:

- :class:`~fields.buffers.core.ArrayBuffer` — an in-memory buffer for plain :class:`~numpy.ndarray`.
- :class:`~fields.buffers.core.HDF5Buffer` — a persistent, disk-backed buffer based on :class:`h5py.Dataset`.

Buffers can be created in multiple ways, depending on the format of your input data and your desired backend. The most
generic of these approaches is the :meth:`~fields.buffers.base.BufferBase.from_array`. This tries to coerce an input object
(:class:`list`, :class:`~numpy.ndarray`, etc.) into a compatible buffer type:

.. code-block:: python

    from pymetric.fields.buffers import ArrayBuffer

    data = [[1, 2], [3, 4]]
    buf = ArrayBuffer.from_array(data,dtype='f8')

This approach has the distinct advantage of clarifying the buffer that will be returned at the expense
of requiring that the user knows that their data is compatible with the particular buffer class.

.. hint::

    In addition to :meth:`~fields.buffers.base.BufferBase.from_array`, there are
    also :meth:`~fields.buffers.base.BufferBase.zeros`, :meth:`~fields.buffers.base.BufferBase.ones`,
    :meth:`~fields.buffers.base.BufferBase.full`,
    and :meth:`~fields.buffers.base.BufferBase.empty` attached to each buffer class.

Resolving Buffer Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, it is useful to let PyMetric decide which buffer class you need based on the type of the
object that needs to be wrapped. This procedure is called **buffer resolution**. At its core, resolution is
a simple procedure; each :class:`~fields.buffers.base.BufferBase` has three critical attributes:

1. The **resolvable classes**: the classes that buffer class can faithfully wrap around.
2. The **core class**: the *single* class that gets wrapped around.
3. The **resolution priority**: dictates at what priority a given buffer class is.

When PyMetric is asked to resolve the correct buffer for a given object, it will seek the *highest* priority class
which can *faithfully* encapsulate the class of the object being resolved. That object is then cast to the **core class**
and wrapped by the buffer class. There are a number of ways to enter the buffer resolution pipeline:

1. Using the :func:`~fields.buffers.base.buffer_from_array` function.
2. Finally, PyMetric provides a number of utility functions in :mod:`~fields.buffers.utilities` like :func:`~fields.buffers.utilities.buffer_zeros`
   or :func:`~fields.buffers.utilities.buffer` which all enter the resolution process.

.. note::

    An initiated reader might ask, "how does PyMetric know what buffers are available?" In fact, this question is a critical
    one if you are extending PyMetric with custom buffer classes. The answer is the use of **buffer registries**. Each entry point
    to the buffer resolution process typically takes two kwargs:

    - ``buffer_class=`` can be used to explicitly set the buffer class to use.
    - ``buffer_registry=`` can tell PyMetric to search through a custom :class:`~fields.buffers.registry.BufferRegistry` class
      for the buffer.

    Custom buffer registries can be used to override the default (``__DEFAULT_BUFFER_REGISTRY__``); into which all new subclasses
    are placed when then are first read by the interpreter.

Operations on Buffers
-------------------------

At their core, buffers behave like "fancy" NumPy arrays. They can be indexed, broadcast,
operated on using NumPy functions, and manipulated using standard array-like semantics.
This allows PyMetric users to interact with buffers in a highly intuitive and flexible way
while preserving backend-specific advantages like disk persistence.

Operationally, computations on buffers act as if they were performed on an equivalent NumPy array.

Indexing and Slicing Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indexing and slicing are pretty simple:
the buffer will forward the indexing operation to the **core class** and then coerce the result into a numpy array.

.. code-block:: python

    buf = ArrayBuffer.full((4, 4), fill_value=10)
    sub = buf[1:3, 1:3]

    assert isinstance(sub, np.ndarray)
    assert sub.shape == (2, 2)


The Buffer API Protocol
-----------------------

This section defines the formal contract that all PyMetric buffer implementations must follow.

The PyMetric buffer protocol is designed to be both rigorous and extensible, enabling consistent behavior across
a variety of numerical backends—including in-memory arrays, unit-aware buffers, and disk-backed datasets—while
retaining full compatibility with NumPy operations and user expectations.

Any custom buffer class used within the PyMetric ecosystem must subclass from :class:`~fields.buffers.base.BufferBase` and implement
the methods, properties, and behaviors described below. These requirements ensure that buffers can participate
uniformly in field operations, differential geometry routines, serialization, and broadcasting logic.

This section serves as the authoritative reference for the PyMetric buffer API. It specifies:

- Required indexing behaviors and access patterns,
- Attribute interface compatibility with NumPy (`shape`, `dtype`, etc.),
- Participation in NumPy ufuncs and `__array_function__` overrides,
- Semantics of transformation methods like `reshape`, `transpose`, etc.,
- How and when buffers should materialize into NumPy arrays,
- Expected fallback behaviors and error signaling.

While PyMetric’s high-level APIs provide convenient wrappers and automatic coercion of raw arrays into buffer instances,
this section is intended for buffer implementers and advanced users who need fine-grained control over backend behavior
and interface guarantees.

Following these guidelines ensures that all buffers can interoperate smoothly across the PyMetric field system, regardless of
their storage format or implementation details.

Buffer Resolution and Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process of constructing buffers in PyMetric is designed to be both user-friendly and extensible across a variety
of numerical backends. This section describes how buffer instances are created from array-like data, and how PyMetric
resolves the appropriate backend when no explicit buffer type is provided.

Buffer construction in PyMetric follows one of two main paths:

1. **Direct instantiation** via a known buffer class (e.g., `ArrayBuffer.from_array(...)`),
2. **Dynamic resolution** via a global or user-defined buffer registry.

All concrete subclasses of :class:`~fields.buffers.base.BufferBase` must implement a class method:

.. code-block:: python

    @classmethod
    def from_array(cls, obj, *args, **kwargs) -> BufferBase:
        ...

This method should take an arbitrary array-like object (list, NumPy array, etc.), coerce it into a
backend-compatible form, and return a valid buffer instance. This is the preferred entry point for constructing
buffers from untrusted or heterogeneous input data.

For convenience, buffer classes should internally distinguish between already-correct types and types that need
conversion, so that users can pass in native arrays directly without needing to wrap or preprocess them.

Example usage:

.. code-block:: python

    buffer = ArrayBuffer.from_array(np.zeros((10, 10)))
    buffer = ArrayBuffer.from_array([[1.0, 2.0]])

Registry-Based Resolution
^^^^^^^^^^^^^^^^^^^^^^^^^

When the buffer class is not explicitly specified, PyMetric uses a dynamic resolution system to automatically determine
the most appropriate backend. This is handled by the :class:`~fields.buffers.registry.BufferRegistry` class, which
maintains a list of registered buffer classes and their resolution preferences.

The resolution process works as follows:

1. Iterate over all registered buffer classes.
2. For each class, check whether the input object matches a supported backend type, as defined by:

   - ``__buffer_resolution_compatible_classes__``: A tuple of supported array-like types (e.g., `(np.ndarray,)`)
   - ``__buffer_resolution_priority__``: An integer indicating resolution precedence (lower = higher priority)

3. Select the first compatible class with the highest priority.
4. Call the class's `from_array()` method to construct the buffer.

This logic allows backends to be registered and prioritized without tightly coupling them to field logic.

Example:

.. code-block:: python

    from fields.buffers.registry import resolve_buffer

    buffer = resolve_buffer(data)  # Automatically dispatches to the right backend

    # Optionally provide a custom registry
    buffer = resolve_buffer(data, registry=my_registry)

Manual and automatic resolution are both valid depending on context. When in doubt, use `from_array()` explicitly for
clarity and control.

Key Resolution Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following class-level attributes control registry-based dispatch:

- ``__buffer_resolution_compatible_classes__``: Tuple of types that the buffer can wrap. Required for resolution.
- ``__buffer_resolution_priority__``: Integer priority used to resolve conflicts when multiple backends can accept the same object.

Concrete buffer classes must set both attributes to participate in registry-based resolution.

Buffer Access Patterns
++++++++++++++++++++++

All PyMetric buffer classes must implement NumPy-compatible data access using the
``__getitem__`` and ``__setitem__`` methods. These methods provide direct read and
write access to the buffer's underlying array and are expected to follow the standard
semantics of `NumPy indexing <https://numpy.org/devdocs/user/basics.indexing.html>`_.

The following indexing behaviors must be supported:

- **Basic indexing** with integers, slices, and ellipsis (e.g., ``buffer[0]``, ``buffer[1:5]``, ``buffer[..., -1]``)
- **Boolean masking** (e.g., ``buffer[mask]`` where ``mask`` is a boolean array)
- **Integer array indexing** (e.g., ``buffer[[0, 2, 4]]``)
- **Multidimensional indexing** (e.g., ``buffer[:, [1, 3]]``)

Assignments via ``__setitem__`` must correctly apply NumPy broadcasting semantics. For example:

.. code-block:: python

    buffer[1:4, :] = 0.0               # Scalar broadcast
    buffer[:, 0] = np.arange(N)       # 1D vector assignment
    buffer[rows, cols] = values       # Indexed writes with broadcasting

Return values from ``__getitem__`` must be NumPy-compatible arrays (typically ``numpy.ndarray``),
or the backend-native array type used in the buffer's representation. Returned slices are not
required to be buffer instances.

.. note::

    While the formal PyMetric API only mandates compatibility with the slightly restricted
    indexing semantics described in `h5py fancy indexing <https://docs.h5py.org/en/stable/high/dataset.html#dataset-fancy>`_,
    all buffer implementations are **strongly encouraged** to support the full range of NumPy fancy
    indexing whenever feasible. This ensures consistency across backends and improves user ergonomics.

Implementations that do not fully support NumPy indexing must document their limitations clearly,
and should raise appropriate exceptions (e.g., ``IndexError``, ``ValueError``) when unsupported
access patterns are attempted.

Buffer Parameters
+++++++++++++++++

All buffer instances in PyMetric must expose the standard structural attributes expected of NumPy-compatible array-like objects.
These attributes enable introspection, broadcasting, and downstream operations without requiring knowledge of the
underlying backend implementation.

Each subclass of :class:`~fields.buffers.base.BufferBase` must implement the following read-only properties:

- :attr:`~fields.buffers.base.BufferBase.shape` (:class:`tuple` of :class:`int`)
  The shape of the buffer, describing the dimensions of the wrapped array. This must match
  ``buffer.as_core().shape`` and is used throughout the system for indexing, broadcasting, reshaping,
  and compatibility with grid structures.

- :attr:`~fields.buffers.base.BufferBase.ndim` (:class:`int`)
  The number of dimensions (i.e., the rank) of the buffer. Equivalent to ``len(buffer.shape)``.

- :attr:`~fields.buffers.base.BufferBase.size` (:class:`int`)
  The total number of elements in the buffer. Computed as the product of all elements in ``shape``.

- :attr:`~fields.buffers.base.BufferBase.dtype` (:class:`numpy.dtype` or equivalent)
  The data type of elements contained in the buffer. Controls precision, casting behavior,
  and memory layout.

- :attr:`~fields.buffers.base.BufferBase.c` (Any)
  The underlying *core* array object used by the buffer backend. This is equivalent to
  :meth:`~fields.buffers.base.BufferBase.as_core` and provides direct access to the backend-native
  representation (e.g., :class:`numpy.ndarray`, or :class:`h5py.Dataset`).

These attributes must behave consistently with their NumPy equivalents, even if the internal data
structure is backed by a more complex or disk-based format.

Backends that wrap proxy arrays—such as unit-tagged buffers or memory-mapped files—must ensure that
these properties return logical dimensions and types, rather than physical storage details.

These parameters are used throughout PyMetric’s field and geometry infrastructure to:
- Validate layout compatibility,
- Infer broadcasting rules,
- Apply transformations,
- Normalize buffer shapes during differential operations.

**Correct and consistent implementation is critical** for PyMetric’s array interoperability model to function as intended.

Buffer Operations
+++++++++++++++++

Buffer instances in PyMetric are designed to behave as NumPy-compatible arrays while maintaining the flexibility
of backend-specific semantics. This allows PyMetric fields and numerical operations to work seamlessly across
in-memory and on-disk representations.

Default Semantics
^^^^^^^^^^^^^^^^^

By default, **all arithmetic and functional operations** (e.g., ``+``, ``*``, ``np.sin``, ``np.mean``) operate by:

1. **Materializing** the buffer into a NumPy array via ``as_array()``,
2. **Performing the operation** on the unwrapped array using standard NumPy logic,
3. **Returning the result** either as:
   - a plain NumPy array (default behavior),
   - or a new buffer of the same type if explicitly requested.

This ensures that even non-NumPy-compatible backends, such as HDF5-backed datasets, can fully participate in
mathematical operations without requiring direct NumPy interoperability.

Numpy Universal Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

PyMetric buffers are fully compatible with `NumPy universal functions (ufuncs) <https://numpy.org/doc/stable/reference/ufuncs.html>`__,
which include element-wise arithmetic, comparisons, trigonometric functions, and more.

All buffer classes inherit support for ufuncs via the base implementation of ``__array_ufunc__`` provided by
:class:`~fields.buffers.base.BufferBase`. This mechanism enables seamless interaction with functions like
``np.add``, ``np.sin``, ``np.abs``, etc., without requiring manual casting or backend-specific logic.

By default, PyMetric buffers:

1. **Materialize their data** using ``as_array()`` (e.g., NumPy array or  HDF5 dataset),
2. **Delegate** the operation to the NumPy ufunc,
3. **Return the result** as a plain NumPy array.

This behavior guarantees correctness even for complex or partially lazy backends, such as those wrapping
HDF5 datasets or proxy arrays.

The only deviation from default NumPy semantics is the support for the ``out=`` keyword. If specified,
``__array_ufunc__`` will:

- Unwrap the output buffer using the ``__array_object__``,
- Place the result directly into the provided output buffer,
- Return the output buffer instance instead of a new NumPy array.

This allows users to **retain the buffer type** and avoid intermediate materialization, particularly in
memory- or I/O-constrained workflows.

.. code-block:: python

    buf = ArrayBuffer.from_array(np.ones((4,)))
    np.multiply(buf, 3.0)                 # returns NumPy array
    np.multiply(buf, 3.0, out=buf)        # updates in place, returns `buf`


Numpy Function Forwarding
^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond universal functions (ufuncs), NumPy defines a large collection of high-level operations—
such as reductions, reshaping, sorting, and linear algebra—via the `array function protocol <https://numpy.org/doc/stable/user/basics.dispatch.html>`__.
These are dispatched through the special method ``__array_function__``.

PyMetric buffers implement this protocol in a way that preserves full interoperability with NumPy while allowing
selective overrides where needed.

By default, all NumPy function calls (e.g., ``np.sum``, ``np.transpose``, ``np.mean``) are **forwarded** to the
buffer’s internal array representation via ``as_core()``. This ensures that:

- The exact backend semantics (e.g. lazy behavior) are preserved,
- Buffers behave like native arrays from the user's perspective.

Unless explicitly overridden, function calls behave as if the buffer were a plain NumPy array.

To enable advanced functionality or optimize backend-specific behavior, buffer classes can override individual
function implementations by populating the ``__array_function_dispatch__`` dictionary. This maps
NumPy functions to custom handlers that receive unwrapped arguments and can return either:

- A plain array (e.g., for scalar reductions),
- A new buffer instance (e.g., for shape-preserving operations).

This allows for precise control over which functions are forwarded and how results are returned.

**Example override (in a buffer subclass):**

.. code-block:: python

    __array_function_dispatch__ = {
        np.sum: custom_sum_handler,
        np.moveaxis: lambda arr, src, dst: arr.transpose(...)),
    }

The ``__array_function__`` implementation behaves as follows:

1. If **all input types are subclasses of** the same buffer class:
   - It checks for an override in ``__array_function_dispatch__``.
   - If found, it calls the custom implementation.

2. If no override is found, or if inputs are mixed types:
   - It unwraps all buffer arguments using ``as_core()``,
   - Delegates to the native NumPy function,
   - Returns the result as a plain array.

This approach provides both flexibility and predictability, and allows for full participation in
NumPy's high-level API without compromising backend fidelity.

Transformation Methods
^^^^^^^^^^^^^^^^^^^^^^

In addition to indexing and universal functions, PyMetric buffers support standard array transformation operations
such as reshaping, flattening, and transposition. These methods are critical for preparing data for numerical operations
and field construction.

All transformation methods implemented in buffer classes (e.g., ``reshape()``, ``flatten()``, ``transpose()``) adhere
to a consistent dispatching pattern that allows users to control whether the result is returned as:

- A native backend array (e.g., ``numpy.ndarray``), or
- A new buffer instance of the same type.

This is governed by the optional keyword argument ``numpy``.

- ``numpy=True`` → Return the result as a raw NumPy array (or backend-native array),
- ``numpy=False`` (default) → Wrap the result in a new buffer instance of the same type.

This dual-mode approach provides flexibility for use cases that require raw arrays
(e.g., downstream NumPy operations or serialization) while preserving full buffer semantics by default.

Example:

.. code-block:: python

    buffer = ArrayBuffer.from_array(np.arange(12).reshape(3, 4))

    buffer.T.shape
    # → (4, 3)       [buffer returned]

    buffer.transpose(numpy=True).shape
    # → (4, 3)       [NumPy array returned]


When ``numpy=False`` (i.e., returning a new buffer instance), the constructor of the buffer may require additional
positional or keyword arguments—especially in complex backends like HDF5. To support this, all transformation
methods also accept:

- ``bargs`` : tuple of positional arguments forwarded to the buffer constructor,
- ``bkwargs`` : dict of keyword arguments forwarded to the buffer constructor.

These allow fine-grained customization of the buffer instantiation process, ensuring compatibility with buffer
constructors that require contextual information (e.g., file handles, unit metadata, etc.).

Example:

.. code-block:: python

    buffer = HDF5Buffer.from_array(np.ones((10, 10)), file="data.h5", path="/original")

    new_buffer = buffer.reshape(100, numpy=False, bargs=("data.h5", "/reshaped"), bkwargs={"create_file": True})


All PyMetric buffer classes must implement the following transformation methods:

- :meth:`~fields.buffers.base.BufferBase.astype`
- :meth:`~fields.buffers.base.BufferBase.conj`
- :meth:`~fields.buffers.base.BufferBase.conjugate`
- :meth:`~fields.buffers.base.BufferBase.copy`
- :meth:`~fields.buffers.base.BufferBase.flatten`
- :meth:`~fields.buffers.base.BufferBase.ravel`
- :meth:`~fields.buffers.base.BufferBase.reshape`
- :meth:`~fields.buffers.base.BufferBase.resize`
- :meth:`~fields.buffers.base.BufferBase.swapaxes`
- :meth:`~fields.buffers.base.BufferBase.transpose`

These methods operate directly on the internal array, using NumPy logic, and then wrap the result back into a buffer
instance when appropriate. The internal helper method ``_cast_numpy_op()`` standardizes this logic across all
transformations, and may be extended by buffer subclasses for backend-specific behaviors.

Subclassing a Custom Buffer
----------------------------

Advanced users may wish to define new buffer types (e.g., for GPU support, cloud storage,
lazy evaluation, etc.). PyMetric provides a simple but robust framework for this.

To create a custom buffer class, inherit from :class:`~fields.buffers.base.BufferBase` and define:

- ``__core_array_types__``: the internal storage format (e.g., `torch.Tensor`, `xarray.DataArray`)
- ``__can_resolve__``: a list of types your buffer knows how to wrap
- ``__resolution_priority__``: an integer priority (higher = preferred)

You must also implement:

- ``__init__(self, array)`` to wrap the storage object
- ``from_array(cls, obj, **kwargs)`` to construct your buffer from flexible input
- Optional: ``zeros``, ``ones``, ``full``, ``empty``, and I/O methods

Example stub:

.. code-block:: python

    class TorchBuffer(BufferBase):
        __core_array_types__ = (torch.Tensor,)
        __can_resolve__ = [torch.Tensor]
        __resolution_priority__ = 40

        def __init__(self, array):
            super().__init__(array)

        @classmethod
        def from_array(cls, obj, **kwargs):
            tensor = torch.tensor(obj)
            return cls(tensor)

Once defined, your buffer will be automatically registered and resolvable by `buffer_from_array`.

.. note::

    If you want to isolate your buffer type from PyMetric’s global resolution pipeline, register it
    with a custom :class:`~fields.buffers.registry.BufferRegistry` and pass it via ``buffer_registry=``.
