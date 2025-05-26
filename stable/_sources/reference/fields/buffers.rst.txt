.. _buffers:

====================
Fields: Data Buffers
====================

The Pisces Geometry **buffer system** provides an abstraction layer over data storage backends
for field components. It allows field operations to remain agnostic to whether data is stored in memory
(via `NumPy <https://numpy.org/doc/stable/index.html>`_), with units
(via `unyt <https://unyt.readthedocs.io/en/stable/>`_), or on disk (e.g., HDF5). This enables modular,
extensible, and backend-agnostic scientific computing.

This document provides an overview of the buffer architecture, key classes, resolution mechanisms, and subclassing guidelines.

.. contents::
   :local:
   :depth: 2

Overview
--------

Buffers are the low-level storage abstraction in the :mod:`~fields` module. They are responsible for:

- Managing actual data (values and memory layout)
- Interfacing with array backends (e.g., NumPy, `unyt`, HDF5)
- Supporting broadcasting, NumPy semantics, and I/O
- Providing a clean interface for the :class:`~fields.components.FieldComponent` system

Each buffer class inherits from :class:`~fields.buffers.base.BufferBase`, which provides a uniform API across backends.

Creating Buffers of Different Types
-----------------------------------

There are currently **three core buffer types**:

- :class:`~fields.buffers.core.ArrayBuffer` — an in-memory buffer for plain :class:`~numpy.ndarray`.
- :class:`~fields.buffers.core.UnytArrayBuffer` — an in-memory buffer for :class:`~unyt.array.unyt_array` objects with physical units.
- :class:`~fields.buffers.core.HDF5Buffer` — a persistent, disk-backed buffer based on :class:`h5py.Dataset`.

Buffers can be created in multiple ways, depending on the format of your input data and your desired backend. The most
generic of these approaches is the :meth:`~fields.buffers.base.BufferBase.from_array`. This tries to coerce an input object
(:class:`list`, :class:`~numpy.ndarray`, :class:`~unyt.array.unyt_array`, etc.) into a compatible buffer type:

.. code-block:: python

    from pymetric.fields.buffers import UnytArrayBuffer

    data = [[1, 2], [3, 4]]
    buf = UnytArrayBuffer.from_array(data,units='keV',dtype='f8')

This approach has the distinct advantage of clarifying the buffer that will be returned at the expense
of requiring that the user knows that their data is compatible with the particular buffer class.

.. hint::

    In addition to :meth:`~fields.buffers.base.BufferBase.from_array`, there are
    also :meth:`~fields.buffers.base.BufferBase.zeros()`, :meth:`~fields.buffers.base.BufferBase.ones()`,
    :meth:`~fields.buffers.base.BufferBase.full()`,
    and :meth:`~fields.buffers.base.BufferBase.empty()` attached to each buffer class.

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
2. Equivalently, each :class:`~fields.buffers.base.BufferBase` has :meth:`~fields.buffers.base.BufferBase.resolve` which
   is a simple alias for option 1.
3. Finally, PyMetric provides a number of utility functions in :mod:`~fields.buffers.utilities` like :func:`~fields.buffers.utilities.zeros`
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
while preserving backend-specific advantages like unit tracking or disk persistence.

To understand buffer operations, it's important to keep track of 3 important classes:

- The **buffer class**: subclass of :class:`~fields.buffers.base.BufferBase` representing some
  dataset.
- The **core class**: the class that the **buffer class** is actually wrapped around. This is what
  the buffer sees behind the scenes.
- The **representation class(es)**: the classes that are actually represented by the buffer.

For example, the :class:`~fields.buffers.core.HDF5Buffer` has only a single **core class**: :class:`h5py.Dataset`,
but its **representation classes** are both :class:`numpy.ndarray` and :class:`unyt.array.unyt_array` depending on whether
or not the underlying dataset has units attached to it.

Operations on buffers hinge very specifically on the nature of these 3 classes.

Indexing and Slicing Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indexing and slicing relies on both the **core class** and the **representation classes**. The approach is pretty simple:
the buffer will forward the indexing operation to the **core class** and then coerce the result into the correct **representation class**.

In most cases, the core class and the representation class are the same; but for classes like :class:`~fields.buffers.core.HDF5Buffer`,
an indexing operation ``buff[0]`` will first fetch ``buff.__array_object__[0]`` (which indexes into the underlying HDF5 dataset) and then
wraps the result to convert it to a :class:`float` or :class:`~unyt.array.unyt_quantity`.

.. code-block:: python

    buf = UnytArrayBuffer.full((4, 4), fill_value=10, units="keV")
    sub = buf[1:3, 1:3]

    assert isinstance(sub, unyt.unyt_array)
    assert sub.shape == (2, 2)

If units are present (as in :class:`~UnytArrayBuffer` and :class:`~HDF5Buffer`), they are preserved in the sliced result unless dimensionality collapses to a scalar, in which case a :class:`~unyt.unyt_quantity` is returned.

Representation Types and Views
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Buffers offer several methods to extract the internal array in various formats:

- :meth:`~fields.buffers.base.BufferBase.as_core()` returns the raw array as stored by
  the backend (e.g., :class:`~numpi.ndarray`, :class:`~unyt.array.unyt_array`, or :class:`h5py.Dataset`).
- :meth:`~fields.buffers.base.BufferBase.as_array()` returns a standard NumPy array (units stripped if needed).
- :meth:`~fields.buffers.base.BufferBase.as_unyt_array()` returns a unit-tagged array (if available).
- :meth:`~fields.buffers.base.BufferBase.as_repr()` is a general-purpose method that returns the
  buffer’s preferred array representation (unyt if possible, raw NumPy otherwise).

These allow flexible interop in numerical code:

.. code-block:: python

    arr = buffer.as_array()         # always NumPy
    tagged = buffer.as_unyt_array() # only works if units are defined
    view = buffer.as_repr()         # representation-aware fallback

Numpy Semantics: Universal Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Universal functions (ufuncs) are core to NumPy’s performance and expressiveness.
Examples include :func:`~numpy.add`, :func:`~numpy.sin`, :func:`~numpy.sqrt`, and many others.
Buffers support full ufunc behavior through the :meth:`~fields.buffers.base.BufferBase.__array_ufunc__` protocol.

PyMetric’s behavior follows these principles:

- All buffer arguments are **converted to their representation type** via :meth:`~fields.buffers.base.BufferBase.as_repr()` before the operation.
- The operation is performed directly on those values.
- If an ``out=`` argument is specified and targets another buffer, PyMetric attempts to:
  - Validate the target buffer’s compatibility.
  - Avoid allocation by performing the operation **in-place** using the buffer’s internal storage.
- If no ``out=`` argument is provided, the result is returned as an instance of the appropriate representation type (e.g., `unyt_array`).

Example:

.. code-block:: python

    from pymetric.fields.buffers import UnytArrayBuffer
    import numpy as np
    import unyt

    # Create a generic unit carrying buffer and take the
    # sqrt.
    buf = UnytArrayBuffer.ones((3, 3), units="m")
    out = np.sqrt(buf) # ! NOT A BUFFER ANYMORE.

    # In-place example
    out_buf = UnytArrayBuffer.empty((3, 3), units="m")
    out = np.add(buf, 5 * unyt.Unit("km"), out=out_buf) # STILL A BUFFER.

.. hint::

    The TLDR of this behavior is as follows: If you use ``out=``, you'll **get another buffer**. Otherwise,
    you're going to get the buffer's **representation type** (the type it represents).

.. note::

    The convention behind this behavior is that the principle priority of buffers is to be
    *backend agnostic*; as such, the safest way to guarantee that behavior is to require users
    to be explicit when working at the buffer level.

Numpy Semantics: NumPy Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To provide full support for NumPy operations, most numpy functions can operate on buffers. The behavior
of this depends somewhat on semantics and on the details of different buffer types. In general, the following
principles should be understood:

- By default, a numpy function called on a buffer will strip the buffer and operation on the
  **representation class**.

  - This is true of *all* numpy functions which operate on multiple arrays in some respect or
    which are not "simple" transformations. While there are some cases where this may be
    a nuisance, the benefit of a standardized behavior considerably outweighs the costs.

- Many standard transformation methods (i.e. :func:`numpy.transpose`, :func:`numpy.reshape`, etc.) are implemented
  as methods of the buffer class. These will (in general) return buffers as output.

  - When called as a **method**, the buffer is transformed to its representation class,
    the operation is performed on that class, and it is then passed back into :meth:`~fields.buffers.base.BufferBase.from_array`.
  - When the equivalent **function** in numpy is called, the same operation occurs; however,
    due to the nature of numpy forwarding, any relevant keyword arguments for :meth:`~fields.buffers.base.BufferBase.from_array`
    are lost. As such, it is generally preferable to simply use the method.

  .. important::

        This behavior can be somewhat odd. For example,

        .. code-block::

            buf = HDF5Buffer.from_array(...)
            np.transpose(buf,units=...)

        will raise an error because ``units=`` is not a permitted keyword argument, but

        .. code-block::

            buf = HDF5Buffer.from_array(...)
            buf.transpose(units=...)

        will work. Thus, the rule of thumb is that the **method provides fine control** and
        the numpy function provides **superficial control**.

.. hint::

      For the HDF5 buffer (:class:`~fields.buffers.core.HDF5Buffer`), :meth:`~fields.buffers.core.HDF5Buffer.from_array`
      requires the user to explicitly provide `file` and `name` arguments. As such, numpy functions like :func:`numpy.ravel`
      cannot be forwarded in the manner described above and therefore fallback to the default behavior.

      This class **does** implement ``inplace=`` in all of its method transformations to expedite
      common use cases. See, for example, :meth:`~fields.buffers.core.HDF5Buffer.reshape`.

Buffer Unit Management
----------------------

All buffers (and by extension fields and components) in PyMetric support **units** in a backend-agnostic way. A buffer may either:

- Be *unitless*, in which case its :attr:`~fields.buffers.base.BufferBase.units` property is ``None``, and its *representation class* is :class:`numpy.ndarray`.
- Be *unit-aware*, in which case its :attr:`~fields.buffers.base.BufferBase.units` property is a :class:`unyt.unit_object.Unit`,
  and its *representation class* is :class:`~unyt.array.unyt_array`.

All of the unit management is performed via `Unyt <https://unyt.readthedocs.io/en/stable/#>`_. There are two paradigms for
unit manipulation of buffers:

1. For *specific* buffer classes which support units, the units of a buffer may be changed in place via *conversion*.
2. For *all* buffer classes, the buffer can be cast to :class:`~unyt.array.unyt_array`, converted, and re-wrapped into
   a new buffer. We call this *unit casting*.

For example, the units borne by the buffers in the following cases are as follows:

.. code-block:: python

    from pymetric import ArrayBuffer, UnytArrayBuffer, HDF5Buffer
    import numpy as np
    import unyt

    # As a numpy array.
    x = np.linspace(-np.pi, np.pi, 100)
    buff_x = ArrayBuffer.from_array(x) # No unit support.
    unyt_buff_x = UnytArrayBuffer.from_array(x) # Only unit support.
    hdf5_buff_x = HDF5Buffer.from_array(x, # Both unit and no unit support.
                                        'test.hdf5',
                                        'tst',
                                        overwrite=True,
                                        create_file=True)
    print(buff_x.units,unyt_buff_x.units,hdf5_buff_x.units)
    # Yields (None, dimensionless, None)

    # As an unyt array.
    x = np.linspace(-np.pi, np.pi, 100) * unyt.Unit("km")
    buff_x = ArrayBuffer.from_array(x) # No unit support.
    unyt_buff_x = UnytArrayBuffer.from_array(x) # Only unit support.
    hdf5_buff_x = HDF5Buffer.from_array(x, # Both unit and no unit support.
                                        'test.hdf5',
                                        'tst',
                                        overwrite=True,
                                        create_file=True)
    print(buff_x.units,unyt_buff_x.units,hdf5_buff_x.units)
    # Yields (None, km, km)

Unit Conversion (In-place)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unit conversion is **in-place**, meaning the buffer is updated without copying or creating a new instance.
This is performed using:

- :meth:`~fields.buffers.base.BufferBase.convert_to_units`
- :meth:`~fields.buffers.base.BufferBase.convert_to_base`

These methods are supported only by buffer types that are unit-aware (e.g., :class:`~fields.buffers.core.UnytArrayBuffer`).
Attempting to call them on a non-unit-supporting buffer (e.g., :class:`~fields.buffers.core.ArrayBuffer`) will raise an error.

.. code-block:: python

    from pymetric import ArrayBuffer, UnytArrayBuffer, HDF5Buffer
    import numpy as np
    import unyt

    x = np.linspace(-np.pi, np.pi, 100) * unyt.Unit("km")
    buff_x = ArrayBuffer.from_array(x)
    unyt_buff_x = UnytArrayBuffer.from_array(x)
    hdf5_buff_x = HDF5Buffer.from_array(x,'test.hdf5','tst',overwrite=True,create_file=True)

    print(buff_x.units, unyt_buff_x.units, hdf5_buff_x.units)
    # Result: None km km

    buff_x.convert_to_units('m')
    # Result: ValueError: Cannot set units for buffer of class ArrayBuffer.
    unyt_buff_x.convert_to_units('m')
    hdf5_buff_x.convert_to_units('m')

    print(buff_x.units, unyt_buff_x.units, hdf5_buff_x.units)
    # Result: None m m

Additionally, if a buffer class supports units, you may directly set the :attr:`~fields.buffers.base.BufferBase.units` attribute, which
results in an unsafe direct setting of the units.

.. code-block:: python

    from pymetric import ArrayBuffer, UnytArrayBuffer, HDF5Buffer
    import numpy as np
    import unyt

    x = np.linspace(-np.pi, np.pi, 100) * unyt.Unit("km")
    unyt_buff_x = UnytArrayBuffer.from_array(x)

    print(unyt_buff_x[0])
    # -3.141592653589793 km
    unyt_buff_x.convert_to_units('m')
    print(unyt_buff_x[0])
    # -3141.592653589793 m
    unyt_buff_x.units = 'keV'
    print(unyt_buff_x[0])
    # -3141.592653589793 keV


Unit Casting (Copy)
^^^^^^^^^^^^^^^^^^^^

Casting to new units always produces a **new buffer** (or optionally, a `unyt_array`).
This is the most flexible and safe option, and is supported by all buffer types.

Casting is performed via:

- :meth:`~fields.buffers.base.BufferBase.in_units`
- :meth:`~fields.buffers.base.BufferBase.to` (alias)
- :meth:`~fields.buffers.base.BufferBase.to_value`

These methods perform the following:

- Convert the buffer to a `unyt_array`
- Apply the requested unit transformation
- Re-wrap the array in a new buffer using resolution logic (unless `as_array=True`)

Examples:

.. code-block:: python

    from pymetric import ArrayBuffer, UnytArrayBuffer, HDF5Buffer
    import numpy as np
    import unyt

    x = np.linspace(0,5, 5) * unyt.Unit("m")
    unyt_buff_x = UnytArrayBuffer.from_array(x)

    # Cast to value (numpy array)
    print(unyt_buff_x.to_value("m"))
    # [0.   1.25 2.5  3.75 5.  ]

    # Convert (cast) to pc with buffer
    # resolution.
    print(type(unyt_buff_x.to('pc')))
    # <class 'pymetric.fields.buffers.core.UnytArrayBuffer'>

    # Force the output buffer type.
    print(type(unyt_buff_x.to('pc',buffer_class=ArrayBuffer)))
    # <class 'pymetric.fields.buffers.core.UnytArrayBuffer'>

.. hint::

    If you only need to change units temporarily or want a non-buffer output, use:

    .. code-block:: python

        arr = buf.in_units("erg", as_array=True)

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
- Optional: ``zeros()``, ``ones()``, ``full()``, ``empty()``, and I/O methods
- Optional: ``units`` property if your buffer handles units

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

Once defined, your buffer will be automatically registered and resolvable by `buffer_from_array()`.

.. note::

    If you want to isolate your buffer type from PyMetric’s global resolution pipeline, register it
    with a custom :class:`~fields.buffers.registry.BufferRegistry` and pass it via ``buffer_registry=``.


See Also
--------

- :class:`~fields.buffers.base.BufferBase` — Abstract base class
- :mod:`~fields.buffers.core` — Core buffer implementations
- :mod:`~fields.buffers.registry` — Registry system
- :mod:`~fields.buffers.utilities` — Helper functions for dynamic buffer generation
- :mod:`~fields.components` — FieldComponent framework (buffer consumers)
