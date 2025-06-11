.. _fields_numpy:

============================
Fields: NumPy Compatibility
============================

Numpy Semantics for Buffers
---------------------------

The lowest level class used in constructing fields is the :class:`~pymetric.fields.buffers.base.BufferBase` class,
which provides a universal wrapper around other array-like types in order to provide standardized access and behavior.

For any given ``BufferClass``, there are generically two other relevant classes:

1. The ``ArrayLikeClass`` that ``BufferClass`` wraps around, and
2. The ``RepresentationClass`` that ``BufferClass`` yields when sliced.

For example, the :class:`~pymetric.fields.buffers.core.HDF5Buffer` wraps the :class:`h5py.Dataset` and yields
:class:`numpy.ndarray` when accessed.
:class:`~pymetric.fields.buffers.core.ArrayBuffer` both wraps and yields :class:`numpy.ndarray`.

For these classes, the numpy semantics are relatively intuitive:

    For a buffer class ``BufferClass`` wrapping an underlying array-like class ``ArrayClass`` and yielding
    a ``RepresentationClass``, numpy operations will be delegated in their behavior to the underlying
    ``RepresentationClass``.

.. hint::

    As such, if I perform an operation on a :class:`~pymetric.fields.buffers.core.HDF5Buffer`, the
    underlying representation of that object will be extracted (:class:`numpy.ndarray`) and
    the operation will be performed on the representation.

As such, **most operations break the buffer**. The reason we allow this sort of behavior is as follows:

1. It's easy to re-wrap a representation in a new buffer when needed.
2. For buffers which are lazy-loading or disk-backed, the buffer really handles the semantics of
   IO access, so an operation on the array really isn't a buffer anymore, its just an array.

In-place Operations
'''''''''''''''''''

There is one exception to the above rule: for NumPy **universal functions (ufuncs)**, if ``out=`` is specified
to be a particular buffer, then the operation occurs as expected except that the resulting ``RepresentationClass`` is
coerced back into the underlying buffer and the result is the original ``out`` instance of ``BufferClass``.

This ensures that operations like ``np.sin(buffer,out=buffer)`` and ``buffer += 1`` do not break the buffer's typing.

This rule **only applies to ufuncs**; however, a similar behavior can be obtained in other operations by simply
passing the underlying ``.__array_object__`` instead of the entire buffer.

.. important::

    For some buffers, specifically those which have disk-based underlying array-like objects like HDF5 datasets,
    fully in-place writing is not possible. These will therefore fail with a corresponding error. To circumvent this,
    one can simply compute without the ``out=`` argument and then assign.


Numpy Semantics for Components
------------------------------

At the next layer of abstraction is the :class:`~pymetric.fields.components.FieldComponent`, which again has
3 critical attributes that determine how it interacts with numpy:

- The ``buffer``: The underlying :class:`~pymetric.fields.buffers.core.HDF5Buffer` class that contains the
  data referenced in the component.
- The ``axes``: The coordinate axes of the component.
- The ``grid``: The grid on which the field component exists.

The numpy semantics for :class:`~pymetric.fields.components.FieldComponent` objects are more complicated, reflecting
the fact that one need now handle not only the ``buffer``, but also the ``axes`` and ``grid``.

NumPy Universal Functions on Components
'''''''''''''''''''''''''''''''''''''''

Numpy provides a number of **universal functions** called `ufuncs <https://numpy.org/doc/stable/reference/ufuncs.html>`__,
which have very well controlled behavior. These operations occur element-by-element with broadcasting, type casting, etc. For
each ufunc, there are a variety of "modes", which can alter behavior.

When ufuncs are performed on :class:`~pymetric.fields.components.FieldComponent`, the class will always **attempt**
to maintain the output as a :class:`~pymetric.fields.components.FieldComponent` albeit with a potentially different
set of axes. The rules for ufuncs are as follows:

- Ufunc is binary (2 inputs):

  - Only 1 of the inputs is a :class:`~pymetric.fields.components.FieldComponent`

    When an operation occurs between a :class:`~pymetric.fields.components.FieldComponent` and a different class,
    we perform the operation under the assertion that we **never add new axes**. The operation is performed and afterward
    the shape is checked to determine if the leading axes of the result still match the grid shape along the original axes.

    If so, then the result will be a :class:`~pymetric.fields.components.FieldComponent` with the same set of axes
    and grid. Otherwise, the operation is delegated entirely to the underlying ``buffer`` and the result will reflect the
    ufunc rules for the ``buffer`` alone.

  - Both inputs are :class:`~pymetric.fields.components.FieldComponent`

    If both inputs are :class:`~pymetric.fields.components.FieldComponent`, then they are broadcast to the union
    of their axes and then the operation is performed, always resulting in a new :class:`~pymetric.fields.components.FieldComponent`
    of their conjoined axes.

    For these to be compatible, they must have identical grids.

- Ufunc is unitary (1 input):

  If the ufunc is a unitary operation and the ufunc is in ``__call__`` mode, then the shape cannot be altered
  by the function. Therefore, the ufunc is delegated to the underlying ``buffer`` of the component and the returned
  with the same grid and axes as the original.

  For example, ``np.sin(component)`` will return a component with data ``np.sin(buffer)`` and the same set
  of axes and grid.









Numpy Semantics for Fields
--------------------------
