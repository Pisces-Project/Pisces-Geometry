"""
Utility functions for working with array-based coordinate grids.

This module provides helper functions used inPyMetric for manipulating and analyzing
multi-dimensional arrays, particularly in the context of numerical coordinate systems and
differential operations.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


def normalize_index(index: int, axis_size: int) -> int:
    """
    Coerce a possibly negative index into a valid non-negative index.

    Parameters
    ----------
    index : int
        Index to normalize (may be negative).
    axis_size : int
        Size of the axis the index refers to.

    Returns
    -------
    int
        Equivalent non-negative index.
    """
    if not -axis_size <= index < axis_size:
        raise IndexError(f"Index {index} out of bounds for axis of size {axis_size}")
    return index % axis_size


def broadcast_labels(*inputs):
    """
    Broadcast shapes and propagate semantic axis labels, with singleton suppression.

    Each input is a tuple `(shape, labels)`:
      - `shape` is a standard tuple of integers.
      - `labels` is a sequence of axis names or `None`. If `None`, the shape is treated as unlabeled.

    Axis labels are propagated to the output broadcast shape only if:
      - They come from a dimension with size > 1 (i.e., not a singleton),
      - And there are no conflicting labels on that axis from other inputs.

    If multiple **non-singleton** labeled inputs contribute different labels to the same axis,
    a `ValueError` is raised. Labels from singleton dimensions are ignored unless no dominant label exists.

    Parameters
    ----------
    *inputs : Tuple[(shape, labels)]
        Input shapes and optional axis labels. Each shape must match the length of its label list
        if labels are provided.

    Returns
    -------
    broadcast_shape : Tuple[int, ...]
        The final broadcasted shape.
    broadcast_labels : Tuple[Optional[str], ...]
        The labels associated with each axis in the broadcasted shape. May contain None.

    Raises
    ------
    ValueError
        If label lengths don't match their shapes, or if conflicting non-singleton labels
        exist for a broadcasted axis.

    Examples
    --------
    >>> broadcast_labels(((1, 5, 4), ['r', 'phi', 'z']), ((10, 5, 4), [None, 'phi', 'z']))
    ((10, 5, 4), (None, 'phi', 'z'))

    >>> broadcast_labels(((3, 4), ['x', 'y']), ((1, 4), ['z', 'y']))
    ValueError: conflicting non-singleton labels on axis 0
    """
    # Normalize shapes and labels
    shapes = [tuple(s) for s, _ in inputs]
    labels = [
        tuple(lab) if lab is not None else (None,) * len(s) for (s, lab) in inputs
    ]

    # Validate input consistency
    for shape, lab in zip(shapes, labels):
        if len(shape) != len(lab):
            raise ValueError(
                f"Labels must match shape: got shape={shape}, labels={lab}"
            )

    # Compute the broadcasted shape
    broadcast_shape = np.broadcast_shapes(*shapes)
    ndim = len(broadcast_shape)

    # Pad shapes and labels to align from the right
    padded_shapes = [(1,) * (ndim - len(s)) + s for s in shapes]
    padded_labels = [(None,) * (ndim - len(lab)) + lab for lab in labels]

    # Resolve output labels axis by axis
    result_labels = []
    for axis_index, _ in enumerate(broadcast_shape):
        axis_labels = np.array(
            [lbls[axis_index] for lbls in padded_labels], dtype=object
        )
        axis_sizes = np.array([shp[axis_index] for shp in padded_shapes], dtype=int)

        # Filter labels that come from non-singleton dimensions
        dominant_labels = {
            lbl
            for lbl, sz in zip(axis_labels, axis_sizes)
            if sz != 1 and lbl is not None
        }

        if len(dominant_labels) > 1:
            raise ValueError(
                f"Conflicting labels at broadcasted axis {axis_index}: {dominant_labels}"
            )
        elif len(dominant_labels) == 1:
            result_labels.append(dominant_labels.pop())
        else:
            result_labels.append(None)

    return broadcast_shape, tuple(result_labels)


def apply_ufunc_to_labels(
    ufunc: np.ufunc,
    method: str,
    *inputs: Tuple[Tuple[Sequence[int], Optional[Sequence[str]]], ...],
    **kwargs,
):
    """
    Infer the output shape and axis labels of a ufunc applied to labeled input shapes.

    This function simulates the structural behavior of NumPy ufuncs while tracking
    axis labels across operations. It is designed to work alongside tensor-aware
    data structures like labeled fields, where axis semantics must be preserved or
    transformed according to the broadcasting and reduction rules of NumPy.

    Parameters
    ----------
    ufunc : np.ufunc
        The NumPy ufunc being applied (e.g., np.add, np.multiply, np.sum).
    method : str
        The method being invoked on the ufunc, such as '__call__', 'reduce',
        'reduceat', 'accumulate', 'outer', or 'at'.
    *inputs :
        One or more (shape, label) pairs.

        - `shape` must be a sequence of integers representing the array shape.
        - `labels` must be a sequence of strings or `None` with the same length
          as the shape, or `None` to indicate an unlabeled input.
    **kwargs :
        Additional keyword arguments specific to the ufunc method:

        - axis : int or tuple of int (for 'reduce', 'reduceat')
        - keepdims : bool (for 'reduce')
        - indices : sequence of int (for 'reduceat')

    Returns
    -------
    output_shape : tuple of int
        The resulting shape after applying the ufunc method.
    output_labels : tuple of str or None
        The axis labels associated with the resulting shape. Labels are propagated
        from inputs where possible, with singleton-suppression and axis elimination
        rules enforced appropriately.

    Raises
    ------
    ValueError
        If the number of inputs is inconsistent with the ufunc arity or method requirements,
        or if required keyword arguments are missing or malformed.
    NotImplementedError
        If the requested ufunc method is not supported or not defined on the given ufunc.

    Examples
    --------
    >>> apply_ufunc_to_labels(np.add, '__call__',
    ...     ((3, 1, 4), ['x', 'y', 'z']),
    ...     ((1, 5, 4), [None, 'y', 'z']))
    ((3, 5, 4), ('x', 'y', 'z'))

    >>> apply_ufunc_to_labels(np.add, 'reduce',
    ...     ((3, 4, 5), ['x', 'y', 'z']),
    ...     axis=1, keepdims=False)
    ((3, 5), ('x', 'z'))

    >>> apply_ufunc_to_labels(np.add, 'outer',
    ...     ((3,), ['x']),
    ...     ((4,), ['y']))
    ((3, 4), ('x', 'y'))
    """
    # Check that the ufunc has the anticipated
    # method available.
    if not hasattr(ufunc, method):
        raise NotImplementedError(f"{method} method is not supported by ufunc {ufunc}.")

    # Now dispatch the behavior by which
    # ufunc is being performed and on which items.
    if method == "__call__":
        # This is the cornerstone methods. We can simply check the nin
        # of the ufunc to check the length of inputs before proceeding.
        if len(inputs) != ufunc.nin:
            raise ValueError(
                f"Expected {ufunc.nin} inputs for {ufunc}, got {len(inputs)}"
            )

        # Special case: matmul (matrix multiplication has custom shape rules)
        if ufunc is np.matmul:
            # normalize the shapes and the labels from the inputs.
            shapes = [tuple(s) for s, _ in inputs]
            labels = [
                tuple(lab) if lab is not None else (None,) * len(s)
                for (s, lab) in inputs
            ]

            # Partition shapes into the grid and matrix components.
            gshapeA, mshapeA, gshapeB, mshapeB = (
                shapes[0][:-2],
                shapes[0][-2:],
                shapes[1][:-2],
                shapes[1][-2:],
            )
            glabelsA, mlabelsA, glabelsB, mlabelsB = (
                labels[0][:-2],
                labels[0][-2:],
                labels[1][:-2],
                labels[1][-2:],
            )

            # broadcast the labels for the grid axes.
            gshape, glabels = broadcast_labels((gshapeA, glabelsA), (gshapeB, glabelsB))
            mshape, mlabels = (mshapeA[0], mshapeB[1]), (mlabelsA[0], mlabelsB[1])
            shape, labels = gshape + mshape, glabels + mlabels
            return shape, labels

        shape, labels = broadcast_labels(*inputs)
        return shape, labels
    elif method == "reduce":
        # This is the reduction protocol. We only ever expect 1 input
        # and then we use the specified axes to determine the shapes and
        # the labeling.
        if len(inputs) != 1:
            raise ValueError(f"Expected 1 input for method 'reduce', got {len(inputs)}")

        shape, label = inputs[0]

        # Extract the axis and the keepdims as these dicate the resulting
        # shape and labels.
        axis = kwargs.pop("axis", 0)
        keepdims = kwargs.pop("keepdims", False)

        # Use axis and keepdims to determine the relevant resulting labels and
        # shape.
        if axis is None:
            # We have no axes, we return a scalar and all labels die.
            return (), ()
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)

        # We now propagate forward using the keepdims.
        if keepdims:
            return (
                tuple(s if i not in axis else 1 for i, s in enumerate(shape)),
                tuple(l if i not in axis else None for i, l in enumerate(label)),
            )
        else:
            return (
                tuple(s for i, s in enumerate(shape) if i not in axis),
                tuple(l for i, l in enumerate(label) if i not in axis),
            )

    elif method == "accumulate":
        # Accumulate keeps shape and labels unchanged
        if len(inputs) != 1:
            raise ValueError(
                f"Expected 1 input for method 'accumulate', got {len(inputs)}"
            )

        return broadcast_labels(*inputs)

    elif method == "reduceat":
        # Reduce at specific indices / indices ranges. Begin by validating that
        # we got a single input.
        if len(inputs) != 1:
            raise ValueError(
                f"Expected 1 input for method 'reduceat', got {len(inputs)}"
            )

        # Now extract the indices and the axis so that we
        # can proceed.
        axis = kwargs.pop("axis", 0)
        indices = kwargs.pop("indices", None)
        if indices is None:
            raise ValueError("reduceat requires 'indices' kwarg.")

        # Now extract the shape and the relevant labels.
        shape, label = inputs[0]
        return (
            tuple(s if i != axis else len(indices) for i, s in enumerate(shape)),
            tuple(l if i != axis else None for i, l in enumerate(label)),
        )

    elif method == "outer":
        if len(inputs) != 2:
            raise ValueError("outer requires exactly two input arrays.")

        (shape1, labels1), (shape2, labels2) = inputs
        out_shape = tuple(shape1) + tuple(shape2)
        out_labels = tuple(labels1 or [None] * len(shape1)) + tuple(
            labels2 or [None] * len(shape2)
        )
        return out_shape, out_labels

    elif method == "at":
        # In-place update; output shape/labels unchanged
        return broadcast_labels(*inputs)

    else:
        raise NotImplementedError(f"Ufunc method '{method}' is not supported.")
