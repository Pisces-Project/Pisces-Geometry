"""
Utility functions for interacting with ``sympy``.
"""
import sympy as sp
import numpy as np
from typing import List, Dict, Any, Union, Callable
from pisces_geometry.utilities.logging import pg_params, pg_log

def lambdify_expression(
    expression: Union[str, sp.Basic],
    axes: List[sp.Symbol],
    parameters: Dict[str, Any] = None
) -> Callable:
    r"""
    Convert a `Sympy <https://docs.sympy.org/latest/index.html>`_ expression (scalar, Matrix, or N-dimensional Array) or a string
    into a callable function. The returned function depends on the given ``axes``
    and optional ``parameters``, and returns either:

        - A scalar or array of shape matching the input arguments (if the expression is scalar).
        - A 2D or higher-dimensional NumPy array (if the expression is a :py:class:`sympy.matrices.Matrix` or N-dim Array).

    The returned function utilizes numpy functions to ensure that it is relatively optimized for use on array valued
    arguments.

    .. note::

        While :py:func:`lambdify_expression` is designed to perform relatively well on array inputs, it should
        not be called in large loops as the call overhead can be quite large.

    Parameters
    ----------
    expression : str or sympy.Basic
        The expression to be converted into a callable function. This may be any valid ``sympy`` expression or a string
        which can be parsed directly to a valid sympy expression using :py:func:`sympy.sympify`. In ``expression`` is a
        sympy matrix or other non-scalar object, each element is lambdified independently and then combined again in
        the callable that is returned.
    axes : list of :py:class:`~sympy.core.symbol.Symbol`
        The list of coordinate symbols which will become the arguments of the resulting function. This should be a list
        of valid :py:class:`~sympy.core.symbol.Symbol` objects, each corresponding to an independent variable in the expression. The
        order of the ``axes`` will determine the expected order of inputs when using the resulting callable function.
    parameters : dict, optional
        A mapping from symbol name to numerical value (or another valid Sympy expression)
        which will be substituted into the final expression. Defaults to ``None``.

    Returns
    -------
    Callable
        A function ``f(*axes)`` that evaluates the expression numerically:

        - **If the expression is scalar**: returns float or NumPy array.
        - **If the expression is Matrix/Array**: returns a NumPy array of shape
          ``(rows, cols)`` or the corresponding multi-dimensional shape. If the first
          argument is an array of length N, then the result generally has shape
          ``(N, rows, cols, ...)`` etc.

    Raises
    ------
    ValueError

        - If sympifying the expression fails.
        - If there are unrecognized symbols that do not match any of the axes or parameters.
        - If a constant expression cannot be converted to float.

    Examples
    --------

    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> expr = sp.Matrix([[1, 0], [x, 1]])
    >>> f = lambdify_expression(expr, axes=[x])
    >>> import numpy as np
    >>> print(f(2.0))
    [[1. 0.]
     [2. 1.]]
    >>> # For an array of x-values, you'll get shape (N, 2, 2).
    >>> xs = np.linspace(0, 1, 3)
    >>> print(f(xs).shape)
    (3, 2, 2)

    Notes
    -----
    This operation proceeds in the following basic steps:

    1. Parse and sympify string expressions.
    2. Substitute known parameters.
    3. Validate that any remaining free symbols are among the provided `axes`.
    4. Identify whether the expression is a scalar, Matrix, or Array.
    5. Recursively lambdify each entry (for Matrix/Array), or do a scalar lambdify.
    6. Return a function that evaluates those sub-lambdas and aggregates them
       into a NumPy array of the correct shape (for Matrix/Array) or a scalar/array (for scalar expressions).

    """
    # Ensure that parameters is a dictionary and not None by default. This
    # ensures that the kwarg is immutable.
    parameters = parameters or {}
    # Parse the expression if it is a string.
    if isinstance(expression, str):
        try:
            expression = sp.sympify(expression)
        except (sp.SympifyError, TypeError) as e:
            raise ValueError(
                f"Failed to parse/convert expression to a Sympy object. Expression: {expression}"
            ) from e

    # Substitute the parameters into the bound expression and then check for any missing symbols
    # which are not axes but remain in the expression after the substitution.
    bound_expr = expression.subs(parameters)

    free_symbols = bound_expr.free_symbols
    recognized_symbols = set(axes) | {sp.Symbol(k) for k in parameters.keys()}
    unrecognized = free_symbols - recognized_symbols
    if unrecognized:
        raise ValueError(
            f"The expression contains unrecognized symbols: {unrecognized}. "
            "Ensure all symbols are in 'axes' or 'parameters'."
        )

    # Direct the function to perform differently depending on whether
    # the expression is multidimensional or not.
    if isinstance(bound_expr, sp.MatrixBase):
        return _lambdify_matrix(bound_expr, axes, parameters)
    elif isinstance(bound_expr, sp.NDimArray):
        return _lambdify_ndarray(bound_expr, axes, parameters)
    else:
        # It's a scalar expression => proceed with scalar logic
        return _lambdify_scalar(bound_expr, axes)


def _lambdify_scalar(bound_expr: sp.Basic, axes: List[sp.Symbol]) -> Callable:
    """
    Lambdify a scalar Sympy expression that depends on given axes.
    If the expression is constant, return a function that broadcasts.
    Otherwise, return a standard lambdified function.
    """
    # Determine the ops count so that we can determine if we are interested
    # in trying to check for .is_constant or if that will be too taxing.
    __ops_count__ = sp.count_ops(bound_expr, visual=False)
    __will_check_flag__ = __ops_count__ <= pg_params['skip_constant_checks']

    if __will_check_flag__:
        # Handle constant expressions
        if bound_expr.is_constant():
            try:
                constant_value = float(sp.simplify(bound_expr))
            except (TypeError, ValueError):
                raise ValueError(
                    f"The expression simplifies to a constant but could not be converted to float: {bound_expr}"
                )

            def _constant_function(*args, _cv=constant_value):
                # If no args, just return the constant
                if not args:
                    return _cv
                # Otherwise broadcast to the shape of the first argument
                return np.full_like(args[0], _cv)

            return _constant_function
    else:
        # We don't perform the bound_expr check.
        pg_log.debug("Skipping .is_constant check for bound expression: %s.",bound_expr)
        pass

    # If non-constant, do a normal lambdify
    try:
        func = sp.lambdify(axes, bound_expr, "numpy")
    except Exception as e:
        raise ValueError(
            f"Failed to lambdify expression with axes={axes}.\n"
            f"Sympy expression: {bound_expr}"
        ) from e

    return func


def _lambdify_matrix(mat_expr: sp.MatrixBase, axes: List[sp.Symbol], parameters: Dict[str, Any]) -> Callable:
    """
    Recursively lambdify each element of a :py:class:`sympy.matrices.Matrix` (2D).
    Return a function that, given arrays for the axes, returns a
    NumPy array of shape (rows, cols) or (N, rows, cols), etc.
    """
    rows, cols = mat_expr.shape

    # Build per-element scalar lambdas
    element_funcs = []
    for i in range(rows):
        row_funcs = []
        for j in range(cols):
            row_funcs.append(_lambdify_scalar(mat_expr[i, j], axes))
        element_funcs.append(row_funcs)

    def matrix_func(*args):
        """
        Evaluate each element for the given arguments in *args.
        Return a NumPy array of shape (rows, cols) if scalars,
        or (N, rows, cols) if the first argument is a 1D array of length N, etc.
        """
        # Evaluate each element
        evaluated = [[f(*args) for f in row] for row in element_funcs]

        # Convert to a numeric array
        # If any element is an array, this will become an object array initially
        arr = np.array(evaluated, dtype=object)

        # If we see that arr[0,0] is an array, that means vectorized input => stack
        if hasattr(arr[0, 0], "shape") and arr[0, 0].shape != ():
            # shape of each element is e.g. (N, ...)
            # We'll broadcast it into shape (N, rows, cols)
            # A simple approach: stack row by row
            stacked_rows = []
            for row in arr:
                # row is a list of arrays => shape (N,) each
                stacked_rows.append(np.stack(row, axis=-1))  # shape (N, cols)
            mat_3d = np.stack(stacked_rows, axis=-2)  # shape (N, rows, cols)
            return mat_3d.astype(float)
        else:
            # All elements are scalar
            return arr.astype(float)

    return matrix_func


def _lambdify_ndarray(nd_expr: sp.NDimArray, axes: List[sp.Symbol], parameters: Dict[str, Any]) -> Callable:
    """
    Recursively lambdify each element of a sympy N-dimensional array.
    Return a function that, given arrays for the axes, returns a
    NumPy array of shape matching the original n-dim shape in nd_expr.
    """
    shape = nd_expr.shape

    # Build a list of scalar-lambdas for each element in flatten order
    sub_exprs = list(nd_expr)  # Flatten into 1D
    sub_funcs = [_lambdify_scalar(e, axes) for e in sub_exprs]

    def array_func(*args):
        """
        Evaluate each element for the given arguments in *args,
        then reshape into the original shape of nd_expr.
        If vectorized input, we add a leading dimension for the length.
        """
        # Evaluate each scalar-lambda
        evaluated_list = [f(*args) for f in sub_funcs]  # each can be scalar or array

        # Turn into an object array of shape = flatten of shape
        arr = np.asarray(evaluated_list, dtype=object)

        # Check if we have vectorized outputs
        first_item = arr.flat[0]
        if hasattr(first_item, "shape") and first_item.shape != ():
            # We have an array in each element => build final array shape
            # e.g. if first_item.shape == (N,), final => (N, shape0, shape1, shape2, ...)
            # We'll do something similar to the matrix approach, but for N-dim
            # We'll stack along a new axis 0.
            # We can do a multi-step approach:
            #    1. gather each element into shape (N,...)
            #    2. create shape (N, *shape)
            # Then each item in arr is shape (N,).
            # So let's build a big list of flattened arrays:

            flattened_arrays = [a for a in arr.flat]
            stacked = np.stack(flattened_arrays, axis=-1)  # shape (N, <number of elements>)
            # Now reshape to (N, *original_nd_shape)
            final_nd = stacked.reshape((stacked.shape[0],) + shape)
            return final_nd.astype(float)

        # If all are scalars => just cast to float
        return arr.astype(float)

    return array_func