"""
Type hints for Pisces-Geometry types.
"""
from typing import Literal, Tuple, Union

import numpy as np
from numpy import typing as npt

IndexType = Union[
    int,
    slice,
    Tuple[Union[int, slice, npt.NDArray[np.bool_], npt.NDArray[np.integer]], ...],
    npt.NDArray[np.bool_],
    npt.NDArray[np.integer],
]
BasisAlias = Literal["covariant", "contravariant"]
