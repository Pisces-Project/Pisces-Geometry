from typing import Literal, Tuple, Union

import numpy as np
import numpy.typing as npt

BasisAlias = Literal["unit", "covariant", "contravariant"]
IndexType = Union[
    int,
    slice,
    Tuple[Union[int, slice, npt.NDArray[np.bool_], npt.NDArray[np.integer]], ...],
    npt.NDArray[np.bool_],
    npt.NDArray[np.integer],
]
