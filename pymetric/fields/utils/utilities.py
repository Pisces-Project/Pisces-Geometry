"""
Utility functions for fields.
"""
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from ._typing import Signature, SignatureInput


def validate_rank_signature(
    rank: int, signature: Optional["SignatureInput"] = None
) -> "Signature":
    """
    Validate that a tensor signature is consistent with a given tensor rank. If it is not provided,
    create a fully contravariant signature.

    Parameters
    ----------
    rank : int
        The rank of the tensor (i.e., number of element-wise dimensions).
    signature : int or list of int, optional
        A sequence of values indicating index variance:

        - `1` for contravariant (upper index)
        - `-1` for covariant (lower index)

    Returns
    -------
    tuple of int
        The corrected / standardized tensor signature.

    Raises
    ------
    ValueError
        If the signature length does not match the rank, or if any value in the
        signature is not -1 or 1.
    """
    if signature is None:
        return (1,) * rank

    if isinstance(signature, int):
        if signature not in (-1, 1):
            raise ValueError(f"Signature value must be 1 or -1, got: {signature}")
        return (signature,) * rank

    # Now assume it's a sequence
    signature = tuple(signature)
    if len(signature) != rank:
        raise ValueError(
            f"Signature length {len(signature)} does not match tensor rank {rank}."
        )
    if any(s not in (-1, 1) for s in signature):
        raise ValueError(
            f"Signature values must be either 1 (contravariant) or -1 (covariant), got: {signature}."
        )

    return signature


def signature_to_tensor_class(signature: "SignatureInput") -> Tuple[int, int]:
    """
    Convert a tensor signature to its :math:`(p, q)` form.

    The :math:`(p, q)` notation describes the number of contravariant (upper) and
    covariant (lower) indices in a tensor:

    - `p`: number of `1`s in the signature (contravariant indices)
    - `q`: number of `-1`s in the signature (covariant indices)

    Parameters
    ----------
    signature : sequence of int
        The tensor signature, typically a tuple of `1` and `-1`.

    Returns
    -------
    tuple of int
        A tuple `(p, q)` where:

        - `p` is the count of contravariant indices
        - `q` is the count of covariant indices

    Raises
    ------
    ValueError
        If the signature contains values other than `1` or `-1`.
    """
    # Validate the signature.
    signature = validate_rank_signature(signature)

    # Compute the sums.
    p = sum(1 for s in signature if s == 1)
    q = sum(1 for s in signature if s == -1)
    return p, q
