from typing import Dict, Literal, Sequence, Tuple, Union

from pymetric.fields.components import FieldComponent

ComponentIndex = Union[int, Tuple[int, ...]]
ComponentDictionary = Dict[ComponentIndex, FieldComponent]
SignatureElement = Literal[-1, 1]
SignatureInput = Union[SignatureElement, Sequence[SignatureElement]]
Signature = Tuple[SignatureElement, ...]
