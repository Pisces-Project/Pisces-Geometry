"""
Modular mix-in classes for enhancing grid functionality / organization in Pisces Geometry.

This submodule defines reusable mixins that extend the behavior of grid classes without
requiring deep inheritance hierarchies. Additionally, all grid mathematics (including support
for both dense and sparse tensor representations) is present in the mathematics mixin modules.
"""
from .chunking import GridChunkingMixin
from .core import GridIOMixin, GridPlotMixin, GridUtilsMixin
from .mathops import DenseMathOpsMixin
