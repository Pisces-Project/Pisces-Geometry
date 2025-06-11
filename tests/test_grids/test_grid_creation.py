"""
Pytesting module for checking grid creation semantics and functionality.

For each of the grid subclasses, we have a creation `factory` which is written
in this module and which comprises the core of the test. The `factory` must take a
`coordinate_system` input as its first argument and return a grid object generated
from that coordinate system.
"""
import pytest
from .utils import __grid_factories__, __all_grid_classes_params__



# =============================================== #
# Generator Tests                                 #
# =============================================== #
@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)
def test_grid_construction(
    cs_flag, grid_class, coordinate_systems
):
    # Run the grid class factory to create the grid.
    _ = __grid_factories__[grid_class](coordinate_systems[cs_flag])



