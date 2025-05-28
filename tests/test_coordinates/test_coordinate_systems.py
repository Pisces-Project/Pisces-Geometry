"""
Testing suite for the various coordinate systems in PyMetric.
"""
import pytest


def test_coordinate_system_initialization(cs_flag, coordinate_systems):
    # Check basic parameters of each of the coordinate systems.
    cs = coordinate_systems[cs_flag]

    # Assert on basic attributes.
    assert hasattr(cs, "__AXES__"), "No __AXES__"
    assert hasattr(cs, "__PARAMETERS__"), "No __PARAMETERS__"
    assert cs.ndim == len(cs.__AXES__)

    # Ensure that we can retrieve relevant symbols.
    __symbols_checked__ = [
        "metric_tensor",
        "inverse_metric_tensor",
        "Lterm",
        "Dterm",
        "metric_density",
    ]
    for _sym_ in __symbols_checked__:
        try:
            _ = cs.get_expression(_sym_)
        except:
            raise ValueError(f"Missing required symbol ({cs_flag},{_sym_}).")
