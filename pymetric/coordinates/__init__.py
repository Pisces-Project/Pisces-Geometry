"""
Coordinate system support in PyMetric.

The :py:mod:`coordinates` module handles all of the different coordinate systems inPyMetric as well
as the backend theory, symbolic manipulations, etc. required to make each system usable for physical systems. For a
detailed guide to this module and its contents, see :ref:`coordinates_user`.
"""
__all__ = [
    "CartesianCoordinateSystem1D",
    "CartesianCoordinateSystem2D",
    "CartesianCoordinateSystem3D",
    "SphericalCoordinateSystem",
    "PolarCoordinateSystem",
    "CylindricalCoordinateSystem",
    "OblateSpheroidalCoordinateSystem",
    "ProlateSpheroidalCoordinateSystem",
    "EllipticCylindricalCoordinateSystem",
    "OblateHomoeoidalCoordinateSystem",
    "ProlateHomoeoidalCoordinateSystem",
    "ConicCoordinateSystem",
    "OrthogonalCoordinateSystem",
    "CurvilinearCoordinateSystem",
]
from .coordinate_systems import (
    CartesianCoordinateSystem1D,
    CartesianCoordinateSystem2D,
    CartesianCoordinateSystem3D,
    ConicCoordinateSystem,
    CylindricalCoordinateSystem,
    EllipticCylindricalCoordinateSystem,
    OblateHomoeoidalCoordinateSystem,
    OblateSpheroidalCoordinateSystem,
    PolarCoordinateSystem,
    ProlateHomoeoidalCoordinateSystem,
    ProlateSpheroidalCoordinateSystem,
    SphericalCoordinateSystem,
)
from .core import CurvilinearCoordinateSystem, OrthogonalCoordinateSystem
