"""Pisces-Geometry configuration management and logging utilities.

This module defines global configuration parameters used throughout Pisces-Geometry.
These include logger control, logging verbosity, and certain performance or validation
tuning knobs.

You can import and modify the `pg_params` dictionary to customize behavior at runtime.

Example:
    >>> from pisces_geometry.config import pg_params
    >>> pg_params["logger_level"] = logging.WARNING
"""

import logging

#: Default Pisces-Geometry configuration parameters.
#:
#: - ``disable_logger`` (bool): If True, disables all logging output.
#: - ``logger_level`` (int): The default Python logging level.
#: - ``skip_constant_checks`` (int): Max elements to skip when checking if an array is constant.
pg_params = {
    "disable_logger": False,  # Disable all logging output if True
    "logger_level": logging.DEBUG,  # Default logging level
    "skip_constant_checks": 5,  # Max number of elements to skip when checking for constancy
}
