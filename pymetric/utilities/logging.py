"""
Logging utilities.
"""
import logging
import sys

from pymetric.utilities.config import pg_params

# Set up the logger with the correct formatting and
# format.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(name)-3s : [%(levelname)-9s] %(asctime)s %(message)s")
)
pg_log = logging.getLogger("pymetric")
""" Logger: The default logger for Pisces-Geometry."""

pg_log.setLevel(pg_params["logger_level"])
pg_log.disabled = pg_params["disable_logger"]
pg_log.addHandler(_handler)
