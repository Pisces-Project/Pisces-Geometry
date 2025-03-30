"""
Logging utilities.
"""
import logging
import sys

# Define the PG parameters for logging.
pg_params = {
    'disable_logger': False,
    'logger_level': logging.DEBUG,
    'skip_constant_checks': 5,
}
""" dict: Logging parameters for Pisces-Geometry.
"""

# Set up the logger with the correct formatting and
# format.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(name)-3s : [%(levelname)-9s] %(asctime)s %(message)s"))
pg_log = logging.getLogger('pisces_geometry')
""" Logger: The default logger for Pisces-Geometry."""

pg_log.setLevel(pg_params['logger_level'])
pg_log.disabled = pg_params['disable_logger']
pg_log.addHandler(_handler)





