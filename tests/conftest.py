"""
Test suite configuration file for PyMetric.
"""
import shutil
import tempfile

import pytest


@pytest.fixture(scope="session")
def tempdir():
    """
    Session-wide scratch directory for tests that need persistent temporary storage.

    Returns
    -------
    Path
        A temporary directory path for test scratch usage.
    """
    scrap_dir = tempfile.mkdtemp(prefix="pymetric_temp_")
    yield scrap_dir
    shutil.rmtree(scrap_dir)
