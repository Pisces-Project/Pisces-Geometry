
Pisces-Geometry
===============

|isort| |black| |Pre-Commit| |docformatter| |NUMPSTYLE| |COMMIT| |CONTRIBUTORS| |docs|

.. raw:: html

   <hr style="height:2px;background-color:black">

The Pisces-Geometry package was originally developed as part of the backend for the `Pisces <https://github.com/Pisces-Project/Pisces>`_ project but has since evolved
into a fully self-contained codebase. It provides a seamless interface for performing coordinate-dependent operations in Python,
including coordinate transformations, differential operations, and equation-of-motion solutions.

Additionally, Pisces-Geometry enables the construction of data structures that inherently respect and understand an
underlying coordinate system and grid structure, ensuring efficient and intuitive handling of curvilinear and
structured grids.

.. raw:: html

   <hr style="color:black">

Installation
============

Pisces-Geometry is written in Python 3.8 and is compatible with Python 3.8+ with continued support for older versions of Python.
For instructions on installation and getting started, check out the :ref:`getting_started` page.

Contents
========
.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   api
   reference/index
   examples
   getting_started


Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |yt-project| image:: https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet"
   :target: https://yt-project.org

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen
   :target: https://eliza-diggins.github.io/Pisces

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
   :target: https://github.com/psf/black

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. |CONTRIBUTORS| image:: https://img.shields.io/github/contributors/eliza-diggins/Pisces
    :target: https://github.com/eliza-diggins/Pisces/graphs/contributors

.. |COMMIT| image:: https://img.shields.io/github/last-commit/eliza-diggins/Pisces

.. |NUMPSTYLE| image:: https://img.shields.io/badge/%20style-numpy-459db9
    :target: https://numpydoc.readthedocs.io/en/latest/format.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba
    :target: https://github.com/PyCQA/docformatter
