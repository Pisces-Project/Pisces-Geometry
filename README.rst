.. image:: docs/source/images/PyMetric.png
   :width: 300px
   :align: center

PyMetric
===============

+-------------------+----------------------------------------------------------+
| **Code**          | |black| |isort| |Pre-Commit| |Xenon|                     |
+-------------------+----------------------------------------------------------+
| **Documentation** | |docs| |NUMPSTYLE| |docformatter|                        |
+-------------------+----------------------------------------------------------+
| **GitHub**        | |Contributors| |Commits| |Tests|                         |
+-------------------+----------------------------------------------------------+
| **PyPi**          | |PyPi| |PyVersion| |Wheel| |License|                     |
+-------------------+----------------------------------------------------------+

PyMetric began as the backend for the `Pisces project <https://github.com/Pisces-Project/Pisces>`__ and has grown
into a self-contained package. It provides a seamless interface for performing coordinate-dependent operations in Python—ranging
from coordinate transformations and differential operations to solving equations of motion. In addition, it offers robust
data structures that natively respect and understand underlying coordinate systems and grid architectures, enabling efficient
handling of both curvilinear and structured grids.

Installation
------------

PyMetric requires Python 3.9 or newer. To install the most recent stable version of the code, use ``pip``:

.. code-block:: shell

    $ pip install pymetric-lib

The active development version can be obtained with

.. code-block:: shell

    $ pip install git+https://github.com/Pisces-Project/PyMetric

Dependencies
------------

PyMetric depends on several packages, which will be installed automatically:

- `numpy <http://www.numpy.org>`__: Numerical operations
- `scipy <http://www.scipy.org>`__: Interpolation and curve fitting
- `h5py <http://www.h5py.org>`__: HDF5 file interaction
- `tqdm <https://tqdm.github.io>`__: Progress bars
- `sympy <https://docs.sympy.org/latest/index.html>`__: Symbolic mathematics.

In addition to the standard versions of the codebase, there are a few special dependency versions
of the code.

- ``pymetric-lib[docs]`` will install all of the relevant Sphinx dependencies.
- ``pymetric-lib[test]`` will install all of the dependencies for running the testing suite.

Development
-----------

PyMetric is open source. Community contributions are welcome! Fork this repository to make your own modifications,
or submit an issue to suggest new features or report bugs.

Acknowledgment
--------------

If you use PyMetric for academic work, please include a statement in your publication similar to:

    This work made use of PyMetric, a geometry and mathematics framework for computational physics,
    developed by Eliza Diggins and available at https://github.com/Pisces-Project/PyMetric


.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen
   :target: https://eliza-diggins.github.io/pisces/build/html/index.html

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://pre-commit.com/

.. |Xenon| image:: https://img.shields.io/badge/Xenon-enabled-red
   :target: https://xenon.readthedocs.io/en/latest/

.. |Tests| image:: https://github.com/Pisces-Project/PyMetric/actions/workflows/run_tests.yml/badge.svg

.. |Contributors| image:: https://img.shields.io/github/contributors/Pisces-Project/PyMetric
   :target: https://github.com/Eliza-Diggins/pisces/graphs/contributors

.. |Commits| image:: https://img.shields.io/github/last-commit/Pisces-Project/PyMetric

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
   :target: https://github.com/psf/black

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/

.. |NUMPSTYLE| image:: https://img.shields.io/badge/%20style-numpy-459db9
    :target: https://numpydoc.readthedocs.io/en/latest/format.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba
    :target: https://github.com/PyCQA/docformatter

.. |License| image:: https://img.shields.io/pypi/l/pymetric-lib
.. |Wheel| image:: https://img.shields.io/pypi/wheel/pymetric-lib
.. |PyVersion| image:: https://img.shields.io/pypi/pyversions/pymetric-lib
.. |PyPi| image:: https://img.shields.io/pypi/v/pymetric-lib
