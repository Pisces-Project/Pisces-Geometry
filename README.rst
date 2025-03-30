.. image:: docs/source/images/pisces.png
   :width: 200px
   :align: center

Pisces-Geometry
===============

+-------------------+----------------------------------------------------------+
| **Code**          | |black| |isort| |yt-project| |Pre-Commit|                |
+-------------------+----------------------------------------------------------+
| **Documentation** | |docs|                                                   |
+-------------------+----------------------------------------------------------+
| **GitHub**        | |Contributors| |Commits|                                 |
+-------------------+----------------------------------------------------------+
| **PyPi**          | Coming Soon                                              |
+-------------------+----------------------------------------------------------+

`Pisces-Geometry <https://github.com/Pisces-Project/Pisces-Geometry>`_ is a python library which forms the backend of
the more expansive astrophysical modeling software `Pisces <https://github.com/Pisces-Project/Pisces>`_. The Pisces-Geometry
library is designed to support computational physics applications in a variety of geometric contexts. By providing a highly
extensible geometry engine based on differential geometry, Pisces-Geometry makes implementing custom geometries and computing
physics calculations within them relatively simple.

Installation
------------

.. note::

    Pisces-Geometry is in active development. Report any installation or usage issues on the GitHub issues page.

Pisces-Geometry requires Python 3.8 or newer. Install from source by cloning the repository:

.. code-block:: shell

    $ pip install git+https://github.com/Pisces-Project/Pisces-Geometry

Dependencies
------------

Pisces depends on several packages, which will be installed automatically:

- `unyt <http://unyt.readthedocs.org>`_: Unit and quantity manipulations
- `numpy <http://www.numpy.org>`_: Numerical operations
- `scipy <http://www.scipy.org>`_: Interpolation and curve fitting
- `h5py <http://www.h5py.org>`_: HDF5 file interaction
- `tqdm <https://tqdm.github.io>`_: Progress bars
- `ruamel.yaml <https://yaml.readthedocs.io>`_: YAML support

Development
-----------

Pisces-Geometry is open source. Community contributions are welcome! Fork this repository to make your own modifications,
or submit an issue to suggest new features or report bugs.

Acknowledgment
--------------

If you use Pisces for academic work, please include a statement in your publication similar to:

    Initial conditions were generated using Pisces-Geometry (version), an initial conditions generator for astrophysical
    simulations, developed by Eliza Diggins and available at https://github.com/Pisces-Project/Pisces-Geometry

.. |yt-project| image:: https://img.shields.io/badge/works%20with-yt-blueviolet
   :target: https://yt-project.org

.. |Pylint| image:: https://github.com/Pisces-Project/Pisces-Geometry/actions/workflows/pylint.yml/badge.svg
   :target: https://pylint.pycqa.org/

.. |coverage| image:: https://coveralls.io/repos/github/Pisces-Project/Pisces-Geometry/pisces/badge.svg
   :target: https://coveralls.io/github/Eliza-Diggins/pisces

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen
   :target: https://eliza-diggins.github.io/pisces/build/html/index.html

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://pre-commit.com/

.. |Issues| image:: https://img.shields.io/github/issues/Pisces-Project/Pisces-Geometry
   :target: https://github.com/Eliza-Diggins/pisces/issues

.. |Contributors| image:: https://img.shields.io/github/contributors/Pisces-Project/Pisces-Geometry
   :target: https://github.com/Eliza-Diggins/pisces/graphs/contributors

.. |Commits| image:: https://img.shields.io/github/last-commit/Pisces-Project/Pisces-Geometry


.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
   :target: https://github.com/psf/black

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/


.. |NUMPSTYLE| image:: https://img.shields.io/badge/%20style-numpy-459db9
    :target: https://numpydoc.readthedocs.io/en/latest/format.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba
    :target: https://github.com/PyCQA/docformatter
