.. image:: ./images/PyMetric.png
   :width: 300px
   :align: center

.. _getting_started:

=====================
Quick-start Guide
=====================

Welcome to **PyMetric**!  This page gets you from zero to useful in a
few minutes.

.. contents::
   :local:
   :depth: 2


Installation
------------

PyMetric is published on **PyPI** and can be installed like any other
Python library:

.. code-block:: bash

   $ pip install pymetric-lib

If you prefer the bleeding edge, install directly from source:

.. code-block:: bash

   $ git clone https://github.com/Pisces-Project/PyMetric
   $ cd PyMetric
   $ pip install .

Requirements
^^^^^^^^^^^^

PyMetric is pure-Python but relies on a small scientific-stack:

+-------------------+---------------+
| **Package**       | **Version**   |
+===================+===============+
| Python            | ≥ 3.9         |
+-------------------+---------------+
| NumPy             | ≥ 1.25        |
+-------------------+---------------+
| SciPy             | ≥ 1.11        |
+-------------------+---------------+
| unyt (optional)   | ≥ 2.9         |
+-------------------+---------------+
| h5py (optional)   | ≥ 3.10        |
+-------------------+---------------+

The *optional* dependencies are only needed for unit-aware buffers
(`unyt`) or on-disk HDF5 storage (`h5py`).  Pip handles everything
automatically when you install with the ``[full]`` extra:

.. code-block:: bash

   $ pip install "pymetric[full]"


Getting Started
---------------



Cookbooks
----------




Getting Help
------------

* **Documentation** – https://pymetric.readthedocs.io
* **Discussion / Q&A** – GitHub *Discussions* tab
  *(short how-to questions are welcome)*
* **Bugs / feature requests** – open an issue at
  https://github.com/Pisces-Project/PyMetric/issues


Contributing to PyMetric
------------------------

Pull requests are warmly welcomed!  To get started:

1. Fork the repository and create a topic branch.
2. Install the dev dependencies:

   .. code-block:: bash

      $ pip install -r requirements-dev.txt

3. Run the test-suite (``pytest``) and static checks (``ruff``).
4. Commit with conventional messages and open a PR against ``main``.
5. Make sure **all** CI checks pass; the reviewers will take it from
   there.

For detailed guidelines, see :doc:`/developer_guide/contributing`.
