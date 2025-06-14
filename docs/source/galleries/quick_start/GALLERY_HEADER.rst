:orphan:

.. image:: ../images/PyMetric.png
   :width: 300px
   :align: center

.. _quickstart:
==========================
PyMetric Quickstart Guide
==========================

Welcome to the **PyMetric Quickstart Guide**!

This guide helps you quickly install, configure, and begin using **PyMetric**. Pymetric is a
flexible framework for geometry-aware scientific computing. Whether you're installing
it for development, documentation, or basic usage, this guide will get you up and running.

.. contents::
   :local:
   :depth: 2


.. _installation:
Installation
------------

Currently, PyMetric is hosted on PyPI and on Github. For standard installations,
we suggest installing the stable package version from PyPI using pip or conda. To
install the development version of the package, you can install directly from the source
code on github:

.. tab-set::

   .. tab-item:: 📦 PyPI (Stable)

      The recommended way to install **PyMetric** is via PyPI:

      .. code-block:: bash

         pip install pymetric-lib

      This installs the latest stable release, suitable for most users. Additional
      options are available (see advanced installation).

   .. tab-item:: 🧪 Development (GitHub)

      To install the latest development version directly from GitHub:

      .. code-block:: bash

         pip install git+https://github.com/pisces-project/pymetric

      Alternatively, clone and install locally:

      .. code-block:: bash

         git https://github.com/pisces-project/pymetric
         cd pymetric
         pip install -e .

      This is the suggested approach if you are intent on developing the
      code base.

   .. tab-item:: 📚 Conda (Experimental)

      If you're using **Conda**, you can install via `pip` in a conda environment:

      .. code-block:: bash

         conda create -n pymetric-env python=3.11
         conda activate pymetric-env
         pip install pymetric

      (A native conda package is not yet maintained, so this uses pip within conda.)

You can check that installation has worked correctly by running

.. code-block:: bash

    $ pip show pymetric
    Name: pymetric
    Version: <the installed version>
    Summary: A high-performance library for structured differential geometry and physical field manipulation.
    Home-page:
    Author:
    Author-email: Eliza Diggins <eliza.diggins@berkeley.edu>
    License: GPL-3.0-or-later
    Location: [source location]
    Requires: h5py, matplotlib, numpy, scipy, sympy, tqdm
    Required-by:

Dependencies
++++++++++++

PyMetric strives to use a minimal set of dependencies to streamline usage without
requiring a lot of effort to install. Below is the list of required dependencies:

+----------------+-----------+--------------------------------------------+
| Package        | Version   | Description                                |
+================+===========+============================================+
| numpy          | >=1.22    | Core numerical array processing            |
+----------------+-----------+--------------------------------------------+
| scipy          | >=1.10    | Scientific computing and numerical tools   |
+----------------+-----------+--------------------------------------------+
| h5py           | >=3.0     | HDF5 file format support                   |
+----------------+-----------+--------------------------------------------+
| sympy          | >=1.14.0  | Symbolic mathematics and algebra           |
+----------------+-----------+--------------------------------------------+
| matplotlib     | any       | Plotting and visualization                 |
+----------------+-----------+--------------------------------------------+
| tqdm           | any       | Progress bars for loops and scripts        |
+----------------+-----------+--------------------------------------------+

In addition, a number of additional dependency groups are available for more
advanced needs. Specifically,

PyMetric supports several **optional dependency groups** for specific workflows:

.. tab-set::

   .. tab-item:: 🧪 Development `[dev]`

      To install:

      .. code-block:: bash

         pip install pymetric-lib[dev]

      Includes tools for formatting, linting, and development workflows.

      +----------------+---------------------------+
      | Package        | Purpose                   |
      +================+===========================+
      | pytest         | Test framework            |
      +----------------+---------------------------+
      | pytest-cov     | Test coverage reporting   |
      +----------------+---------------------------+
      | black          | Code formatter            |
      +----------------+---------------------------+
      | mypy           | Static type checker       |
      +----------------+---------------------------+
      | pre-commit     | Git hook management       |
      +----------------+---------------------------+
      | jupyter        | Interactive notebooks     |
      +----------------+---------------------------+

   .. tab-item:: 📚 Documentation `[docs]`

      To install:

      .. code-block:: bash

         pip install pymetric-lib[docs]

      Includes packages required to build, style, and preview documentation.

      +------------------------------+-------------------------------------------+
      | Package                      | Purpose                                   |
      +==============================+===========================================+
      | sphinx                       | Core documentation generator              |
      +------------------------------+-------------------------------------------+
      | numpydoc                     | NumPy-style docstring parser              |
      +------------------------------+-------------------------------------------+
      | myst-parser                  | Markdown support via MyST                 |
      +------------------------------+-------------------------------------------+
      | sphinx-gallery               | Auto-build galleries from example scripts |
      +------------------------------+-------------------------------------------+
      | sphinx-design                | Responsive design components (tabs, etc.) |
      +------------------------------+-------------------------------------------+
      | jupyter                      | Notebook integration                      |
      +------------------------------+-------------------------------------------+
      | sphinxcontrib-*              | Various builder integrations (HTML, Qt)   |
      +------------------------------+-------------------------------------------+

   .. tab-item:: 🧪 Testing `[test]`

      To install:

      .. code-block:: bash

         pip install pymetric-lib[test]

      A minimal environment to run the test suite and property-based tests.

      +----------------+------------------------------+
      | Package        | Purpose                      |
      +================+==============================+
      | pytest         | Core test runner             |
      +----------------+------------------------------+
      | pytest-xdist   | Parallel test execution      |
      +----------------+------------------------------+
      | pytest-cov     | Test coverage metrics        |
      +----------------+------------------------------+
      | hypothesis     | Property-based testing       |
      +----------------+------------------------------+



Getting Help
------------

If you encounter issues using **PyMetric**, or have questions about its functionality:

- 💬 **Search or open an issue** on our `GitHub issue tracker <https://github.com/pisces-project/pymetric/issues>`_.
- 📧 **Contact us directly** by emailing `eliza.diggins@berkeley.edu <mailto:eliza.diggins@berkeley.edu>`__ for questions,
  bug reports, or suggestions.
- 📖 Refer to the full documentation for API details, examples, and conceptual guides.

We’re happy to help you resolve installation problems, clarify behavior, or explore new use cases!

Help Develop PyMetric
---------------------

Contributions are welcome and encouraged! Whether you're fixing typos, adding examples, writing tests, or developing new features,
you can help improve **PyMetric** for everyone.

To setup PyMetric for development, start by creating a fork of the repository in your own
github. In your own branch, Identify the **current development branch** of the code. This is
the branch with name ``dev-v...``. This branch will eventually be merged into
the production branch. You should create a new branch from the development branch with a name
indicating the specific features / fixes you're working on.

To set up the development environment, you should first create a virtual environment

.. code-block:: python

    python -m venv ./.venv

In that venv, install the development and documentation requirements with

.. code-block:: python

    pip install -e ./pymetric[dev]

Ensure that ``precommit`` is installed and configure it on the git with

.. code-block:: python

    pre-commit install

.. hint::

    To streamline this, we provide a makefile which automates these processes. To use this approach,
    modify the ``Makefile`` to correctly point to your base python:

    .. code-block:: makefile

        # ----------------------- #
        # Metadata for PYTHON     #
        # ----------------------- #
        # These settings MAY need to be modified by new
        # users in order to get everything working vis-a-vis
        # the make ... command style.
        #
        # If you're just a user, you DON'T want to be here. You should
        # install via pip install pymetric instead.
        # The python command from which to build the venv
        PYTHON := python3
        # Directory to build the .venv in.
        VENV_DIR := .venv


    Once configured, you can simply do the following:

    .. code-block:: bash

        $ make venv-build
        $ make dev-branch
        $ make precommit-install


You're now ready to start implementing new features!


Once you've implemented the features you want to include, there are two things that need
to be done:

1. Create the relevant issues in the `GitHub issue tracker <https://github.com/pisces-project/pymetric/issues>`_.

   - In the comments, note that you're working on development and link your forked repository.

2. Create a pull request to merge your development branch into the current development branch.

   Before we will accept a pull-request, the following must all be working

   1. 🧼 Run formatting and lint checks with pre-commit.
   2. 🧪 Run the test suite:

      .. code-block:: bash

            make test

   3. 📚 Build the documentation locally:

      .. code-block:: bash

          make docs

If you’re not sure where to start, check the
`GitHub issues <https://github.com/pisces-project/pymetric/issues>`__ labeled "**good first issue**" or feel
free to ask questions by opening a discussion or emailing the maintainer directly `here <eliza.diggins@berkeley.edu>`__.
We’d love your help building a powerful, flexible tool for computational geometry and physical modeling!
