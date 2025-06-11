# PyMetric Test Suite

This directory contains the complete test suite for the **PyMetric** library. It is organized into modular
subdirectories that mirror the core structure of the codebase, with dedicated testing for coordinates, differential
geometry, grids, fields, and buffer operations.

This suite uses **[`pytest`](https://docs.pytest.org/)** as the primary test runner and supports a wide
range of symbolic, numerical, and structural validation tests.

---

## 📁 Module Structure

The testing module is broken down into a number of submodules, each responsible
for testing a specific part of the larger library. Within the leafs of the directory
structure, there are typically

- A number of ``test___.py`` to test different aspects of the code.
- A ``README.md`` with specific instructions on how the testing in that module behaves.
- A ``utils.py`` file with testing configuration / utility functions for the testing.
- A ``__init__.py`` which is blank to allow importing of the ``units.py`` file.

The overall directory is organized as follows:

```text

tests/
├── conftest.py                 # Global fixtures and pytest configuration
├── README.md                   # This file
│
├── test_differential_geometry/ # Tests the ``differential_geometry`` module.
│   ├── test_dense_ops_utils.py
│   ├── test_dependence.py
│   ├── test_symbolic.py
|
├── test_coordinates/           # Tests the ``coordinates`` module.
│   ├── README.md
│   ├── test_coordinates_build.py
│   ├── test_coordinates_io.py
│   ├── utils.py
│
├── test_grids/                # Tests the ``grids`` module.
│   ├── README.md
│   ├── test_grid_creation.py
│   ├── test_grid_io.py
│   ├── utils.py
│
├── test_fields/               # Tests for fields, buffers, components
│   ├── test_buffers/
│   │   ├── test_buffer_creation.py
│   │   ├── test_buffer_numpy_ops.py
│   │   └── README.md
│   ├── test_components/
│   │   ├── test_comp_creation.py
│   │   ├── test_comp_semantics.py
│   │   └── README.md
│   ├── test_fields/
│   │   ├── test_dense_fields.py
|   |   └── README.md
│   └── test_mathops/          # (Reserved for field-level math ops)
```

The ``README`` files in each subdirectory provide more detail about exactly what
the different testing files contain.

## Global Configuration

The PyMetric test suite uses a centralized configuration system defined in ``tests/conftest.py`` to manage fixtures,
coordinate system registration, and custom pytest behavior. This configuration enables parameterized testing across
all supported coordinate systems and provides reusable grid and geometry fixtures for consistent and flexible test behavior.

### Coordinate System Registry

All supported coordinate systems are defined in a global dictionary:

.. code-block:: python

    __pymetric_all_coordinate_systems__ = {
        "cartesian2D": (CartesianCoordinateSystem2D, [[0, 0], [1, 1]]),
        "spherical": (SphericalCoordinateSystem, [[0, 0, 0], [1, π, 2π]]),
        ...
    }

Each entry includes:

- A constructor or class for the coordinate system
- A default bounding box used when constructing a ``UniformGrid``

This registry is used to initialize test cases and fixtures automatically, and can be filtered at runtime.

### CLI Options for pytest

You can control which coordinate systems are included in test runs using the ``--coord-systems`` option:

.. code-block:: bash

    pytest --coord-systems=all                   # (default) test all systems
    pytest --coord-systems=spherical,cartesian2D

This allows you to selectively test subsets of the system during debugging or performance runs.

### Fixtures Provided

The configuration provides the following reusable fixtures:

- ``cs_flag`` – A parameterized flag corresponding to each selected coordinate system (used internally for test generation)
- ``coordinate_system_flag`` – A session-scoped fixture that returns the value of ``--coord-systems``
- ``coordinate_systems`` – A dictionary mapping each selected coordinate system name to an initialized object
- ``uniform_grids`` – A dictionary of ``UniformGrid`` instances constructed from each selected coordinate system and bounding box

These fixtures can be used in any test module by simply declaring them in the function signature:

.. code-block:: python

    def test_bbox_shape_match(uniform_grids):
        for name, grid in uniform_grids.items():
            assert grid.shape == (10,) * grid.ndim

Using these shared fixtures helps ensure consistency across test modules and avoids redundant construction logic.

## Example Usage

Run all tests on the default grid set:

.. code-block:: bash

    pytest

Run only spherical and cylindrical coordinate system tests:

.. code-block:: bash

    pytest --coord-systems=spherical,cylindrical

Filter by test name and coordinate system:

.. code-block:: bash

    pytest -k "test_metric_properties" --coord-systems=polar

This configuration enables modular, efficient, and easily debuggable test runs across the full geometry engine.

## 🧭 Submodule README Template
Each testing submodule in the tests/ directory should contain a ``README.md`` file that describes the structure,
contents, and intended coverage of that submodule. These READMEs help contributors understand what is being
tested and where to add new tests.

``README.md`` files should follow the format of ``README_SKELETON.md``.

##  Writing Custom Tests

When contributing new tests to the PyMetric suite, follow these best practices
to ensure **consistency**, **modularity**, and **speed**.

### Use Shared Fixtures

Always use the fixtures defined in `conftest.py` for coordinate systems and grids:

```python
def test_centered_grid_shape(uniform_grids):
    for name, grid in uniform_grids.items():
        assert grid.center == "cell"
```

>  **Avoid** recreating coordinate systems or grids from scratch in each test.

### ⚡ Keep It Fast

- Tests should be **performant** and suitable for rapid iteration.
- Use **small, fixed-resolution** domains (e.g., `10×10×10`) by default.
- **Avoid symbolic simplification** during tests unless it’s essential for correctness.

### Use `utils.py` for Boilerplate

Each test submodule should provide a `utils.py` file containing shared test logic and helpers.
Centralizing repeated code improves readability and maintainability.

```python
# In test_grids/utils.py
def check_shape(grid, expected_shape):
    assert grid.shape == expected_shape
```

> Reuse common routines across files instead of duplicating test logic.

### Target Specific Behavior

Each `test_*.py` file should validate **one coherent behavioral domain**. For example:

| File                             | Purpose                                                |
|----------------------------------|--------------------------------------------------------|
| `test_grid_creation.py`         | Grid instantiation and domain layout validation        |
| `test_comp_semantics.py`        | Component broadcasting and axis matching logic         |
| `test_coordinate_system_io.py`  | I/O serialization and `.from_metadata_dict()` tests    |

> Do not mix unrelated behavior (e.g., IO + math ops) in a single file.

### Use Parametrization

Use `pytest.mark.parametrize` or coordinate system fixtures to increase coverage **without duplicating test code**.

```python
@pytest.mark.parametrize("center", ["cell", "vertex"])
def test_centering_behavior(center, coordinate_systems):
    ...
```
