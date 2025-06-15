# PyMetric Testing: Grids

---

This testing suite targets the ``grids`` module of PyMetric and performs tests over the
following broad regimes of behavior:

- [Build]: (``./test_grids_build``) tests generation of grids.
- [IO]: (``./tests_grids_io``) tests writing and reading grids.
- [Math Operations]: (``./tests_grids_dmops``) tests dense math operations.


---

## 📐 Testing Structure

### 🔧 Fixtures Summary

List the fixtures that are available for this test module (e.g. imported from `conftest.py`, or defined locally):



### 🧰 Utilities

Summarize the helper functions in `utils.py`:

- **Grid Factories** can be used to generate a standard grid from a given
  coordinate system. There is one for each of the grid classes:
  - ``__grid_factory_UniformGrid__``
  - ``__grid_factory_GenericGrid__``

  These are bundled into ``__grid_factories__``.

- ``__grid_io_protocols__`` provides tupled parameters for each of th e
  io protocols, including the reading function, writing function, and name.

### 🧪 Parameterizations

List any `pytest.mark.parametrize` or dynamic `pytest_generate_tests` behavior used in the module:

Tests can be parameterized over

- The **grid class** using ``__all_grid_classes_params__``.
- The **IO Protocol** using ``__grid_io_protocols__``.
- The **Coordinate System** using ``cs_flag``.

---

## ✅ Tests

| File                                                   | Purpose                                        |
|--------------------------------------------------------|------------------------------------------------|
| ``./test_grids_build``                                 | tests generation of grids.                     |
| ``./tests_grids_io``                                   | tests writing and reading grids.               |
| ``./tests_grids_dmops``                                |tests dense math operations.                    |
| `utils.py`                                             | Contains test utilities and shared assertions  |

Use this table to describe what each file is testing. Stick to one feature set per file for modularity.

---

## 📝 Notes

- Always use global fixtures when available to avoid redundant setup.
- Keep test resolution small (e.g., 10×10×10) unless stress testing.
- Add new helper logic to `utils.py` when reused across multiple files.
