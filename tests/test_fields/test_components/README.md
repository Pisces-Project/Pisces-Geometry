# PyMetric Testing: ``fields.components``

---

This testing suite targets the ``fields.components`` of PyMetric and performs tests over the
following broad regimes of behavior:

- **build**: Check that components can be built correctly using constructors and functions.

---

## 📐 Testing Structure

### 🔧 Fixtures Summary

Fixtures used in this module:

- `coordinate_systems`: Dictionary of all initialized coordinate systems filtered by the `--coord-systems` flag.
- `uniform_grids`: Dictionary of `UniformGrid` instances constructed from each coordinate system.
- `component_test_directory`: Temporary directory used for buffer tests (especially HDF5-backed).
- [Other fixtures defined in `utils.py` or locally]

### 🧰 Utilities

Helper functions defined in `utils.py`:

- `__func_ArrayBuffer_args_kwargs_factory__`: Returns empty args/kwargs for constructing `ArrayBuffer`.
- `__func_HDF5Buffer_args_kwargs_factory__`: Generates args/kwargs for HDF5-backed buffer from file path.
- `__from_func_args_factories__`: Maps buffer classes to their corresponding argument factories.

### 🧪 Parameterizations

This test module uses the following parameterizations:

- `@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)`: Tests logic across all registered buffer classes (e.g., `ArrayBuffer`, `HDF5Buffer`).
- `@pytest.mark.parametrize("method", ["ones", "zeros", "full", "empty"])`: Evaluates standard constructors in `FieldComponent`.
- `@pytest.mark.parametrize("cs_flag", ...)`: Dynamically selects coordinate systems through `cs_flag` fixture.

---

## ✅ Tests

| File                         | Purpose                                                  |
|------------------------------|----------------------------------------------------------|
| `test_components_build.py`   | Tests construction of `FieldComponent`                   |
| `utils.py`                   | Provides reusable fixtures and buffer argument factories |

---

## 📝 Notes

- Prefer global fixtures like `uniform_grids` and `coordinate_systems` to avoid repeated setup.
- Use small domains (e.g. 10×10×10) for faster runtime unless explicitly stress testing.
- Validate output arrays using `assert_allclose` against true analytic expressions.
- Extend `utils.py` for additional backend testing (e.g., `UnytBuffer`) or format compatibility.

---
