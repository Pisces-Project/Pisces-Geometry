# PyMetric Testing: Buffers

---

This testing suite targets the **buffer system** of PyMetric and performs tests over the 
following broad regimes of behavior:

- `test_buffer_creation.py`: Verifies buffer instantiation using `from_array()` for all buffer classes.
- `test_buffer_numpy_ops.py`: Validates NumPy ufunc compatibility and method overrides (e.g. `reshape`, `astype`) for all buffer types.

---

## 📐 Testing Structure

### 🔧 Fixtures Summary

The following fixtures are used across this module:

- `test_array`: A `2×2` NumPy array of ones, used as the base test input for buffer creation.
- `buffer_test_directory`: A temporary directory for safe creation of files used by `HDF5Buffer`.

These are defined in `utils.py`.

### 🧰 Utilities

The `utils.py` file provides the following helper logic:

- `__all_buffer_classes__`: List of available buffer classes to test, currently includes `ArrayBuffer` and `HDF5Buffer`.
- `__from_array_args_factories__`: Dictionary mapping buffer class → factory function to generate `from_array()` arguments.
- `__all_numpy_builtin_methods__`: A curated set of NumPy-like methods (e.g., `reshape`, `astype`, etc.) to test via parametrize.

These utilities enable generic and reusable parameterized tests across different buffer backends.

### 🧪 Parameterizations

- `@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)`: Ensures all tests run over both `ArrayBuffer` and `HDF5Buffer`.
- `@pytest.mark.parametrize("ufunc", [...])`: Applies various NumPy ufuncs (e.g., `np.add`, `np.sqrt`) to verify array interface behavior.
- `@pytest.mark.parametrize("method_name,args,kwargs", __all_numpy_builtin_methods__)`: Tests each supported transformation method under both buffer-return and NumPy-return modes.

---

## ✅ Tests

| File                         | Purpose                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `test_buffer_creation.py`    | Validates that all buffer types correctly construct from arrays         |
| `test_buffer_numpy_ops.py`   | Verifies NumPy ufunc compatibility and buffer method consistency        |
| `utils.py`                   | Contains parameter definitions, fixtures, and argument factories        |

---

## 📝 Notes

- Always use fixtures from `utils.py` and avoid hardcoding test arrays or paths.
- Each test file focuses on a specific axis of buffer behavior (e.g., I/O, NumPy ops).
- Add new buffer types to `__all_buffer_classes__` and define a corresponding `from_array` factory.
- New method coverage should be added to `__all_numpy_builtin_methods__`.

To run just this test suite:

```bash
pytest ./tests/test_fields/test_buffers/
```