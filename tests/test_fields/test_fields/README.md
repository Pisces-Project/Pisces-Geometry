# PyMetric Testing: `test_fields.test_fields`

---

This testing suite targets the **`DenseField` class and NumPy compatibility layer** of PyMetric and performs tests over the following broad regimes of behavior:

- **Construction**: Validates that `DenseField` objects can be instantiated using `zeros`, `ones`, `full`, and `empty` constructors across buffer types.
- **Function Evaluation**: Checks that `DenseField.from_function` correctly populates fields using user-defined mathematical expressions over the grid.
- **NumPy Interoperability**: Ensures that `DenseField` supports NumPy ufuncs (`add`, `multiply`, `sqrt`, etc.) and behaves predictably with `out=` arguments and type preservation.

---

## 📐 Testing Structure

### 🔧 Fixtures Summary

Available fixtures for this module:

- `coordinate_systems`: Dictionary of initialized coordinate systems filtered by `--coord-systems`.
- `uniform_grids`: Dictionary of `UniformGrid` instances created from each coordinate system.
- `dense_test_directory`: Temporary directory for I/O tests, especially HDF5-backed buffers.

### 🧰 Utilities

Helper functions defined in `utils.py`:

- `__func_ArrayBuffer_args_kwargs_factory__`: Provides dummy args/kwargs for `ArrayBuffer` construction.
- `__func_HDF5Buffer_args_kwargs_factory__`: Returns file-based args/kwargs for `HDF5Buffer` using a temp directory.
- `__from_func_args_factories__`: Maps buffer types to their respective argument factories.
- `__all_numpy_builtin_methods__`: (Future use) List of common NumPy method parameters for testing field method mirrors.

---

## 🧪 Parameterizations

This test module uses the following parameterized configurations:

- `@pytest.mark.parametrize("buffer_class", __all_buffer_classes__)`: Tests across `ArrayBuffer` and `HDF5Buffer`.
- `@pytest.mark.parametrize("method", ["ones", "zeros", "full", "empty"])`: Covers all core DenseField constructors.
- `@pytest.mark.parametrize("ufunc", [...])`: Applies ufuncs like `np.add`, `np.multiply`, `np.sqrt`, etc. to field instances.

---

## ✅ Tests

| File                      | Purpose                                                                      |
|---------------------------|-------------------------------------------------------------------------------|
| `test_dense_build.py`     | Validates construction of `DenseField` via constructors and `from_function`  |
| `test_dense_numpy_ops.py` | Tests support for NumPy ufuncs (e.g., `np.add`, `np.abs`) with and without `out=` |
| `utils.py`                | Contains fixture and factory helpers for buffer setup and I/O configuration |

---

## 📝 Notes

- All tests use small grid sizes and scalar fields to reduce runtime overhead.
- When testing ufuncs, results are compared between buffer-backed output and NumPy-native evaluation.
- This module does not yet test reduction ufuncs (e.g., `np.sum`) or axis-based behavior—those should be added separately.
- Add coverage for broadcasting, type promotion, and slicing semantics as the DenseField class evolves.
