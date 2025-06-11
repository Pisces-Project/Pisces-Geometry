# PyMetric Testing: [Module Name]

---

This testing suite targets the **[module]** of PyMetric and performs tests over the
following broad regimes of behavior:

- [Test Area 1]: [Short description of what this file tests]
- [Test Area 2]: [Short description]
- [Test Area 3]: [Etc.]

Replace the placeholders above with actual test names and functions.

---

## 📐 Testing Structure

### 🔧 Fixtures Summary

List the fixtures that are available for this test module (e.g. imported from `conftest.py`, or defined locally):

- `coordinate_systems`: Dictionary of all initialized coordinate systems filtered by the `--coord-systems` flag.
- `uniform_grids`: Dictionary of `UniformGrid` instances constructed from each coordinate system.
- [Other fixtures defined in `utils.py` or locally]

### 🧰 Utilities

Summarize the helper functions in `utils.py`:

- `make_dummy_grid(...)`: Creates a basic test grid with arbitrary metadata
- `check_metadata_roundtrip(...)`: Confirms that I/O functions are reversible
- [Etc.]

### 🧪 Parameterizations

List any `pytest.mark.parametrize` or dynamic `pytest_generate_tests` behavior used in the module:

- Parametrization over `center=["cell", "vertex"]` for grid centering logic
- Parametrization over `cs_flag` for coordinate system-specific logic
- [Etc.]

---

## ✅ Tests

| File                          | Purpose                                                          |
|-------------------------------|------------------------------------------------------------------|
| `test_example_behavior.py`    | Validates [X] functionality of [module]                          |
| `test_io.py`                  | Tests JSON, YAML, and HDF5 metadata serialization and parsing    |
| `test_symbolic.py`            | Tests symbolic representations and metric simplification         |
| `utils.py`                    | Contains test utilities and shared assertions                   |

Use this table to describe what each file is testing. Stick to one feature set per file for modularity.

---

## 📝 Notes

- Always use global fixtures when available to avoid redundant setup.
- Keep test resolution small (e.g., 10×10×10) unless stress testing.
- Add new helper logic to `utils.py` when reused across multiple files.
