# PyMetric Testing: test_coordinates

---

This testing suite targets the **coordinates** module of PyMetric and performs tests over the
following broad regimes of behavior:

- Coordinate System Construction: Validates instantiation and structural attributes of all supported coordinate systems.
- Coordinate System I/O: Tests YAML, JSON, and HDF5 serialization and deserialization routines for correctness.
- Coordinate System Metadata: Ensures coordinate systems preserve internal fields and properties on roundtrip.

---

## 📐 Testing Structure

### 🔧 Fixtures Summary

- `coordinate_system_flag`: Returns the value of the `--coord-systems` flag (defaults to `all`).
- `coordinate_systems`: Dictionary of all initialized coordinate systems filtered by the CLI flag.
- `cs_flag`: Automatically parameterized flag representing one coordinate system per test.
- `coordinate_io_temp_dir`: A module-scoped temporary directory for I/O protocol testing.

### 🧰 Utilities

This module defines helper constants in `utils.py`:

- `__coordinate_system_io_protocols__`: A list of all supported I/O protocols in the form:
  ```python
  [
      pytest.param("HDF5", 'from_hdf5', 'to_hdf5'),
      pytest.param("YAML", 'from_yaml', 'to_yaml'),
      pytest.param("JSON", 'from_json', 'to_json'),
  ]
  ```
  Used to parameterize I/O tests dynamically across protocol formats.

### 🧪 Parameterizations

- `@pytest.mark.parametrize('protocol, from_method, to_method', __coordinate_system_io_protocols__)` — iterates I/O test over all supported formats.
- `cs_flag` — injected via `pytest_generate_tests()` based on the selected `--coord-systems` flag.

---

## ✅ Tests

| File                          | Purpose                                                                 |
|-------------------------------|-------------------------------------------------------------------------|
| `test_coordinates_build.py`   | Validates initialization of all built-in coordinate system classes.     |
| `test_coordinates_io.py`      | Ensures disk-based serialization (YAML, JSON, HDF5) preserves structure. |
| `utils.py`                    | Contains `__coordinate_system_io_protocols__` used for I/O param tests. |

---

## 📝 Notes

- Always use the shared fixtures for coordinate system setup and configuration.
- Roundtrip I/O tests check structural equivalence only (not identity of memory references).
- Avoid embedding large symbolic expressions directly — prefer simplified versions unless stress-testing.
- Tests are executed for all coordinate systems unless filtered using:

```bash
pytest --coord-systems=cartesian2D,spherical
```

To test only specific cases.

```bash
pytest -k "test_coordinates_io"
```

To run I/O tests only.
