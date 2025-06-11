# PyMetric Testing: `test_fields.test_mathops`

---

This testing suite targets the **field-level differential geometry operations** module of PyMetric and performs tests over the following broad regimes of behavior:

- **Gradient Computation**: Verifies that the `DenseTensorField.gradient()` method executes across all supported grid types and coordinate systems.
- **Grid Coverage**: Confirms behavior consistency for Cartesian, spherical, and oblate homoeoidal coordinates.
- **Smoke Testing**: Ensures that field instantiation and differential operator application do not raise errors, validating core API connectivity.

---

## 📐 Testing Structure

### 🔧 Fixtures Summary

Fixtures used in this test module:

- `coordinate_systems`: Dictionary of all initialized coordinate systems filtered by the `--coord-systems` flag.
- [Others indirectly used via `__grid_factories__` or param fixtures from `test_grids.utils`]

### 🧰 Utilities

No local or `utils.py` helper functions are used in this module.

---

## 🧪 Parameterizations

This test module uses the following parameterizations:

- `@pytest.mark.parametrize("grid_class", __all_grid_classes_params__)`: Tests across all registered grid types (e.g., uniform, scaled, generic).
- `@pytest.mark.parametrize("coordinate_system", __coordinate_systems_run__)`: Dynamically selects among spherical, Cartesian3D, and oblate homoeoidal systems.

---

## ✅ Tests

| File                      | Purpose                                                                 |
|---------------------------|-------------------------------------------------------------------------|
| `test_fields_dmops.py`    | Validates `DenseTensorField.gradient()` behavior across grids and coordinates |
| `utils.py`                | *None present or used in this module*                                   |

---

## 📝 Notes

- These tests are not meant to check numerical correctness (except in other modules); they are execution tests ensuring that the field-based gradient API integrates correctly with PyMetric's grid and coordinate infrastructure.
- Use small field shapes (e.g., zero-order tensors on modest resolution grids) for lightweight testing.
- Extend this suite with divergence and Laplacian tests when those methods are implemented at the `DenseTensorField` level.
