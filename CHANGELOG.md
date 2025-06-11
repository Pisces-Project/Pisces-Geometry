# Features

- Added `containers` module with environments for field creation / retention.
  - Added `containers.FieldContainer` as the generic base class of the module.

# Testing

- Added `tests/test_other` for generic testing of non-module code.
- Added `tests/test_other/test_containers` to test the new `containers` module.
  - `test_field_container_build` tests that we can use ``.zeros`` to build containers.

# Bug Fixes

- Fixed issue with some generators (like `.zeros`) where ``buffer_args=None`` led to
  issues when calling as ``(*buffer_args)``.

# Minor Changes
