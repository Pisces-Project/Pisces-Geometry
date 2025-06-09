# test_buffers

This directory contains tests for the buffer API implementation in the PyMetric `fields` module. The tests ensure that all buffer classes conform to the expected interface and behavior, including:

- NumPy-compatible data access via `__getitem__` and `__setitem__`
- Support for standard indexing patterns (basic, boolean, integer array, multidimensional)
- Correct broadcasting and assignment semantics
- Implementation of standard buffer attributes (`shape`, `dtype`, `ndim`, `size`, etc.)
- Interoperability with various backends (NumPy, unyt, HDF5, etc.)
- Forwarding of NumPy universal functions and operations

## Structure

- Each test file targets a specific buffer backend or feature.
- Tests are written using `pytest` and are designed to be backend-agnostic where possible.

## Running the Tests

From the project root, run:

```bash
pytest tests/test_fields/test_buffers
```

## Contributing

- Add new tests for any new buffer features or backends.
- Ensure all tests pass before submitting changes.
- Follow the project’s coding and documentation standards.