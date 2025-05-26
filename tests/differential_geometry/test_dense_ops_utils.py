"""
Tests for the `pymetric.differential_geometry.dense_utils` module and the
`pymetric.differential_geometry.dense_ops` module.

This test suite validates tensor index manipulations, contractions,
signature adjustments, and volume element computations on dense NumPy arrays.
"""

import numpy as np
import pytest

from pymetric.differential_geometry import dense_ops as dop
from pymetric.differential_geometry import dense_utils as du


# -------------------------- #
# Fixtures                   #
# -------------------------- #
@pytest.fixture(scope="module")
def identity_metric():
    """Fixture for a full identity metric tensor (3x3) broadcast over a 2x2 grid."""
    return np.broadcast_to(np.eye(3), (2, 2, 3, 3))


@pytest.fixture(scope="module")
def diagonal_metric():
    """Fixture for a diagonal metric (vector of ones) over a 2x2 grid."""
    return np.ones((2, 2, 3))


@pytest.fixture(scope="module")
def vector_field():
    """Fixture for a contravariant vector field with a non-zero θ component."""
    field = np.zeros((2, 2, 3))
    field[..., 1] = 1.0  # e.g., v^θ = 1
    return field


@pytest.fixture(scope="module")
def tensor_field():
    """Fixture for a rank-2 tensor with diagonal [1, 2, 3] for trace testing."""
    field = np.zeros((2, 2, 3, 3))
    for i in range(3):
        field[..., i, i] = i + 1
    return field


# -------------------------- #
# Tests: Utils               #
# -------------------------- #
def test_dense_contract_with_full_metric(vector_field, identity_metric):
    """
    Contraction with identity full metric should return the same vector field.
    """
    result = du.dense_contract_with_metric(
        vector_field, identity_metric, index=0, rank=1
    )
    np.testing.assert_allclose(result, vector_field)


def test_dense_contract_with_diagonal_metric(vector_field, diagonal_metric):
    """
    Contraction with diagonal metric should elementwise multiply the field.
    """
    expected = vector_field * diagonal_metric
    result = du.dense_contract_with_metric(
        vector_field, diagonal_metric, index=0, rank=1
    )
    np.testing.assert_allclose(result, expected)


def test_dense_raise_lower_index(vector_field, diagonal_metric):
    """
    Raising then lowering an index should yield the original field.
    """
    raised = du.dense_raise_index(
        vector_field, index=0, rank=1, inverse_metric_field=diagonal_metric
    )
    lowered = du.dense_lower_index(
        raised, index=0, rank=1, metric_field=diagonal_metric
    )
    np.testing.assert_allclose(lowered, vector_field)


def test_dense_adjust_tensor_signature(vector_field, diagonal_metric):
    """
    Adjusting signature using diagonal metric should match expected scaling.
    """
    sig = np.array([+1])
    result, new_sig = du.dense_adjust_tensor_signature(
        vector_field, [0], sig, metric_field=diagonal_metric
    )
    expected = vector_field * diagonal_metric
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(new_sig, [-1])


def test_tensor_trace_diagonal(tensor_field):
    """
    Tensor trace over [0, 1] axes should sum the diagonal components [1+2+3=6].
    """
    sig = np.array([+1, -1])
    trace = du.dense_compute_tensor_trace(
        tensor_field, indices=(0, 1), tensor_signature=sig
    )
    expected = np.full((2, 2), 6.0)
    np.testing.assert_allclose(trace, expected)


def test_volume_element_full(identity_metric):
    """
    Volume element of identity full metric should be 1 everywhere.
    """
    vol = du.dense_compute_volume_element(identity_metric, "full")
    expected = np.ones((2, 2))
    np.testing.assert_allclose(vol, expected)


def test_volume_element_diagonal(diagonal_metric):
    """
    Volume element of diagonal metric of ones should be 1 everywhere.
    """
    vol = du.dense_compute_volume_element(diagonal_metric, "diag")
    expected = np.ones((2, 2))
    np.testing.assert_allclose(vol, expected)


# -------------------------- #
# Tests: Dense Operations    #
# -------------------------- #
# In this part of the testing module, we test the
# actual operations in `dense_ops`. For each of these, we'll
# test a cartesian case and a spherical case where the answer is
# analytically known.
def test_dense_gradient_cartesian_scalar():
    """
    Test covariant gradient of scalar field f(x, y) = x^2 + y^2 in Cartesian coordinates.
    ∇f = [2x, 2y]
    """
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y, indexing="ij")
    F = X**2 + Y**2

    grad = dop.dense_gradient(F, 0, 3, *[x, y], basis="covariant")

    np.testing.assert_allclose(grad[..., 0], 2 * X)
    np.testing.assert_allclose(grad[..., 1], 2 * Y)


def test_dense_divergence_cartesian_vector():
    """
    Test divergence of vector field V = [x, y] in Cartesian coordinates.
    ∇·V = 2
    """
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y, indexing="ij")

    V = np.stack([X, Y], axis=-1)
    D = np.zeros_like(V)  # Cartesian => D = 0

    div = dop.dense_vector_divergence(V, D, x, y, basis="contravariant")
    np.testing.assert_allclose(div, 2.0)


def test_dense_laplacian_cartesian_scalar():
    """
    Test Laplacian of f(x, y) = x^2 + y^2 in Cartesian coordinates.
    Δf = 4
    """
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y, indexing="ij")
    F = X**2 + Y**2

    # F-term and inverse metric for Cartesian
    Fterm = np.zeros(F.shape + (2,))
    inverse_metric = np.ones(F.shape + (2,))

    lap = dop.dense_scalar_laplacian(F, Fterm, inverse_metric, 0, 2, *[x, y])
    np.testing.assert_allclose(lap, 4.0)


def test_dense_gradient_spherical_scalar():
    """
    In spherical coordinates, r^m cos(n theta) has contravariant gradient
    (m r^(m-1) cos(n theta), -n r^(m-2) sin(n theta)). We'll check this
    for m = 2, n = 3.
    """
    # Setup
    m, n = 2, 3
    r = np.linspace(0.1, 1.0, 100)
    theta = np.linspace(0.1, np.pi - 0.1, 100)
    R, THETA = np.meshgrid(r, theta, indexing="ij")
    F = R**m * np.cos(n * THETA)

    # Build the inverse metric.
    inv_metric = np.stack([np.ones_like(R), 1 / R**2], axis=-1)

    # compute the contravariant gradient.
    grad = dop.dense_gradient(
        F, 0, 2, r, theta, basis="contravariant", inverse_metric_field=inv_metric
    )

    # Check against the expected values.
    egrad_r = (m * R ** (m - 1)) * np.cos(n * THETA)
    egrad_t = -n * (R ** (m - 2)) * np.sin(n * THETA)

    np.testing.assert_allclose(grad[..., 0], egrad_r, rtol=1e-2)
    np.testing.assert_allclose(grad[..., 1], egrad_t, rtol=1e-2)
