"""
Tests for the :mod:`differential_geometry.symbolic` module.
"""
import pytest
import sympy as sp

from pymetric.differential_geometry import symbolic as sym


# ================================= #
# Setup Fixture                     #
# ================================= #
# Testing for this module relies on using the
# spherical coordinate system to check against known results
# for various computations.
@pytest.fixture(scope="module")
def coordinate_symbols():
    """
    Fixture to create coordinate symbols used in spherical coordinates.

    Returns
    -------
    list of sympy.Symbol
        The coordinate symbols [r, theta, phi], each declared positive for stability
        in metric-related expressions.
    """
    r, theta, phi = sp.symbols("r theta phi", positive=True)
    return [r, theta, phi]


@pytest.fixture(scope="module")
def metrics(coordinate_symbols):
    """
    Fixture to construct both full and diagonal metric representations,
    along with their inverses, for a spherical coordinate system.

    Parameters
    ----------
    coordinate_symbols : list of sympy.Symbol
        Coordinate variables [r, theta, phi].

    Returns
    -------
    tuple
        (metric_full, metric_diag, inv_metric_full, inv_metric_diag), where:

        - metric_full : sympy.Matrix (3x3) — full g_{μν} matrix
        - metric_diag : sympy.Array (length 3) — diagonal entries for orthogonal system
        - inv_metric_full : sympy.Matrix — inverse of full metric
        - inv_metric_diag : sympy.Array — inverse of diagonal metric
    """
    r, theta, phi = coordinate_symbols
    metric_full = sp.Matrix(
        [[1, 0, 0], [0, r**2, 0], [0, 0, (r * sp.sin(theta)) ** 2]]
    )
    metric_diag = sp.Array([1, r**2, (r * sp.sin(theta)) ** 2])
    inv_metric_full = metric_full.inv()
    inv_metric_diag = sp.Array([1, 1 / r**2, 1 / (r**2 * sp.sin(theta) ** 2)])
    return metric_full, metric_diag, inv_metric_full, inv_metric_diag


@pytest.fixture(scope="module")
def metric_density(coordinate_symbols):
    """
    Fixture to return the symbolic metric density ρ = sqrt(det(g)) for spherical coordinates.

    This corresponds to:
        ρ = r² sin(θ)

    Parameters
    ----------
    coordinate_symbols : list of sympy.Symbol
        Coordinate variables [r, theta, phi].

    Returns
    -------
    sympy.Basic
        The symbolic metric density.
    """
    r, theta, _ = coordinate_symbols
    return r**2 * sp.sin(theta)


@pytest.fixture(scope="module")
def scalar_field(coordinate_symbols):
    """
    Fixture to provide a sample scalar field for testing gradient/Laplacian computations.

    The field is defined as:
        φ(r, θ) = r² sin(θ)

    Parameters
    ----------
    coordinate_symbols : list of sympy.Symbol
        Coordinate variables [r, theta, phi].

    Returns
    -------
    sympy.Basic
        The scalar field φ.
    """
    r, theta, _ = coordinate_symbols
    return r**2 * sp.sin(theta)


@pytest.fixture(scope="module")
def vector_field(coordinate_symbols):
    """
    Fixture to return a sample vector field for divergence tests.

    The field is defined in spherical coordinates as:
        F^μ = [0, r, 0]

    Parameters
    ----------
    coordinate_symbols : list of sympy.Symbol
        Coordinate variables [r, theta, phi].

    Returns
    -------
    sympy.Array
        A contravariant vector field.
    """
    r, _, _ = coordinate_symbols
    return sp.Array([0, r, 0])


# ================================= #
# Tests                             #
# ================================= #
# These are now the actual testing functions for
# the various elements of the module.


# --- invert_metric --- #
def test_invert_metric_full(metrics):
    """
    Test that `invert_metric` correctly computes the inverse of a full 3x3 metric matrix.

    This test uses the full metric tensor for spherical coordinates:
        g_{μν} = diag(1, r², r² sin²θ)

    Expected:
        The result should exactly match SymPy's matrix inverse.

    Parameters
    ----------
    metrics : tuple
        Provided by the `metrics` fixture. Contains:
        (metric_full, metric_diag, inv_metric_full, inv_metric_diag)
    """
    metric_full, _, inv_metric_full, _ = metrics
    result = sym.invert_metric(metric_full)
    assert result == inv_metric_full


def test_invert_metric_diag(metrics):
    """
    Test that `invert_metric` correctly computes the inverse of a diagonal metric.

    This test uses a simplified diagonal form of the spherical metric:
        g_{μμ} = [1, r², r² sin²θ]

    Expected:
        The result should be:
        g^{μμ} = [1, 1/r², 1/(r² sin²θ)]

    Parameters
    ----------
    metrics : tuple
        Provided by the `metrics` fixture. Contains:
        (metric_full, metric_diag, inv_metric_full, inv_metric_diag)
    """
    _, metric_diag, _, inv_metric_diag = metrics
    result = sym.invert_metric(metric_diag)
    for i in range(3):
        assert sp.simplify(result[i] - inv_metric_diag[i]) == 0


# --- compute_metric_density --- #
def test_compute_metric_density(metrics):
    """
    Test symbolic computation of metric density ρ = sqrt(det(g)) from a diagonal metric.

    This uses the diagonal form of the spherical coordinate metric:
        g_diag = [1, r², r² sin²θ]

    Expected result:
        ρ = r² sin(θ)

    Parameters
    ----------
    metrics : tuple
        Fixture providing metric components; we use `metric_diag`.
    """
    _, metric_diag, _, _ = metrics
    r, theta = sp.symbols("r theta", positive=True)
    expected = r**2 * sp.Abs(sp.sin(theta))
    result = sym.compute_metric_density(metric_diag)
    assert sp.simplify(result - expected) == 0


def test_compute_Dterm(metric_density, coordinate_symbols):
    """
    Test symbolic computation of D-term from the metric density.

    The expected D-term for spherical coordinates is:
        D = [2/r, 1/tan(θ), 0]

    Parameters
    ----------
    metric_density : sympy.Basic
        Fixture providing the metric density.
    coordinate_symbols : list
        Fixture providing the [r, θ, φ] symbols.
    """
    r, theta, _ = coordinate_symbols
    expected = sp.Array([2 / r, 1 / sp.tan(theta), 0])
    result = sym.compute_Dterm(metric_density, coordinate_symbols)
    for i in range(3):
        assert sp.simplify(result[i] - expected[i]) == 0


def test_compute_Lterm_full(metrics, metric_density, coordinate_symbols):
    """
    Test symbolic computation of the L-term using a full inverse metric.

    The expected L-term for spherical coordinates is:
        L = [2/r, 1/(r² tan(θ)), 0]

    Parameters
    ----------
    metrics : tuple
        Fixture providing inverse metric.
    metric_density : sympy.Basic
        Fixture for the metric density.
    coordinate_symbols : list
        Coordinate axes [r, θ, φ].
    """
    _, _, inv_metric_full, _ = metrics
    r, theta, _ = coordinate_symbols
    expected = sp.Array([2 / r, 1 / (r**2 * sp.tan(theta)), 0])
    result = sym.compute_Lterm(inv_metric_full, metric_density, coordinate_symbols)
    for i in range(3):
        assert sp.simplify(result[i] - expected[i]) == 0


def test_compute_Lterm_diag(metrics, metric_density, coordinate_symbols):
    """
    Test symbolic computation of the L-term using a diagonal inverse metric.

    Parameters
    ----------
    metrics : tuple
        Fixture providing diagonal inverse metric.
    metric_density : sympy.Basic
        Fixture for the metric density.
    coordinate_symbols : list
        Coordinate axes [r, θ, φ].
    """
    _, _, _, inv_metric_diag = metrics
    r, theta, _ = coordinate_symbols
    expected = sp.Array([2 / r, 1 / (r**2 * sp.tan(theta)), 0])
    result = sym.compute_Lterm(inv_metric_diag, metric_density, coordinate_symbols)
    for i in range(3):
        assert sp.simplify(result[i] - expected[i]) == 0


def test_compute_gradient_covariant(scalar_field, coordinate_symbols):
    """
    Test symbolic computation of the gradient in covariant basis.

    φ(r, θ) = r² sin(θ)
    ∂_μ φ = [2r sin(θ), r² cos(θ), 0]

    Parameters
    ----------
    scalar_field : sympy.Basic
        Fixture defining the scalar field.
    coordinate_symbols : list
        Coordinate axes [r, θ, φ].
    """
    r, theta, _ = coordinate_symbols
    expected = sp.Array([2 * r * sp.sin(theta), r**2 * sp.cos(theta), 0])
    result = sym.compute_gradient(scalar_field, coordinate_symbols, basis="covariant")
    for i in range(3):
        assert sp.simplify(result[i] - expected[i]) == 0


def test_compute_gradient_contravariant(scalar_field, coordinate_symbols, metrics):
    """
    Test symbolic gradient in contravariant basis using inverse metric.

    φ(r, θ) = r² sin(θ)

    Parameters
    ----------
    scalar_field : sympy.Basic
        Fixture defining the scalar field.
    coordinate_symbols : list
        Coordinate axes [r, θ, φ].
    metrics : tuple
        Provides inverse metric.
    """
    r, theta, _ = coordinate_symbols
    _, _, inv_metric_full, _ = metrics
    expected = sp.Array([2 * r * sp.sin(theta), sp.cos(theta), 0])
    result = sym.compute_gradient(
        scalar_field,
        coordinate_symbols,
        basis="contravariant",
        inverse_metric=inv_metric_full,
    )
    for i in range(3):
        assert sp.simplify(result[i] - expected[i]) == 0


def test_compute_divergence(vector_field, coordinate_symbols, metric_density):
    """
    Test symbolic computation of divergence of a vector field.

    Vector field: F^μ = [0, r, 0]
    Expected divergence: ∇·F = r / tan(θ)

    Parameters
    ----------
    vector_field : sympy.Array
        Vector fixture.
    coordinate_symbols : list
        Coordinate axes.
    metric_density : sympy.Basic
        Metric density ρ.
    """
    r, theta, _ = coordinate_symbols
    expected = r / sp.tan(theta)
    result = sym.compute_divergence(
        vector_field, coordinate_symbols, metric_density=metric_density
    )
    assert sp.simplify(result - expected) == 0


def test_compute_laplacian(scalar_field, coordinate_symbols, metrics, metric_density):
    """
    Test symbolic computation of Laplacian ∇²φ of a scalar field.

    φ = r² sin(θ)
    Expected: ∇²φ = 4 sin(θ) + 1 / sin(θ)

    Parameters
    ----------
    scalar_field : sympy.Basic
        The scalar field φ.
    coordinate_symbols : list
        Coordinate axes.
    metrics : tuple
        Provides inverse metric.
    metric_density : sympy.Basic
        Metric density ρ.
    """
    r, theta, _ = coordinate_symbols
    _, _, inv_metric_full, _ = metrics
    expected = 4 * sp.sin(theta) + 1 / sp.sin(theta)
    result = sym.compute_laplacian(
        scalar_field,
        coordinate_symbols,
        inverse_metric=inv_metric_full,
        metric_density=metric_density,
    )
    assert sp.simplify(result - expected) == 0


def test_tensor_gradient_and_laplacian_shapes():
    """
    Test tensor gradient and Laplacian shape correctness for rank-2 tensor field.

    The tensor is 2x2 with functional entries, using 2D polar coordinates.

    Verifies:

    - Gradient output shape is (2, 2, 2)
    - Laplacian output shape is (2, 2)
    """
    r, theta = sp.symbols("r theta", positive=True)
    coords = [r, theta]
    tensor = sp.Array(
        [
            [sp.Function("T0")(r, theta), sp.Function("T1")(r, theta)],
            [sp.Function("T2")(r, theta), sp.Function("T3")(r, theta)],
        ]
    )
    grad = sym.compute_tensor_gradient(tensor, coords)
    lap = sym.compute_tensor_laplacian(
        tensor,
        coords,
        inverse_metric=sp.Matrix([[1, 0], [0, 1 / r**2]]),
        metric_density=sym.compute_metric_density(sp.Array([1, r**2])),
    )
    assert grad.shape == (2, 2, 2)
    assert lap.shape == (2, 2)
