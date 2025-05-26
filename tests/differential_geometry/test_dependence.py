"""
Testing module for :mod:`differential_geometry.dependence` to establish
the correctness of the dependence tracking support in PyMetric.
"""
import pytest
import sympy as sp

from pymetric.coordinates import (
    CartesianCoordinateSystem2D,
    OblateHomoeoidalCoordinateSystem,
    SphericalCoordinateSystem,
)
from pymetric.differential_geometry import dependence as dep

# ================================== #
# Setup                              #
# ================================== #
__all_coordinate_systems_params__ = [
    pytest.param("spherical", marks=pytest.mark.spherical),
    pytest.param("cartesian", marks=pytest.mark.cartesian),
    pytest.param("oblate_spherical", marks=pytest.mark.obs),
    pytest.param("oblate_non_spherical", marks=pytest.mark.obns),
]


# ================================== #
# Fixtures                           #
# ================================== #
# These are the coordinate system marks we use for each test.
# We use Cartesian for a null metric, Spherical for a diagonal metric, and
# then the two homoeoidal coordinate systems to confirm the behavior of non-diagonal
# metrics.
#
# To avoid reloading coordinate systems, we load them all as fixtures.
@pytest.fixture(scope="module")
def coordinate_systems():
    """The test coordinate systems."""
    return {
        "spherical": SphericalCoordinateSystem(),
        "cartesian": CartesianCoordinateSystem2D(),
        "oblate_spherical": OblateHomoeoidalCoordinateSystem(ecc=0),
        "oblate_non_spherical": OblateHomoeoidalCoordinateSystem(ecc=0.5),
    }


# ================================== #
# Non-Tensor Dense Dependence Tests  #
# ================================== #
# The tests here are for the DenseDependenceObject and
# test the non-tensorial methods of the class.
@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_dense_scalar_construction_and_proxy(coordinate_systems, cs_key):
    """
    Test that we can correctly construct scalar dependence objects from
    each of the coordinate systems.
    """
    # Extract the coordinate systems.
    cs = coordinate_systems[cs_key]

    # Construct the scalar dependence.
    dep_obj = dep.DenseDependenceObject(cs, (), dependent_axes=cs.__AXES__[:1])

    # Confirm properties
    assert dep_obj.is_scalar
    assert dep_obj.rank == 0
    assert dep_obj.dependent_axes == cs.__AXES__[:1]

    # Check symbolic proxy
    proxy = dep_obj.symbolic_proxy
    assert isinstance(proxy, sp.Basic)
    assert all(str(sym) in str(proxy) for sym in dep_obj.axes_symbols)


@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_dense_array_construction_and_proxy(coordinate_systems, cs_key):
    cs = coordinate_systems[cs_key]
    dep_obj = dep.DenseDependenceObject(cs, (3, 3, 3), dependent_axes=cs.__AXES__[:1])

    # Confirm properties
    assert not dep_obj.is_scalar
    assert dep_obj.rank == 3
    assert dep_obj.dependent_axes == cs.__AXES__[:1]

    # Check symbolic proxy
    proxy = dep_obj.symbolic_proxy
    assert isinstance(proxy, sp.DenseNDimArray)
    assert all(str(sym) in str(proxy) for sym in dep_obj.axes_symbols)


@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_elementwise_derivatives_dependence(coordinate_systems, cs_key):
    """
    In this test, we confirm that all of the coordinate systems preserve
    their dependence under elementwise differentiation. We check that
    the covariant case preserves and the contravariant case changes.
    """
    # Setup coordinate systems and
    # dependence objects.
    cs = coordinate_systems[cs_key]
    dep_obj = dep.DenseDependenceObject(cs, (3,), dependent_axes=cs.__AXES__[1])

    # -- Compute gradient dependence -- #
    co_grad_obj = dep_obj.element_wise_gradient(basis="covariant")
    contra_grad_obj = dep_obj.element_wise_gradient(basis="contravariant")

    # -- Perform Checks -- #
    # The covariant case should have exactly the same dependence as
    # the original dense object did.
    __answer_key__ = {
        "spherical": dict(co=["theta"], contra=["r", "theta"]),
        "cartesian": dict(co=["y"], contra=["y"]),
        "oblate_spherical": dict(co=["theta"], contra=["xi", "theta"]),
        "oblate_non_spherical": dict(co=["theta"], contra=["xi", "theta"]),
    }

    __answer__ = __answer_key__[cs_key]

    assert set(__answer__["co"]) == set(co_grad_obj.dependent_axes)
    assert set(__answer__["contra"]) == set(contra_grad_obj.dependent_axes)


@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_elementwise_laplacian_dependence(coordinate_systems, cs_key):
    """
    In this test, we confirm the dependence changes when computing
    elementwise Laplacians.
    """
    # Setup coordinate systems and
    # dependence objects.
    cs = coordinate_systems[cs_key]
    dep_obj = dep.DenseDependenceObject(cs, (3,), dependent_axes=cs.__AXES__[1])

    # -- Compute gradient dependence -- #
    lap_obj = dep_obj.element_wise_laplacian()

    # -- Perform Checks -- #
    # The covariant case should have exactly the same dependence as
    # the original dense object did.
    __answer_key__ = {
        "spherical": ["r", "theta"],
        "cartesian": ["y"],
        "oblate_spherical": ["xi", "theta"],
        "oblate_non_spherical": ["xi", "theta"],
    }

    __answer__ = __answer_key__[cs_key]

    assert set(__answer__) == set(lap_obj.dependent_axes)


# ================================== #
# Tensor Dense Dependence Tests      #
# ================================== #
# The tests here are for the DenseTensorDependence and
# test the tensorial methods of the class.
@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_tensor_dependence_construction_and_proxy(coordinate_systems, cs_key):
    """
    Confirm that DenseTensorDependence can be constructed with correct shape and symbolic proxy.
    """
    cs = coordinate_systems[cs_key]
    tensor = dep.DenseTensorDependence(cs, 2, dependent_axes=cs.__AXES__[:2])

    assert tensor.rank == 2
    assert tensor.shape == (cs.ndim, cs.ndim)
    assert tensor.dependent_axes == cs.__AXES__[:2]

    proxy = tensor.symbolic_proxy
    assert isinstance(proxy, sp.DenseNDimArray)
    assert proxy.shape == tensor.shape
    assert all(str(sym) in str(proxy) for sym in tensor.axes_symbols)


@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_tensor_index_raising_lowering_updates_dependence(coordinate_systems, cs_key):
    """
    Test that raising and lowering a tensor index modifies the dependence
    appropriately by incorporating metric dependence.

    Since metric tensors vary by coordinate system, the resulting dependence
    after raising and lowering differs:

    Cartesian:          ['x']           -> ['x']
    Spherical:          ['r']           -> ['r', 'theta']
    Oblate Spherical:   ['xi']          -> ['xi', 'theta']
    Oblate Non-Spherical: ['xi']        -> ['xi', 'theta']
    """
    cs = coordinate_systems[cs_key]
    original = dep.DenseTensorDependence(cs, 1, dependent_axes=[cs.__AXES__[0]])

    raised = original.raise_index(0)
    lowered = raised.lower_index(0)

    # Expected dependence sets after metric contraction
    expected_dependence = {
        "cartesian": ["x"],
        "spherical": ["r", "theta"],
        "oblate_spherical": ["xi", "theta"],
        "oblate_non_spherical": ["xi", "theta"],
    }

    assert set(raised.dependent_axes) == set(expected_dependence[cs_key])
    assert set(lowered.dependent_axes) == set(expected_dependence[cs_key])


@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_tensor_gradient_and_laplacian(coordinate_systems, cs_key):
    """
    Test that tensor gradient and Laplacian return valid dependence objects
    and that their symbolic coordinate dependence is correctly updated.

    Rank increases by one for gradients, and remains the same for Laplacians.

    Expected dependence:
    --------------------
    Original: ['r'] or ['x'] or ['xi'] (depends on cs)
    Gradient:
        Cartesian → ['x'] (contra) or ['x'] (co)
        Spherical → ['r', 'theta'] (contra) or ['r'] (co)
        Oblate (any) → ['xi', 'theta'] or ['xi'] (co)
    Laplacian:
        Cartesian → ['x']
        Spherical → ['r', 'theta']
        Oblate (any) → ['xi', 'theta']
    """
    # Create the coordinate system and
    # the dependence objects.
    cs = coordinate_systems[cs_key]
    tensor = dep.DenseTensorDependence(cs, 1, dependent_axes=[cs.__AXES__[1]])

    # --- Perform the computations --- #
    co_grad_obj, contra_grad_obj = tensor.gradient(basis="covariant"), tensor.gradient(
        basis="contravariant"
    )
    lap_obj = tensor.laplacian()

    # --- Structural assertions --- #
    assert isinstance(co_grad_obj, dep.DenseTensorDependence)
    assert isinstance(contra_grad_obj, dep.DenseTensorDependence)
    assert isinstance(lap_obj, dep.DenseTensorDependence)

    assert co_grad_obj.rank == tensor.rank + 1
    assert contra_grad_obj.rank == tensor.rank + 1
    assert lap_obj.rank == tensor.rank

    # --- Symbolic structure checks --- #
    assert isinstance(co_grad_obj.symbolic_proxy, sp.DenseNDimArray)
    assert isinstance(contra_grad_obj.symbolic_proxy, sp.DenseNDimArray)
    assert isinstance(lap_obj.symbolic_proxy, sp.DenseNDimArray)

    # --- Coordinate dependence expectations --- #
    expected_dependence = {
        "cartesian": dict(co=["y"], contra=["y"], lap=["y"]),
        "spherical": dict(co=["theta"], contra=["r", "theta"], lap=["r", "theta"]),
        "oblate_spherical": dict(
            co=["theta"], contra=["xi", "theta"], lap=["xi", "theta"]
        ),
        "oblate_non_spherical": dict(
            co=["theta"], contra=["xi", "theta"], lap=["xi", "theta"]
        ),
    }

    expected = expected_dependence[cs_key]

    # --- Check that the dependent axes match expectation --- #
    assert set(co_grad_obj.dependent_axes) == set(
        expected["co"]
    ), f"Co gradient mismatch for {cs_key}"
    assert set(contra_grad_obj.dependent_axes) == set(
        expected["contra"]
    ), f"Contra gradient mismatch for {cs_key}"
    assert set(lap_obj.dependent_axes) == set(
        expected["lap"]
    ), f"Laplacian mismatch for {cs_key}"


@pytest.mark.parametrize("cs_key", __all_coordinate_systems_params__)
def test_tensor_divergence_on_vector_only(coordinate_systems, cs_key):
    """
    Test divergence is valid for rank-1 tensors and raises for others.
    """
    cs = coordinate_systems[cs_key]

    # Valid case: rank 1
    v = dep.DenseTensorDependence(cs, 1, dependent_axes=cs.__AXES__[:1])
    div = v.divergence()
    assert isinstance(div, dep.DenseTensorDependence)
    assert div.rank == 0

    # Invalid case: rank 2
    t = dep.DenseTensorDependence(cs, 2, dependent_axes=cs.__AXES__[:2])
    with pytest.raises(ValueError, match="Divergence only defined for rank‑1 tensors"):
        t.divergence()
