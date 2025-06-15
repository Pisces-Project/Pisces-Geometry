"""
Testing suite to ensure that containers can be generated
and interacted with successfully.
"""
from pymetric.containers import FieldContainer


def test_field_container_build(uniform_grids):
    """
    In this test, we simply create a Field Container using a dense
    field over the Cartesian3D grid.
    """
    # Extract the grid
    grid = uniform_grids["cartesian3D"]

    # Construct the container using the grid.
    container = FieldContainer(grid)

    # Now we can add fields.
    _ = container.zeros("test_field", ["x", "y"])
