from pymetric.coordinates import CartesianCoordinateSystem2D
from pymetric.grids import UniformGrid
import matplotlib.pyplot as plt
import numpy as np

csys = CartesianCoordinateSystem2D()
bbox = [[0.0, 1.0], [0.0, 1.0]]  # [lower_bounds, upper_bounds]
shape = [100, 100]  # number of cells per axis

grid = UniformGrid(
    csys,
    bbox,
    shape,
    ghost_zones=[[2, 2], [2, 2]]  # 2 ghost zones on each side
)

X,Y = grid.compute_domain_mesh()
R = np.sqrt(X**2 + Y**2)
Z = np.sin(10*R)

plt.imshow(Z)
plt.show()