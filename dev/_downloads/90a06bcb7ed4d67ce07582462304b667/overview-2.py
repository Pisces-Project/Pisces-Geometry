from pymetric import UniformGrid, CartesianCoordinateSystem2D
import matplotlib.pyplot as plt
import numpy as np

# Build the coordinate system
# and the grid.
cs = CartesianCoordinateSystem2D()
bbox = [[-1,1],[-1,1]]
dd = [1000,1000]
grid = UniformGrid(cs, bbox, dd,
                   chunk_size=(10,10),
                   center='cell')

# Generate the function Z(X,Y).
func = lambda x,y: np.sin(10*np.sqrt(x**2+y**2))
Z = grid.compute_function_on_grid(func)

# Compute the covariant gradient.
gradZ = grid.dense_covariant_gradient(Z,['x','y'])

# Plot the x and y components.
fig,axes = plt.subplots(1,3,figsize=(10,4),sharex=True,sharey=True)
axes[0].imshow(Z.T,extent=(-1,1,-1,1),cmap='inferno')
axes[1].imshow(gradZ[...,0].T,extent=(-1,1,-1,1),cmap='inferno')
axes[2].imshow(gradZ[...,1].T,extent=(-1,1,-1,1),cmap='inferno')
plt.show()