import numpy as np
from pymetric import FieldComponent, CartesianCoordinateSystem2D, GenericGrid
import matplotlib.pyplot as plt

# Create the coordinate system and the grid.
cs = CartesianCoordinateSystem2D()
x, y = (np.linspace(0,1,100),
        np.linspace(0,1,100))
g = GenericGrid(cs, [x, y])

# Define a function of the coords.
func = lambda _x,_y: np.sin(10*np.sqrt(_x**2+_y**2))

# Create the dense field from the function.
f = FieldComponent.from_function(func, g, ['x','y'])

fig,axes = plt.subplots(1,1)
Q = axes.imshow(f[...].T,extent=(0,1,0,1))
axes.set_xlabel('x')
axes.set_ylabel('y')
plt.colorbar(Q,ax=axes)
plt.show()