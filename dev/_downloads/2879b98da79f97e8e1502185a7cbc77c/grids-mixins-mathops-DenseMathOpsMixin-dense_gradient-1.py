from pymetric.grids.core import UniformGrid
from pymetric.coordinates import CartesianCoordinateSystem2D
import matplotlib.pyplot as plt
cs = CartesianCoordinateSystem2D()
grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]],center='cell')
X,Y = grid.compute_domain_mesh(origin='global')
Z = np.sin((X**2+Y**2))
derivatives = grid.dense_gradient(Z,['x','y'],basis='contravariant')
fig,axes = plt.subplots(1,3,sharey=True,sharex=True,figsize=(7,3))
_ = axes[0].imshow(Z.T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
_ = axes[1].imshow(derivatives[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
_ = axes[2].imshow(derivatives[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
plt.show()
