from pymetric.grids.core import UniformGrid
from pymetric.coordinates import CartesianCoordinateSystem2D
import matplotlib.pyplot as plt
cs = CartesianCoordinateSystem2D()
grid = UniformGrid(cs,[[-1,-1],[1,1]],[500,500],chunk_size=[50,50],ghost_zones=[[2,2],[2,2]],center='cell')
X,Y = grid.compute_domain_mesh(origin='global')
Z = np.stack([np.sin((X**2+Y**2)),np.sin(5*(X**2+Y**2))],axis=-1) # (504,504,2)
derivatives = grid.dense_element_wise_partial_derivatives(Z,['x','y'])
fig,axes = plt.subplots(2,3,sharey=True,sharex=True,figsize=(7,6))
_ = axes[0,0].imshow(Z[...,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
_ = axes[0,1].imshow(derivatives[...,0,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
_ = axes[0,2].imshow(derivatives[...,0,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
_ = axes[1,0].imshow(Z[...,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-1,vmax=1,cmap='coolwarm')
_ = axes[1,1].imshow(derivatives[...,1,0].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
_ = axes[1,2].imshow(derivatives[...,1,1].T,origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
plt.show()
