from pymetric.grids.core import UniformGrid
from pymetric.coordinates import SphericalCoordinateSystem
import matplotlib.pyplot as plt
cs = SphericalCoordinateSystem()
grid = UniformGrid(cs,[[0,1],[0,np.pi],[0,2*np.pi]],
                  [500,50,50],
                  chunk_size=[50,50,50],
                  ghost_zones=[[2,2],[2,2],[2,2]])
R, THETA = grid.compute_domain_mesh(origin='global',axes=['r','theta'])
Z = (R**2) * np.cos(2*THETA)
derivatives_cont = grid.dense_contravariant_gradient(Z,['r','theta'])
derivatives_co = grid.dense_covariant_gradient(Z,['r','theta'])
fig,axes = plt.subplots(2,3,sharey=True,sharex=True,figsize=(7,6))
_ = axes[0,0].imshow(Z.T                 ,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
_ = axes[0,1].imshow(derivatives_co[...,0].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
_ = axes[0,2].imshow(derivatives_cont[...,0].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
_ = axes[1,0].imshow(Z.T                 ,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
_ = axes[1,1].imshow(derivatives_co[...,1].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
_ = axes[1,2].imshow(derivatives_cont[...,1].T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-2,vmax=2,cmap='coolwarm')
plt.show()
