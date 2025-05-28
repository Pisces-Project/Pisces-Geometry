from pymetric.grids.core import UniformGrid
from pymetric.coordinates import SphericalCoordinateSystem
import matplotlib.pyplot as plt
cs = SphericalCoordinateSystem()
grid = UniformGrid(cs,[[0,0,0],[1,np.pi,2*np.pi]],
                  [500,50,50],
                  chunk_size=[50,50,50],
                  ghost_zones=[2,2,2],center='cell')
R, THETA = grid.compute_domain_mesh(origin='global',axes=['r','theta'])
Z = np.stack([R * np.cos(THETA), np.sin(THETA), np.zeros_like(R)],axis=-1)
Z.shape
# Expected:
## (504, 54, 3)
DivZ = grid.dense_vector_divergence_contravariant(Z,['r','theta'],in_chunks=True)
fig,axes = plt.subplots(1,1)
_ = axes.imshow(DivZ.T,aspect='auto',origin='lower',extent=grid.gbbox.T.ravel(),vmin=-5,vmax=5,cmap='coolwarm')
plt.show()
