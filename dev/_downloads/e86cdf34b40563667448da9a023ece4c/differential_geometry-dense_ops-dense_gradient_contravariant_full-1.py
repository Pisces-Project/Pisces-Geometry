from scipy.interpolate import RegularGridInterpolator
from pymetric.differential_geometry.dense_ops import (dense_gradient_contravariant_full,
dense_gradient_covariant)
import matplotlib.pyplot as plt
r = np.linspace(1e-4,1,1000)
theta = np.linspace(0,np.pi,30)
R,THETA = np.meshgrid(r,theta,indexing='ij')
Z = np.sin(10*R) * np.cos(THETA)**2
interpZ = RegularGridInterpolator((r,theta),Z,bounds_error=False)
gradZ = dense_gradient_covariant(Z,0,2,r,theta)
metric = np.zeros(R.shape + (2,2))
metric[:,:,0,0] = 1
metric[:,:,1,1] = 1/R**2
gradZcontra = dense_gradient_contravariant_full(Z,metric,0,2,r,theta)
interpZr = RegularGridInterpolator((r,theta),gradZ[...,0],bounds_error=False)
interpZtheta = RegularGridInterpolator((r,theta),gradZ[...,1],bounds_error=False)
interpCZr = RegularGridInterpolator((r,theta),gradZcontra[...,0],bounds_error=False)
interpCZtheta = RegularGridInterpolator((r,theta),gradZcontra[...,1],bounds_error=False)
bound = 1/np.sqrt(2)
x,y = np.linspace(-bound,bound,100),np.linspace(-bound,bound,100)
X,Y = np.meshgrid(x,y,indexing='ij')
RG = np.sqrt(X**2+Y**2)
THETAG = np.arccos(Y/RG)
grid_points = np.stack([RG.ravel(),THETAG.ravel()],axis=1)
Zgrid = interpZ(grid_points).reshape(RG.shape)
Zrgrid = interpZr(grid_points).reshape(RG.shape)
Zthetagrid = interpZtheta(grid_points).reshape(RG.shape)
ZCrgrid = interpCZr(grid_points).reshape(RG.shape)
ZCthetagrid = interpCZtheta(grid_points).reshape(RG.shape)
#
fig,axes = plt.subplots(2,2, sharex=True, sharey=True)
_ = axes[0,0].imshow(Zrgrid.T    ,extent=[-bound,bound,-bound,bound], vmin=-3,vmax=3,cmap='seismic',origin='lower')
_ = axes[0,1].imshow(Zthetagrid.T,extent=[-bound,bound,-bound,bound], vmin=-3,vmax=3,cmap='seismic',origin='lower')
_ = axes[1,0].imshow(ZCrgrid.T    ,extent=[-bound,bound,-bound,bound],vmin=-3,vmax=3,cmap='seismic',origin='lower')
P = axes[1,1].imshow(ZCthetagrid.T,extent=[-bound,bound,-bound,bound],vmin=-3,vmax=3,cmap='seismic',origin='lower')
_ = plt.colorbar(P,ax=axes)
plt.show()
