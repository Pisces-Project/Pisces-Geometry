import numpy as np
from pymetric.differential_geometry.dense_ops import dense_gradient_covariant
import matplotlib.pyplot as plt
x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
X,Y = np.meshgrid(x,y)
Z = np.sin(X**2 + Y**2)
grad = dense_gradient_covariant(Z,0,2,x,y)
fig,axes = plt.subplots(2,2,sharex=True,sharey=True)
axes[0,1].set_visible(False)
_ = axes[0,0].imshow(Z.T, origin='lower', vmin=-1,vmax=1,extent=(-1,1,-1,1))
_ = axes[1,0].imshow(grad[...,0].T, origin='lower', vmin=-1,vmax=1,extent=(-1,1,-1,1))
_ = axes[1,1].imshow(grad[...,1].T, origin='lower', vmin=-1,vmax=1,extent=(-1,1,-1,1))
_ = plt.show()
