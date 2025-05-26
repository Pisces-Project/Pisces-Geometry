import numpy as np
import matplotlib.pyplot as plt
from pymetric.differential_geometry.dense_ops import dense_scalar_laplacian_diag
r = np.linspace(0.01, 1.0, 100)
theta = np.linspace(0.1, np.pi - 0.1, 100)  # avoid tan(theta)=0
R, THETA = np.meshgrid(r, theta, indexing='ij')
phi = R**2 * np.cos(THETA)
Fterm = np.zeros(R.shape + (2,))
Fterm[:,:,0] = 2 / R
Fterm[:,:,1] = 1 / (R**2 * np.tan(THETA))
IM = np.zeros(R.shape + (2,))
IM[..., 0] = 1
IM[..., 1] = 1 / R**2
lap = dense_scalar_laplacian_diag(phi, Fterm, IM, 0,2,r,theta)
_ = plt.imshow(lap.T, origin='lower', extent=[0.01, 1.0, 0.1, np.pi - 0.1], aspect='auto', cmap='viridis')
_ = plt.colorbar(label=r"Laplacian $\Delta \phi$")
_ = plt.title(r"Laplacian of $\phi(r, \theta) = r^2 \cos(\theta)$")
_ = plt.xlabel("r")
_ = plt.ylabel(r"$\theta$")
_ = plt.tight_layout()
plt.show()
