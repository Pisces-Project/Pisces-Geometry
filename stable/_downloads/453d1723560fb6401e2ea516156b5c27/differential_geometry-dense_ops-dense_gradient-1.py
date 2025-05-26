import numpy as np
import matplotlib.pyplot as plt
from pymetric.differential_geometry.dense_ops import dense_gradient
r = np.linspace(0.01, 1.0, 100)
theta = np.linspace(0, np.pi, 100)
R, THETA = np.meshgrid(r, theta, indexing='ij')
F = R**2 * np.sin(THETA)
IM = np.zeros(R.shape + (2,))
IM[..., 0] = 1            # g^rr = 1
IM[..., 1] = 1 / R**2     # g^thetatheta = 1/r^2
grad_cov = dense_gradient(F, 0, 2, r, theta, basis='covariant', edge_order=2)
grad_contra = dense_gradient(F, 0, 2, r, theta, basis='contravariant', inverse_metric_field=IM, edge_order=2)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(grad_cov[..., 1].T, origin='lower', extent=[0.01, 1.0, 0, np.pi], aspect='auto')
_ = axes[0].set_title(r'Covariant Gradient $(\partial_\theta f)$')
_ = fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(grad_contra[..., 1].T, origin='lower', extent=[0.01, 1.0, 0, np.pi], aspect='auto')
_ = axes[1].set_title(r'Contravariant Gradient $(r^{-2} \; \partial_\theta f)$')
_ = fig.colorbar(im1, ax=axes[1])
for ax in axes:
    _ = ax.set_xlabel("r")
    _ = ax.set_ylabel("theta")
plt.tight_layout()
plt.show()
