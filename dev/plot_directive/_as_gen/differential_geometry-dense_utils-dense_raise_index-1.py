import numpy as np
import matplotlib.pyplot as plt
from pymetric.differential_geometry.dense_utils import dense_raise_index
r = np.linspace(1, 2, 100)
theta = np.linspace(0.1, np.pi - 0.1, 100)
R, THETA = np.meshgrid(r, theta, indexing="ij")
v_cov = np.zeros(R.shape + (2,))
v_cov[..., 1] = R  # non-zero only in theta direction
g_inv = np.zeros(R.shape + (2,))
g_inv[..., 0] = 1
g_inv[..., 1] = 1 / R**2
v_contra = dense_raise_index(v_cov, index=0, rank=1, inverse_metric_field=g_inv)
_ = plt.figure(figsize=(6, 4))
im = plt.imshow(v_contra[..., 1].T, extent=[1, 2, 0.1, np.pi - 0.1], aspect="auto", origin="lower")
_ = plt.colorbar(im, label="Raised $v^\theta$")
_ = plt.xlabel("r")
_ = plt.ylabel(r"$\theta$")
_ = plt.title(r"Contravariant Component $v^\theta = r$")
_ = plt.tight_layout()
_ = plt.show()
