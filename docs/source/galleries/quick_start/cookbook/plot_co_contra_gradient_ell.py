"""
======================================
Computing Field Gradients: Ellipsoidal
======================================

In this cookbook, we'll show how to compute covariant and contravariant gradients
of a scalar field in PyMetric for complex coordinate systems.
"""
import matplotlib.pyplot as plt
import numpy as np

# %%
# To set the stage for this example, we'll consider a common type of scenario in computational
# astrophysics: computing fields from potentials! This time, we'll consider the tricky case of
# **ellipsoidal coordinates**, specifically oblate spheroidal and homoeoidal coordinates.
#
# .. note::
#
#   In this example, we're using both spheroidal and homoeoidal coordinate systems. The difference
#   between the two is that spheroidal coordinate systems are constructed of **confocal** ellipsoids,
#   while those in homoeoidal coordinate systems are **concentric**.
#
#   As we will see, the homoeoidal case is **much more challenging** because it is not orthogonal.
#
# Step 1: Setting Up the System
# -------------------------------
# Like almost any PyMetric workflow, we need to start by building a coordinate system and
# grid. In this case, we'll consider two different coordinate systems:
#
# 1. :class:`coordinates.coordinate_systems.OblateSpheroidalCoordinateSystem`: confocal ellipsoids.
# 2. :class:`coordinates.coordinate_systems.OblateHomoeoidalCoordinateSystem`: concentric ellipsoids.
#
# For both, we'll want good angular and radial resolution in the grids, but we can rely on azimuthal symmetry
# to get away with low resolution in the :math:`\phi` coordinate.
import pymetric as pym

# Create the coordinate systems.#
cs_hom = pym.OblateHomoeoidalCoordinateSystem(ecc=0.6)
cs_sph = pym.OblateSpheroidalCoordinateSystem(a=0.5)

# Plot the coordinate system contours.
x, z = np.linspace(-1, 1, 800), np.linspace(-1, 1, 800)
X, Z = np.meshgrid(x, z)

R_HOM, _, _ = cs_hom.from_cartesian(X, 0, Z)
R_SPH, _, _ = cs_sph.from_cartesian(X, 0, Z)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

axes[0].contour(X, Z, R_HOM, levels=20)
axes[1].contour(X, Z, R_SPH, levels=20)

plt.show()

# %%
# Now that the coordinate systems are generated, we can build the grids pretty easily. For the
# homoeoidal grid, we'll construct :math:`(r,\theta,\phi)` in a grid with a maximal effective radius
# of :math:`1`.

bbox = [[0, 0, 0], [1, np.pi, 2 * np.pi]]
r = np.geomspace(1e-2, 1, 1000)

# Note: for phi and theta, we go from 0 -> np.pi (2*np.pi) and then cut of the edges
# so that we can create 20 points in between 0 and the upper bound. This ensures
# the metric is never degenerate.
theta, phi = np.linspace(0, np.pi, 1000)[1:-1], np.linspace(0, 2 * np.pi, 22)[1:-1]
grid_hom = pym.GenericGrid(
    cs_hom, [r, theta, phi], ghost_zones=1, center="cell", bbox=bbox
)

# %%
# For the spheroidal case, we need :math:`\mu, \nu, \phi` which (coincidentally) can be exactly the same
# that we used for the homoeoidal case.

bbox = [[0, 0, 0], [1, np.pi, 2 * np.pi]]
mu = np.geomspace(1e-2, 1, 1000)

# Note: for phi and theta, we go from 0 -> np.pi (2*np.pi) and then cut of the edges
# so that we can create 20 points in between 0 and the upper bound. This ensures
# the metric is never degenerate.
nu, phi = np.linspace(0, np.pi, 200)[1:-1], np.linspace(0, 2 * np.pi, 22)[1:-1]
grid_sph = pym.GenericGrid(
    cs_sph, [mu, nu, phi], ghost_zones=1, center="cell", bbox=bbox
)

# %%
# Step 2: Build the Potential Field
# ---------------------------------
# The next important step is to construct the potential field as a function of the radius. For this
# case, we'll use a `NFW profile <https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile>`__:
#
# .. math::
#
#       \Phi = \frac{-4\pi G \rho_0 R_s^3}{r} \ln\left(1+\frac{r}{R_s}\right).
#
# We'll use a scale free coordinate system (:math:`G=1`) to simplify things:


# Create the function for the potential.
def potential_function(_r, R_s=100, rho_0=1):
    return -4 * np.pi * rho_0 * R_s**3 * (np.log(1 + (_r / R_s)) / _r)


# Create a dense field from the function.
PhiHom = pym.DenseTensorField.from_function(
    potential_function, grid_hom, ["xi"], rho_0=1, R_s=1e-1
)
PhiSph = pym.DenseTensorField.from_function(
    potential_function, grid_sph, ["mu"], rho_0=1, R_s=1e-1
)

# Plot the field.
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

axes[0].loglog(r, -PhiHom[...], color="k", label="NFW Potential (Homoeoidal)")
axes[1].loglog(r, -PhiSph[...], color="k", label="NFW Potential (Spheroidal)")
axes[0].set_xlabel("Radius, [kpc]")
axes[0].set_ylabel("Potential")
axes[1].set_xlabel("Radius, [kpc]")
plt.show()

# %%
# Looking at these as a function of their respective radius coordinates clearly makes them look
# nearly identical, but if we instead compute the potential on the X-Z grid from earlier, we see how
# different the fields are. This is an example of the interpolation feature possessed by grids!
HomInterp = PhiHom.grid.construct_domain_interpolator(
    PhiHom[...], ["xi"], method="linear", fill_value=None
)
SphInterp = PhiSph.grid.construct_domain_interpolator(
    PhiSph[...], ["mu"], method="linear", fill_value=None
)

XI, _, _ = cs_hom.from_cartesian(X, 0, Z)
MU, _, _ = cs_sph.from_cartesian(X, 0, Z)

Z_HOM, Z_SPH = HomInterp(XI.ravel()).reshape(XI.shape), SphInterp(MU.ravel()).reshape(
    MU.shape
)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

Q = axes[0].imshow(
    Z_HOM, cmap="inferno", origin="lower", aspect="equal", extent=(-1, 1, -1, 1)
)
axes[1].imshow(
    Z_SPH, cmap="inferno", origin="lower", aspect="equal", extent=(-1, 1, -1, 1)
)
plt.colorbar(Q, label="Potential", ax=axes)
plt.show()

# %%
# Step 3: Compute the Gradient
# ---------------------------------
# To compute the gradients, all we need to do is call the :meth:`~fields.DenseTensorField.gradient` method!
gradPhi_HOM_cov, grad_Phi_SPH_cov = PhiHom.gradient(basis="covariant"), PhiSph.gradient(
    basis="covariant"
)
gradPhi_HOM_con, grad_Phi_SPH_con = PhiHom.gradient(
    basis="contravariant"
), PhiSph.gradient(basis="contravariant")

print(gradPhi_HOM_cov.shape, grad_Phi_SPH_cov.shape)
print(gradPhi_HOM_con.shape, grad_Phi_SPH_con.shape)

# %%
# Notice that the the gradients are different shapes. This is a byproduct of the internal dependence
# calculations performed by Pymetric. To see why, recall that the covariant gradient is :math:`\nabla_i \Phi = \partial_i \Phi`,
# while the contravariant gradient is :math:`\nabla^i \Phi = g^{i\mu} \partial_\mu \Phi`. Thus, if the metric introduces
# additional dependence, then the field axes will expand to reflect this.
#
# We can check out the metric tensor from the coordinate system:
print(cs_hom.metric_tensor_symbol[0, 0])
print(cs_sph.metric_tensor_symbol[0])

# %%
# Each of the coordinate systems has a :math:`\nu` or :math:`\theta` dependence in the ``(0,0)`` component of
# the metric tensor!
#
# Now, we can look at the range of different radial components we have over different values of the angular
# variable:

grad_r_hom = gradPhi_HOM_con[..., 0]  # shape: (r, θ, φ)
grad_r_sph = grad_Phi_SPH_con[..., 0]


# Get coordinates for meshgrid
# sphinx_gallery_thumbnail_number = 2
RR, TT = np.meshgrid(r, theta, indexing="ij")
MM, NN = np.meshgrid(mu, nu, indexing="ij")

# Compute min/max over angular coordinate (axis=1)
min_hom, max_hom = np.min(grad_r_hom, axis=1), np.max(grad_r_hom, axis=1)
min_sph, max_sph = np.min(grad_r_sph, axis=1), np.max(grad_r_sph, axis=1)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 9), gridspec_kw=dict(hspace=0.4))

# Top: 2D heatmaps

im0 = axes[0, 0].pcolormesh(
    RR, TT, grad_r_hom, shading="auto", cmap="viridis", norm="log"
)
axes[0, 0].set_title("Radial gradient (Homoeoidal)")
axes[0, 0].set_xlabel(r"$r$")
axes[0, 0].set_ylabel(r"$\theta$")
fig.colorbar(im0, ax=axes[0, 0], label="Radial Gradient")

im1 = axes[0, 1].pcolormesh(
    MM, NN, grad_r_sph, shading="auto", cmap="viridis", norm="log"
)
axes[0, 1].set_title("Radial gradient (Spheroidal)")
axes[0, 1].set_xlabel(r"$\mu$")
axes[0, 1].set_ylabel(r"$\nu$")
fig.colorbar(im1, ax=axes[0, 1], label="Radial Gradient")

# Bottom: fill_between for min/max
axes[1, 0].fill_between(r, min_hom, max_hom, alpha=0.4, color="purple")
axes[1, 0].plot(r, min_hom, color="k", lw=1, label="min")
axes[1, 0].plot(r, max_hom, color="k", lw=1, label="max")
axes[1, 0].set_title(r"Range of $\nabla^r \Phi$ over $\Theta$ (Homoeoidal)")
axes[1, 0].set_xlabel(r"$r$")
axes[1, 0].set_ylabel(r"$g^{\xi\nu} \partial_\nu \Phi(\xi)$")
axes[1, 0].legend()

axes[1, 1].fill_between(mu, min_sph, max_sph, alpha=0.4, color="green")
axes[1, 1].loglog(mu, min_sph, color="k", lw=1, label="min")
axes[1, 1].loglog(mu, max_sph, color="k", lw=1, label="max")
axes[1, 1].set_title(r"Range of $\nabla^r \Phi$ over $\nu$ (Spheroidal)")
axes[1, 1].set_xlabel(r"$\mu$")
axes[1, 1].set_ylabel(r"$g^{\mu\nu} \partial_\nu \Phi(\mu)$")
axes[1, 1].legend()

plt.show()
