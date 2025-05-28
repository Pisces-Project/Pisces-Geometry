"""
======================================================
Oblate Homoeoidal Coordinates: Effect of Eccentricity
======================================================

Visualize how the radial coordinate lines (R-contours) in the OblateHomoeoidalCoordinateSystem
change as a function of eccentricity.
"""
import matplotlib.pyplot as plt

# %%
# In this example, we'll showcase a more complex coordinate system:
# :class:`~coordinates.coordinate_systems.OblateHomoeoidalCoordinateSystem`.
#
# The oblate homoeoidal coordinate system is a fully curvilinear coordinate system
# (non-diagonal metric) which inherits the angular coordinates of spherical coordinates
# but instead relies on a modified effective radius composed of con-centric ellipsoids.
#
# In this example, we'll show off how to make these coordinate systems
# and then display some of its properties.
import numpy as np

from pymetric import OblateHomoeoidalCoordinateSystem

# %%
# To visualize, we'll use an *x-z* grid and then convert it into the
# relevant coordinate systems with different eccentricities. We can then
# plot contours for the effective radius :math:`\xi`.

# Create the cartesian grid.
x = np.linspace(-1.5, 1.5, 200)
z = np.linspace(-1.5, 1.5, 200)
X, Z = np.meshgrid(x, z)

# %%
# The class requires an eccentricity parameter when
# initialized, so we'll iterate through each of them,
# create a coordinate system class, and then perform
# the conversions.
eccentricities = [0.0, 0.1, 0.3, 0.6, 0.9, 0.99]

fig, axes = plt.subplots(
    int(np.ceil(len(eccentricities) / 3)), 3, sharex=True, sharey=True
)

for i, ecc in enumerate(eccentricities):
    ax = axes.ravel()[i]
    csys = OblateHomoeoidalCoordinateSystem(ecc=ecc)

    # Convert Cartesian (x, z) to native coordinates (λ, μ, φ)
    Lambda, Mu, Phi = csys.from_cartesian(X, 0, Z)

    # Plot λ contours (these correspond to elliptical shells)
    contour = ax.contour(X, Z, Lambda, levels=15, cmap="viridis")
    ax.set_title(f"$\\varepsilon = {ecc}$")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("z")

fig.suptitle(r"Oblate Homoeoidal Coordinate Contours ($\lambda$-lines)", fontsize=14)
plt.tight_layout()
plt.show()
