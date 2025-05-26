import matplotlib.pyplot as plt
from pymetric.coordinates.coordinate_systems import SphericalCoordinateSystem
#
# Initialize the coordinate system:
#
coordinate_system = SphericalCoordinateSystem()
#
# Define radial and angular ranges:
#
r_vals = np.linspace(0, 1, 6)  # Radial distances
theta_vals = np.linspace(0, np.pi, 12)  # Angular values
phi = 0  # Fix the azimuthal angle
#
# Plot circles (constant r):
#
for r in r_vals:
    theta = np.linspace(0, np.pi, 200)
    coords = [r * np.ones_like(theta), theta, np.full_like(theta, phi)]
    cartesian = coordinate_system._convert_native_to_cartesian(*coords)
    _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)
#
# Plot radial lines (constant theta):
#
for theta in theta_vals:
    r = np.linspace(0, 1, 200)
    coords = [r, theta * np.ones_like(r), np.full_like(r, phi)]
    cartesian = coordinate_system._convert_native_to_cartesian(*coords)
    _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)
#
_ = plt.title('Spherical Coordinate System (Slice)')
_ = plt.xlabel('x')
_ = plt.ylabel('z')
_ = plt.axis('equal')
plt.show()
