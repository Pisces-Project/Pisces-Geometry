"""
User-facing coordinate systems in Pisces-Geometry.

This module contains all the standard coordinate systems that might be of interest for various
use cases.
"""
from pisces_geometry.coordinates.base import class_expression, _CoordinateSystemBase
from pisces_geometry.coordinates.core import OrthogonalCoordinateSystem, CurvilinearCoordinateSystem
import sympy as sp
import numpy as np

class CartesianCoordinateSystem1D(OrthogonalCoordinateSystem):
    r"""
    1 Dimensional Cartesian coordinate system.

    Conversion to and from Cartesian coordinates is trivial as the coordinates are already in this form:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad (x) \to (x) \\
           \text{From Cartesian:} & \quad (x) \to (x)
       \end{aligned}

    Notes
    -----
    Lame coefficients for each axis in the Cartesian coordinate system are all equal to 1. This lack of scaling factors is
    why the system is often preferred in Euclidean geometry calculations.

    +----------+-------------------------+
    | Axis     | Lame Coefficient        |
    +==========+=========================+
    | :math:`x`| :math:`1`               |
    +----------+-------------------------+
    """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['x']
    __PARAMETERS__ = {}

    @staticmethod
    def __construct_metric_tensor_symbol__(x,**kwargs):
        return sp.Array([1])

    def _convert_cartesian_to_native(self, x):
        return x

    def _convert_native_to_cartesian(self, x):
        return x

class CartesianCoordinateSystem2D(OrthogonalCoordinateSystem):
    r"""
    2 Dimensional Cartesian coordinate system.

    Conversion to and from Cartesian coordinates is trivial as the coordinates are already in this form:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad (x,y) \to (x,y) \\
           \text{From Cartesian:} & \quad (x,y) \to (x,y)
       \end{aligned}

    Notes
    -----
    Lame coefficients for each axis in the Cartesian coordinate system are all equal to 1. This lack of scaling factors is
    why the system is often preferred in Euclidean geometry calculations.

    +----------+-------------------------+
    | Axis     | Lame Coefficient        |
    +==========+=========================+
    | :math:`x`| :math:`1`               |
    +----------+-------------------------+
    | :math:`y`| :math:`1`               |
    +----------+-------------------------+

    Examples
    --------
    The Cartesian 2D coordinate system is visualized as a regular grid of constant x and y values.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces_geometry.coordinates.coordinate_systems import CartesianCoordinateSystem2D

        Initialize the coordinate system:

        >>> coordinate_system = CartesianCoordinateSystem2D()

        Define a grid of points:

        >>> x_vals = np.linspace(-1, 1, 10)  # x values
        >>> y_vals = np.linspace(-1, 1, 10)  # y values
        >>> x, y = np.meshgrid(x_vals, y_vals)

        Plot the grid:

        >>> for i in range(len(x_vals)):
        ...     _ = plt.plot(x[:, i], y[:, i], 'k-', lw=0.5)
        ...     _ = plt.plot(x[i, :], y[i, :], 'k-', lw=0.5)

        >>> _ = plt.title('Cartesian 2D Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('y')
        >>> _ = plt.axis('equal')
        >>> plt.show()
    """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['x','y']
    __PARAMETERS__ = {}


    @staticmethod
    def __construct_metric_tensor_symbol__(x,y,**kwargs):
        return sp.Array([1,1])

    def _convert_cartesian_to_native(self, x,y):
        return x,y

    def _convert_native_to_cartesian(self, x,y):
        return x,y

class CartesianCoordinateSystem3D(OrthogonalCoordinateSystem):
    r"""
    3 Dimensional Cartesian coordinate system.

    The Cartesian coordinate system represents a flat, orthogonal system without curvature. It is defined by the coordinates
    :math:`(x, y, z)` in 3D space. In this system, each coordinate axis is a straight line, and all basis vectors are unit vectors,
    meaning that the system does not scale or curve.

    Conversion to and from Cartesian coordinates is trivial as the coordinates are already in this form:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad (x, y, z) \to (x, y, z) \\
           \text{From Cartesian:} & \quad (x, y, z) \to (x, y, z)
       \end{aligned}

    Notes
    -----
    Lame coefficients for each axis in the Cartesian coordinate system are all equal to 1. This lack of scaling factors is
    why the system is often preferred in Euclidean geometry calculations.

    +----------+-------------------------+
    | Axis     | Lame Coefficient        |
    +==========+=========================+
    | :math:`x`| :math:`1`               |
    +----------+-------------------------+
    | :math:`y`| :math:`1`               |
    +----------+-------------------------+
    | :math:`z`| :math:`1`               |
    +----------+-------------------------+

    Examples
    --------

    We can now plot the function:

    .. plot::
        :include-source:

        Let's begin by initializing the Cartesian coordinate system:

        >>> from pisces_geometry.coordinates.coordinate_systems import CartesianCoordinateSystem3D
        >>> coordinate_system = CartesianCoordinateSystem3D()

        We can now initialize a Cartesian grid. We'll use a slice in X-Y:

        >>> grid = np.mgrid[-1:1:100j,-1:1:100j,-1:1:3j]
        >>> grid = np.moveaxis(grid,0,-1) # fix the grid ordering to meet our standard

        Let's now create a function on this geometry.

        >>> func = lambda x,y: np.cos(y)*np.sin(x*y)
        >>> Z = func(grid[...,0],grid[...,1])

        >>> import matplotlib.pyplot as plt
        >>> image_array = Z[:,:,1].T
        >>> plt.imshow(image_array,origin='lower',extent=(-1,1,-1,1),cmap='inferno') # doctest: +SKIP

    """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['x','y','z']
    __PARAMETERS__ = {}


    @staticmethod
    def __construct_metric_tensor_symbol__(x,y,z,**kwargs):
        return sp.Array([1,1,1])

    def _convert_cartesian_to_native(self, x,y,z):
        return x,y

    def _convert_native_to_cartesian(self, x,y,z):
        return x,y

class SphericalCoordinateSystem(OrthogonalCoordinateSystem):
    r"""
    3 Dimensional Spherical coordinate system.

    The spherical coordinate system is defined by the coordinates :math:`(r, \theta, \phi)`, where :math:`r` is the radial distance,
    :math:`\theta` is the polar angle, and :math:`\phi` is the azimuthal angle. This system is ideal for spherical symmetries,
    such as in gravitational and electrostatic fields.

    Conversion between spherical and Cartesian coordinates follows these standard transformations:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = r \sin(\theta) \cos(\phi), \\
                                & \quad y = r \sin(\theta) \sin(\phi), \\
                                & \quad z = r \cos(\theta) \\
           \text{From Cartesian:} & \quad r = \sqrt{x^2 + y^2 + z^2}, \\
                                  & \quad \theta = \arccos\left(\frac{z}{r}\right), \\
                                  & \quad \phi = \arctan2(y, x)
       \end{aligned}

    Notes
    -----
    The Lame coefficients for each axis in the spherical coordinate system are provided in the table below, reflecting the
    scaling factors due to curvature in radial and angular directions.

    +-------------------+------------------------------+
    | Axis              | Lame Coefficient             |
    +===================+==============================+
    |:math:`r`          | :math:`1`                    |
    +-------------------+------------------------------+
    | :math:`\theta`    | :math:`r`                    |
    +-------------------+------------------------------+
    | :math:`\phi`      | :math:`r \sin(\theta)`       |
    +-------------------+------------------------------+

    Examples
    --------
    The Spherical coordinate system is visualized with circles (constant r) and lines radiating from the origin (constant theta) on a 2D slice.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces_geometry.coordinates.coordinate_systems import SphericalCoordinateSystem

        Initialize the coordinate system:

        >>> coordinate_system = SphericalCoordinateSystem()

        Define radial and angular ranges:

        >>> r_vals = np.linspace(0, 1, 6)  # Radial distances
        >>> theta_vals = np.linspace(0, np.pi, 12)  # Angular values
        >>> phi = 0  # Fix the azimuthal angle

        Plot circles (constant r):

        >>> for r in r_vals:
        ...     theta = np.linspace(0, np.pi, 200)
        ...     coords = [r * np.ones_like(theta), theta, np.full_like(theta, phi)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)

        Plot radial lines (constant theta):

        >>> for theta in theta_vals:
        ...     r = np.linspace(0, 1, 200)
        ...     coords = [r, theta * np.ones_like(r), np.full_like(r, phi)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)

        >>> _ = plt.title('Spherical Coordinate System (Slice)')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()
    """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['r','theta','phi']
    __PARAMETERS__ = {}

    @staticmethod
    def __construct_metric_tensor_symbol__(r,theta,phi,**kwargs):
        return sp.Array([1,r**2,(r*sp.sin(theta))**2])

    def _convert_cartesian_to_native(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        return r,theta,phi


    def _convert_native_to_cartesian(self, r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return x,y,z

class PolarCoordinateSystem(OrthogonalCoordinateSystem):
    r"""
    2 Dimensional Polar coordinate system.

    The polar coordinate system is used for 2D spaces with rotational symmetry, defined by the coordinates :math:`(r, \theta)`,
    where :math:`r` represents the radial distance from the origin and :math:`\theta` represents the angle from a reference axis.

    Conversion between polar and Cartesian coordinates is as follows:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = r \cos(\theta), \\
                                & \quad y = r \sin(\theta) \\
           \text{From Cartesian:} & \quad r = \sqrt{x^2 + y^2}, \\
                                  & \quad \theta = \arctan2(y, x)
       \end{aligned}

    Notes
    -----
    The Lame coefficients for each axis in the polar coordinate system provide a scaling factor as shown below.

    +-----------------+-------------------------+
    | Axis            | Lame Coefficient        |
    +=================+=========================+
    | :math:`r`       | :math:`1`               |
    +-----------------+-------------------------+
    | :math:`\theta`  | :math:`r`               |
    +-----------------+-------------------------+

    Examples
    --------
    The Polar coordinate system is visualized as concentric circles (constant r) and radial lines (constant theta).

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces_geometry.coordinates.coordinate_systems import PolarCoordinateSystem

        Initialize the coordinate system:

        >>> coordinate_system = PolarCoordinateSystem()

        Define the radial and angular ranges:

        >>> r_vals = np.linspace(0, 1, 6)  # Radial distances
        >>> theta_vals = np.linspace(0, 2 * np.pi, 12)  # Angular values

        Plot concentric circles (constant r):

        >>> for r in r_vals:
        ...     theta = np.linspace(0, 2 * np.pi, 200)
        ...     coords = [r * np.ones_like(theta), theta]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(*cartesian, 'k-', lw=0.5)

        Plot radial lines (constant theta):

        >>> for theta in theta_vals:
        ...     r = np.linspace(0, 1, 200)
        ...     coords = [r, theta * np.ones_like(r)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(*cartesian, 'k-', lw=0.5)

        >>> _ = plt.title('Polar Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('y')
        >>> _ = plt.axis('equal')
        >>> plt.show()
        """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['r','theta']
    __PARAMETERS__ = {}

    @staticmethod
    def __construct_metric_tensor_symbol__(r,theta,**kwargs):
        return sp.Array([1,r**2])

    def _convert_native_to_cartesian(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def _convert_cartesian_to_native(self, x,y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

class CylindricalCoordinateSystem(OrthogonalCoordinateSystem):
    r"""
    3 Dimensional Cylindrical coordinate system.

    The cylindrical coordinate system is defined by the coordinates :math:`(\rho, \phi, z)`, where :math:`\rho` is the radial
    distance in the xy-plane, :math:`\phi` is the azimuthal angle, and :math:`z` is the height along the z-axis.
    This system is commonly used in problems with axial symmetry, such as electromagnetic fields around a wire.

    Conversion between cylindrical and Cartesian coordinates is defined as:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = \rho \cos(\phi), \\
                                & \quad y = \rho \sin(\phi), \\
                                & \quad z = z \\
           \text{From Cartesian:} & \quad \rho = \sqrt{x^2 + y^2}, \\
                                  & \quad \phi = \arctan2(y, x), \\
                                  & \quad z = z
       \end{aligned}

    Notes
    -----
    The Lame coefficients for each axis in the cylindrical coordinate system are detailed below, reflecting scaling
    factors associated with the radial and angular coordinates.

    +-----------------+----------------------------+
    | Axis            | Lame Coefficient           |
    +=================+============================+
    | :math:`\rho`    | :math:`1`                  |
    +-----------------+----------------------------+
    | :math:`\phi`    | :math:`\rho`               |
    +-----------------+----------------------------+
    | :math:`z`       |  :math:`1`                 |
    +-----------------+----------------------------+

    Examples
    --------
    The Cylindrical coordinate system is visualized as concentric circles (constant rho) and vertical lines (constant z) on a 2D slice.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from pisces_geometry.coordinates.coordinate_systems import CylindricalCoordinateSystem

        Initialize the coordinate system:

        >>> coordinate_system = CylindricalCoordinateSystem()

        Define the radial and angular ranges:

        >>> rho_vals = np.linspace(0,1,10)  # Radial distances
        >>> phi_vals = np.linspace(0, 2 * np.pi, 12)  # Angular values
        >>> z = 0  # Fix the z-coordinate

        Plot concentric circles (constant rho):

        >>> for rho in rho_vals[2:]:
        ...     phi = np.linspace(0, 2 * np.pi, 200)
        ...     coords = [rho * np.ones_like(phi), phi, np.full_like(phi, z)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(*cartesian[:-1],'k', lw=0.5)

        Plot radial lines (constant phi):

        >>> for phi in phi_vals:
        ...     pass
        ...     rho = np.linspace(0, 1, 200)
        ...     coords = [rho, phi * np.ones_like(rho), np.full_like(rho, z)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(*cartesian[:-1], 'k-', lw=0.5)

        >>> _ = plt.title('Cylindrical Coordinate System (Slice)')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('y')
        >>> _ = plt.axis('equal')
        >>> plt.show()
        """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['rho','phi','z']
    __PARAMETERS__ = {}

    def _convert_native_to_cartesian(self, rho,phi,z):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y, z
    def _convert_cartesian_to_native(self, x,y,z):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi, z

    @staticmethod
    def __construct_metric_tensor_symbol__(r,theta,phi,**kwargs):
        return sp.Array([1,r**2,1])

class OblateSpheroidalCoordinateSystem(OrthogonalCoordinateSystem):
    r"""
    3 Dimensional Oblate Spheroidal coordinate system.

    Oblate Spheroidal coordinates are defined using the hyperbolic coordinates :math:`\mu` and :math:`\nu` such that the surfaces of constant :math:`\mu` are oblate spheroids.

    Conversion to Cartesian coordinates is given by:

    .. math::

       \begin{aligned}
           x &= a \cosh(\mu) \cos(\nu) \cos(\phi), \\
           y &= a \cosh(\mu) \cos(\nu) \sin(\phi), \\
           z &= a \sinh(\mu) \sin(\nu).
       \end{aligned}

    Conversion from Cartesian coordinates:

    .. math::

       \begin{aligned}
           \mu &= \operatorname{arccosh}\left(\frac{\sqrt{x^2 + y^2 + z^2 + a^2}}{2a}\right), \\
           \nu &= \arcsin\left(\frac{z}{a \sinh(\mu)}\right), \\
           \phi &= \arctan2(y, x).
       \end{aligned}

    Parameters
    ----------
    a : float
        Semi-major axis defining the scale of the coordinate system.

    Notes
    -----
    The Lamé coefficients in this system are:

    +---------------+---------------------------------------------+
    | Axis          | Lamé Coefficient                            |
    +===============+=============================================+
    | :math:`\mu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\nu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\phi`  | :math:`a \cosh(\mu) \cos(\nu)`              |
    +---------------+---------------------------------------------+

    Examples
    --------
    The Oblate Spheroidal coordinate system is visualized by converting level surfaces to Cartesian coordinates.
    This plot shows an axial slice of constant :math:`\phi = 0`.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces_geometry.coordinates.coordinate_systems import OblateSpheroidalCoordinateSystem

        Initialize the coordinate system:

        >>> a = 1.0  # Semi-major axis
        >>> coordinate_system = OblateSpheroidalCoordinateSystem(a=a)

        Define the coordinate ranges:

        >>> mu_vals = np.linspace(0, 2, 6)  # Range of mu values
        >>> nu_vals = np.linspace(-np.pi / 2, np.pi / 2, 12)  # Range of nu values
        >>> phi = 0  # Fix the azimuthal angle

        Plot constant :math:`\mu` surfaces:

        >>> for mu in mu_vals:
        ...     nu = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ...     coords = [mu * np.ones_like(nu), nu, np.full_like(nu, phi)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)

        Plot constant :math:`\nu` surfaces:

        >>> for nu in nu_vals:
        ...     mu = np.linspace(0, 2, 200)
        ...     coords = [mu, np.full_like(mu, nu), np.full_like(mu, phi)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)

        >>> _ = plt.title('Oblate Spheroidal Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()

    """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['mu','nu','phi']
    __PARAMETERS__ = {'a': 1.0}

    def _convert_native_to_cartesian(self, mu,nu,phi):
        a = self.parameters['a']
        x = a * np.cosh(mu) * np.cos(nu) * np.cos(phi)
        y = a * np.cosh(mu) * np.cos(nu) * np.sin(phi)
        z = a * np.sinh(mu) * np.sin(nu)
        return x, y, z

    def _convert_cartesian_to_native(self, x,y,z):
        a = self.parameters['a']
        rho = np.sqrt(x**2 + y**2)
        d1_2, d2_2 = (rho + a) ** 2 + z**2, (rho - a) ** 2 + z**2
        mu = np.arccosh((np.sqrt(d1_2) + np.sqrt(d2_2)) / (2 * a))
        nu = np.arccos((np.sqrt(d1_2) - np.sqrt(d2_2)) / (2 * a))
        phi = np.arctan2(y, x)
        return mu, nu, phi

    @staticmethod
    def __construct_metric_tensor_symbol__(mu,nu,phi,a=1.0):
        return sp.Array([(a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2))**2,
                         (a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2))**2,
                         (a * sp.cosh(mu) * sp.cos(nu))**2])

class ProlateSpheroidalCoordinateSystem(OrthogonalCoordinateSystem):
    r"""
    3 Dimensional Prolate Spheroidal coordinate system.

    Prolate Spheroidal coordinates are defined using the hyperbolic coordinates :math:`\mu` and :math:`\nu` such that the surfaces of constant :math:`\mu` are prolate spheroids.

    Conversion to Cartesian coordinates is given by:

    .. math::

       \begin{aligned}
           x &= a \sinh(\mu) \sin(\nu) \cos(\phi), \\
           y &= a \sinh(\mu) \sin(\nu) \sin(\phi), \\
           z &= a \cosh(\mu) \cos(\nu).
       \end{aligned}

    Conversion from Cartesian coordinates:

    .. math::

       \begin{aligned}
           \mu &= \operatorname{arccosh}\left(\frac{\sqrt{x^2 + y^2 + z^2 - a^2}}{2a}\right), \\
           \nu &= \arccos\left(\frac{z}{a \cosh(\mu)}\right), \\
           \phi &= \arctan2(y, x).
       \end{aligned}

    Parameters
    ----------
    a : float
        Semi-major axis defining the scale of the coordinate system.

    Notes
    -----
    The Lamé coefficients in this system are:

    +---------------+---------------------------------------------+
    | Axis          | Lamé Coefficient                            |
    +===============+=============================================+
    | :math:`\mu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\nu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\phi`  | :math:`a \sinh(\mu) \sin(\nu)`              |
    +---------------+---------------------------------------------+

    Examples
    --------
    The Prolate Spheroidal coordinate system is visualized by converting level surfaces to Cartesian coordinates.
    This plot shows an axial slice of constant :math:`\phi = 0`.

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from pisces_geometry.coordinates.coordinate_systems import ProlateSpheroidalCoordinateSystem

        Initialize the coordinate system:

        >>> a = 1.0  # Semi-major axis
        >>> coordinate_system = ProlateSpheroidalCoordinateSystem(a=a)

        Define the coordinate ranges:

        >>> mu_vals = np.linspace(0, 2, 6)  # Range of mu values
        >>> nu_vals = np.linspace(0, np.pi, 12)  # Range of nu values
        >>> phi = 0  # Fix the azimuthal angle

        Plot constant :math:`\mu` surfaces:

        >>> for mu in mu_vals:
        ...     nu = np.linspace(0, np.pi, 200)
        ...     coords = [mu * np.ones_like(nu), nu, np.full_like(nu, phi)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)

        Plot constant :math:`\nu` surfaces:

        >>> for nu in nu_vals:
        ...     mu = np.linspace(0, 2, 200)
        ...     coords = [mu, np.full_like(mu, nu), np.full_like(mu, phi)]
        ...     cartesian = coordinate_system._convert_native_to_cartesian(*coords)
        ...     _ = plt.plot(cartesian[0],cartesian[2], 'k-', lw=0.5)

        >>> _ = plt.title('Prolate Spheroidal Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()
    """
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['mu','nu','phi']
    __PARAMETERS__ = {'a': 1.0}

    def _convert_native_to_cartesian(self,mu,nu,phi):
        a = self.parameters['a']
        x = a * np.sinh(mu) * np.sin(nu) * np.cos(phi)
        y = a * np.sinh(mu) * np.sin(nu) * np.sin(phi)
        z = a * np.cosh(mu) * np.cos(nu)
        return x, y, z

    def _convert_cartesian_to_native(self, x,y,z):
        a = self.parameters['a']
        rho = np.sqrt(x**2 + y**2)
        d1_2, d2_2 = (rho) ** 2 + (z + a) ** 2, (rho) ** 2 + (z - a) ** 2
        mu = np.arccosh((np.sqrt(d1_2) + np.sqrt(d2_2)) / (2 * a))
        nu = np.arccos((np.sqrt(d1_2) - np.sqrt(d2_2)) / (2 * a))
        phi = np.arctan2(y, x)
        return mu, nu, phi

    @staticmethod
    def __construct_metric_tensor_symbol__(mu,nu,phi,a=1.0):
        return sp.Array([(a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2))**2,
                         (a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2))**2,
                         (a * sp.sinh(mu) * sp.sin(nu))**2])

class EllipticCylindricalCoordinateSystem(OrthogonalCoordinateSystem):
    r"""
    3 Dimensional Elliptic Cylindrical coordinate system.

    The elliptic cylindrical coordinate system is a 3D orthogonal coordinate system defined by
    coordinates :math:`(\mu, \nu, z)`, where:

    - :math:`\mu \in [0, \infty)` is a radial-like coordinate (confocal ellipses)
    - :math:`\nu \in [0, 2\pi)` is an angular coordinate (confocal hyperbolae)
    - :math:`z \in (-\infty, \infty)` is the standard height coordinate (same as Cartesian)

    The Cartesian coordinates are given by:

    .. math::
        x = a \cosh(\mu) \cos(\nu)\\
        y = a \sinh(\mu) \sin(\nu)\\
        z = z

    The Lamé coefficients are:

    +-------------+---------------------------------------------+
    | Axis        | Lamé Coefficient                            |
    +=============+=============================================+
    | :math:`\mu` | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +-------------+---------------------------------------------+
    | :math:`\nu` | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +-------------+---------------------------------------------+
    | :math:`z`   | :math:`1`                                   |
    +-------------+---------------------------------------------+

    Parameters
    ----------
    a : float
        Focal half-distance (scaling parameter), must be nonzero.
    """

    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['mu', 'nu', 'z']
    __PARAMETERS__ = {'a': 1.0}

    @staticmethod
    def __construct_metric_tensor_symbol__(mu, nu, z, a=1.0):
        h = a * sp.sqrt(sp.sinh(mu)**2 + sp.sin(nu)**2)
        return sp.Array([h**2, h**2, 1])

    def _convert_native_to_cartesian(self, mu, nu, z):
        a = self.__parameters__['a']
        x = a * np.cosh(mu) * np.cos(nu)
        y = a * np.sinh(mu) * np.sin(nu)
        return x, y, z

    def _convert_cartesian_to_native(self, x, y, z):
        a = self.__parameters__['a']
        cosh_mu = np.sqrt((x / a)**2 + (y / a)**2 +
                          np.sqrt(((x / a)**2 + (y / a)**2)**2 - (4 * x**2 * y**2) / a**4)) / 2
        sinh_mu = np.sqrt(cosh_mu**2 - 1)
        mu = np.arccosh(cosh_mu)
        tan_nu = y / (a * sinh_mu)
        cos_nu = x / (a * cosh_mu)
        nu = np.arctan2(tan_nu, cos_nu)
        return mu, nu, z

class OblateHomoeoidalCoordinateSystem(CurvilinearCoordinateSystem):
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['xi','theta','phi']
    __PARAMETERS__ = {'ecc': 0.0}

    def _setup_parameters(self,**kwargs):
        # Start by creating a carbon-copy of the default parameters.
        _params = super()._setup_parameters(**kwargs)

        # Create the scale factor attribute as well.
        _params['scale'] = 1/np.sqrt(1-_params['ecc']**2)

        return _params

    @staticmethod
    def __construct_metric_tensor_symbol__(xi,theta,phi,ecc=0.0):
        st, ct = sp.sin(theta), sp.cos(theta)
        omega = sp.sqrt(1-(ecc*st)**2)
        domega_dtheta = sp.diff(omega, theta)

        return sp.Matrix([[1/omega**2                  ,(-xi*domega_dtheta)/omega**3                     , 0],
                          [(-xi*domega_dtheta)/omega**3, ((xi**2)*(omega**2 + domega_dtheta**2))/omega**4, 0],
                          [0                           ,0                                                , (xi*st/omega)**2]])

    @staticmethod
    def __construct_inverse_metric_tensor_symbol__(xi,theta,phi,ecc=0.0):
        st, ct = sp.sin(theta), sp.cos(theta)
        omega = sp.sqrt(1-(ecc*st)**2)
        domega_dtheta = (st*ct)/omega

        return sp.Matrix([[omega**2+domega_dtheta**2   ,(omega*domega_dtheta)/xi, 0                  ],
                          [(omega*domega_dtheta)/xi    , (omega/xi)**2          , 0                  ],
                          [0                           ,0                       , (omega/(st*xi))**2]])


    def _convert_native_to_cartesian(self, xi, theta, phi):
        # Fetch the numerical conversion function.
        ecc = self.__parameters__['ecc']
        _st, _ct, _cp, _sp = np.sin(theta), np.cos(theta), np.cos(phi), np.sin(phi)
        _omega = np.sqrt(1-(ecc*_st)**2)
        # COMPUTE inversion
        x = xi * _st * _cp / _omega
        y = xi * _st * _sp / _omega
        z = xi * _ct / _omega
        return x, y, z

    def _convert_cartesian_to_native(self, x,y,z):
        scale = self.__parameters__['scale']
        xi = np.sqrt(
            (x/scale) ** 2 + (y/scale) ** 2 + z ** 2
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return xi, theta, phi

class ProlateHomoeoidalCoordinateSystem(CurvilinearCoordinateSystem):
    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['xi','theta','phi']
    __PARAMETERS__ = {'ecc': 0.0}

    def _setup_parameters(self,**kwargs):
        # Start by creating a carbon-copy of the default parameters.
        _params = super()._setup_parameters(**kwargs)

        # Create the scale factor attribute as well.
        _params['scale'] = np.sqrt(1-_params['ecc']**2)

        return _params

    @staticmethod
    def __construct_metric_tensor_symbol__(xi,theta,phi,ecc=0.0):
        st, ct = sp.sin(theta), sp.cos(theta)
        omega = sp.sqrt(1-(ecc*ct)**2)
        domega_dtheta = sp.diff(omega, theta)

        return sp.Matrix([[1/omega**2                  ,(-xi*domega_dtheta)/omega**3                     , 0],
                          [(-xi*domega_dtheta)/omega**3, ((xi**2)*(omega**2 + domega_dtheta**2))/omega**4, 0],
                          [0                           ,0                                                , (xi*st/omega)**2]])

    @staticmethod
    def __construct_inverse_metric_tensor_symbol__(xi,theta,phi,ecc=0.0):
        st, ct = sp.sin(theta), sp.cos(theta)
        omega = sp.sqrt(1-(ecc*ct)**2)
        domega_dtheta = (st*ct)/omega

        return sp.Matrix([[omega**2+domega_dtheta**2   ,(omega*domega_dtheta)/xi, 0                  ],
                          [(omega*domega_dtheta)/xi    , (omega/xi)**2          , 0                  ],
                          [0                           ,0                       , (omega/(st*xi))**2]])


    def _convert_native_to_cartesian(self, xi, theta, phi):
        # Fetch the numerical conversion function.
        ecc = self.__parameters__['ecc']
        _st, _ct, _cp, _sp = np.sin(theta), np.cos(theta), np.cos(phi), np.sin(phi)
        _omega = np.sqrt(1-(ecc*_ct)**2)
        # COMPUTE inversion
        x = xi * _st * _cp / _omega
        y = xi * _st * _sp / _omega
        z = xi * _ct / _omega
        return x, y, z

    def _convert_cartesian_to_native(self, x,y,z):
        scale = self.__parameters__['scale']
        xi = np.sqrt(
            x ** 2 + y ** 2 + (z/scale) ** 2
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return xi, theta, phi

class ConicCoordinateSystem(OrthogonalCoordinateSystem):
    r"""
    3 Dimensional Conic coordinate system.

    The conic coordinate system is a 3D orthogonal coordinate system derived from intersecting conical surfaces.
    It is defined by coordinates :math:`(\mu, \nu, \phi)`, where:

    - :math:`\mu \in (0, \pi)` is the angle between a point and a reference axis
    - :math:`\nu \in (0, \pi)` is the angle from a second family of intersecting cones
    - :math:`\phi \in [0, 2\pi)` is the azimuthal angle (rotation around the z-axis)

    The Cartesian coordinates are given by:

    .. math::
        x = a \frac{\sin(\mu) \sin(\nu)}{\cos(\mu) - \cos(\nu)} \cos(\phi)

        y = a \frac{\sin(\mu) \sin(\nu)}{\cos(\mu) - \cos(\nu)} \sin(\phi)

        z = a \frac{\sin(\mu + \nu)}{\cos(\mu) - \cos(\nu)}

    Parameters
    ----------
    a : float
        Focal distance parameter, must be nonzero.
    """

    __is_abstract__ = False
    __setup_point__ = 'init'
    __AXES__ = ['mu', 'nu', 'phi']
    __PARAMETERS__ = {'a': 1.0}

    @staticmethod
    def __construct_metric_tensor_symbol__(mu, nu, phi, a=1.0):
        denom = (sp.cos(mu) - sp.cos(nu))
        r = a * sp.sin(mu) * sp.sin(nu) / denom
        z = a * sp.sin(mu + nu) / denom
        h_mu = a * sp.sqrt(1 - sp.cos(mu)**2) * sp.sqrt(1 - sp.cos(nu)**2) / denom**2
        h_nu = h_mu
        h_phi = r
        return sp.Array([h_mu**2, h_nu**2, h_phi**2])

    def _convert_native_to_cartesian(self, mu, nu, phi):
        a = self.__parameters__['a']
        denom = np.cos(mu) - np.cos(nu)
        r = a * np.sin(mu) * np.sin(nu) / denom
        z = a * np.sin(mu + nu) / denom
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y, z

    def _convert_cartesian_to_native(self, x, y, z):
        raise NotImplementedError("Inverse transformation for conic coordinates is not trivial and currently not implemented.")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pisces_geometry.differential_geometry import compute_gradient

    u = ProlateHomoeoidalCoordinateSystem(ecc=0.0)
    print(u.get_expression('Dterm'))