import numpy as np
from .mesh import Mesh2D


def calcDispersion2DSquare(mesh: Mesh2D, t=1., tp=0., tpp=0.) -> np.ndarray:
    """
    Compute the tight-binding dispersion for a 2D square lattice.

    Evaluates :math:`\\epsilon(\\mathbf{k})` on the k-grid held by ``mesh``,
    including nearest-, next-nearest- (diagonal), and third-nearest-neighbor
    hopping:

    .. math::

        \\epsilon(\\mathbf{k}) = -2t\\,(\\cos(k_x) + \\cos(k_y))
            - 4t'\\,\\cos(k_x)\\cos(k_y)
            - 2t''\\,(\\cos(2 k_x) + \\cos(2 k_y))

    :param mesh: Two-dimensional k-grid on which to evaluate the dispersion.
    :type mesh: Mesh2D
    :param t: Nearest-neighbor hopping amplitude. Defaults to 1.
    :type t: float
    :param tp: Next-nearest-neighbor (diagonal) hopping amplitude. Defaults to 0.
    :type tp: float
    :param tpp: Third-nearest-neighbor hopping amplitude. Defaults to 0.
    :type tpp: float
    :return: The dispersion evaluated on the mesh's k-grid, with the same shape
        as ``mesh.k1``.
    :rtype: numpy.ndarray
    """
    k1 = mesh.k1
    k2 = mesh.k2

    return -2*t*( np.cos(2*np.pi*k1) + np.cos(2*np.pi*k2) ) -4*tp*(np.cos(2*np.pi*k1)*np.cos(2*np.pi*k2)) - 2*tpp*(np.cos(4*np.pi*k1)+np.cos(4*np.pi*k2))

