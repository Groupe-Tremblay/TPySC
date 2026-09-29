from scipy.optimize import brentq

from tpysc.mesh import Mesh2D
import numpy as np


def calcGiwnk(mesh: Mesh2D, z) -> "np.ndarray | complex":
    """
    Calculate a ferminonic Matsubara Green's function in the form 1/(iwn - z).

    :param mesh: The 2D mesh defining the Matsubara frequency grid and temperature.
    :type mesh: Mesh2D
    :param z: The quantity subtracted from iwn (e.g. a dispersion or self-energy
        array), broadcastable against the fermionic Matsubara frequencies.
    :type z: numpy.ndarray or complex
    :return: The Green's function 1/(iwn - z), with Matsubara frequency as the
        leading axis.
    :rtype: numpy.ndarray or complex
    """
    iwn = 1j * mesh.IR_basis_set.wn_f * np.pi * mesh.T
    return 1 / (iwn[:, None, None] - z)


def calcNfromG(mesh: Mesh2D, z) -> "float | np.ndarray":
    """
    Calculate the density from the fermionic Green's function.

    :param mesh: The 2D mesh containing the Matsubara frequency grid and temperature.
    :type mesh: Mesh2D
    :param z: The quantity subtracted from iwn when building the Green's function
        (e.g. a dispersion shifted by the chemical potential), passed through to
        :func:`calcGiwnk`.
    :type z: numpy.ndarray or complex
    :return: The density computed from the equal-time Green's function.
    :rtype: float or numpy.ndarray
    """
    giwnk = calcGiwnk(mesh, z)
    g_tau0 = -mesh.trace('F', giwnk, 1 / mesh.T)
    return 2 * g_tau0.real


def transform_g_to_direct_space(mesh: Mesh2D, greens_function) -> tuple:
    """
    Converts a Green's function G(k, iwn) into real space G(±r, tau).

    :param mesh: The 2D mesh providing the frequency-to-time and momentum-to-real-
        space transforms.
    :type mesh: Mesh2D
    :param greens_function: The Green's function in momentum and Matsubara
        frequency space, G(k, iwn).
    :type greens_function: numpy.ndarray
    :return: A tuple ``(g_tau_r, g_tau_mr)`` of the Green's function in real space
        and imaginary time, evaluated at +r and -r respectively.
    :rtype: tuple
    """
    g = mesh.wn_to_tau('F', greens_function)
    g_tau_r = mesh.k_to_r(g)
    g_tau_mr = mesh.k_to_mr(g)
    return g_tau_r, g_tau_mr
