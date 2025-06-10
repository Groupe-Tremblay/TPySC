from scipy.optimize import brentq

from tpysc.mesh import Mesh2D
import numpy as np


def calcGiwnk(mesh: Mesh2D, z):
    """
    Calculate a general Green's function in the form 1/(iwn - z).
    """
    return 1 / (mesh.iwn_f[:, None, None] - z)


def calcNfromG(mesh: Mesh2D, z):
    """
    Calculate the density from the Green's function and an input chemical potential
    """
    giwnk = calcGiwnk(mesh, z)
    g_tau0 = -mesh.trace('F', giwnk, 1 / mesh.T)
    return 2 * g_tau0.real


def transform_g_to_direct_space(mesh: Mesh2D, greens_function) -> tuple:
    """
    Converts a Green's function G(k, iwn) into real space G(±r, tau).
    """
    g = mesh.wn_to_tau('F', greens_function)
    g_tau_r = mesh.k_to_r(g)
    g_tau_mr = mesh.k_to_mr(g)
    return g_tau_r, g_tau_mr