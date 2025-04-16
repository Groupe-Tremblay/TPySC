import numpy as np
from .mesh import Mesh2D

"""
Date: June 22, 2023
"""
def calcDispersion2DSquare(mesh: Mesh2D, t=1., tp=0., tpp=0.):
    k1 = mesh.k1
    k2 = mesh.k2
    nk = mesh.nk

    return -2*t*( np.cos(2*np.pi*k1) + np.cos(2*np.pi*k2) ).reshape(nk) -4*tp*(np.cos(2*np.pi*k1)*np.cos(2*np.pi*k2)).reshape(nk) - 2*tpp*(np.cos(4*np.pi*k1)+np.cos(4*np.pi*k2)).reshape(nk)


def calcDispersion2DTriangle(k1, k2, nk, t=1., tp=-1.):
    return -2*t*( np.cos(2*np.pi*k1) + np.cos(2*np.pi*k2) ).reshape(nk) -2*tp*(np.cos(2*np.pi*k1+2*np.pi*k2)).reshape(nk)

