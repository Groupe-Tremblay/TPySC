from scipy.optimize import brentq

from tpysc.mesh import Mesh2D
import numpy as np

"""
Date: June 23, 2023
"""
class GF:
    def __init__(self, mesh: Mesh2D):
        """
        Class to create an interacting Green function
        Inputs:
        mesh: Mesh object from the Mesh.py file
        n: input density
        selfEnergy: Table with the self-energy if calculating G, or None if calculating G0
        Credits for part of the code: Niklas Witt
        """
        # Initialize the input quantities
        self.mesh = mesh

        # Initialize the various dependencies
        self._gtaur = None
        self._gtaumr = None
        self.giwnk = None


    @property
    def gtaur(self) -> np.ndarray:
        # TODO Doc
        if self._gtaur is None:
            g = self.mesh.k_to_r(self.giwnk)
            self._gtaur = self.mesh.wn_to_tau('F', g)

        return self._gtaur


    @property
    def gtaumr(self) -> np.ndarray:
        # TODO Doc
        if self._gtaumr is None:
            g  = self.mesh.k_to_mr(self.giwnk)
            self._gtaumr = self.mesh.wn_to_tau('F', g)

        return self._gtaumr


    def calcGiwnk(self, z):
        """
        Calculate a general Green's function in the form 1/(iwn - z).
        """
        return 1 / (self.mesh.iwn_f[:, None, None] - z)


    def calcNfromG(self, z):
        """
        Calculate the density from the Green's function and an input chemical potential
        """
        self.giwnk = self.calcGiwnk(z)
        print(self.giwnk.shape)
        g_tau0 = -self.mesh.trace(self.giwnk, 'F', 1 / self.mesh.T)
        print(g_tau0.shape)
        return 2 * g_tau0.real







