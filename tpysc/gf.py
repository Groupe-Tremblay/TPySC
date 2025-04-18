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
        self.gtaur = None
        self.gtaumr = None
        self.giwnk = None


    def calcGiwnkFromMu(self, mu):
        """
        Calculate Green function G(iwn,k) from an input chemical potential
        """
        if self.selfEnergy is not None:
            self.giwnk = 1./(self.mesh.iwn_f_ - (self.mesh.ek_ - mu) - self.selfEnergy)
        else:
            self.giwnk = 1./(self.mesh.iwn_f_ - (self.mesh.ek_ - mu))

    def calcGtaur(self):
        """
        Calculate real space Green function G(tau,r) [for calculating chi0 and sigma]
        """
        # Fourier transform
        # Calculation of G
        gtaur = self.mesh.k_to_r(self.giwnk)
        self.gtaur = self.mesh.wn_to_tau('F', gtaur)

    def calcGtaumr(self):
        """
        Calculate real space Green function G(tau,-r) [for calculating chi0 and sigma]
        """
        # Fourier transform
        # Calculation of G
        gtaumr = self.mesh.k_to_mr(self.giwnk)
        self.gtaumr = self.mesh.wn_to_tau('F', gtaumr)


    def calcGiwnk(self, z):
        """
        Calculate a general Green's function in the form 1/(iwn - z).
        """
        # return 1 / (self.mesh.iwn_f[:, None, None] - z[None, :, :])
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







