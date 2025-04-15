from scipy.optimize import brentq

from tpysc.Dispersions import *
from tpysc.Mesh import *

"""
Date: June 23, 2023
"""
class GF:
    def __init__(self, mesh, n, selfEnergy=None):
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
        self.n = n

        # Set the self-energy
        self.selfEnergy = selfEnergy

        # Calculate the chemical potential
        self.calcMu()

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

    def calcNfromG(self, mu):
        """
        Calculate the density from the Green's function and an input chemical potential
        """
        self.calcGiwnkFromMu(mu)

        gio  = np.sum(self.giwnk,axis=1)/self.mesh.nk
        g_l  = self.mesh.IR_basis_set.smpl_wn_f.fit(gio)
        g_tau0 = -self.mesh.IR_basis_set.basis_f.u(1/self.mesh.T)@g_l

        return 2*g_tau0.real

    def calcMu(self):
        """
        Calculate the chemical potential for the Green's function
        """
        self.mu = brentq(lambda m: self.calcNfromG(m)-self.n, np.amin(self.mesh.ek), np.amax(self.mesh.ek), disp=True)







