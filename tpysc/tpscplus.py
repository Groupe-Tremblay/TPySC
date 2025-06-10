from .tpsc import Tpsc
from .mesh import Mesh2D
import numpy as np
from .gf import calcGiwnk, calcNfromG, transform_g_to_direct_space
from scipy.optimize import brentq


class TpscPlus:

    def __init__(self,
                 mesh: Mesh2D,
                 dispersion: np.ndarray,
                 U: float,
                 n: float,
                 ):
        """
        TODO Documentation
        """
        self.tpsc_obj = Tpsc(mesh, dispersion, U, n)

        self.g2 = None
        self.self_energy = None



    def solve(self,
              alpha: float = 0.5, # TODO Déterminer la valeur de ça
              msd2precision: float = 1e-5,
              msdInfprecision: float = 1e-3,
              iter_max: int = 1_000,
              iter_min: int = 30,
              anderson_acc: bool = False
              ) -> None:
        """
        TODO Documentation
        """
        # First do a regular TPSC procedure.
        self.tpsc_obj.solve()

        # TODO Do the TPSC+ loop.
        converged = False
        for i in range(iter_max):
            if anderson_acc == False:

                if i > 0 and alpha > 0:

                    self.self_energy = (1 - alpha) * self.tpsc_obj.self_energy + (alpha) * self.self_energy

                    # Compute the new G2
                    dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
                    self.mu2 = brentq(lambda m: calcNfromG(self.mesh, self.dispersion[None, :, :] - m + self.self_energy) - self.n, dispersion_min, dispersion_max, disp=True)
                    self.g2 = calcGiwnk(self.mesh, self.dispersion[None, :, :] - self.mu2 + self.self_energy)
                else:
                    self.g2 = self.tpsc_obj.g2
                    self.self_energy = self.tpsc_obj.self_energy


                # Update chi2
                self.calc_chi2()

                # Calculate Usp and Uch from the TPSC ansatz
                self.tpsc_obj.calc_usp() # XXX This is not the right function!
                self.tpsc_obj.calc_uch()

                # Calculate the double occupancy
                self.docc = self.tpsc_obj.calc_double_occupancy()
            else:
                pass
                # TODO Anderson acceleration that does not suck
            break
            # ===========
            # Check the convergence
            #correct the norm based on alpha
        #     delta_i = tpsc0.deltaout
        #     preUsp = tpsc0.prevUsp
        #     norm = np.linalg.norm((delta_ip1-delta_i)/delta_i) / (1 - alpha)
        #     norm_inf = np.max(np.abs((delta_ip1-delta_i)/delta_i)) / (1 - alpha)


        #     norm_conditions = (norm < msd2precision) or (norm_inf < msdInfprecision)

        #     if norm_conditions:
        #         if tpsc0.deltap == True and tpsc0.prevUsp == 0 and i > iter_min:
        #             converged = True
        #             nIteFinal = i+1
        #             break
        #         else :
        #             converged = False


        #     delta_ip1= delta_i

        # # TODO PUT THIS IN LOG
        # if converged:
        #     print("The TPSC+ calculation has converged after {} iterations.".format(nIteFinal))
        # else:
        #     print("The TPSC+ calculation has not converged after {} iterations.".format(iter_max))

        # return self.converged
        # ===========


    def calc_chi2(self):
        """
        TODO Documentation
        """
        g2_tau_r, g2_tau_mr = transform_g_to_direct_space(self.mesh, self.g2)

        V = self.g1_tau_r * g2_tau_mr[::-1, :] + g2_tau_r * self.g1_tau_mr[::-1, :]

        # Fourier transform
        V = self.mesh.r_to_k(V)
        self.chi2 = self.mesh.tau_to_wn('B', V)

        # TODO calculer la trace de cet affaire là



    # --- Wrapper of the Tpsc class ---
    @property
    def mesh(self):
        return self.tpsc_obj.mesh


    @property
    def dispersion(self):
        return self.tpsc_obj.dispersion


    @property
    def g1(self):
        return self.tpsc_obj.g1


    @property
    def g1_tau_r(self):
        return self.tpsc_obj.g1_tau_r


    @property
    def g1_tau_mr(self):
        return self.tpsc_obj.g1_tau_mr


    @property
    def chi2(self):
        return self.tpsc_obj.chi1


    @chi2.setter
    def chi2(self, value):
        self.tpsc_obj.chi1 = value # XXX MAKE SURE THIS IS COPIED


    @property
    def mu1(self):
        return self.tpsc_obj.mu1


    @property
    def docc(self):
        return self.tpsc_obj.docc

    @docc.setter
    def docc(self, value):
        self.tpsc_obj.docc = value