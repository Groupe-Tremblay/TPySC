from .tpsc import Tpsc
from .mesh import Mesh2D
import numpy as np
from .gf import calcGiwnk, calcNfromG, transform_g_to_direct_space
from scipy.optimize import brentq
import logging

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
        self.mu2 = None

        self.main_results  = {}

        self.trace_chi2 = None
        self.converged = False

        self.usp_crit = -1
        self.delta = -1


    def solve(self,
              alpha: float = 0.5, # TODO Déterminer la valeur de ça
              msd2precision: float = 1e-5,
              msdInfprecision: float = 1e-3,
              iter_max: int = 1_000,
              iter_min: int = 30,
              usp_max: float = 0.,
              usp_prev_T: float = 0.,
              self_energy: np.ndarray = None,
              new_temp: float = 0,
              ) -> None:
        """
        TODO Documentation
        """
        logging.basicConfig(level=logging.DEBUG)
        logging.info("Start of TPSC+ calculations.")

        self.usp_max = usp_max
        self.prev_usp = 0. # XXX This seems deprecated
        self.usp_prev_T = usp_prev_T # XXX This also seems deprecated

        # First do a regular TPSC procedure.
        # Calculate the Green function G1 at the first level of approximation of TPSC.
        self.tpsc_obj.calc_g1()

        if self_energy is None:
            self.tpsc_obj.calc_chi1()
            self.tpsc_obj.calc_usp()
        else:
            # Set the self-energy.
            if np.shape(self_energy)[0] < len(self.mesh.iwn_f): # Check if len(selfE) < len(iwn).
                diffshape = len(self.mesh.iwn_f) - np.shape(self_energy)[0] # If so, gets the difference in lengths.
                self.self_energy = np.zeros(self.mesh.shape, dtype=complex) # Create empty array to fill.
                self.self_energy[diffshape//2:-diffshape//2,:] = self_energy # Fill it with known values (approximative).

            # Compute the new G2.
            dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
            self.mu2 = brentq(lambda m: calcNfromG(self.mesh, self.dispersion[None, :, :] - m + self.self_energy) - self.n, dispersion_min, dispersion_max, disp=True)
            self.g2 = calcGiwnk(self.mesh, self.dispersion[None, :, :] - self.mu2 + self.self_energy)

            # Update chi2.
            self.calc_chi2()

            # Calculate Usp and Uch from the TPSC ansatz.
            self.tpsc_obj.calc_usp() # XXX This could be changed?

        self.tpsc_obj.calc_uch() # XXX This is wrong

        # Calculate the spin and charge susceptibilities.
        self.tpsc_obj.chisp = self.tpsc_obj.calc_chisp(self.Usp)
        self.tpsc_obj.chich = self.tpsc_obj.calc_chich(self.tpsc_obj.Uch)

        # Calculate the double occupancy.
        self.docc = self.tpsc_obj.calc_double_occupancy()

        # Perform the second level approx as usual.
        self.tpsc_obj.calc_second_level_approx()


        # TODO Comment this
        delta_ip1 = 1. - 0.5 * self.tpsc_obj.Usp * self.chi2

        # XXX This makes no sense
        self.newTemp = new_temp

        # Do the TPSC+ loop.
        logging.info("Start of TPSC+ self-consistent loop.")
        for i in range(iter_max):
            print(i) # CONVERGERS ON ITERATION 0 (PROBLEM)

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

            # Calculate Usp and Uch from the TPSC ansatz.
            self.calc_usp()
            self.tpsc_obj.calc_uch()

            # Calculate the spin and charge susceptibilities.
            self.tpsc_obj.chisp = self.tpsc_obj.calc_chisp(self.Usp)
            self.tpsc_obj.chich = self.tpsc_obj.calc_chich(self.tpsc_obj.Uch)

            # Calculate the double occupancy.
            self.docc = self.tpsc_obj.calc_double_occupancy()

            # Perform the second level approx as usual.
            self.tpsc_obj.calc_second_level_approx()

            # Check the convergence
            # delta_i = self.delta_out
            # norm = np.linalg.norm((delta_ip1 - delta_i) / delta_i) / (1 - alpha)
            # norm_inf = np.max(np.abs((delta_ip1 - delta_i) / delta_i)) / (1 - alpha) # TODO Recompute norm.

            ucrit_difference = (self.usp_crit - self.previous_usp_crit) / self.previous_usp_crit
            delta_difference = (self.delta - self.previous_delta) / self.previous_delta

            # norm_conditions = (norm < msd2precision) or (norm_inf < msdInfprecision)
            conditions = (np.abs(ucrit_difference) < 1e-10) or (np.abs(delta_difference) < 1e-10) # TODO Make this adjustable.

            # logging.debug(f"Iteration #{i}, {norm}, {norm_inf}")
            # TODO Use new and improved convergence condition
            # if norm_conditions:)
            #     if (self.delta_p == True) and (self.prev_usp == 0) and (i > iter_min:
            #         self.converged = True
            #         break
            if conditions and self.delta > 0:
                self.converged = True
                break

            # delta_ip1 = delta_i

        if self.converged:
            logging.info("The TPSC+ calculation has converged after {} iterations.".format(i+1))
        else:
            logging.error("The TPSC+ calculation has not converged after {} iterations.".format(iter_max))

        # Update to last values of G^(2) and self-energy.
        self.g2 = self.tpsc_obj.g2
        self.self_energy = self.tpsc_obj.self_energy
        # Check consistency
        self.trace_chi2 = self.mesh.trace('B', self.chi2)
        self.tpsc_obj.check_self_consistency()

        # Prepare output
        self.main_results = {
            "Usp" : self.Usp,
            "Uch" : self.Uch,
            "doubleocc" : self.docc,
            "Trace_chi2" : self.trace_chi2,
            "Trace_Self2_G1" : self.tpsc_obj.trace_self_g1,
            "Trace_Self2_G2" : self.tpsc_obj.trace_self_g2,
            "Exact_Trace_Self2_G" : self.tpsc_obj.exact_trace_self_g,
            "mu1" : self.mu1,
            "mu2" : self.mu2,
            "converged": self.converged
        }
        return self.main_results


    def calc_usp(self, gamma: float = 0.8):
        """
        Docstring for calc_usp

        :param self: Description
        """
        usp_min = 1e-6
        small_num_for_usp_max = 1e-9

        # Relevant parameters TODO use as function params
        U = self.tpsc_obj.U
        n = self.n

        # Compute the trace of chi2 squared.
        trace_chi2_sq = self.mesh.trace('B', self.chi2 * np.conj(self.chi2))

        # Get the two possible upper bounds for usp
        usp_max_abs = U / (1 + U * trace_chi2_sq / (n*n))
        usp_crit = 2 / np.amax(self.chi2).real

        # Find the upper bound for Usp that best suits the situation
        if usp_max_abs < usp_crit: # Typically away from the critical regime
            if self.mesh.trace('B', self.calc_chisp(usp_max_abs)).real - self.calc_sum_rule_chisp(usp_max_abs) > 0:
                usp_max_h = usp_max_abs
            else:
                usp_max_h = usp_crit - small_num_for_usp_max
        else: # In the critical regime
            usp_max_h = usp_crit - small_num_for_usp_max

        # Yury added a while loop to have an upper bound on the positive side of the function's crossing for brentq.
        # TODO Check if this is redundant with the previous if statement
        temp_usp_max_h_rule = self.mesh.trace('B', self.calc_chisp(usp_max_h)).real - self.calc_sum_rule_chisp(usp_max_h)
        usp_braketed = True
        while temp_usp_max_h_rule < 0:
            small_num_for_usp_max /= 10
            if small_num_for_usp_max < 1e-12: # Condition for max iteration
                usp_braketed = False
                break
            usp_max_h = usp_crit - small_num_for_usp_max
            temp_usp_max_h_rule = self.mesh.trace('B', self.calc_chisp(usp_max_h)).real - self.calc_sum_rule_chisp(usp_max_h)

        # Yury added a while loop to select Uspmin on the "right" side of the functions' crossing for brentq.
        while self.mesh.trace('B', self.calc_chisp(usp_min)).real - self.calc_sum_rule_chisp(usp_min) > 0:
             usp_min = gamma * usp_min
             if usp_min < 0.05 * usp_max_h:
                usp_min = 1e-6
                break

        # This should not happen
        if usp_max_h < usp_min:
            usp_min = 1e-6

        if usp_braketed and self.mesh.trace('B', self.calc_chisp(usp_min)).real - self.calc_sum_rule_chisp(usp_min) < 0:
            self.Usp = brentq(lambda m: self.mesh.trace('B', self.calc_chisp(m)).real - self.calc_sum_rule_chisp(m), usp_min, usp_max_h, disp=True)
        else:
            usp_braketed = False
            self.Usp = usp_max_h * gamma

        # Setting new values for next iteration
        self.previous_usp_crit = self.usp_crit # Keep the previous delta in memory for convergence.
        self.usp_crit = usp_crit

        self.previous_delta = self.delta # Keep the previous delta in memory for convergence.
        self.delta = 1 - self.Usp / usp_crit



        # if self.delta > 0:
        #     self.delta_p = True
        # else:
        #     self.delta_p = False


    def calc_chi2(self):
        """
        TODO Documentation
        """
        g2_tau_r, g2_tau_mr = transform_g_to_direct_space(self.mesh, self.g2)
        V = self.tpsc_obj.g1_tau_r * g2_tau_mr[::-1, :] + g2_tau_r * self.tpsc_obj.g1_tau_mr[::-1, :]

        # Fourier transform (r, tau) -> (k, iwn)
        V = self.mesh.r_to_k(V)
        self.chi2 = self.mesh.tau_to_wn('B', V).real


    def __str__(self) -> str:
        # if self.main_results is {}:
        #     return "TPSC was not run, please run the TPSC before printing the results."

        string = ""
        for key,value in self.main_results.items():
            string += f"{key:<20}: {value:5e}\n"

        return string

    # --- Wrapper of the Tpsc class ---
    @property
    def mesh(self):
        return self.tpsc_obj.mesh


    @property
    def dispersion(self):
        return self.tpsc_obj.dispersion


    @property
    def n(self):
        return self.tpsc_obj.n


    @property
    def g1(self):
        return self.tpsc_obj.g1


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
    def Usp(self):
        return self.tpsc_obj.Usp


    @Usp.setter
    def Usp(self, value):
        self.tpsc_obj.Usp = value


    @property
    def Uch(self):
        return self.tpsc_obj.Uch


    @property
    def docc(self):
        return self.tpsc_obj.docc


    @docc.setter
    def docc(self, value):
        self.tpsc_obj.docc = value


    def calc_sum_rule_chisp(self, usp: float):
        return self.tpsc_obj.calc_sum_rule_chisp(usp)

    def calc_chisp(self, usp: float):
        return self.tpsc_obj.calc_chisp(usp)