from .gf import GF
from .mesh import Mesh2D
import matplotlib.pyplot as plt
import json
import numpy as np

from scipy.optimize import brentq

class TPSC:
    """
    Class to set up a TPSC calculation.
    Calculation is carried using the ``run()`` method.

    :param n: Density
    :type n: double
    :param U: Hubbard interaction
    :type U: double
    :param t: First neighbour hopping
    :type t: double
    :param tp: Second neighbour hopping
    :type tp: double
    :param tpp: Third neighbour hopping
    :type tpp: double
    :param nkx: Number of k-points in one space direction
    :type nkx: int
    :param dispersion_scheme: Dispersion scheme (either `triangle` or `square`)
    :type dispersion_scheme: str
    :param T: Temperature
    :type T: double
    :param wmax_mult: For IR basis, multiple of bandwidth to use as wmax (must be greater than 1)
    :type wmax_mult: double, optional
    :param IR_tol: For IR basis, tolerance of intermediate representation (default = 1e-12)
    :type IR_tol: double, optional
    """
    def __init__(self,
                 mesh: Mesh2D,
                 dispersion: np.ndarray,
                 U: float,
                 n: float,
                 ):

        self.mesh = mesh
        self.dispersion = dispersion

        self.n = n
        self.U = U

        # Member to hold the results
        self.g1 = None
        self.g2 = None
        self.mu1 = None
        self.mu2 = None
        self.selfEnergy = None
        self.main_results = {}
        self.Uch = -1.0
        self.Usp = -1.0
        self.docc = -1.0

        self.traceSG1 = None
        self.traceSG2 = None


    def calc_first_level_approx(self):
        """
        Do the first level of approximation of TPSC.
        This calculates chi1, and then obtains chisp and chich from the sum rules and the TPSC ansatz.

        :meta private:
        """
        # Calculate chi1
        self.calc_chi1()

        # Calculate Usp and Uch from the TPSC ansatz
        self.calc_usp()
        self.calc_uch()

        # Calculate the double occupancy
        self.docc = self.calc_double_occupancy()

        # Calculate the spin correlation length
        self.calc_xisp_commensurate()
        return

    def calc_chi1(self):
        """
        Function to calculate chi1(q,iqn).
        This also calculates the trace of chi1(q,iqn) as a consistency check.

        :meta private:
        """
        # Calculate the Green function G1 at the first level of approximation of TPSC
        self.g1 = GF(self.mesh)
        # Compute mu^(1)
        dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
        self.mu1 = brentq(lambda m: self.g1.calcNfromG(self.dispersion[None, :, :] - m) - self.n, dispersion_min, dispersion_max, disp=True)
        self.g1.giwnk = self.g1.calcGiwnk(self.dispersion - self.mu1)

        # Calculate chi1(tau,r)
        self.chi1 = 2.*self.g1.gtaur * self.g1.gtaumr[::-1, :]

        # Fourier transform to (q,iqn)
        self.chi1 = self.mesh.r_to_k(self.chi1)
        self.chi1 = self.mesh.tau_to_wn('B', self.chi1)

        self.traceChi1 = self.mesh.trace(self.chi1, 'B')


    def calc_usp(self):
        """
        Function to compute Usp from chi1 and the sum rule.

        :meta private:
        """
        # Bounds on the value of Usp
        Uspmin = 0.
        Uspmax = 2./np.amax(self.chi1).real-1e-7 # Note: the 1e-7 is chosen for stability purposes

        # Calculate Usp
        self.Usp = brentq(lambda m: self.calc_sum_chisp(m)-self.calc_sum_rule_chisp(m), Uspmin, Uspmax, disp=True)


    def calc_uch(self, Uchmin=0., Uchmax=100.):
        """
        Function to compute Uch from chi1 and the sum rule.
        Note: calc_usp has to be called before this function.

        :meta private:
        """
        # Calculate Uch
        self.Uch = brentq(lambda m: self.calc_sum_chich(m)-self.calc_sum_rule_chich(self.Usp), Uchmin, Uchmax, disp=True)


    def calc_sum_chisp(self, Usp):
        """
        Function to compute the trace of chisp(q) = chi1(q)/(1-Usp/2*chi1(q)).
        Also sets chisp(q,iqn) and finds the maximal value.

        :meta private:
        """
        self.chisp = self.chi1 / (1 - 0.5 * Usp * self.chi1)
        self.chispmax = np.amax(self.chisp)
        return self.mesh.trace(self.chisp, 'B').real


    def calc_sum_chich(self, Uch):
        """
        Function to compute the trace of chich(1) = chi1(q)/(1+Uch/2*chi1(q)).
        Also sets chich(q,iqn).

        :meta private:
        """
        self.chich = self.chi1/(1+0.5*Uch*self.chi1)
        return self.mesh.trace(self.chich, 'B').real


    def calc_double_occupancy(self):
        """
        Function to compute the double occupancy.
        Note: the function calc_usp has to be called before this one
        The TPSC ansatz we use here satisfies the particle-hole symmetry with:
        n<1: Usp = U<n_up n_dn>/(<n_up><n_dn>)
        n>1: Usp = U<(1-n_up)(1-n_dn)>/(<(1-n_up)><(1-n_dn)>)

        :meta private:
        """
        if (self.n<1):
            return self.Usp/self.U*self.n*self.n/4
        else:
            return self.Usp/(4*self.U)*(2-self.n)*(2-self.n)-1+self.n


    def calc_sum_rule_chisp(self, Usp):
        """
        Calculate the spin susceptibility sum rule for a specific Usp and U.
        The TPSC ansatz we use here satisfies the particle-hole symmetry with:
        n<1: Usp = U<n_up n_dn>/(<n_up><n_dn>)
        n>1: Usp = U<(1-n_up)(1-n_dn)>/(<(1-n_up)><(1-n_dn)>)

        :meta private:
        """
        if self.n<1:
            return self.n - Usp/self.U*self.n*self.n/2
        else:
            return self.n - Usp/(2*self.U)*(2-self.n)*(2-self.n)+2-2*self.n


    def calc_sum_rule_chich(self, Usp):
        """
        Calculate the charge susceptibility sum rule for a specific Usp and U.
        The TPSC ansatz we use here satisfies the particle-hole symmetry with:
        n<1: Usp = U<n_up n_dn>/(<n_up><n_dn>)
        n>1: Usp = U<(1-n_up)(1-n_dn)>/(<(1-n_up)><(1-n_dn)>)

        :meta private:
        """
        if self.n<1:
            return self.n + Usp/self.U*self.n*self.n/2 - self.n*self.n
        else:
            return self.n + Usp/(2*self.U)*(2-self.n)*(2-self.n)-2+2*self.n - self.n*self.n


    def calc_xisp_commensurate(self):
        """
        Compute the spin correlation length from commensurate spin fluctuations at Q=(pi,pi).
        This calculates the width at half maximum of the spin susceptibility ONLY if its maximal value is at (pi,pi).
        If the spin susceptibility maximum is not at (pi,pi) (incommensurate spin fluctuations), this function returns -1.

        :meta private:
        """
        # Set the default value
        self.xisp = -1
        qy = 0
        qx = int(self.mesh.nk1/2)
        qHM = 0
        q0 = 0
        index_peak = np.argmax(self.chisp[self.mesh.iw0_b])
        # Calculate the spin susceptibility from commensurate fluctuations
        if (index_peak == int(self.mesh.nk1*qx+qx)):
            chispmax = self.chisp[self.mesh.iw0_b, index_peak].real
            chisphalf = self.chisp[self.mesh.iw0_b,int(self.mesh.nk1*qx)+qy].real
            chisptemp = chisphalf
            while (chisphalf < chispmax/2 and qy < self.mesh.nk1/2):
                chisptemp = chisphalf
                qy = qy+1
                chisphalf = self.chisp[self.mesh.iw0_b, int(self.mesh.nk1*qx)+qy].real
            if qy>0:
                q0 = 2*np.pi*(qy-1)/self.mesh.nk1
                qHM = 2*np.pi/self.mesh.nk1*(chispmax/2 - chisptemp)/(chisphalf - chisptemp)
            self.xisp = 1/(np.pi - qHM - q0)


    def calc_second_level_approx(self):
        """
        Function to calculate the self-energy in the second level of approximation of TPSC.
        Important: The function calc_first_level_approx must be called before this one.
        Note: The Hartree term (Un/2) is not included here.
        The TPSC self-energy is: U/8 sum_q(3chi_sp(q)U_sp + chi_ch(q)U_ch)G1(k+q).
        We define V(q) =  U/8(3chi_sp(q)U_sp + chi_ch(q)U_ch) and compute 1/2(V(r)*G(-r)+V(-r)G(r)).

        :meta private:
        """
        # Get V(iqn,q)
        V = self.U/8*(3.*self.Usp*(self.chisp)+self.Uch*(self.chich))

        # Get V(tau,r)
        Vp = self.mesh.k_to_r(V)
        Vm = self.mesh.k_to_mr(V)
        Vp = self.mesh.wn_to_tau('B', Vp)
        Vm = self.mesh.wn_to_tau('B', Vm)

        # Calculate the self-energy in (r,tau) space
        self.selfEnergy = 0.5*(Vm*self.g1.gtaur+Vp*self.g1.gtaumr)

        # Fourier transform
        self.selfEnergy = self.mesh.r_to_k(self.selfEnergy)
        self.selfEnergy = self.mesh.tau_to_wn('F', self.selfEnergy)

        # Calculate G2
        self.g2 = GF(self.mesh)
        dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
        self.mu2 = brentq(lambda m: self.g2.calcNfromG(self.dispersion[None, :, :] - m + self.selfEnergy) - self.n, dispersion_min, dispersion_max, disp=True)
        self.g2.giwnk = self.g2.calcGiwnk(self.dispersion[None, :, :] - self.mu2 + self.selfEnergy)
        return


    def check_self_consistency(self):
        """
        Function to check the self-consistency between one- and two-particle quantities through:
        Tr[Self-Energy*Green's function] = U<n_up n_dn> - Un^2/4
        The -Un^2/4 term on the right hand side is due to the fact that the Hartree term is not included in the self-energy.
        In TPSC, the self-consistency check is exact when computed with the Green's function at the first level of approximation,
        but it is not with the Green's function G2. The discrepancy between the exact result and the trace with G2 is
        a check of the validity of the TPSC calculation.

        :meta private:
        """
        # Calculate the traces
        self.traceSG1 = self.mesh.trace(self.selfEnergy * self.g1.giwnk, 'F')
        self.traceSG2 = self.mesh.trace(self.selfEnergy * self.g2.giwnk, 'F')

        # Calculate the expected result
        self.exactTraceSelfG = self.U*self.docc-self.U*self.n*self.n/4


    def solve(self):
        """
        Run the TPSC method

        :return: A dictionary containing main TPSC output
        :rtype: dict
        """
        # Make the calculation
        self.calc_first_level_approx()
        self.calc_second_level_approx()
        self.check_self_consistency()

        # Prepare output
        self.main_results = {
            "Usp" : self.Usp,
            "Uch" : self.Uch,
            "doubleocc" : self.docc,
            "Trace_chi1" : self.traceChi1,
            "Trace_Self2_G1" : self.traceSG1,
            "Trace_Self2_G2" : self.traceSG2,
            "Exact_Trace_Self2_G" : self.exactTraceSelfG,
            "mu1" : self.mu1,
            "mu2" : self.mu2,
        }
        return self.main_results


    def printResults(self):
        """
        Print results to screen
        """
        if self.main_results is None:
            print("TPSC was not run, please run the TPSC before printing the results.")
            return
        for key,value in self.main_results.items():
            print(f"{key}: {value:5e}")
        return


    def writeResultsJSON(self, filename):
        """
        Write the results in a JSON file

        :param filename: The name of the output JSON file
        :type filename: str
        """
        if self.main_results is None:
            print("TPSC was not run, please run the TPSC before printing the results.")
            return
        out_results = {
            "Usp" : self.Usp,
            "Uch" : self.Uch,
            "doubleocc" : self.docc,
            "Trace_chi1" : [self.traceChi1.real, self.traceChi1.imag],
            "Trace_Self2_G1" : [self.traceSG1.real, self.traceSG1.imag],
            "Trace_Self2_G2" : [self.traceSG2.real, self.traceSG2.imag],
            "Exact_Trace_Self2_G" : self.exactTraceSelfG,
            "mu1" : self.mu1,
            "mu2" : self.mu2,
        }
        with open(filename, 'w') as outfile:
           outfile.write(json.dumps(out_results, indent=4))
        return


    def plotSelfEnergyVsWn(self, coordinates, Wn_range, show=True, ax=None):
        """
        Evaluate and plot the self-energy as a function of Wn for a specific k-point.

        :param coordinates: The k-point to evaluate the self-energy
        :type coordinates: list
        :param Wn_range: x-axis range
        :type Wn_range: list
        :param show: Whether or not showing the grah
        :type show: bool, optional
        :param ax: A matplotlib axis to plot the grah (default, create a graph)
        :type ax: optional
        :return: A matplotlib axis objet containing the graph
        """
        inds = np.arange(Wn_range[0], Wn_range[1]+1, 1)
        ind_kpoint_node = self.mesh.get_ind_kpt(coordinates[0], coordinates[1])
        vals_s2 = self.mesh.get_specific_wn("F", self.selfEnergy[:,ind_kpoint_node], inds)
        if ax is None:
            fig,ax = plt.subplots()
        ax.set_title("Self-energy node")
        ax.plot(inds, vals_s2.real, 'b', label="Re")
        ax.plot(inds, vals_s2.imag, 'r', label="Im")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$\Sigma$")
        ax.grid()
        ax.legend(loc='best')
        if show:
            plt.show()
        return ax
