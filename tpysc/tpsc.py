from .gf import calcGiwnk, calcNfromG, transform_g_to_direct_space
from .mesh import Mesh2D
import matplotlib.pyplot as plt
import json
import numpy as np
import logging

from scipy.optimize import brentq

logger = logging.getLogger(__name__)

class Tpsc:
    """
    Set up and run a Two-Particle Self-Consistent (TPSC) calculation.

    The calculation is performed using the :meth:`solve` method, which runs the
    first and second levels of approximation and checks self-consistency between
    one- and two-particle quantities.

    :ivar mesh: Two-dimensional momentum/frequency mesh used for the calculation.
    :vartype mesh: Mesh2D
    :ivar dispersion: Array containing the dispersion values defined on the mesh.
    :vartype dispersion: numpy.ndarray
    :ivar g1: Green's function at the first level of approximation, G1(k, iwn).
        ``None`` until :meth:`calc_g1` has been called.
    :vartype g1: numpy.ndarray or None
    :ivar g1_tau_r: G1(r, tau), the first-level Green's function transformed to
        direct (real space, imaginary time) space. ``None`` until :meth:`calc_g1`
        has been called.
    :vartype g1_tau_r: numpy.ndarray or None
    :ivar g1_tau_mr: G1(-r, tau), the first-level Green's function transformed to
        direct space with the real-space coordinate mirrored. ``None`` until
        :meth:`calc_g1` has been called.
    :vartype g1_tau_mr: numpy.ndarray or None
    :ivar mu1: Chemical potential at the first level of approximation. ``None``
        until :meth:`calc_g1` has been called.
    :vartype mu1: float or None
    :ivar chi1: Irreducible susceptibility chi1(q, iqn). ``None`` until
        :meth:`calc_chi1` has been called.
    :vartype chi1: numpy.ndarray or None
    :ivar trace_chi1: Trace of chi1(q, iqn), computed as a consistency check.
        Not set until :meth:`calc_first_level_approx` has been called.
    :vartype trace_chi1: complex
    :ivar Usp: Irreducible spin vertex. Set to -1.0 as a default value until
        :meth:`calc_usp` has been called.
    :vartype Usp: float
    :ivar chisp: Spin susceptibility chisp(q, iqn). Not set until
        :meth:`calc_first_level_approx` has been called.
    :vartype chisp: numpy.ndarray
    :ivar docc: Double occupancy. Set to -1.0 as a default value until
        :meth:`calc_double_occupancy` has been called.
    :vartype docc: float
    :ivar Uch: Irreducible charge vertex. Set to -1.0 as a default value until
        :meth:`calc_uch` has been called.
    :vartype Uch: float
    :ivar chich: Charge susceptibility chich(q, iqn). Not set until
        :meth:`calc_first_level_approx` has been called.
    :vartype chich: numpy.ndarray
    :ivar g2: Green's function at the second level of approximation, G2(k, iwn).
        ``None`` until :meth:`calc_second_level_approx` has been called.
    :vartype g2: numpy.ndarray or None
    :ivar mu2: Chemical potential at the second level of approximation. ``None``
        until :meth:`calc_second_level_approx` has been called.
    :vartype mu2: float or None
    :ivar self_energy: Second-level (TPSC) self-energy, Sigma(k, iwn), excluding
        the Hartree term. ``None`` until :meth:`calc_second_level_approx` has
        been called.
    :vartype self_energy: numpy.ndarray or None
    :ivar main_results: Dictionary summarizing the results of the TPSC
        calculation, populated by :meth:`solve`.
    :vartype main_results: dict
    :ivar trace_self_g1: Trace of the product of the self-energy with G1,
        Tr[Sigma * G1]. ``None`` until :meth:`check_self_consistency` has been
        called.
    :vartype trace_self_g1: complex or None
    :ivar trace_self_g2: Trace of the product of the self-energy with G2,
        Tr[Sigma * G2]. ``None`` until :meth:`check_self_consistency` has been
        called.
    :vartype trace_self_g2: complex or None
    :ivar exact_trace_self_g: Expected value of Tr[Sigma * G1] from the exact
        sum rule, U * docc - U * n^2 / 4, used by :meth:`check_self_consistency`
        as a self-consistency check. Not set until :meth:`check_self_consistency`
        has been called.
    :vartype exact_trace_self_g: float
    """

    def __init__(self,
                 mesh: Mesh2D,
                 dispersion: np.ndarray,
                 ) -> None:
        """
        Initialize a TPSC calculation.

        :param mesh: Two-dimensional momentum/frequency mesh used for the calculation.
        :type mesh: Mesh2D
        :param dispersion: Array containing the dispersion values defined on the mesh.
        :type dispersion: numpy.ndarray
        """

        self.mesh = mesh
        self.dispersion = dispersion

        # Member to hold the results
        self.g1 = None
        self.g1_tau_r = None
        self.g1_tau_mr = None
        self.mu1 = None
        self.chi1 = None
        self.Usp = -1.0
        self.docc = -1.0
        self.Uch = -1.0
        self.g2 = None
        self.mu2 = None
        self.self_energy = None
        self.main_results = {}
        self.trace_self_g1 = None
        self.trace_self_g2 = None


    def calc_first_level_approx(self, n: float, U: float) -> None:
        """
        Do the first level of approximation of TPSC.
        This calculates chi1, and then obtains chisp and chich from the sum rules and the TPSC ansatz.

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float

        :meta private:
        """
        # Calculate the Green function G1 at the first level of approximation of TPSC.
        self.calc_g1(n)

        # Calculate chi1 and its trace.
        self.calc_chi1()
        self.trace_chi1 = self.mesh.trace('B', self.chi1)

        # Calculate Usp and Uch from the TPSC ansatz.
        logger.info("Computing irreducible spin vertex Usp...")
        self.Usp = self.calc_usp(n, U)
        logger.info("Computing irreducible charge vertex Uch...")
        self.Uch = self.calc_uch(n, U)

        # Calculate the spin and charge susceptibilities.
        self.chisp = self.calc_chisp(self.Usp)
        self.chich = self.calc_chich(self.Uch)

        # Calculate the double occupancy.
        self.docc = self.calc_double_occupancy(n, U)


    def calc_g1(self, n: float) -> None:
        """
        Compute the first-level Green's function G1 and its real-space transforms.

        Finds the chemical potential ``mu1`` that yields the target density ``n`` by
        root-finding on the non-interacting density, then computes G1(k, iwn) and
        its Fourier transforms to direct space, G1(r, tau) and G1(-r, tau).

        :param n: Electron filling (density per site).
        :type n: float
        """

        # Compute mu^(1)
        dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
        self.mu1 = brentq(lambda m: calcNfromG(self.mesh, self.dispersion[None, :, :] - m) - n, dispersion_min, dispersion_max, disp=True)
        self.g1 = calcGiwnk(self.mesh, self.dispersion - self.mu1)

        # Compute Fourier transforms
        self.g1_tau_r, self.g1_tau_mr = transform_g_to_direct_space(self.mesh, self.g1)


    def calc_chi1(self) -> None:
        """
        Function to calculate chi1(q,iqn).
        This also calculates the trace of chi1(q,iqn) as a consistency check.
        """
        # Calculate chi1(tau,r)
        self.chi1 = 2. * self.g1_tau_r * self.g1_tau_mr[::-1, :]

        # Fourier transform to (q,iqn)
        self.chi1 = self.mesh.r_to_k(self.chi1)
        self.chi1 = self.mesh.tau_to_wn('B', self.chi1)


    def calc_usp(self, n: float, U: float) -> float:
        """
        Function to compute Usp from chi1 and the sum rule.

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :return: The irreducible spin vertex Usp solving the spin susceptibility sum rule.
        :rtype: float
        """
        # Bounds on the value of Usp
        Uspmin = 0.
        Uspmax = 2./np.amax(self.chi1).real-1e-7 # Note: the 1e-7 is chosen for stability purposes

        # Calculate Usp
        return brentq(lambda usp: self.mesh.trace('B', self.calc_chisp(usp)).real - self.calc_sum_rule_chisp(usp, n, U),
                          Uspmin,
                          Uspmax,
                          disp=True)


    def calc_uch(self, n: float, U: float, Uchmin=0., Uchmax=100.) -> float:
        """
        Function to compute Uch from chi1 and the sum rule.
        Note: calc_usp has to be called before this function.

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :param Uchmin: Lower bound of the search interval for Uch. Defaults to 0.
        :type Uchmin: float
        :param Uchmax: Upper bound of the search interval for Uch. Defaults to 100.
        :type Uchmax: float
        :return: The irreducible charge vertex Uch solving the charge susceptibility sum rule.
        :rtype: float
        """
        # Calculate Uch
        return brentq(lambda u: self.mesh.trace('B', self.calc_chich(u)).real-self.calc_sum_rule_chich(self.Usp, n, U),
                    Uchmin,
                    Uchmax,
                    disp=True)


    def calc_chisp(self, usp) -> np.ndarray:
        """
        Computes chisp(q) = chi1(q)/(1 - Usp/2 * chi1(q)).

        :param usp: The irreducible spin vertex.
        :type usp: float
        :return: The spin susceptibility chisp(q, iqn).
        :rtype: numpy.ndarray
        """
        return  self.chi1 / (1 - 0.5 * usp * self.chi1)


    def calc_chich(self, uch) -> np.ndarray:
        """
        Computes chich(q) = chi1(q)/(1 + Uch/2 * chi1(q)).

        :param uch: The irreducible charge vertex.
        :type uch: float
        :return: The charge susceptibility chich(q, iqn).
        :rtype: numpy.ndarray
        """
        return  self.chi1 / (1 + 0.5 * uch * self.chi1)


    def calc_double_occupancy(self, n: float, U: float) -> float:
        """
        Compute the double occupancy :math:`\\langle n_\\uparrow n_\\downarrow \\rangle`.

        :meth:`calc_usp` must be called before this method. The TPSC ansatz
        used here satisfies particle-hole symmetry, giving a piecewise
        expression in terms of the irreducible spin vertex ``Usp``:

        .. math::

            \\langle n_\\uparrow n_\\downarrow \\rangle =
            \\begin{cases}
                \\dfrac{U_\\mathrm{sp}}{4U}\\, n^2 & \\text{if} ~ n < 1 \\\\
                \\dfrac{U_\\mathrm{sp}}{4U}\\, (2-n)^2 - 1 + n & \\text{if} ~ n \\geq 1
            \\end{cases}

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :return: The double occupancy.
        :rtype: float
        """
        if (n < 1):
            return self.Usp /U * n * n / 4
        else:
            return self.Usp / (4 * U) * (2 - n) * (2 - n) - 1 + n


    def calc_sum_rule_chisp(self, Usp: float, n: float, U: float) -> float:
        """
        Calculate the spin susceptibility sum rule for a specific Usp and U.

        The TPSC ansatz satisfies particle-hole symmetry according to:

        * For n < 1: :math:`U_{sp} = U \\frac{\\langle n_{\\uparrow} n_{\\downarrow} \\rangle}{\\langle n_{\\uparrow} \\rangle \\langle n_{\\downarrow} \\rangle}`
        * For n > 1: :math:`U_{sp} = U \\frac{\\langle (1-n_{\\uparrow})(1-n_{\\downarrow}) \\rangle}{\\langle (1-n_{\\uparrow}) \\rangle \\langle (1-n_{\\downarrow}) \\rangle}`

        :param Usp: The irreducible spin vertex.
        :type Usp: float
        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :return: The value of the spin susceptibility sum rule evaluated at Usp.
        :rtype: float

        :meta private:
        """
        if n < 1:
            return n - Usp / U * n * n / 2
        else:
            return n - Usp / (2 * U) * (2 - n) * (2 - n) + 2 - 2 * n


    def calc_sum_rule_chich(self, Usp: float, n: float, U: float) -> float:
        """
        Calculate the charge susceptibility sum rule for a specific Usp and U.

        The TPSC ansatz satisfies particle-hole symmetry according to:

        * For n < 1: :math:`U_{sp} = U \\frac{\\langle n_{\\uparrow} n_{\\downarrow} \\rangle}{\\langle n_{\\uparrow} \\rangle \\langle n_{\\downarrow} \\rangle}`
        * For n > 1: :math:`U_{sp} = U \\frac{\\langle (1-n_{\\uparrow})(1-n_{\\downarrow}) \\rangle}{\\langle (1-n_{\\uparrow}) \\rangle \\langle (1-n_{\\downarrow}) \\rangle}`

        :param Usp: The irreducible spin vertex.
        :type Usp: float
        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :return: The value of the charge susceptibility sum rule evaluated at Usp.
        :rtype: float

        :meta private:
        """
        if n < 1:
            return n + Usp/U*n*n/2 - n*n
        else:
            return n + Usp/(2 * U)*(2-n)*(2-n)-2+2*n - n*n


    def calc_xisp_commensurate(self) -> "float | None":
        """
        Compute the spin correlation length from commensurate spin fluctuations at Q=(pi,pi).
        This calculates the width at half maximum of the spin susceptibility ONLY if its maximal value is at (pi,pi).
        If the spin susceptibility maximum is not at (pi,pi) (incommensurate spin fluctuations), this function returns -1.
        """
        # Set the default value
        qx = int(self.mesh.nk1/2)
        qy = 0
        qHM = 0
        q0 = 0
        index_peak = np.unravel_index(self.chisp[self.mesh.iw0_b].argmax(), self.chisp[self.mesh.iw0_b].shape)


        if (index_peak != (qx, qx)): # Abort if peak is not at Q=(pi, pi)
            self.xisp = -1
            return self.xisp

        # Calculate the spin susceptibility from commensurate fluctuations
        chispmax = self.chisp[self.mesh.iw0_b, qx, qx].real
        chisphalf = self.chisp[self.mesh.iw0_b, qx, qy].real

        # Calculate the spin susceptibility from commensurate fluctuations
        while (chisphalf < chispmax/2 and qy < self.mesh.nk1/2):
            chisptemp = chisphalf
            qy += 1
            chisphalf = self.chisp[self.mesh.iw0_b, qx, qy].real

        if qy>0:
            q0 = 2*np.pi*(qy-1)/self.mesh.nk1
            qHM = 2*np.pi/self.mesh.nk1*(chispmax/2 - chisptemp)/(chisphalf - chisptemp)
        self.xisp = 1/(np.pi - qHM - q0)


    def calc_second_level_approx(self, n: float, U: float) -> None:
        """
        Function to calculate the self-energy in the second level of approximation of TPSC.
        Important: The function calc_first_level_approx must be called before this one.
        Note: The Hartree term (Un/2) is not included here.
        The TPSC self-energy is: U/8 sum_q(3chi_sp(q)U_sp + chi_ch(q)U_ch)G1(k+q).
        We define V(q) =  U/8(3chi_sp(q)U_sp + chi_ch(q)U_ch) and compute 1/2(V(r)*G(-r)+V(-r)G(r)).

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float

        :meta private:
        """
        logger.info("Computing self-energy...")
        # Get V(iqn,q)
        V = U / 8. * (3.*self.Usp*(self.chisp)+self.Uch*(self.chich))

        # Get V(tau,r)
        Vp = self.mesh.k_to_r(V)
        Vm = self.mesh.k_to_mr(V)
        Vp = self.mesh.wn_to_tau('B', Vp)
        Vm = self.mesh.wn_to_tau('B', Vm)

        # Calculate the self-energy in (r,tau) space
        self.self_energy = 0.5*(Vm * self.g1_tau_r + Vp * self.g1_tau_mr)

        # Fourier transform
        self.self_energy = self.mesh.r_to_k(self.self_energy)
        self.self_energy = self.mesh.tau_to_wn('F', self.self_energy)

        # Calculate G2
        dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
        self.mu2 = brentq(lambda m: calcNfromG(self.mesh, self.dispersion[None, :, :] - m + self.self_energy) - n, dispersion_min, dispersion_max, disp=True)
        self.g2 = calcGiwnk(self.mesh, self.dispersion[None, :, :] - self.mu2 + self.self_energy)


    def check_self_consistency(self, n: float, U: float) -> None:
        """
        Function to check the self-consistency between one- and two-particle quantities through:
        Tr[Self-Energy*Green's function] = U<n_up n_dn> - Un^2/4
        The -Un^2/4 term on the right hand side is due to the fact that the Hartree term is not included in the self-energy.
        In TPSC, the self-consistency check is exact when computed with the Green's function at the first level of approximation,
        but it is not with the Green's function G2. The discrepancy between the exact result and the trace with G2 is
        a check of the validity of the TPSC calculation.

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float

        :meta private:
        """
        # Calculate the traces
        self.trace_self_g1 = self.mesh.trace('F', self.self_energy * self.g1)
        self.trace_self_g2 = self.mesh.trace('F', self.self_energy * self.g2)

        # Calculate the expected result
        self.exact_trace_self_g = U * self.docc - U * n * n / 4


    def solve(self, n: float, U: float,) -> dict:
        """
        Run the TPSC method

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :return: A dictionary containing main TPSC output
        :rtype: dict
        """
        logger.info(
            "Start of TPSC calculations (n=%.4f, U=%.4f, T=%.4f)",
            n, U, self.mesh.T,
        )
        self.calc_first_level_approx(n, U)
        self.calc_second_level_approx(n, U)
        self.check_self_consistency(n, U)
        logger.info("End of TPSC calculations.")

        # Prepare output
        self.main_results = {
            "Usp" : self.Usp,
            "Uch" : self.Uch,
            "doubleocc" : self.docc,
            "Trace_chi1" : self.trace_chi1,
            "Trace_Self2_G1" : self.trace_self_g1,
            "Trace_Self2_G2" : self.trace_self_g2,
            "Exact_Trace_Self2_G" : self.exact_trace_self_g,
            "mu1" : self.mu1,
            "mu2" : self.mu2,
        }
        return self.main_results


    def __str__(self) -> str:
        """
        Return a human-readable summary of the main TPSC results.

        :return: A formatted multi-line string listing each entry of :attr:`main_results`.
        :rtype: str
        """
        if not self.main_results:
            return "TPSC was not run, please run the TPSC before printing the results."

        string = ""
        for key,value in self.main_results.items():
            string += f"{key:<20}: {value:5e}\n"

        return string


    def writeResultsJSON(self, filename) -> None:
        """
        Write the results in a JSON file

        :param filename: The name of the output JSON file
        :type filename: str
        """
        if not self.main_results:
            logger.warning("TPSC was not run; no results to write to %s.", filename)
            return
        out_results = {
            "Usp" : self.Usp,
            "Uch" : self.Uch,
            "doubleocc" : self.docc,
            "Trace_chi" : [self.trace_chi1.real, self.trace_chi1.imag],
            "Trace_Self2_G1" : [self.trace_self_g1.real, self.trace_self_g1.imag],
            "Trace_Self2_G2" : [self.trace_self_g2.real, self.trace_self_g2.imag],
            "Exact_Trace_Self2_G" : self.exact_trace_self_g,
            "mu1" : self.mu1,
            "mu2" : self.mu2,
        }
        with open(filename, 'w') as outfile:
           outfile.write(json.dumps(out_results, indent=4))