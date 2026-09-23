from .tpsc import Tpsc
from .mesh import Mesh2D
import numpy as np
from .gf import calcGiwnk, calcNfromG, transform_g_to_direct_space
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)

class TpscPlus:
    """
    Set up and run a TPSC+ calculation, an extension of TPSC with self-energy
    feedback through a self-consistent loop.

    Wraps a :class:`Tpsc` instance to reuse its first-level TPSC results, then
    iterates over the second-level self-energy to compute converged values of
    Usp, Uch, and the double occupancy.

    :ivar tpsc_obj: The underlying TPSC calculation providing the first-level
        results that TPSC+ builds on.
    :vartype tpsc_obj: Tpsc
    :ivar g2: Green's function at the second level of approximation, G2(k, iwn).
        ``None`` until :meth:`solve` has been called.
    :vartype g2: numpy.ndarray or None
    :ivar self_energy: TPSC+ self-energy, Sigma(k, iwn). ``None`` until
        :meth:`solve` has been called.
    :vartype self_energy: numpy.ndarray or None
    :ivar mu2: Chemical potential at the second level of approximation. ``None``
        until :meth:`solve` has been called.
    :vartype mu2: float or None
    :ivar main_results: Dictionary summarizing the results of the TPSC+
        calculation, populated by :meth:`solve`.
    :vartype main_results: dict
    :ivar trace_chi2: Trace of chi2(q, iqn), computed as a consistency check.
        ``None`` until :meth:`solve` has been called.
    :vartype trace_chi2: complex or None
    :ivar converged: Whether the self-consistent loop converged within
        ``iter_max`` iterations.
    :vartype converged: bool
    :ivar usp_crit: Upper bound on Usp above which the spin susceptibility
        diverges, tracked across iterations for the convergence check. Set to
        -1 as a default value until :meth:`calc_usp` has been called.
    :vartype usp_crit: float
    :ivar delta: Distance of Usp from ``usp_crit``, ``1 - Usp / usp_crit``,
        tracked across iterations for the convergence check. Set to -1 as a
        default value until :meth:`calc_usp` has been called.
    :vartype delta: float
    """

    def __init__(self,
                 mesh: Mesh2D,
                 dispersion: np.ndarray,
                 ):
        """
        Initialize a TPSC+ calculation.

        :param mesh: Two-dimensional momentum/frequency mesh used for the calculation.
        :type mesh: Mesh2D
        :param dispersion: Array containing the dispersion values defined on the mesh.
        :type dispersion: numpy.ndarray
        """
        self.tpsc_obj = Tpsc(mesh, dispersion,)

        self.g2 = None
        self.self_energy = None
        self.mu2 = None

        self.main_results  = {}

        self.trace_chi2 = None
        self.converged = False

        self.usp_crit = -1
        self.delta = -1


    def solve(self,
            n: float,
            U: float,
            alpha: float = 0.5, # TODO Déterminer la valeur de ça
            iter_max: int = 1_000,
            self_energy: np.ndarray = None,
            ) -> dict:
        """
        Run the TPSC+ method: a TPSC calculation followed by a self-consistent loop
        over the second-level self-energy.

        If ``self_energy`` is not given, it first runs a TPSC approximation (G1,
        chi1, Usp) via the wrapped :attr:`tpsc_obj`. Otherwise, it uses the
        supplied self-energy directly, first re-casting it onto the current
        Matsubara-frequency mesh if it was computed on a smaller one. It then
        iterates, mixing the self-energy between iterations with weight ``alpha``
        and recomputing G2, chi2, Usp, Uch, and the double occupancy each time,
        until Usp and the double occupancy stop changing between iterations or
        ``iter_max`` iterations are reached.

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :param alpha: Mixing parameter for the self-energy between iterations,
            ``self_energy = (1 - alpha) * new_self_energy + alpha * old_self_energy``.
            Defaults to 0.5.
        :type alpha: float
        :param iter_max: Maximum number of self-consistent loop iterations.
            Defaults to 1000.
        :type iter_max: int
        :param self_energy: Initial self-energy to seed the calculation with,
            Sigma(k, iwn). If ``None`` (the default), the self-energy is instead
            computed from a first-level TPSC approximation.
        :type self_energy: numpy.ndarray or None
        :return: A dictionary containing the main TPSC+ output.
        :rtype: dict
        """
        logger.info("Start of TPSC+ calculations.")

        # First do a regular TPSC procedure.
        # Calculate the Green function G1 at the first level of approximation of TPSC.
        self.tpsc_obj.calc_g1(n)

        if self_energy is None:
            self.tpsc_obj.calc_chi1()
            self.Usp = self.tpsc_obj.calc_usp(n, U)
        else:
            logger.info("Self-energy already set.")

            # Set the self-energy.
            if np.shape(self_energy)[0] < len(self.mesh.IR_basis_set.wn_f): # Check if len(selfE) < len(iwn).
                logger.info("Casting self energy on new mesh.")

                diffshape = len(self.mesh.IR_basis_set.wn_f) - np.shape(self_energy)[0] # If so, gets the difference in lengths.
                shape_of_mesh = (len(self.mesh.IR_basis_set.wn_f), self.mesh.nk1, self.mesh.nk2)
                self.self_energy = np.zeros(shape_of_mesh, dtype=complex) # Create empty array to fill.
                self.self_energy[diffshape//2:-diffshape//2,:] = self_energy # Fill it with known values (approximative).

            # Compute the new G2.
            dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
            self.mu2 = brentq(lambda m: calcNfromG(self.mesh, self.dispersion[None, :, :] - m + self.self_energy) - n, dispersion_min, dispersion_max, disp=True)
            self.g2 = calcGiwnk(self.mesh, self.dispersion[None, :, :] - self.mu2 + self.self_energy)

            # Update chi2.
            self.calc_chi2()

            # Calculate Usp and Uch from the TPSC ansatz.
            self.tpsc_obj.calc_usp() # XXX This might have to be changed

        self.Uch = self.tpsc_obj.calc_uch(n, U)

        # Calculate the spin and charge susceptibilities.
        self.tpsc_obj.chisp = self.tpsc_obj.calc_chisp(self.Usp)
        self.tpsc_obj.chich = self.tpsc_obj.calc_chich(self.Uch)

        # Calculate the double occupancy.
        self.docc = self.tpsc_obj.calc_double_occupancy(n, U)
        # Perform the second level approx as usual.
        self.tpsc_obj.calc_second_level_approx(n, U)

        # Do the TPSC+ loop.
        logger.info("Start of TPSC+ self-consistent loop...")
        for i in range(iter_max):

            if i > 0 and alpha > 0:
                self.self_energy = (1 - alpha) * self.tpsc_obj.self_energy + (alpha) * self.self_energy

                # Compute the new G2
                dispersion_min, dispersion_max = np.amin(self.dispersion), np.amax(self.dispersion)
                self.mu2 = brentq(lambda m: calcNfromG(self.mesh, self.dispersion[None, :, :] - m + self.self_energy) - n, dispersion_min, dispersion_max, disp=True)
                self.g2 = calcGiwnk(self.mesh, self.dispersion[None, :, :] - self.mu2 + self.self_energy)
            else:
                self.g2 = self.tpsc_obj.g2
                self.self_energy = self.tpsc_obj.self_energy

            # Update chi2
            self.calc_chi2()

            # Calculate Usp and Uch from the TPSC ansatz.
            self.calc_usp(n, U)
            self.Uch = self.tpsc_obj.calc_uch(n, U)

            # Calculate the spin and charge susceptibilities.
            self.tpsc_obj.chisp = self.tpsc_obj.calc_chisp(self.Usp)
            self.tpsc_obj.chich = self.tpsc_obj.calc_chich(self.Uch)

            # Calculate the double occupancy.
            self.docc = self.tpsc_obj.calc_double_occupancy(n, U)

            # Perform the second level approx.
            self.tpsc_obj.calc_second_level_approx(n, U)

            # Check the convergence
            # delta_i = self.delta_out
            ucrit_difference = (self.usp_crit - self.previous_usp_crit) / self.previous_usp_crit
            delta_difference = (self.delta - self.previous_delta) / self.previous_delta

            conditions = (np.abs(ucrit_difference) < 1e-10) or (np.abs(delta_difference) < 1e-10) # TODO Make this adjustable.

            if conditions and self.delta > 0:
                self.converged = True
                break

        if self.converged:
            logger.info(
                "The TPSC+ calculation has converged after %d iterations.", i + 1
            )
        else:
            logger.error(
                "The TPSC+ calculation has not converged after %d iterations.",
                iter_max,
            )

        # Update to last values of G^(2) and self-energy.
        self.g2 = self.tpsc_obj.g2
        self.self_energy = self.tpsc_obj.self_energy
        # Check consistency
        self.trace_chi2 = self.mesh.trace('B', self.chi2)
        self.tpsc_obj.check_self_consistency(n, U)

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


    def calc_usp(self,
                n: float,
                U: float,
                gamma: float = 0.8):
        """
        Compute Usp for TPSC+ from chi2 and the sum rule.

        Determines search bounds for Usp from the maximum of chi2 and the sum-rule
        crossing, then finds Usp by root-finding on the spin susceptibility sum rule
        within those bounds. Also updates the critical value ``usp_crit`` (the
        upper bound on Usp above which chi2 diverges) and ``delta`` (the distance of
        Usp from ``usp_crit``), keeping their previous values around for the
        convergence check in :meth:`solve`.

        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :param gamma: Multiplicative factor used to shrink the lower search bound
            for Usp when it is not on the expected side of the sum-rule crossing.
            Defaults to 0.8.
        :type gamma: float
        """
        usp_min = 1e-6
        small_num_for_usp_max = 1e-9

        # Compute the trace of chi2 squared.
        trace_chi2_sq = self.mesh.trace('B', self.chi2 * np.conj(self.chi2))

        # Get the two possible upper bounds for usp
        usp_max_abs = U / (1 + U * trace_chi2_sq / (n*n))
        usp_crit = 2 / np.amax(self.chi2).real

        # Find the upper bound for Usp that best suits the situation
        if usp_max_abs < usp_crit: # Typically away from the critical regime
            if self.mesh.trace('B', self.calc_chisp(usp_max_abs)).real - self.calc_sum_rule_chisp(usp_max_abs, n, U) > 0:
                usp_max_h = usp_max_abs
            else:
                usp_max_h = usp_crit - small_num_for_usp_max
        else: # In the critical regime
            usp_max_h = usp_crit - small_num_for_usp_max

        # Yury added a while loop to have an upper bound on the positive side of the function's crossing for brentq.
        # TODO Check if this is redundant with the previous if statement
        temp_usp_max_h_rule = self.mesh.trace('B', self.calc_chisp(usp_max_h)).real - self.calc_sum_rule_chisp(usp_max_h, n, U)
        usp_braketed = True
        while temp_usp_max_h_rule < 0:
            small_num_for_usp_max /= 10
            if small_num_for_usp_max < 1e-12: # Condition for max iteration
                usp_braketed = False
                break
            usp_max_h = usp_crit - small_num_for_usp_max
            temp_usp_max_h_rule = self.mesh.trace('B', self.calc_chisp(usp_max_h)).real - self.calc_sum_rule_chisp(usp_max_h, n, U)

        # Yury added a while loop to select Uspmin on the "right" side of the functions' crossing for brentq.
        while self.mesh.trace('B', self.calc_chisp(usp_min)).real - self.calc_sum_rule_chisp(usp_min, n, U) > 0:
             usp_min = gamma * usp_min
             if usp_min < 0.05 * usp_max_h:
                usp_min = 1e-6
                break

        # This should not happen
        if usp_max_h < usp_min:
            usp_min = 1e-6

        if usp_braketed and self.mesh.trace('B', self.calc_chisp(usp_min)).real - self.calc_sum_rule_chisp(usp_min, n, U) < 0:
            self.Usp = brentq(lambda m: self.mesh.trace('B', self.calc_chisp(m)).real - self.calc_sum_rule_chisp(m, n, U), usp_min, usp_max_h, disp=True)
        else:
            usp_braketed = False
            self.Usp = usp_max_h * gamma

        # Setting new values for next iteration
        self.previous_usp_crit = self.usp_crit # Keep the previous delta in memory for convergence.
        self.usp_crit = usp_crit

        self.previous_delta = self.delta # Keep the previous delta in memory for convergence.
        self.delta = 1 - self.Usp / usp_crit


    def calc_chi2(self):
        """
        Compute the irreducible particle-hole response function chi2(q, iqn) from
        G1 and G2.

        chi2 plays the same role at the level of TPSC+ that chi1 plays at
        the level of TPSC, which is why it is stored in :attr:`tpsc_obj`'s
        ``chi1`` slot (see :attr:`chi2`). Transforms G2 to direct space and
        combines it with the first-level direct-space Green's function G1 to build
        chi2(r, tau), then Fourier transforms the result to momentum and
        Matsubara-frequency space and stores it (real part) in :attr:`chi2`.
        """
        g2_tau_r, g2_tau_mr = transform_g_to_direct_space(self.mesh, self.g2)
        V = self.tpsc_obj.g1_tau_r * g2_tau_mr[::-1, :] + g2_tau_r * self.tpsc_obj.g1_tau_mr[::-1, :]

        # Fourier transform (r, tau) -> (k, iwn)
        V = self.mesh.r_to_k(V)
        self.chi2 = self.mesh.tau_to_wn('B', V).real


    def __str__(self) -> str:
        """
        Return a human-readable summary of the main TPSC+ results.

        :return: A formatted multi-line string listing each entry of :attr:`main_results`.
        :rtype: str
        """
        if not self.main_results:
            return "TPSC+ was not run, please run the TPSC+ before printing the results."

        string = ""
        for key,value in self.main_results.items():
            string += f"{key:<20}: {value:5e}\n"

        return string

    # --- Wrapper of the Tpsc class ---
    @property
    def mesh(self):
        """
        Two-dimensional momentum/frequency mesh used for the calculation, forwarded
        from :attr:`tpsc_obj`.

        :rtype: Mesh2D
        """
        return self.tpsc_obj.mesh


    @property
    def dispersion(self):
        """
        Array containing the dispersion values defined on the mesh, forwarded from
        :attr:`tpsc_obj`.

        :rtype: numpy.ndarray
        """
        return self.tpsc_obj.dispersion


    @property
    def g1(self):
        """
        First-level Green's function G1(k, iwn), forwarded from :attr:`tpsc_obj`.

        :rtype: numpy.ndarray or None
        """
        return self.tpsc_obj.g1


    @property
    def chi2(self):
        """
        Irreducible particle-hole response function chi2(q, iqn) at the second
        level of TPSC+, computed by :meth:`calc_chi2`.

        chi2 plays the same role here that chi1 plays at the level of TPSC,
        so TPSC+ deliberately reuses :attr:`tpsc_obj`'s ``chi1`` slot to store it
        instead of keeping a separate attribute; this getter and the corresponding
        setter simply read and write ``tpsc_obj.chi1``.

        :rtype: numpy.ndarray or None
        """
        return self.tpsc_obj.chi1


    @chi2.setter
    def chi2(self, value):
        """
        Set the irreducible particle-hole response function chi2(q, iqn).

        Stores ``value`` in :attr:`tpsc_obj`'s ``chi1`` slot, which TPSC+
        deliberately reuses to hold chi2 (see :attr:`chi2`'s getter).

        :param value: The new chi2(q, iqn) to store.
        :type value: numpy.ndarray
        """
        self.tpsc_obj.chi1 = value # XXX MAKE SURE THIS IS COPIED


    @property
    def mu1(self):
        """
        Chemical potential at the first level of approximation, forwarded from
        :attr:`tpsc_obj`.

        :rtype: float or None
        """
        return self.tpsc_obj.mu1


    @property
    def Usp(self):
        """
        Irreducible spin vertex, forwarded from :attr:`tpsc_obj`.

        :rtype: float
        """
        return self.tpsc_obj.Usp


    @Usp.setter
    def Usp(self, value):
        """
        Set the irreducible spin vertex, forwarded to :attr:`tpsc_obj`.

        :param value: The new irreducible spin vertex.
        :type value: float
        """
        self.tpsc_obj.Usp = value


    @property
    def Uch(self):
        """
        Irreducible charge vertex, forwarded from :attr:`tpsc_obj`.

        :rtype: float
        """
        return self.tpsc_obj.Uch


    @Uch.setter
    def Uch(self, value):
        """
        Set the irreducible charge vertex, forwarded to :attr:`tpsc_obj`.

        :param value: The new irreducible charge vertex.
        :type value: float
        """
        self.tpsc_obj.Uch = value


    @property
    def docc(self):
        """
        Double occupancy, forwarded from :attr:`tpsc_obj`.

        :rtype: float
        """
        return self.tpsc_obj.docc


    @docc.setter
    def docc(self, value):
        """
        Set the double occupancy, forwarded to :attr:`tpsc_obj`.

        :param value: The new double occupancy.
        :type value: float
        """
        self.tpsc_obj.docc = value


    def calc_sum_rule_chisp(self, usp: float, n: float, U: float):
        """
        Calculate the spin susceptibility sum rule for a specific Usp and U.

        Delegates to :meth:`Tpsc.calc_sum_rule_chisp` on the wrapped :attr:`tpsc_obj`.

        :param usp: The irreducible spin vertex.
        :type usp: float
        :param n: Electron filling (density per site).
        :type n: float
        :param U: On-site Hubbard interaction strength.
        :type U: float
        :return: The value of the spin susceptibility sum rule evaluated at Usp.
        :rtype: float

        :meta private:
        """
        return self.tpsc_obj.calc_sum_rule_chisp(usp, n, U)


    def calc_chisp(self, usp: float):
        """
        Compute chisp(q) = chi2(q) / (1 - Usp/2 * chi2(q)).

        Delegates to :meth:`Tpsc.calc_chisp` on the wrapped :attr:`tpsc_obj`.

        :param usp: The irreducible spin vertex.
        :type usp: float
        :return: The spin susceptibility chisp(q, iqn).
        :rtype: numpy.ndarray
        """
        return self.tpsc_obj.calc_chisp(usp)