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


    def solve(self,
              alpha: float = 0.5, # TODO Déterminer la valeur de ça
              msd2precision: float = 1e-5,
              msdInfprecision: float = 1e-3,
              iter_max: int = 1_000,
              iter_min: int = 30,
              anderson_acc: bool = False,
              usp_max: float = 0.,
              ) -> None:
        """
        TODO Documentation
        """
        logging.basicConfig(level=logging.INFO)
        logging.info("Start of TPSC+ calculations.")

        # First do a regular TPSC procedure.
        tpsc_results = self.tpsc_obj.solve()

        # TODO Comment this
        self.usp_max = usp_max
        self.prev_usp = 0
        delta_ip1 = 1. - 0.5 * self.tpsc_obj.Usp * self.chi2

        # TODO Do the TPSC+ loop.
        logging.info("Start of TPSC+ self-consistent loop.")
        for i in range(iter_max):
            if anderson_acc == False: # Regular TPSC+ loop.

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

            else:
                pass
                # TODO Anderson acceleration that does not suck


            # ===========
            # Check the convergence
            #correct the norm based on alpha
            delta_i = self.delta_out

            norm = np.linalg.norm((delta_ip1 - delta_i) / delta_i) / (1 - alpha)
            norm_inf = np.max(np.abs((delta_ip1 - delta_i) / delta_i)) / (1 - alpha)

            norm_conditions = (norm < msd2precision) or (norm_inf < msdInfprecision)

            if norm_conditions:
                if (self.delta_p == True) and (self.prev_usp == 0) and (i > iter_min):
                    self.converged = True
                    nIteFinal = i + 1
                    break
                else :
                    self.converged = False

            delta_ip1 = delta_i


        if self.converged:
            logging.info("The TPSC+ calculation has converged after {} iterations.".format(nIteFinal))
        else:
            logging.error("The TPSC+ calculation has not converged after {} iterations.".format(iter_max))

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
        #===========


    def calc_usp(self):
        """
        Function to compute Usp from chi0 and the sum rule.
        """

        # ---- Setting the initial values ----
        usp_min = 0.
        usp_max_h = 2. / np.amax(self.chi2).real - 1e-7  # to escape 0-division -- Warning : smaller than this can create other problems in the calculations such as the calculation of Mu2.
        if self.usp_max == 0. :
            self.usp_max = usp_max_h - 1e-5


        # ----- Calculation of Usp with brentq. -----
        #       - If f(usp_min) and f(usp_max) have the same sign, we set Usp with the usp_guess and a proportion (gamma) of delta.
        #           This way, the algorithm can continue and, of course, it will not converge with this Usp.
        #           But it will at least provide a value of Usp that can make it to the next iteration being a possible value that is less than usp_max.
        #       - I saw that sometimes the first iterations of TPSC+ does not have a solution, there is not crossing between sumChisp and sumruleChisp.
        #           Probably because the other values of Usp, Uch and double occupation are not optimized.
        f_usp_min = self.mesh.trace('B', self.calc_chisp(usp_min)).real - self.calc_sum_rule_chisp(usp_min)
        f_usp_max = self.mesh.trace('B', self.calc_chisp(usp_max_h)).real - self.calc_sum_rule_chisp(usp_max_h)

        if f_usp_min * f_usp_max < 0:
            if self.prev_usp > 0 :
                self.prev_usp = self.prev_usp - 1

            self.Usp = brentq(lambda u: self.mesh.trace('B', self.calc_chisp(u)).real- self.calc_sum_rule_chisp(u),
                              usp_min,
                              usp_max_h,
                              disp=True)

        else:
            self.guess_Usp_flag = True
            self.prev_usp = 5

            # ----  Choosing from which technique we guess the value of usp_min ----
            #       1. The first one is From the Previous Temperature and from the interpolation of log Delta and 1/T.
            #       2. The second one is from the previous Temperature and when there is no logdelta provided (For example the first temperatures done in a calculations)
            #           It uses the values of Usp and usp_max to guess usp_min
            #       3. Third one is from the previous iteration :
            #           It uses the values of Usp and usp_max to guess usp_min
            #       4. Fourth one is when none of the above is the case

            if self.newTemp and self.logdelta!=0.:
                gamma = 0.8 # gamma can be between 0 and 1.
                print("self.newTemp and self.logdelta!=0.")
                # Guess a value of Usp from the interpolation with the log(Delta)
                # Delta = 1 - Usp/usp_max. Supposedly the relation with the temperature is log(Delta) \propto 1/T
                usp_guess = usp_max_h * (1. - np.exp(self.logdelta)).real - 1e-5

                # Set the real initial guess value lower than usp_guess so it is on the other side of the functions crossing.
                # If Usp is too near the real answer, it can be on the same side as usp_max and brentq does not find the roots.
                usp_min = usp_guess - self.delta_out_1
                # Of course, if that makes it so usp_min becomes negative, we put it back to its inital value.

                if usp_min < 0.:
                    usp_min = 0.
                elif self.newTemp :
                    logging.debug("In calc_usp: newTemp")
                    usp_guess = usp_max_h - gamma*(self.usp_max-self.Usp_prevT)
                    usp_min = usp_guess - self.delta_out_1

                    if not self.delta_p:
                        # If self.delta_p is Fasle, that means that Usp is larger than usp_max. Which is not good : Chisp would then be negative.
                        # We give a new guess to Usp so it is smaller than usp_max hopefully
                        usp_guess = usp_max_h - gamma * (self.Usp_prevT - self.usp_max)
                        usp_min = usp_guess + self.delta_out_1

                elif self.delta_out_1!=0. : # XXX WILL NEVER ENTER HERE (newTemp has already been checked to be True, see elif above)
                    print("self.delta_out_1!=0.")
                    usp_guess = usp_max_h - self.gamma*(self.usp_max-self.Usp_prevT)
                    usp_min = usp_guess*self.gamma
                    if not self.delta_p: # If Delta is smaller than 0, the above equation would make the new guess for Usp go even beyond usp_max again, so we changed the sign.
                        usp_guess = usp_max_h - self.gamma*(self.Usp_prevT-self.usp_max)
                        usp_min = usp_guess + self.delta_out_1

                else :
                    print("else")
                    usp_guess = usp_max_h*self.gamma # Arbitrary chosen to remove Usp*0.1 so it is proportionnal and smaller.


                # Making sure that usp_min is on the "right" side of the functions' crossing for brentq,
                # If not, we reinitialize usp_min to 0.

                if (self.calc_sum_chisp(usp_min) - self.calc_sum_rule_chisp(usp_min)) > 0:
                    usp_min = 0.
                if usp_max_h < usp_min:
                    usp_min = 0.

                # ------------- Setting the value of Usp -------------------
                # -- 1st option : Retry with the new values of usp_min
                f_usp_min = self.calc_sum_chisp(usp_min) - self.calc_sum_rule_chisp(usp_min)
                f_usp_max = self.calc_sum_chisp(usp_max_h) - self.calc_sum_rule_chisp(usp_max_h)
                if f_usp_min * f_usp_max < 0:
                    print("Usp found with Brentq with guessed usp_min")
                    self.Usp = brentq(lambda u: self.calc_sum_chisp(u)-self.calcSumRuleChisp(u), usp_min, usp_max_h, disp=True)

                # -- 2nd option : Set Usp from an eduacted guess
                elif usp_guess > 0 and usp_guess < usp_max_h :
                    print("Usp found with Usp guess")
                    self.Usp = usp_guess
                # -- 3rd option : Set Usp with a normally valid value, but not the most recommended one.
                else :
                    print(f"Usp found with a percentage {gamma:.2f} of usp_max" )
                    self.newTemp = self.newTemp + 5
                    self.Usp = usp_max_h*gamma


        #  ---- Setting the new values for the next iteration ----
        self.delta_out = 1. - 0.5 * self.Usp * self.chi2 # To use for the convergence check.
        self.usp_max = usp_max_h # Save the actual usp_max for the next iteration
        self.delta_out_1 = 1. - self.Usp/(self.usp_max + 1e-7) # Delta at Q=q_max. + Recover the exact value of usp_max to it.


        # ---- Checking the validity of the value of Usp found. ----
        # If self.delta_p is True, the algorithm can converge.
        # If self.delta_p is False, the calculations continue but the algorithm cannot converge until Delta is positive.
        self.delta_p = (self.delta_out_1 > 0.)
        if self.delta_p == False:
            logging.warning("Usp larger than usp_max") # TODO Check if this is a warning or an error


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