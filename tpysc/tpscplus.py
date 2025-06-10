from .tpsc import Tpsc
from .mesh import Mesh2D
import numpy as np


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


    def solve(self,
              iter_max: int = 1_000,
              anderson_acc: bool = False
              ) -> None:
        """
        TODO Documentation
        """
        # First do a regular TPSC procedure.
        self.tpsc_obj.solve()

        # TODO Do the TPSC+ loop.
        for i in range(iter_max):
            if anderson_acc == False:
                # TODO update the G2 function

                # Calculate Usp and Uch from the TPSC ansatz
                self.tpsc_obj.calc_usp() # XXX This is not the right function!
                self.tpsc_obj.calc_uch()

                # Calculate the double occupancy
                self.docc = self.tpsc_obj.calc_double_occupancy()
            else:
                pass
                # TODO Anderson acceleration that does not suck


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
    def mu1(self):
        return self.tpsc_obj.mu1


    @property
    def docc(self):
        return self.tpsc_obj.docc