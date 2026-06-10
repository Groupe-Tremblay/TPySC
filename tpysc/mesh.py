import numpy as np
import sparse_ir
from scipy.interpolate import BarycentricInterpolator
import h5py


class Mesh2D:
    """
    Holding class for k-mesh and sparsely sampled imaginary time 'tau' / Matsubara frequency 'iw_n' grids.
    Additionally it defines the Fourier transform routines 'r <-> k'  and 'tau <-> l <-> wn'.
    This is valid for the 2D case
    Credit for the basics: Niklas Witt
    https://spm-lab.github.io/sparse-ir-tutorial/src/TPSC_py.html
    """
    def __init__(self,
                 nk1: int,
                 T: float,
                 wmax: float,
                 IR_tol: float = 1e-12
                 ):

        # Compute the bandwidth and define the IR basis
        self.T = T # Temperature
        IR_basis_set = sparse_ir.FiniteTempBasisSet(1./self.T, wmax, eps=IR_tol)

        self.IR_basis_set = IR_basis_set
        self.T = T

        # Generate k-mesh and dispersion
        self.nk1, self.nk2, self.nk = nk1, nk1, nk1*nk1
        self.k1, self.k2 = np.meshgrid(np.arange(self.nk1)/self.nk1, np.arange(self.nk2)/self.nk2)

        # Lowest Matsubara frequency index
        self.iw0_f = np.where(self.IR_basis_set.wn_f == 1)[0][0]
        self.iw0_b = np.where(self.IR_basis_set.wn_b == 0)[0][0]

        ### Generate a frequency-momentum grid for iw_n.
        self.iwn_f = 1j * self.IR_basis_set.wn_f * np.pi * self.T # TODO This is redundant


    def smpl_obj(self, statistics):
        """ Return sampling object for a given statistic """
        smpl_tau = {'F': self.IR_basis_set.smpl_tau_f, 'B': self.IR_basis_set.smpl_tau_b}[statistics]
        smpl_wn  = {'F': self.IR_basis_set.smpl_wn_f,  'B': self.IR_basis_set.smpl_wn_b }[statistics]
        return smpl_tau, smpl_wn


    def tau_to_wn(self, statistics, obj_tau):
        """ Fourier transform from tau to iw_n via IR basis """
        smpl_tau, smpl_wn = self.smpl_obj(statistics)

        obj_l   = smpl_tau.fit(obj_tau, axis=0)
        obj_wn  = smpl_wn.evaluate(obj_l, axis=0)
        return obj_wn


    def wn_to_tau(self, statistics, obj_wn):
        """ Fourier transform from tau to iw_n via IR basis """
        smpl_tau, smpl_wn = self.smpl_obj(statistics)

        obj_l   = smpl_wn.fit(obj_wn, axis=0)
        obj_tau = smpl_tau.evaluate(obj_l, axis=0)
        return obj_tau


    def k_to_r(self, obj_k):
        """ Fourier transform from k-space to real space """
        obj_r = np.fft.ifftn(obj_k,axes=(1,2))
        return obj_r


    def k_to_mr(self, obj_k):
        """ Fourier transform from k-space to real space (with a - sign) """
        obj_r = np.fft.fftn(obj_k, axes=(1,2), norm="forward")
        return obj_r


    def r_to_k(self, obj_r):
        """ Fourier transform from real space to k-space """
        obj_k = np.fft.fftn(obj_r,axes=(1,2))
        return obj_k


    def get_specific_wn(self, statistics, obj_wn, n_array):
        """
        Routine that takes a sparsely-sampled wn object and a list of
        matsubara frequency indices (n=0, ±1, ±2, ...) and evaluates the
        object at those frequencies. If obj_wn is multi-dimensional, it is
        assumed that the wn axis is the first one.
        """
        # We make sure the n_array is a numpy array of integers. If not, we
        # convert if to that (if possible)
        if not isinstance(n_array, np.ndarray):
            if isinstance(n_array, (float, int)):
                n_array = np.array([n_array], dtype=int)
            elif isinstance(n_array, list):
                n_array = np.array(n_array, dtype=int)
            else:
                print("ERROR: Wrong type of n_array passed as argument. Leaving...")
                exit(1)

        # We calculate the reduced wn's for the given statistics
        if statistics.lower() == 'f':
            wn_array = 2*n_array + 1
            basis_l = self.IR_basis_set.basis_f
        elif statistics.lower() == 'b':
            wn_array = 2*n_array
            basis_l = self.IR_basis_set.basis_b
        else:
            print("ERROR: Wrong statistics passed as argument")
            exit(1)

        # We calculate obj_l with the correct sampling object
        smpl_wn = self.smpl_obj(statistics=statistics)[1]
        obj_l = smpl_wn.fit(obj_wn, axis=0)

        # We evaluate obj_l on the specified reduced matsubara frequencies
        # using the uhat_l(iwn) basis functions
        calculated_obj_wn =  np.einsum("ij, i... -> j...", basis_l.uhat(wn_array), obj_l)

        # We remove any length-one axis from the resulting array:
        return np.squeeze(calculated_obj_wn)


    def extrapolate_fermionic_zero_freq(self, obj_wn, n_freqs: int=4, eta: float=0.001):
        """
        Extrapolate a fermionic function to zero frequency using barycentric Lagrange interpolation
        for the first n_freqs Matsubara frequencies.

        :param obj_wn: The fermionic function object to extrapolate.
        :type obj_wn: object
        :param n_freqs: Number of Matsubara frequencies to use for interpolation.
                        Defaults to 4.
        :type n_freqs: int
        :param eta: Small imaginary frequency offset for extrapolation (typically used to avoid
                    exact zero). Defaults to 0.001.
        :type eta: float
        :return: The extrapolated function value at frequency i*eta.
        :rtype: float or array-like

        .. note::
        The small offset eta helps avoid numerical issues
        at exactly zero frequency.
        """
        # We evaluate the first few frequencies
        indices = np.arange(n_freqs, dtype='int')
        freq_interp = (2*indices+1)*np.pi*self.T
        evaluated_data = self.get_specific_wn('F', obj_wn, indices)

        # We use our routine to evaluate the zero-frequency correlation function
        interpolation_object = BarycentricInterpolator(freq_interp, evaluated_data, axis=0)
        return interpolation_object(eta)


    def trace(self, statistic: str, obj,  tau_value: float = 0) -> float:
        """
            TODO Documentation
        """
        trace = np.sum(obj, axis=(1,2)) / self.nk
        if statistic.lower() == 'f':
            trace_l = self.IR_basis_set.smpl_wn_f.fit(trace)
            return self.IR_basis_set.basis_f.u(tau_value) @ trace_l
        elif statistic.lower() == 'b':
            trace_l = self.IR_basis_set.smpl_wn_b.fit(trace)
            return self.IR_basis_set.basis_b.u(tau_value) @ trace_l


    def get_ind_kpt(self, kx, ky):
        """
        Returns the index corresponding to a given k-point
        in the Brillouin zone (0,0) -> (2pi, 2pi) by finding the closest
        k-point in the mesh.
        """

        # We calculate the corresponding k-point in the (0,0)->(2pi, 2pi)
        # range
        kx %= 2*np.pi
        ky %= 2*np.pi

        # We normalize the k-point, since we store k/2pi in the arrays
        kx /= 2*np.pi
        ky /= 2*np.pi

        # We find the distance squared from the point (kx, ky) of every k-point
        # in the BZ
        dist2_arr = ((self.k1 - kx)**2 + (self.k2 - ky)**2).reshape(self.nk)

        # We find the index for the k-point which has the minimum distance
        # squared from (kx, ky)
        return dist2_arr.argmin()


    def save_k_grid_function(self, target_file: str, data_label: str, obj: np.ndarray) -> None:
         """
        Save a k-space grid array to an HDF5 file.

        For complex-valued arrays, the real and imaginary parts are stored
        separately in named datasets within a group for easier visualization.
        Real-valued arrays are stored directly as a single dataset.

        :param target_file: Path to the HDF5 file where data will be saved.
        :type target_file: str
        :param data_label: Key or group name for the dataset(s) in the HDF5 file.
        :type data_label: str
        :param obj: The k-space grid array to save. Should be a 2D array.
        :type obj: np.ndarray
        :return: None
        :rtype: None

        .. note::
            This function currently saves only the subset obj[:(self.nk1//2), :(self.nk1//2)].
            This behavior is flagged for optimization in the source code.

        .. todo::
            Verify that the input is truly a grid structure before saving.
            Optimize the slicing operation for large arrays.
        """
        # TODO Check that is is really a grid

        save_obj = obj[:(self.nk1//2),:(self.nk1//2)] # TODO This can be optimized
        with h5py.File(target_file, "w") as f:
            if np.iscomplexobj(obj): # Seperate real and complex part for ease of vizualisation.
                grp = f.create_group(data_label)
                grp.create_dataset("real", data=save_obj.real)
                grp.create_dataset("imag", data=save_obj.imag)
            else:
                f.create_dataset(data_label, data=save_obj)


    @property
    def shape(self) -> float: # TODO This is not very rigourous
        return (len(self.iwn_f), self.nk1, self.nk1)