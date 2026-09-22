import numpy as np
import sparse_ir
from scipy.interpolate import BarycentricInterpolator
import h5py
import logging

logger = logging.getLogger(__name__)


class Mesh2D:
    """
    Hold the k-grid and the sparsely sampled imaginary-time and
    Matsubara-frequency grids for a 2D system.

    Defines the Fourier transform routines between real space and k-space
    ('r' <-> 'k') and between imaginary time, the intermediate
    representation, and Matsubara frequency ('tau' <-> 'l' <-> 'iw_n').
    Valid only for the 2D case.

    Credit for the basics: Niklas Witt,
    https://spm-lab.github.io/sparse-ir-tutorial/src/TPSC_py.html

    :ivar T: Temperature.
    :vartype T: float
    :ivar IR_basis_set: Intermediate representation basis set used for
        fermionic and bosonic sampling.
    :vartype IR_basis_set: sparse_ir.FiniteTempBasisSet
    :ivar nk1: Number of k-points along the first dimension.
    :vartype nk1: int
    :ivar nk2: Number of k-points along the second dimension. Equals nk1.
    :vartype nk2: int
    :ivar nk: Total number of k-points in the mesh (nk1 * nk2).
    :vartype nk: int
    :ivar k1: Meshgrid of k-point coordinates along the first dimension,
        normalized to [0, 1) (i.e. k/2*pi).
    :vartype k1: np.ndarray
    :ivar k2: Meshgrid of k-point coordinates along the second dimension,
        normalized to [0, 1) (i.e. k/2*pi).
    :vartype k2: np.ndarray
    :ivar iw0_f: Index of the lowest fermionic Matsubara frequency (n=1)
        in the IR basis.
    :vartype iw0_f: int
    :ivar iw0_b: Index of the lowest (zero) bosonic Matsubara frequency
        in the IR basis.
    :vartype iw0_b: int
    :ivar iwn_f: Fermionic Matsubara frequencies i*wn.
    :vartype iwn_f: np.ndarray
    """
    def __init__(self,
                 nk1: int,
                 T: float,
                 wmax: float,
                 IR_tol: float = 1e-12
                 ):
        """
        Build the k-grid and the intermediate representation basis for a 2D
        system at a given temperature.

        :param nk1: Number of k-points along each dimension of the square
            k-grid.
        :type nk1: int
        :param T: Temperature.
        :type T: float
        :param wmax: Maximum frequency (bandwidth cutoff) for the
            intermediate representation basis.
        :type wmax: float
        :param IR_tol: Tolerance for the intermediate representation basis.
            Defaults to 1e-12.
        :type IR_tol: float
        """

        # Compute the bandwidth and define the IR basis
        self.T = T # Temperature
        IR_basis_set = sparse_ir.FiniteTempBasisSet(1./self.T, wmax, eps=IR_tol)

        self.IR_basis_set = IR_basis_set
        self.T = T

        # Generate k-grid and dispersion
        self.nk1, self.nk2, self.nk = nk1, nk1, nk1*nk1 # TODO Make this more flexible
        self.k1, self.k2 = np.meshgrid(np.arange(self.nk1)/self.nk1, np.arange(self.nk2)/self.nk2)

        # Lowest Matsubara frequency index
        self.iw0_f = np.where(self.IR_basis_set.wn_f == 1)[0][0]
        self.iw0_b = np.where(self.IR_basis_set.wn_b == 0)[0][0]

        ### Generate a frequency-momentum grid for iw_n.
        self.iwn_f = 1j * self.IR_basis_set.wn_f * np.pi * self.T # TODO This is redundant


    def smpl_obj(self, statistics: str) -> tuple[sparse_ir.TauSampling, sparse_ir.MatsubaraSampling]:
        """
        Return the tau and Matsubara-frequency sampling objects for a given
        statistic.

        :param statistics: Statistic type, 'f' for fermionic or 'b' for
            bosonic. Case-insensitive.
        :type statistics: str
        :return: The tau sampling object and the Matsubara-frequency sampling
            object.
        :rtype: tuple[sparse_ir.TauSampling, sparse_ir.MatsubaraSampling]
        :raises KeyError: If statistics is not 'f' or 'b' (case-insensitive).

        :meta private:
        """
        statistics = statistics.upper()
        smpl_tau = {'F': self.IR_basis_set.smpl_tau_f, 'B': self.IR_basis_set.smpl_tau_b}[statistics]
        smpl_wn  = {'F': self.IR_basis_set.smpl_wn_f,  'B': self.IR_basis_set.smpl_wn_b }[statistics]
        return smpl_tau, smpl_wn


    def tau_to_wn(self, statistics: str, obj_tau: np.ndarray) -> np.ndarray:
        """
        Fourier transform an object from imaginary time to Matsubara frequency,
        via the intermediate representation basis.

        :param statistics: Statistic type, 'f' for fermionic or 'b' for
            bosonic. Case-insensitive.
        :type statistics: str
        :param obj_tau: Object sampled on the sparse tau grid, with the tau
            axis first.
        :type obj_tau: np.ndarray
        :return: The object evaluated on the sparse Matsubara-frequency grid.
        :rtype: np.ndarray
        """
        smpl_tau, smpl_wn = self.smpl_obj(statistics)

        obj_l   = smpl_tau.fit(obj_tau, axis=0)
        obj_wn  = smpl_wn.evaluate(obj_l, axis=0)
        return obj_wn


    def wn_to_tau(self, statistics: str, obj_wn: np.ndarray) -> np.ndarray:
        """
        Fourier transform an object from Matsubara frequency to imaginary time,
        via the intermediate representation basis.

        :param statistics: Statistic type, 'f' for fermionic or 'b' for
            bosonic. Case-insensitive.
        :type statistics: str
        :param obj_wn: Object sampled on the sparse Matsubara-frequency grid,
            with the frequency axis first.
        :type obj_wn: np.ndarray
        :return: The object evaluated on the sparse tau grid.
        :rtype: np.ndarray
        """
        smpl_tau, smpl_wn = self.smpl_obj(statistics)

        obj_l   = smpl_wn.fit(obj_wn, axis=0)
        obj_tau = smpl_tau.evaluate(obj_l, axis=0)
        return obj_tau


    def k_to_r(self, obj_k: np.ndarray) -> np.ndarray:
        """
        Fourier transform an object from k-space to real space.

        :param obj_k: Object defined on the k-grid, with the k-axes second and
            third (axes 1 and 2).
        :type obj_k: np.ndarray
        :return: The object transformed to real space.
        :rtype: np.ndarray
        """
        obj_r = np.fft.ifftn(obj_k,axes=(1,2))
        return obj_r


    def k_to_mr(self, obj_k: np.ndarray) -> np.ndarray:
        """
        Fourier transform an object from k-space to real space, using the
        opposite sign convention from k_to_r (a -i k.r kernel, normalized by
        1/N).

        :param obj_k: Object defined on the k-mesh, with the k-axes second and
            third (axes 1 and 2).
        :type obj_k: np.ndarray
        :return: The object transformed to real space.
        :rtype: np.ndarray
        """
        obj_r = np.fft.fftn(obj_k, axes=(1,2), norm="forward")
        return obj_r


    def r_to_k(self, obj_r: np.ndarray) -> np.ndarray:
        """
        Fourier transform an object from real space to k-space.

        :param obj_r: Object defined on the real-space mesh, with the
            real-space axes second and third (axes 1 and 2).
        :type obj_r: np.ndarray
        :return: The object transformed to k-space.
        :rtype: np.ndarray
        """
        obj_k = np.fft.fftn(obj_r,axes=(1,2))
        return obj_k


    def get_specific_wn(self, statistics: str, obj_wn: np.ndarray, n_array: "int | float | list | np.ndarray") -> np.ndarray:
        """
        Evaluate a sparsely-sampled Matsubara-frequency object at specific
        Matsubara frequency indices.

        Takes a list of Matsubara frequency indices (n=0, ±1, ±2, ...) and
        evaluates obj_wn at those frequencies. If obj_wn is multi-dimensional,
        the wn axis must be the first one.

        :param statistics: Statistic type, 'f' for fermionic or 'b' for
            bosonic. Case-insensitive.
        :type statistics: str
        :param obj_wn: Object sampled on the sparse Matsubara-frequency grid,
            with the frequency axis first.
        :type obj_wn: np.ndarray
        :param n_array: Matsubara frequency indices n at which to evaluate the
            object. Accepts a scalar, a list, or a numpy array.
        :type n_array: int or float or list or np.ndarray
        :return: The object evaluated at the requested Matsubara frequencies,
            with any length-one axis removed.
        :rtype: np.ndarray
        :raises TypeError: If n_array is not an int, float, list, or
            np.ndarray.
        :raises ValueError: If statistics is not 'f' or 'b'
            (case-insensitive).
        """
        # We make sure the n_array is a numpy array of integers. If not, we
        # convert if to that (if possible)
        if not isinstance(n_array, np.ndarray):
            if isinstance(n_array, (float, int)):
                n_array = np.array([n_array], dtype=int)
            elif isinstance(n_array, list):
                n_array = np.array(n_array, dtype=int)
            else:
                msg = (
                    f"n_array must be an int, float, list, or np.ndarray, "
                    f"got {type(n_array).__name__}."
                )
                logger.error(msg)
                raise TypeError(msg)

        # We calculate the reduced wn's for the given statistics
        if statistics.lower() == 'f':
            wn_array = 2*n_array + 1
            basis_l = self.IR_basis_set.basis_f
        elif statistics.lower() == 'b':
            wn_array = 2*n_array
            basis_l = self.IR_basis_set.basis_b
        else:
            msg = f"statistics must be 'f' or 'b', got {statistics!r}."
            logger.error(msg)
            raise ValueError(msg)

        # We calculate obj_l with the correct sampling object
        smpl_wn = self.smpl_obj(statistics=statistics)[1]
        obj_l = smpl_wn.fit(obj_wn, axis=0)

        # We evaluate obj_l on the specified reduced matsubara frequencies
        # using the uhat_l(iwn) basis functions
        calculated_obj_wn =  np.einsum("ij, i... -> j...", basis_l.uhat(wn_array), obj_l)

        # We remove any length-one axis from the resulting array:
        return np.squeeze(calculated_obj_wn)


    def extrapolate_fermionic_zero_freq(self, obj_wn: np.ndarray, n_freqs: int=4, eta: float=0.001) -> "float | np.ndarray":
        """
        Extrapolate a fermionic function to zero frequency, using barycentric
        Lagrange interpolation over the first n_freqs Matsubara frequencies.

        :param obj_wn: The fermionic function to extrapolate.
        :type obj_wn: np.ndarray
        :param n_freqs: Number of Matsubara frequencies to use for
            interpolation. Defaults to 4.
        :type n_freqs: int
        :param eta: Small imaginary frequency offset for the extrapolation.
            Defaults to 0.001.
        :type eta: float
        :return: The extrapolated function value at frequency i*eta.
        :rtype: float or np.ndarray

        .. note::

            The offset eta avoids the numerical issues of evaluating exactly
            at zero frequency.
        """
        # We evaluate the first few frequencies
        indices = np.arange(n_freqs, dtype='int')
        freq_interp = (2*indices+1)*np.pi*self.T
        evaluated_data = self.get_specific_wn('F', obj_wn, indices)

        # We use our routine to evaluate the zero-frequency correlation function
        interpolation_object = BarycentricInterpolator(freq_interp, evaluated_data, axis=0)
        return interpolation_object(eta)


    def trace(self, statistic: str, obj: np.ndarray, tau_value: float = 0) -> float:
        """
        Compute the sum over wavevectors and Matsubara frequencies of obj.

        Average obj over the k-grid, then sum over Matsubara frequency via the
        intermediate representation basis by evaluating it near tau=0. tau_value
        must be 0 or beta (1/T), since the basis functions are defined only for
        tau in [0, beta]; these give the tau=0+ and tau=0- limits of obj's imaginary-time
        counterpart, which is discontinuous at tau=0.

        .. math::

            \\mathrm{tr}\mathrm{O} = \\frac{T}{N}\\sum_{\\mathbf{k}}\\sum_{n}
                O(\\mathbf{k}, i\\omega_n)\\, e^{i\\omega_n 0^\\pm}

        :param statistic: Statistic type, 'f' for fermionic or 'b' for
            bosonic. Case-insensitive.
        :type statistic: str
        :param obj: Object sampled on the sparse Matsubara-frequency grid and
            defined on the k-grid, with the Matsubara-frequency axis first and
            the k-axes second and third (axes 1 and 2).
        :type obj: np.ndarray
        :param tau_value: Imaginary time at which to evaluate the trace: 0 for
            the tau=0+ limit, or beta (1/T) for the tau=0- limit. Defaults to 0.
        :type tau_value: float
        :return: The trace evaluated at tau_value.
        :rtype: float
        :raises ValueError: If statistic is not 'f' or 'b' (case-insensitive).
        """
        trace = np.sum(obj, axis=(1,2)) / self.nk
        if statistic.lower() == 'f':
            trace_l = self.IR_basis_set.smpl_wn_f.fit(trace)
            return self.IR_basis_set.basis_f.u(tau_value) @ trace_l
        elif statistic.lower() == 'b':
            trace_l = self.IR_basis_set.smpl_wn_b.fit(trace)
            return self.IR_basis_set.basis_b.u(tau_value) @ trace_l
        else:
            raise ValueError(f"statistic must be 'f' or 'b', got {statistic!r}.")


    def get_ind_kpt(self, kx: float, ky: float) -> int:
        """
        Return the mesh index of the k-point closest to (kx, ky) in the
        Brillouin zone.

        :param kx: x-component of the k-point, in radians.
        :type kx: float
        :param ky: y-component of the k-point, in radians.
        :type ky: float
        :return: Flattened index of the closest k-point in the mesh.
        :rtype: int
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
        return int(dist2_arr.argmin())


    def save_k_grid_function(self, target_file: str, data_label: str, obj: np.ndarray, io_mode: str='a') -> None:
        """
        Save a k-space grid array to an HDF5 file.

        For complex-valued arrays, the real and imaginary parts are stored
        separately in named datasets within a group for easier visualization.
        Real-valued arrays are stored directly as a single dataset.

        :param target_file: Path to the HDF5 file.
        :type target_file: str
        :param data_label: Key or group name for the dataset(s) in the HDF5 file.
        :type data_label: str
        :param obj: The k-space grid array to save. Should be a 2D array.
        :type obj: np.ndarray
        :param io_mode: Mode in which to open the HDF5 file ('a' to
            append, 'w' to overwrite). Defaults to 'a'.
        :type io_mode: str

        .. note::
            This function currently saves only the subset obj[:(self.nk1//2), :(self.nk1//2)].
            This behavior is flagged for optimization.

        .. todo::
            Verify that the input is truly a grid structure before saving.
            Optimize the slicing operation for large arrays.
        """
        # TODO Check that is is really a grid
        save_obj = obj[:(self.nk1//2),:(self.nk1//2)] # TODO This can be optimized

        self.__save_hdf__(target_file, data_label, save_obj, io_mode)


    def save_wn_function(self, target_file: str, data_label: str, obj: np.ndarray, io_mode: str) -> None:
        """
        Save a Matsubara-frequency or imaginary-time grid array to an HDF5
        file.

        :param target_file: Path to the HDF5 file.
        :type target_file: str
        :param data_label: Key or group name for the dataset(s) in the HDF5
            file.
        :type data_label: str
        :param obj: The grid array to save.
        :type obj: np.ndarray
        :param io_mode: Mode in which to open the HDF5 file ('a' to
            append, 'w' to overwrite).
        :type io_mode: str
        """
        self.__save_hdf__(target_file, data_label, obj, io_mode)


    def __save_hdf__(self, target_file: str, data_label: str, obj: np.ndarray, io_mode: str) -> None:
        """
        Write an array to an HDF5 file under a given label, splitting
        complex-valued arrays into real and imaginary parts.

        If data_label already exists in the file and io_mode is 'a', the
        existing dataset or group is deleted before writing. A complex-valued
        array is stored as a group holding ``real`` and ``imag`` datasets; a
        real-valued array is stored directly as a single dataset.

        :param target_file: Path to the HDF5 file where data will be saved.
        :type target_file: str
        :param data_label: Key or group name for the dataset(s) in the HDF5
            file.
        :type data_label: str
        :param obj: The array to save.
        :type obj: np.ndarray
        :param io_mode: Mode in which to open the HDF5 file ('a' to
            append, 'w' to overwrite).
        :type io_mode: str

        :meta private:
        """
        with h5py.File(target_file, io_mode) as f:
            # Delete the data if it is already inside the file
            if (data_label in f) and io_mode == 'a':
                del f[data_label]

            if np.iscomplexobj(obj): # Seperate real and complex part for ease of vizualisation.
                grp = f.create_group(data_label)
                grp.create_dataset("real", data=obj.real)
                grp.create_dataset("imag", data=obj.imag)
            else:
                f.create_dataset(data_label, data=obj)