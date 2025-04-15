import numpy as np
"""
Date: June 22, 2023
"""
class Mesh2D:
    """
    Holding class for k-mesh and sparsely sampled imaginary time 'tau' / Matsubara frequency 'iw_n' grids.
    Additionally it defines the Fourier transform routines 'r <-> k'  and 'tau <-> l <-> wn'.
    This is valid for the 2D case
    Requires an input dispersion
    Credit for the basics: Niklas Witt
    https://spm-lab.github.io/sparse-ir-tutorial/src/TPSC_py.html
    """
    def __init__(self,IR_basis_set, nk1, nk2, T, dispersion):
        self.IR_basis_set = IR_basis_set
        self.T = T

        # Generate k-mesh and dispersion
        self.nk1, self.nk2, self.nk = nk1, nk2, nk1*nk2
        self.k1, self.k2 = np.meshgrid(np.arange(self.nk1)/self.nk1, np.arange(self.nk2)/self.nk2)
        self.ek = dispersion

        # Lowest Matsubara frequency index
        self.iw0_f = np.where(self.IR_basis_set.wn_f == 1)[0][0]
        self.iw0_b = np.where(self.IR_basis_set.wn_b == 0)[0][0]

        ### Generate a frequency-momentum grid for iw_n and ek (in preparation for calculating the Green function)
        # frequency mesh (for Green function)
        self.iwn_f = 1j * self.IR_basis_set.wn_f * np.pi * self.T
        self.iwn_f_ = np.tensordot(self.iwn_f, np.ones(self.nk), axes=0)

        # ek mesh
        self.ek_ = np.tensordot(np.ones(len(self.iwn_f)), self.ek, axes=0)
    
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

    def k_to_r(self,obj_k):
        """ Fourier transform from k-space to real space """
        obj_k = obj_k.reshape(-1, self.nk1, self.nk2)
        obj_r = np.fft.ifftn(obj_k,axes=(1,2))
        obj_r = obj_r.reshape(-1, self.nk)
        return obj_r
    
    def k_to_mr(self,obj_k):
        """ Fourier transform from k-space to real space (with a - sign) """
        obj_k = obj_k.reshape(-1, self.nk1, self.nk2)
        obj_r = np.fft.fftn(obj_k, axes=(1,2), norm="forward")
        obj_r = obj_r.reshape(-1, self.nk)
        return obj_r

    def r_to_k(self,obj_r):
        """ Fourier transform from real space to k-space """
        obj_r = obj_r.reshape(-1, self.nk1, self.nk2)
        obj_k = np.fft.fftn(obj_r,axes=(1,2))
        obj_k = obj_k.reshape(-1, self.nk)
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
    
    def _lagrange_extrapolation_zero_freq_nth_order(self, xs, ys):
        """
        Routine that evaluates the Lagrange polynomial passing through the points
        xs=[x1, x2, ..., xn+1], ys=[y1, y2, ..., yn+1]. The expected shape of the arguments is
        xs: 1D array of frequencies
        ys: array of datapoints to extrapolate to 0 frequency. If ys is multidimensional,
            it is assumed that the first dimension is the frequency dependence
        """
        val = np.zeros_like(ys[0,...])
        for i in range(ys.shape[0]):
            prod_temp=1
            for j in range(ys.shape[0]):
                if j != i:
                    prod_temp *= -xs[j] / (xs[i] - xs[j])
            val += prod_temp * ys[i,...]
        return val

    def extrapolate_zero_freq(self, obj_wn, n_freqs):
        """
        Routine that uses a Lagrange extrapolation of the n_freqs first
        Matsubara frequencies to extrapolate a fermionic correlation
        function to wn=0.
        """
        # We evaluate the first few frequencies
        indices = np.arange(n_freqs, dtype='int')
        frequencies_interpolation = (2*indices+1)*np.pi*self.T

        evaluated_data = self.get_specific_wn('F', obj_wn, indices)
        
        # We use our routine to evaluate the zero-frequency correlation function
        return self._lagrange_extrapolation_zero_freq_nth_order(frequencies_interpolation, evaluated_data)

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