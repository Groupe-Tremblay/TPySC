import numpy as np


def pade(omega_n: list,
			f_n: np.ndarray,
			wmin: float,
			wmax: float,
			nbr_w: int,
			eta:float=0.001,
			epsilon=1e-10):
	"""
    Perform analytical continuation using Padé approximant via continued fractions.

    This function reconstructs a real-frequency spectral function from Matsubara frequency data
    using the Padé approximant method. The algorithm builds a continued fraction representation
    of the function and evaluates it at real frequencies with a small broadening parameter.

    :param omega_n: Matsubara frequencies (imaginary axis frequencies in energy units).
    :type omega_n: list

    :param f_n: Spectral function values at Matsubara frequencies, shape (N,).
    :type f_n: np.ndarray

    :param wmin: Minimum real frequency (in energy units) for the output grid.
    :type wmin: float

    :param wmax: Maximum real frequency (in energy units) for the output grid.
    :type wmax: float

    :param nbr_w: Number of points in the output real frequency grid. Output frequencies are
                  linearly spaced between ``wmin`` and ``wmax``.
    :type nbr_w: int

    :param eta: Broadening parameter (in energy units). Adds an imaginary part to the real
                frequencies to regularize the continuation: :math:`z = \omega + i\eta`.
                Typical values: 0.001-0.01. Default: 0.001.
    :type eta: float, optional

    :param epsilon: Regularization parameter to avoid division by zero in the recurrence relations.
                    Added to denominators when they become too small. Should be much smaller than
                    the smallest expected function value. Default: 1e-10.
    :type epsilon: float, optional

    :returns: Tuple containing:

        - **omega** (np.ndarray): Real frequency grid, shape (nbr_w,). Linearly spaced from
          ``wmin`` to ``wmax``.
        - **P** (np.ndarray): Analytically continued function values at real frequencies,
          shape (nbr_w,). Complex-valued array.

    :rtype: tuple[np.ndarray, np.ndarray]

    :raises ZeroDivisionError: If regularization fails (epsilon too small or data corrupted). # TODO

    .. warning::
        The broadening parameter ``eta`` affects the imaginary part of the output. Smaller values
        give sharper features but may amplify noise; larger values smooth features but reduce
        resolution. Choose based on data quality and desired spectral resolution.
    """
	N = len(omega_n)

	# We only keep rows of the g matrix.
	g_prev = (f_n[0] - f_n) / ((omega_n - omega_n[0]) * 1.j * f_n + epsilon)
	a = np.zeros(N, dtype=complex)
	a[0] = f_n[0]
	a[1] = g_prev[1]
	for k in range(2, N): # TODO I believe this can be compressed even further
		g = (g_prev[k-1] - g_prev) / ((omega_n - omega_n[k-1]) * 1.j * g_prev + epsilon)
		g_prev = g
		a[k] = g[k]

	# Construct the real frequency grid
	omega = np.linspace(wmin, wmax, nbr_w)
	z = omega + 1.j * eta

	# we only keep the previous values of A, B and P.
	A_prev = np.ones(nbr_w, dtype=complex)
	B_prev= 1. + a[1] * (z - 1.j * omega_n[0])
	P = a[0] * A_prev / B_prev # P_1(z)

	# Interpolation procedure.
	for c in range(2, N):
		dz = (z - 1j * omega_n[c-1])
		A_current = 1. + a[c] * dz / A_prev
		B_current = 1. + a[c] * dz / B_prev
		P *= A_current / B_current

		# Update for next iteration.
		A_prev = A_current
		B_prev = B_current

	return omega, P