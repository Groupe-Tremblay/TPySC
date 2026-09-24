import numpy as np
import pytest

from tpysc.gf import calcGiwnk
from tpysc.mesh import Mesh2D


def test_trace_flat_band_occupation(mesh_nk1_8: Mesh2D) -> None:
    """
    For a flat band (``eps = 0``) at ``mu = 0``, the trace of ``-G`` at
    ``tau = 0`` gives the free occupation ``f(0) = 0.5`` (half filling, one
    spin channel).
    """
    eps = np.zeros_like(mesh_nk1_8.k1)
    G = calcGiwnk(mesh_nk1_8, eps)

    occupation = mesh_nk1_8.trace('f', -G, 0)

    assert occupation == pytest.approx(0.5, abs=1e-10)


def test_trace_tau_zero_discontinuity(mesh_nk1_8: Mesh2D) -> None:
    """
    ``G(tau=0^+) - G(tau=0^-) = -1`` is the equal-time anticommutator
    discontinuity of a fermionic Green's function.
    """
    eps = np.zeros_like(mesh_nk1_8.k1)
    beta = mesh_nk1_8.IR_basis_set.beta

    for mu in (0., 0.7, -2.1):
        G = calcGiwnk(mesh_nk1_8, eps - mu)
        trace_0 = mesh_nk1_8.trace('f', G, 0)
        trace_beta = mesh_nk1_8.trace('f', G, beta)

        assert trace_0 + trace_beta == pytest.approx(-1., abs=1e-10)


def test_trace_invalid_statistic(mesh_nk1_8: Mesh2D) -> None:
    """
    ``trace`` raises an error on any statistic other than 'f'/'b', and is
    case-insensitive for the valid ones.
    """
    eps = np.zeros_like(mesh_nk1_8.k1)
    G = calcGiwnk(mesh_nk1_8, eps)

    with pytest.raises(ValueError):
        mesh_nk1_8.trace('x', G, 0)

    assert mesh_nk1_8.trace('f', G, 0) == mesh_nk1_8.trace('F', G, 0)
