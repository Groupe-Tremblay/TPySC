import numpy as np
import pytest

from tpysc.dispersions import calcDispersion2DSquare
from tpysc.mesh import Mesh2D
from tpysc.tpsc import Tpsc


@pytest.fixture
def solved_chi1(mesh_nk1_8: Mesh2D) -> np.ndarray:
    """
    :return: ``chi1(q, iqn)`` for ``t=1, tp=tpp=0`` at half filling
        (``n=1``), the configuration the plan's ``calc_chi1`` checks target.
    :rtype: numpy.ndarray
    """
    dispersion = calcDispersion2DSquare(mesh_nk1_8, t=1., tp=0., tpp=0.)
    obj = Tpsc(mesh_nk1_8, dispersion)
    obj.calc_g1(n=1.)
    obj.calc_chi1()
    return obj.chi1


def test_chi1_real_at_iqn_zero(mesh_nk1_8: Mesh2D, solved_chi1: np.ndarray) -> None:
    """
    ``chi1`` is real to within numerical noise at ``iqn = 0``.
    """
    assert np.max(np.abs(solved_chi1[mesh_nk1_8.iw0_b].imag)) < 1e-10


def test_chi1_symmetric_under_q_negation(mesh_nk1_8: Mesh2D, solved_chi1: np.ndarray) -> None:
    """
    ``chi1`` is symmetric under ``q -> -q``.
    """
    idx = (-np.arange(mesh_nk1_8.nk1)) % mesh_nk1_8.nk1

    assert np.allclose(solved_chi1[:, np.ix_(idx, idx)[0], np.ix_(idx, idx)[1]], solved_chi1)


def test_chi1_peak_at_pi_pi(mesh_nk1_8: Mesh2D, solved_chi1: np.ndarray) -> None:
    """
    ``chi1``'s maximum sits at ``(pi, pi)`` for ``t=1, tp=0, n=1``.
    """
    peak = np.unravel_index(solved_chi1[mesh_nk1_8.iw0_b].real.argmax(), solved_chi1[mesh_nk1_8.iw0_b].shape)

    assert peak == (mesh_nk1_8.nk1 // 2, mesh_nk1_8.nk1 // 2)


def test_chi1_trace_sum_rule(mesh_nk1_8: Mesh2D, solved_chi1: np.ndarray) -> None:
    """
    The trace sum rule ``trace('B', chi1) == n - n**2/2`` holds at
    half filling. ``chi1`` has no ``U`` dependence, so this is a genuine
    physical identity on ``chi1`` alone rather than something specific to
    ``U = 0``.
    """
    n = 1.

    assert mesh_nk1_8.trace('B', solved_chi1) == pytest.approx(n - n**2 / 2, abs=1e-10)
