from types import SimpleNamespace

import numpy as np
import pytest

from tpysc.dispersions import calcDispersion2DSquare
from tpysc.mesh import Mesh2D


def test_high_symmetry_points(mesh_nk1_8: Mesh2D) -> None:
    """
    With ``t=1, tp=tpp=0``, the dispersion takes exact known values at the
    high-symmetry points, all of which sit on an ``nk1=8`` grid.
    """
    eps = calcDispersion2DSquare(mesh_nk1_8, t=1., tp=0., tpp=0.).reshape(mesh_nk1_8.nk)

    gamma = mesh_nk1_8.get_ind_kpt(0., 0.)
    corner = mesh_nk1_8.get_ind_kpt(np.pi, np.pi)
    edge = mesh_nk1_8.get_ind_kpt(np.pi, 0.)

    assert eps[gamma] == pytest.approx(-4.)
    assert eps[corner] == pytest.approx(4.)
    assert eps[edge] == pytest.approx(0.)


def test_bandwidth(mesh_nk1_8: Mesh2D) -> None:
    """
    Pure nearest-neighbour hopping gives a bandwidth of ``8*t``.
    """
    t = 1.7
    eps = calcDispersion2DSquare(mesh_nk1_8, t=t, tp=0., tpp=0.)

    assert eps.max() - eps.min() == pytest.approx(8. * t)


def test_c4_symmetry(mesh_nk1_16: Mesh2D) -> None:
    """
    The dispersion is invariant under ``k1 <-> k2`` on the square grid.
    """
    eps = calcDispersion2DSquare(mesh_nk1_16, t=1., tp=0.3, tpp=0.1)

    assert np.allclose(eps, eps.T)


def test_inversion_symmetry(mesh_nk1_16: Mesh2D) -> None:
    """
    The dispersion is invariant under ``k -> -k``, since it is built purely
    out of cosines.
    """
    eps = calcDispersion2DSquare(mesh_nk1_16, t=1., tp=0.3, tpp=0.1)

    idx = (-np.arange(mesh_nk1_16.nk1)) % mesh_nk1_16.nk1
    eps_negated = eps[np.ix_(idx, idx)]

    assert np.allclose(eps_negated, eps)


def test_periodicity(mesh_nk1_8: Mesh2D) -> None:
    """
    The dispersion is periodic under ``k -> k + 2*pi``, i.e. under
    ``k1 -> k1 + 1`` and ``k2 -> k2 + 1`` in the normalized units the mesh
    stores.
    """
    # calcDispersion2DSquare only reads .k1/.k2, and a k -> k + 2*pi shift
    # isn't a point on the actual k-grid, so a plain namespace stands in for
    # a real Mesh2D here.
    shifted_mesh = SimpleNamespace(k1=mesh_nk1_8.k1 + 1., k2=mesh_nk1_8.k2 + 1.)

    eps = calcDispersion2DSquare(mesh_nk1_8, t=1., tp=0.3, tpp=0.1)
    eps_shifted = calcDispersion2DSquare(shifted_mesh, t=1., tp=0.3, tpp=0.1)

    assert np.allclose(eps_shifted, eps)


def test_particle_hole_symmetry(mesh_nk1_8: Mesh2D) -> None:
    """
    For ``tp = tpp = 0``, ``eps(k + Q) == -eps(k)`` with ``Q = (pi, pi)``,
    which is the identity ``mu1 == 0`` at half filling depends on.
    """
    # Same reasoning as test_periodicity: k + Q isn't a grid point, so a
    # namespace exposing the shifted .k1/.k2 is enough to drive the function.
    shifted_mesh = SimpleNamespace(k1=mesh_nk1_8.k1 + 0.5, k2=mesh_nk1_8.k2 + 0.5)

    eps = calcDispersion2DSquare(mesh_nk1_8, t=1., tp=0., tpp=0.)
    eps_shifted = calcDispersion2DSquare(shifted_mesh, t=1., tp=0., tpp=0.)

    assert np.allclose(eps_shifted, -eps)


def test_linearity_in_hoppings(mesh_nk1_8: Mesh2D) -> None:
    """
    ``eps(t, tp, tpp)`` equals ``t*eps(1,0,0) + tp*eps(0,1,0) + tpp*eps(0,0,1)``.
    """
    t, tp, tpp = 1.3, -0.4, 0.2

    eps = calcDispersion2DSquare(mesh_nk1_8, t=t, tp=tp, tpp=tpp)
    eps_t = calcDispersion2DSquare(mesh_nk1_8, t=1., tp=0., tpp=0.)
    eps_tp = calcDispersion2DSquare(mesh_nk1_8, t=0., tp=1., tpp=0.)
    eps_tpp = calcDispersion2DSquare(mesh_nk1_8, t=0., tp=0., tpp=1.)

    assert np.allclose(eps, t * eps_t + tp * eps_tp + tpp * eps_tpp)


def test_shape(mesh_nk1_8: Mesh2D) -> None:
    """
    The output shape matches ``mesh.k1``.
    """
    eps = calcDispersion2DSquare(mesh_nk1_8, t=1., tp=0.3, tpp=0.1)

    assert eps.shape == mesh_nk1_8.k1.shape
