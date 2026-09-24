from pathlib import Path
import pytest

from tpysc.mesh import Mesh2D

REF_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def ref_tpsc_path() -> Path:
    """
    :return: Path to the golden-test reference file for :class:`tpysc.tpsc.Tpsc`.
    :rtype: Path
    """
    return REF_DIR / "ref_tpsc.json"


@pytest.fixture(scope="session")
def ref_tpscplus_path() -> Path:
    """
    :return: Path to the golden-test reference file for :class:`tpysc.tpscplus.TpscPlus`.
    :rtype: Path
    """
    return REF_DIR / "ref_tpscplus_lahaie.json"


@pytest.fixture(scope="session")
def mesh_nk1_8() -> Mesh2D:
    """
    Small session-scoped k-grid for fast unit tests.

    :return: An 8x8 k-grid at ``T=0.5`` with ``wmax=8``.
    :rtype: Mesh2D
    """
    return Mesh2D(nk1=8, T=0.5, wmax=8)


@pytest.fixture(scope="session")
def mesh_nk1_16() -> Mesh2D:
    """
    Small session-scoped k-grid for fast unit tests.

    :return: A 16x16 k-grid at ``T=0.5`` with ``wmax=8``.
    :rtype: Mesh2D
    """
    return Mesh2D(nk1=16, T=0.5, wmax=8)
