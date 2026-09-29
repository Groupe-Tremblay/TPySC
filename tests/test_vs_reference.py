import json
import numpy as np

from tpysc.dispersions import calcDispersion2DSquare
from tpysc.mesh import Mesh2D
from tpysc.tpsc import Tpsc
from tpysc.tpscplus import TpscPlus


def test_compare_tpsc(ref_tpsc_path):
    """
    A simple test that compare results from the current commit to those of f5dc1ceff8627341739142caea4d4d09678a0b74
    """
    # Run the reference TPSC calculation
    parameters = {
        "mesh": {
            "T" : 0.1,                # Temperature
            "nk1" : 64,               # Number of k-points in one space direction
            "wmax" : 8,               # for IR basis
            "IR_tol" : 1e-12
        },
        "dispersion": {
            "t" : 1,                  # First neighbour hopping
            "tp" : 1,                 # Second neighbour hopping
            "tpp" : 0,                # Third neighbour hopping
        },
        "tpsc": {
            "U" : 2.0,                # Hubbard Interaction
            "n" : 1,                  # Electronic density per unit cell
            },
    }

    mesh = Mesh2D(**parameters["mesh"])
    dispersion = calcDispersion2DSquare(mesh, **parameters["dispersion"])
    obj = Tpsc(mesh, dispersion,)

    results = obj.solve(**parameters["tpsc"])

    # Load the reference results
    with open(ref_tpsc_path, 'r') as reference_filename:
        reference_results = json.load(reference_filename)

    # Compare
    for key in reference_results.keys():
        # Convert list to complex from JSON input
        if type(reference_results[key]) == list:
            reference_results[key] = reference_results[key][0] + 1.j*reference_results[key][1]
        assert np.allclose(results[key], reference_results[key], rtol=1e-06, atol=1e-08)


def test_compare_tpscplus(ref_tpscplus_path):
    """
    A simple test that compare results from the current commit to those of Camille Lahaie's code.
    """
    # Run the reference TPSC+ calculation
    parameters = {
        "mesh": {
            "T" : 0.1,                # Temperature
            "nk1" : 64,               # Number of k-points in one space direction
            "wmax" : 8,               # for IR basis
            "IR_tol" : 1e-12
        },
        "dispersion": {
            "t" : 1,                  # First neighbour hopping
            "tp" : 0,                 # Second neighbour hopping
            "tpp" : 0,                # Third neighbour hopping
        },
        "tpsc": {
            "U" : 2.0,                # Hubbard Interaction
            "n" : 1,                  # Electronic density per unit cell
            },
    }

    mesh = Mesh2D(**parameters["mesh"])
    dispersion = calcDispersion2DSquare(mesh, **parameters["dispersion"])
    obj = TpscPlus(mesh, dispersion)

    results = obj.solve(**parameters["tpsc"])

    # Load the reference results
    with open(ref_tpscplus_path, 'r') as reference_filename:
        reference_results = json.load(reference_filename)

    # Compare
    for key in reference_results.keys():
        # Convert list to complex from JSON input
        if type(reference_results[key]) == list:
            reference_results[key] = reference_results[key][0] + 1.j*reference_results[key][1]
        assert np.allclose(results[key], reference_results[key], rtol=1e-06)