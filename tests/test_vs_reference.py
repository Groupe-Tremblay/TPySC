import tpysc
import json
import numpy as np


def test_compare():
    """
    A simple test that compare results from the current commit to those of f5dc1ceff8627341739142caea4d4d09678a0b74
    """

    # Run the reference TPSC calculation
    parameters = {
        "dispersion_scheme" : "square",  # Dispersion model
        "t" : 1,                  # First neighbour hopping
        "tp" : 1,                 # Second neighbour hopping
        "tpp" : 0,                # Third neighbour hopping
        "T" : 0.1,                # Temperature
        "U" : 2.0,                # ?
        "n" : 1,                  # Density
        "nkx" : 64,               # Number of k-points in one space direction
        "wmax_mult" : 1.25,       # for IR basis, multiple of bandwidth to use as wmax (must be greater than 1)
        "IR_tol" : 1e-12          # # for IR basis, tolerance of intermediate representation
    }

    obj = tpysc.TPSC(**parameters)
    results = obj.run()
    print("this version:", results)

    # Load the reference results
    with open("ref_f5dc1ce.json", 'r') as reference_filename:
        reference_results = json.load(reference_filename)
    print("reference:", reference_results)

    # Compare
    for key in reference_results.keys():
        # Convert list to complex from JSON input
        if type(reference_results[key]) == list:
            reference_results[key] = reference_results[key][0] + 1.j*reference_results[key][1]
        assert np.allclose(results[key], reference_results[key], rtol=1e-06, atol=1e-08)

