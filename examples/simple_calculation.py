import logging

import tpysc

# Enable console logging to see the progress of the calculation.
tpysc.enable_console_logging(level=logging.INFO)

# Pack the TPSC input parameters into dictionaries
parameters = {
    "mesh": {
        "T": 0.1,          # Temperature
        "nk1": 64,         # Number of k-points in one space direction
        "wmax": 8,         # For IR basis
        "IR_tol": 1e-12,   # For IR basis, tolerance of intermediate representation
    },
    "dispersion": {
        "t": 1,            # First neighbour hopping
        "tp": 1,           # Second neighbour hopping
        "tpp": 0,          # Third neighbour hopping
    },
    "tpsc": {
        "U": 2.0,          # On-site Hubbard interaction strength
        "n": 1,            # Electron filling (density per site)
    },
}

mesh = tpysc.Mesh2D(**parameters["mesh"])
dispersion = tpysc.dispersions.calcDispersion2DSquare(mesh, **parameters["dispersion"])

tpsc = tpysc.Tpsc(mesh, dispersion)
tpsc.solve(**parameters["tpsc"])

print(tpsc)
tpsc.writeResultsJSON("main_results.json")
