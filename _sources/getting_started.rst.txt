.. getting_started

Getting started
===============

Installation
------------

TPSC will be soon available on PyPi. For the time being, it can be installed by cloning this repository and running ``pip`` locally:

.. code-block:: bash

    git clone https://github.com/Groupe-Tremblay/TPySC.git
    cd TPSC
    pip install .


Running a solver
----------------

A calculation is set up by building a mesh, computing a dispersion on it, and
solving:

.. code-block:: python

    import tpysc

    mesh = tpysc.Mesh2D(T=0.1, nk1=64, wmax=8, IR_tol=1e-12)
    dispersion = tpysc.dispersions.calcDispersion2DSquare(mesh, t=1, tp=1, tpp=0)
    solver = tpysc.Tpsc(mesh, dispersion)
    results = solver.solve(n=1, U=2.0)

See the :doc:`user_guide` to learn more about the Python interface.
