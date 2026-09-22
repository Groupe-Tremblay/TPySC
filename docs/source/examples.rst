.. examples

Examples
========

A simple calculation
---------------------

This example walks through a minimal TPSC calculation, from setting up the
mesh to writing out the results. Download the full script:
:download:`examples/simple_calculation.py <../../examples/simple_calculation.py>`.

Start by importing ``tpysc`` and, optionally, enabling console logging to
print the progress of the self-consistency loop as it runs:

.. literalinclude:: ../../examples/simple_calculation.py
    :language: python
    :lines: 1-6

Group the TPSC input parameters into three dictionaries: the mesh
(temperature, k-grid resolution and intermediate-representation settings),
the tight-binding dispersion (hopping amplitudes), and the TPSC parameters
themselves (interaction strength and filling):

.. literalinclude:: ../../examples/simple_calculation.py
    :language: python
    :lines: 8-25

Build the mesh and dispersion from those dictionaries: the mesh sets up
the k-grid and the imaginary-time/Matsubara-frequency sampling grids, and
the dispersion evaluates the band energy on it:

.. literalinclude:: ../../examples/simple_calculation.py
    :language: python
    :lines: 27-28

Create a :class:`tpysc.Tpsc` solver from the mesh and dispersion, and run
it to self-consistency:

.. literalinclude:: ../../examples/simple_calculation.py
    :language: python
    :lines: 30-31

Finally, print the solver for a quick summary and write its results to a
JSON file for later post-processing:

.. literalinclude:: ../../examples/simple_calculation.py
    :language: python
    :lines: 33-34
