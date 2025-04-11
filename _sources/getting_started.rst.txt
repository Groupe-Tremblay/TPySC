.. getting_started

Getting started
===============

Installation
------------

TPSC will be soon available on PyPi. For the time being, it can be installed by cloning this repository and running ``pip`` locally:

.. code-block:: bash

    git clone https://github.com/amstremblay/TPSC
    cd TPSC
    pip install .
    
This method installs both the TPSC Python library and the ``TPSC`` executable to be called from the command line.


Use
---

From the command line
#####################

TPSC calculation can be run directly from the command line with parameters supplied in a JSON file.

.. code-block:: bash

    TPSC parameters.json
    
Where the JSON file contains the following data:

.. code-block:: json

    {
        "dispersion_scheme" : "square",
        "t" : 1,
        "tp" : 1,
        "tpp" : 0,
        "T" : 0.1,
        "U" : 2.0,
        "n" : 1,
        "nkx" : 64,
        "wmax_mult" : 1.25,
        "IR_tol" : 1e-12
    }

See :meth:`TPSC.TPSC` for description of the input parameters.


Python library
##############

See the :doc:`user_guide` to learn more about the Python interface.
