<h1 align="center">TPySC</h1>
<p align="center">
A Python library that allows the computation of Hubbard model related functions and quantities (such as the self-energy and Green's function) using the Two-Particle-Self-Consistent (TPSC) approach first described in
[<a href="https://arxiv.org/abs/cond-mat/9702188">Vilk and Tremblay, 1997</a>]. See additional references in the documentation.
</p>

## Table of contents

- [Installation](#installation)
- [Documentation and tutorials](#documentation-and-tutorials)
- [Examples](#examples)
- [Tests](#tests)
- [Licence and citation](#license-and-citation)


## Installation

This package will be soon available on PyPI.
Meanwhile, you can install it by cloning this repository and using pip:

```bash
git clone https://github.com/Groupe-Tremblay/TPySC.git
cd TPySC
pip install .
```

## Documentation and tutorials

See the [online documentation](https://groupe-tremblay.github.io/TPySC/).

Documentation can also be build locally with the sources located in the ``docs`` folder.
To build the documentation locally:

```bash
pip install .[docs]
cd docs
make html
```

You can access the documentation in your browser by opening ``docs/build/html/index.html``.


## Examples

The `examples/` directory contains some examples for you to experiment with and get familiar with different use cases.

A calculation is set up by building a mesh, computing a dispersion on it, and
solving:

```python
import tpysc
import tpysc.dispersions

mesh = tpysc.Mesh2D(T=0.1, nk1=64, wmax=8, IR_tol=1e-12)
dispersion = tpysc.dispersions.calcDispersion2DSquare(mesh, t=1, tp=1, tpp=0)
solver = tpysc.Tpsc(mesh, dispersion)
results = solver.solve(n=1, U=2.0)
```

The returned dictionary can then be post-processed in the same script, for
instance to plot observables.

## Tests

The ``tests`` folder provides automated tests that can be run to ensure TPSC is behaving correctly on your system.
To run the tests:

```bash
pip install .[tests]
cd tests
pytest
```

Every test should pass.
If not, please create an issue with the output of the above code and a description of your system.


## License and citation

This software is released under the MIT License. See [LICENSE](LICENSE) for details.

If you find the intermediate representation, sparse sampling, or this software useful in your research, please consider citing the following papers:

Hiroshi Shinaoka et al., Phys. Rev. B 96, 035147 (2017)
Jia Li et al., Phys. Rev. B 101, 035144 (2020)
Markus Wallerberger et al., SoftwareX 21, 101266 (2023)
If you are discussing sparse sampling in your research specifically, please also consider citing an independently discovered, closely related approach, the MINIMAX isometry method (Merzuk Kaltak and Georg Kresse, Phys. Rev. B 101, 205145, 2020).