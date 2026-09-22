import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TPSC'
copyright = '2023, André-Marie Tremblay, Camille Lahaie, Chloé Gauvin-Ndiaye, Jérôme Leblanc, Moïse Rousseau, Nicolas Martin'
author = 'André-Marie Tremblay, Camille Lahaie, Chloé Gauvin-Ndiaye, Jérôme Leblanc, Moïse Rousseau, Nicolas Martin'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.todo']

autodoc_mock_imports = ["scipy", "numpy", "sparse_ir", "matplotlib", "h5py"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
