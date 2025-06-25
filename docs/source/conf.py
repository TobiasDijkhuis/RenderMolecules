# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "render_molecules"
copyright = "2025, Tobias Dijkhuis"
author = "Tobias Dijkhuis"
release = "0.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = ["bmesh", "bpy"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}

# Insert both class docstring and __init__ docstring into documentation of class
autoclass_content = "both"

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_show_sourcelink = False
