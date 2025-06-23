# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

_HERE = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PACKAGE_DIR = os.path.abspath(os.path.join(_HERE, "..", "../RenderMolecules/"))
_SUBPACKAGE_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "../RenderMolecules/src/RenderMolecules")
)

sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _PACKAGE_DIR)
sys.path.insert(0, _SUBPACKAGE_DIR)

import RenderMolecules

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RenderMolecules"
copyright = "2025, Tobias Dijkhuis"
author = "Tobias Dijkhuis"
release = "0.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]
autodoc_mock_imports = ["bmesh", "bpy"]
autodoc_member_order = ["bysource"]

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
