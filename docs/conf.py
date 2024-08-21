# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../NeuralTSNE"))


project = "NeuralTSNE"
copyright = "2024, Patryk Tajs"
author = "Patryk Tajs"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

smv_branch_whitelist = r"^v[0-9]+\.[0-9]+\.[0-9]+(\.dev[0-9]+)?$"
smv_released_pattern = r"^heads/v[0-9]+\.[0-9]+\.[0-9]+$"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_sidebars = {
    "**": [
        "versioning.html",
    ]
}
