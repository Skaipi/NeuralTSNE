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
    "sphinx.ext.linkcode",
    "sphinx_multiversion",
    "sphinx_github_style",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

smv_tag_whitelist = None
smv_branch_whitelist = r"^v[0-9]+\.[0-9]+\.[0-9]+(\.dev[0-9]+)?$"
smv_remote_whitelist = r"^(origin|upstream)$"
smv_released_pattern = r"^refs/(heads|remotes/[^/]+)/v[0-9]+\.[0-9]+\.[0-9]+$"
smv_prefer_remote_refs = True

add_module_names = False

# linkcode_url = "https://github.com/Minosyx/WIP_NeuralTSNE"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

default_dark_mode = True
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
