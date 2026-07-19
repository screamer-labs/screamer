import os
import sys
import re

# Add project root to the Python path
sys.path.insert(0, os.path.abspath('../../screamer'))  # Adjust the relative path as necessary


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'screamer'
copyright = '2026, Thijs van den Berg'
author = 'Thijs van den Berg'
# Read the version from the installed package metadata so the docs never drift
# from the release (was hardcoded to a stale 0.1.46).
try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version('screamer')
except Exception:
    release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',  # math rendering
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',  # capture type hints
    'myst_nb',  # markdown (via myst-parser) + .ipynb notebook rendering
    "matplotlib.sphinxext.plot_directive",  # include matplotlib plots
    "sphinx_plotly_directive",  # include plotly plots
    "sphinx_exec_code",  # execute python snippets in docs and show output
    "sphinx_design",     # grid layouts, cards, tabs, badges
    # myst-nb replaces the plain 'myst_parser' extension (it loads myst-parser
    # itself, so listing both would conflict) and renders the
    # docs/notebooks/*.ipynb demo notebooks. It supersedes the abandoned
    # 'nbsphinx'/'nbsphinx_link' that were previously configured here.
]


templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # Internal engineering artifacts under docs/ that are not public
    # documentation: the superpowers specs/plans (design + implementation
    # notes for contributors) and the notebooks' developer README (how to run
    # the .ipynb suite locally). The notebooks themselves are published via the
    # Examples toctree; only their README is excluded.
    'superpowers/**',
    'notebooks/README.md',
]
# Roadmap docs (ROADMAP_*.md) are now wired into the sidebar via a
# Roadmap toctree section in index.rst.

# Configure myst-parser to parse math
myst_enable_extensions = [
    "colon_fence",  # Optional extensions, you can enable more if needed
    "amsmath",      # For parsing LaTeX-style math
    "dollarmath"  # allows using $...$ for inline math and $$...$$ for display math
]

# Generate slug anchors for headings h1-h3 so in-page links like [text](#warmup)
# and [text](#ignore) resolve to the matching heading.
myst_heading_anchors = 3

# Disable docutils "smart quotes": keep hyphens as hyphens (no auto em/en dashes
# from -- / ---), straight quotes, and literal ... instead of an ellipsis glyph.
smartquotes = False

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',     # markdown, parsed by myst-nb (which wraps myst-parser)
    '.ipynb': 'myst-nb',  # render Jupyter notebooks
}

# -----------------------------------------------------------------------------
# myst-nb (notebook execution + rendering)
# -----------------------------------------------------------------------------
# The docs/notebooks/*.ipynb files are committed WITHOUT stored outputs, so
# myst-nb executes them at build time and captures fresh outputs (plots,
# printed values) - the same build-time-execution model that sphinx-exec-code
# and sphinx-plotly-directive already use here. The notebooks are seeded and
# deterministic, so output is stable across rebuilds.
nb_execution_mode = "auto"          # execute notebooks that have no stored outputs
nb_execution_timeout = 120          # seconds per notebook (generous; suite runs in ~10s)
nb_execution_raise_on_error = True  # a broken notebook fails the docs build (docs stay honest)


# -----------------------------------------------------------------------------
# HTML Outpiut options
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# -----------------------------------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']
html_css_files = ['css/custom.css']

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],           # top-level toctree captions become nav items
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],   # right "on this page" sidebar
    "show_nav_level": 1,                        # sidebar shows the current section only
    "navigation_depth": 3,
    "icon_links": [
        {"name": "GitHub",
         "url": "https://github.com/screamer-labs/screamer",
         "icon": "fa-brands fa-github"},
        {"name": "PyPI",
         "url": "https://pypi.org/project/screamer/",
         "icon": "fa-solid fa-box"},
    ],
}

# Single-page sections have no sub-pages, so their left "Section Navigation" would
# be an empty box. Drop the primary sidebar on those pages (full-width prose plus
# the right-hand on-this-page TOC).
html_sidebars = {
    "usage": [],
    "changelog": [],
}

# -----------------------------------------------------------------------------
# Plotly
#  -----------------------------------------------------------------------------
plotly_html_show_source_link = False
plotly_html_show_formats = False

# -----------------------------------------------------------------------------
# Code fragment execution (sphinx-exec-code)
# -----------------------------------------------------------------------------
# Point at the project root so exec_code blocks can `import screamer`.
exec_code_working_dir = '..'
exec_code_folders = ['..']
exec_code_example_dir = '../examples/'

# Known cosmetic warning: at startup the extension scans
# `<project>/src/` for Python packages and warns when none are present.
# Our `src/` directory holds C++ source. There is no supported config
# to disable the scan and its logger sits outside Python's standard
# hierarchy so a logging filter does not catch it. The build still
# succeeds and exec_code blocks render correctly. Two cosmetic warnings
# remain in `make docs` output and we accept them.

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------
# Display the type hints but hide the 'self' argument
autodoc_typehints = "description"  # Shows type hints in the description, not in the signature
autodoc_typehints_format = "short"  # Simplifies the displayed types (e.g., 'numpy.ndarray' instead of the full path)
# The public API is flat (everything imports from the top-level `screamer`), so
# render bare class/function names, not their internal module path.
add_module_names = False
autodoc_default_options = {
    'special-members': '__init__, __call__',  # Include both __call__ and __init__ in the documentation    
}
#autodoc_member_order = 'bysource'

def remove_self_and_simplify_signature(app, what, name, obj, options, signature, return_annotation):
    if signature:
        # Remove 'self: <type>, ' pattern
        signature = re.sub(r'\(self: [^,]+, ', '(', signature)
        signature = re.sub(r'\(self: [^,]+\)', '()', signature)
        
        # Simplify numpy type hints in the signature
        signature = signature.replace('numpy.ndarray[numpy.float64]', 'numpy.ndarray')
    
    if return_annotation:
        # Simplify numpy type hints in the signature
        return_annotation = return_annotation.replace('numpy.ndarray[numpy.float64]', 'numpy.ndarray')    

    return signature, return_annotation

def simplify_type_hints(app, what, name, obj, options, lines):
    """
    This function modifies the docstring, specifically the type annotations
    inside the docstring, and simplifies them for better readability.
    """
    # Go through each line of the docstring and replace verbose types
    for i, line in enumerate(lines):
        # Replace verbose numpy types with simpler ones
        lines[i] = line.replace('numpy.ndarray[numpy.float64]', 'numpy.ndarray')
        # You can add more replacements here as needed


def setup(app):
    app.connect('autodoc-process-signature', remove_self_and_simplify_signature)
    app.connect('autodoc-process-docstring', simplify_type_hints)
    pass