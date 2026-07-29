import os
import sys
from datetime import datetime

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

project = "FGPT"
html_title = "FGPT Documentation"

copyright = (
    f"{datetime.now().year}, "
    "S. Sivanesan, K. Ardaneh / CNRS / IPSL / Sorbonne University"
)

author = "Shivamshan Sivanesan, Kazem Ardaneh"

release = "0.1.0"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
]

# Mock only optional/heavy dependencies
autodoc_mock_imports = [
    "rich",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "fparser": ("https://fparser.readthedocs.io/en/stable/", None),
    "rich": ("https://rich.readthedocs.io/en/stable/", None),
}


suppress_warnings = [
    "intersphinx",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
