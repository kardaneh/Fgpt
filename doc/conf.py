import os
import sys
sys.path.insert(0, os.path.abspath('../Fgpt_package'))

project = 'Fgpt_package'
copyright = '2025, Kardaneh'
author = 'Kardaneh'
release = 'dev'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # supports Google/NumPy docstrings
    'sphinx.ext.viewcode',   # adds links to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
