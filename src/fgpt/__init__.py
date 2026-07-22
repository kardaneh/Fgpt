# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""FGPT: Fortran General Purpose Transpiler and Transformer."""

__author__ = "Shivamshan SIVANESAN, Kazem ARDANEH"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
__copyright__ = "2026, CNRS / IPSL / Sorbonne University"
__description__ = "Fortran source analysis, transpilation, and JAX conversion toolkit for automatic differentiation"


from fgpt.autodiff import AutoDiff
from fgpt.core.backends import JaxConverter
from fgpt.core.common import Logger
from fgpt.core.frontend import Extractor, Navigator, Processor
from fgpt.core.transpiler import F2NP, Transformer
from fgpt.isolator import Isolator
from fgpt.version import __version__, __version_info__, get_version

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
    "__author__",
    "__license__",
    "__copyright__",
    "F2NP",
    "Isolator",
    "Extractor",
    "Logger",
    "Navigator",
    "Processor",
    "Transformer",
    "JaxConverter",
    "AutoDiff",
]
