FGPT
====

FGPT (Fortran General-Purpose Transpiler) is a source-to-source transformation
framework designed to convert Fortran programs into Python representations while
preserving their computational structure and semantics.

The framework provides a robust parsing and transformation pipeline based on
`fparser` and Python's Abstract Syntax Tree (AST), enabling the automatic
translation of legacy scientific Fortran codes into modern Python workflows.
FGPT serves as an intermediate compilation layer that can subsequently target
NumPy, JAX, or other Python-based numerical ecosystems.

GitHub repository: https://github.com/kardaneh/FGPT.git

Key Features
------------

* **Fortran 2003/2008 parsing** through the `fparser` infrastructure
* **AST-based transformation pipeline** for robust source-to-source conversion
* **Control-flow translation** including nested loops and conditional statements
* **Intrinsic handling** for selected Fortran intrinsic procedures
* **Dependency analysis and extraction** of routines and call graphs
* **JAX conversion utilities** for generating differentiable numerical kernels
* **Modular architecture** supporting custom transformation passes
* **Extensible framework** for future optimization and backend generation

.. figure:: ../../images/Flowchart_FGPT.png
   :align: center
   :figwidth: 90%

   **Figure 1:** End-to-end workflow for source-to-source translation and
   automatic differentiation. The Fortran program is transformed into a Python
   AST and subsequently unparsed into Python source code. The translated
   implementation is made compatible with the Equinox and JAX ecosystems,
   allowing automatic differentiation to generate adjoint and tangent linear
   models without manual implementation.

Applications
------------

* Modernization of legacy scientific Fortran codes
* Migration of numerical kernels to Python ecosystems
* Automatic generation of JAX-compatible scientific workflows
* Static analysis of large Fortran code bases
* Compiler and transpiler research
* Scientific software sustainability and reproducibility

Transformation Pipeline
-----------------------

FGPT follows a multi-stage transformation workflow:

1. Parse Fortran source code into an FParser syntax tree.
2. Analyze declarations, symbols, and dependencies.
3. Convert Fortran constructs into Python AST nodes.
4. Preserve control-flow semantics and program structure.
5. Apply semantic transformations and intrinsic replacements.
6. Generate executable Python or JAX-compatible code.

Example Capabilities
--------------------

FGPT supports the translation of complex control-flow structures, including:

* Nested `DO` loops
* Nested `IF / ELSEIF / ELSE` blocks
* Procedure calls and dependency resolution
* Array indexing transformations
* Intrinsic function conversion

The generated code is intended to serve as an intermediate representation that
can later be optimized, vectorized, or lowered to specialized numerical backends.

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    overview
    installation
    quickstart

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    architecture
    transformation
    isolator
    extractor
    jax_conversion
    configuration
    benchmark

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    api/modules

.. toctree::
    :maxdepth: 2
    :caption: Developer Guide

    project_structure
    contributing
    testing

# Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
