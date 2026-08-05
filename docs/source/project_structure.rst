Project Structure
=================

The FGPT package is organized as follows. The ``core/`` subpackage mirrors
a compiler pipeline — ``frontend/`` (parsing + symbol resolution),
``analysis/`` (shared static analysis), ``passes/`` (IR-to-IR
transformations), ``transpiler/`` (Fortran → NumPy middle-end), ``backends/``
(target-specific code generation, currently JAX), and ``support/``
(cross-cutting infrastructure) — see :doc:`architecture` for the full
pipeline description.

.. code-block:: text

   fgpt/
   ├── src/
   │   └── fgpt/
   │       ├── __init__.py
   │       ├── __main__.py
   │       ├── cli.py                     # Command-line interface
   │       ├── version.py                 # Package version
   │       ├── isolator.py                # Fortran isolation pipeline
   │       ├── autodiff.py                # JAX/Tapenade conversion pipeline
   │       │
   │       ├── core/
   │       │   ├── frontend/
   |       |   |   ├── __init__.py
   │       │   │   ├── processor.py       # Fortran parser (fparser wrapper)
   │       │   │   ├── extractor.py       # Static analysis and metadata extraction
   │       │   │   └── navigator.py       # Cross-module symbol resolution
   │       │   │
   │       │   ├── analysis/
   |       |   |   ├── __init__.py
   │       │   │   └── shaper.py          # Array shape/dimension analysis
   │       │   │
   │       │   ├── passes/
   |       |   |   ├── __init__.py
   │       │   │   └── modifier.py        # Fortran AST transformation passes
   │       │   │
   │       │   ├── transpiler/
   |       |   |   ├── __init__.py
   │       │   │   ├── transformer.py     # Fortran → Python pipeline
   │       │   │   ├── f2np.py            # Statement/expression-level translation
   │       │   │   └── intrinsic.py       # Fortran intrinsic → NumPy mapping
   │       │   │
   │       │   ├── backends/
   |       |   |   ├── __init__.py
   |       |   |   ├── utils.py                  # Shared helper functions used across backend modules
   |       |   |   └── jax_converter/
   |       |   |       ├── converter.py          # Main entry point: orchestrates conversion of code into JAX representations
   |       |   |       ├── analysis.py           # Static/dynamic analysis utilities (shape inference, dependency tracking, etc.)
   |       |   |       ├── array_updates.py      # Handles array mutation patterns and converts them to JAX-safe updates
   |       |   |       ├── call_rewriting.py     # Rewrites function calls into JAX-compatible primitives or transformations
   |       |   |       ├── conditionals.py       # Transforms if/else logic into JAX control-flow primitives (e.g., lax.cond)
   |       |   |       ├── dynamic_loops.py      # Deals with loops whose bounds depend on runtime values (dynamic control flow)
   |       |   |       ├── loops.py              # Handles static/structured loop transformations
   |       |   |       ├── masking.py            # Implements masking strategies for conditional execution without branching
   |       |   |       ├── scope_utils.py        # Utilities for managing variable scope during transformation/rewrite passes
   |       |   |       └── vectorization.py      # Converts scalar functions into vectorized versions
   │       │   └── common/
   |       |       ├── __init__.py
   │       │       ├── executive.py       # Workflow orchestration
   │       │       ├── logger.py          # Logging infrastructure
   │       │       ├── line_length.py     # Fortran line-length utilities
   │       │       └── utils.py           # Shared helper utilities
   │       │
   │       └── templates/
   │           └── default.yaml
   ├── tests/                    # Unit tests
   │   ├── conftest.py
   │   ├── test_autodiff.py
   │   ├── test_extractor.py
   │   ├── test_f2np.py
   │   ├── test_intrinsic.py
   │   ├── test_jaxconverter.py
   │   ├── test_jax_utils.py
   │   ├── test_navigator.py
   │   ├── test_processor.py
   │   ├── test_shaper.py
   │   ├── test_utils.py
   │   └── test_transformer.py
   |
   ├── notebooks/                         # Example notebooks, tutorials, and development prototypes
   │   ├── autodiff_principles.ipynb      # Introduction to JVP and VJP concepts
   │   ├── prototype.ipynb                # Experimental notebook with autodifferenciation
   │   ├── fortran_to_numpy.ipynb         # F2NP translation examples
   │   ├── jax_converter.ipynb            # JAX conversion pipeline examples
   │   ├── jax_examples.ipynb             # JAX experiments and demonstrations
   │   └── Test_Transformer.ipynb         # Transformer pipeline examples
   |
   ├── docs/                     # Documentation
   │   ├── source/
   │   └── build/
   ├── .github/workflows/        # CI/CD pipelines
   │   ├── ci.yaml
   │   └── docs.yml
   ├── setup                     # Setup file for transformation
   ├── arch-nvhpc_HAL.env
   ├── arch-nvhpc_LEONARDO.env
   ├── arch-nvhpc_spirit.env
   ├── Makefile                  # Makefile to run the isolated procedures in Fortran
   ├── template.yaml             # Predefined templates
   ├── pyproject.toml            # Package configuration
   ├── README.md                 # Project README
   └── LICENSE                   # CC BY-NC-SA 4.0

Module Descriptions
-------------------

Frontend — ``core/frontend/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``processor.py``
     - Fortran parsing utilities using fparser and AST generation helpers
   * - ``extractor.py``
     - Extracts subroutines, variables, loops, and dependency structures from the Fortran AST
   * - ``navigator.py``
     - Traverses module dependencies and resolves cross-file subroutine calls

Shared Analysis — ``core/analysis/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``shaper.py``
     - Resolves implicit array shapes and reconstructs explicit dimensions across call chains

IR-to-IR Passes — ``core/passes/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``modifier.py``
     - Performs source-to-source transformations for vectorization and GPU adaptation

Middle-End Lowering — ``core/transpiler/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``transformer.py``
     - Core AST transformation utilities for rewriting Fortran-to-Python structures
   * - ``f2np.py``
     - Converts Fortran constructs into NumPy-compatible Python code via AST rewriting
   * - ``intrinsic.py``
     - Defines and normalizes Fortran intrinsic functions and their transformation rules to NumPy

JAX Backend — ``core/backends/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``backends/utils.py``
     - Shared helper functions used across backend modules
   * - ``jax_converter/``
     - Subpackage for JAX-specific code transformation pipeline:

        - ``converter.py`` — main conversion entry point
        - ``analysis.py`` — dependency and usage analysis (shape inference, dependency tracking)
        - ``array_updates.py`` — array mutation rewriting
        - ``call_rewriting.py`` — function and subroutine call conversion
        - ``conditionals.py`` — If/Else transformation rules
        - ``dynamic_loops.py`` — loops with dynamic range
        - ``loops.py`` — loop normalization and rewriting utilities
        - ``masking.py`` — handling masked array operations
        - ``scope_utils.py`` — variable scope resolution utilities
        - ``vectorization.py`` — vectorized execution transformations

Support — ``core/support/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``executive.py``
     - High-level orchestration layer for running FGPT pipelines
   * - ``logger.py``
     - Rich logging system with console progress bars and optional file logging
   * - ``line_length.py``
     - Utilities for managing Fortran line-length constraints and formatting rules
   * - ``utils.py``
     - General-purpose AST utilities, helpers, and transformation functions (``ReplaceGlobals``, ``AdjustIndices``)

Top-Level Drivers
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``cli.py``
     - Command-line interface exposing the ``isolate`` and ``autodiff`` subcommands
   * - ``isolator.py``
     - Frontend driver: separates independent code blocks for modular transformation and analysis
   * - ``autodiff.py``
     - Backend driver: AST-based transformations for JAX compatibility and code rewriting for automatic differentiation
   * - ``version.py``
     - Package versioning and metadata definitions

.. note::

   Earlier revisions of this package exposed a flat ``jax_utils.py`` module
   with utilities such as ``VectorizationAnalyzer``, ``ReductionHandler``,
   ``MaybeAddIndexTransformer``, ``WhileVectorToScalar``, and ``Control``.
   These now live in ``core/backends/jax_converter/analysis.py`` and
   ``core/backends/utils.py``; ``jax_utils.py`` no longer exists as a
   top-level module. The ``test_jax_utils.py`` test file name is a
   historical holdover and exercises these relocated utilities.
